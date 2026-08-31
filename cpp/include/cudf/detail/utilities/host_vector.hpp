/*
 *  SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *  SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/detail/utilities/cuda.hpp>
#include <cudf/detail/utilities/host_writable_resource.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/aligned.hpp>

#include <thrust/host_vector.h>

#include <chrono>
#include <cstddef>
#include <limits>
#include <new>  // for bad_alloc
#include <span>

namespace CUDF_EXPORT cudf {
namespace detail {

/**
 * @brief Get the memory resource to be used for pageable memory allocations.
 *
 * @return Reference to the pageable memory resource
 */
CUDF_EXPORT rmm::host_async_resource_ref get_pageable_memory_resource();

/**
 * @brief Whether allocations and deallocations on `mr` are ordered on the stream they are passed.
 *
 * Only libcudf's pageable resource is known not to be stream-ordered; it allocates with
 * `operator new` and frees immediately, ignoring the stream. Any other resource is assumed to be
 * stream-ordered.
 *
 * @param mr The host memory resource to query
 * @return true if the resource is (assumed to be) stream-ordered
 */
inline bool is_stream_ordered_host_resource(rmm::host_async_resource_ref mr)
{
  return mr != get_pageable_memory_resource();
}

/*! \p rmm_host_allocator is a CUDA-specific host memory allocator
 *  that employs \c a `cudf::host_async_resource_ref` for allocation.
 *
 *  \see https://en.cppreference.com/cpp/memory/allocator
 */
template <typename T>
class rmm_host_allocator;

/*! \p rmm_host_allocator is a CUDA-specific host memory allocator
 *  that employs \c an `cudf::host_async_resource_ref` for allocation.
 *
 *  \see https://en.cppreference.com/cpp/memory/allocator
 */
template <>
class rmm_host_allocator<void> {
 public:
  using value_type      = void;            ///< The type of the elements in the allocator
  using pointer         = void*;           ///< The type returned by address() / allocate()
  using const_pointer   = void const*;     ///< The type returned by address()
  using size_type       = std::size_t;     ///< The type used for the size of the allocation
  using difference_type = std::ptrdiff_t;  ///< The type of the distance between two pointers

  /**
   * @brief converts a `rmm_host_allocator<void>` to `rmm_host_allocator<U>`
   */
  template <typename U>
  struct rebind {
    using other = rmm_host_allocator<U>;  ///< The rebound type
  };
};

/*! \p rmm_host_allocator is a CUDA-specific host memory allocator
 *  that employs \c `cudf::host_async_resource_ref` for allocation.
 *
 * The \p rmm_host_allocator provides an interface for host memory allocation through the user
 * provided \c `cudf::host_async_resource_ref`. The \p rmm_host_allocator does not take ownership of
 * this reference and therefore it is the user's responsibility to ensure its lifetime for the
 * duration of the lifetime of the \p rmm_host_allocator.
 *
 *  \see https://en.cppreference.com/cpp/memory/allocator
 */
template <typename T>
class rmm_host_allocator {
 public:
  using value_type      = T;               ///< The type of the elements in the allocator
  using pointer         = T*;              ///< The type returned by address() / allocate()
  using const_pointer   = T const*;        ///< The type returned by address()
  using reference       = T&;              ///< The parameter type for address()
  using const_reference = T const&;        ///< The parameter type for address()
  using size_type       = std::size_t;     ///< The type used for the size of the allocation
  using difference_type = std::ptrdiff_t;  ///< The type of the distance between two pointers

  using propagate_on_container_move_assignment = cuda::std::true_type;

  /**
   * @brief converts a `rmm_host_allocator<T>` to `rmm_host_allocator<U>`
   */
  template <typename U>
  struct rebind {
    using other = rmm_host_allocator<U>;  ///< The rebound type
  };

  /**
   * @brief Cannot declare an empty host allocator.
   */
  rmm_host_allocator() = delete;

  template <class... Properties>
  using async_host_resource_ref = cuda::mr::resource_ref<cuda::mr::host_accessible, Properties...>;

  /**
   * @brief Construct from a `cudf::host_async_resource_ref`
   */
#ifdef __CUDACC__
#pragma nv_exec_check_disable
#endif
  template <typename ResourceType>
  rmm_host_allocator(ResourceType _mr, cuda::stream_ref _stream)
    : mr(std::move(_mr)),
      stream(std::move(_stream)),
      _is_device_accessible{
        cuda::mr::synchronous_resource_with<ResourceType, cuda::mr::device_accessible>},
      _is_stream_ordered{is_stream_ordered_host_resource(mr)},
      _host_writable{get_host_writable_resource(mr)}
  {
  }

#ifdef __CUDACC__
#pragma nv_exec_check_disable
#endif
  rmm_host_allocator(rmm_host_allocator const&) = default;

#ifdef __CUDACC__
#pragma nv_exec_check_disable
#endif
  rmm_host_allocator(rmm_host_allocator&&) = default;

#ifdef __CUDACC__
#pragma nv_exec_check_disable
#endif
  rmm_host_allocator& operator=(rmm_host_allocator const&) = default;

#ifdef __CUDACC__
#pragma nv_exec_check_disable
#endif
  rmm_host_allocator& operator=(rmm_host_allocator&&) = default;

  /**
   * @brief This method allocates storage for objects in host memory.
   *
   *  @param cnt The number of objects to allocate.
   *  @return a \c pointer to the newly allocated objects.
   *  @note This method does not invoke \p value_type's constructor.
   *        It is the responsibility of the caller to initialize the
   *        objects at the returned \c pointer.
   */
  inline pointer allocate(size_type cnt)
  {
    // The resource can make the memory safe to write with a wait narrower than a stream sync
    if (_host_writable != nullptr) { return allocate_host_writable(cnt); }

    auto const result = allocate_async(cnt);
    // A stream-ordered resource can hand back a block whose previous owner still has work in
    // flight on this stream, so the host must not write to it before synchronizing. Resources that
    // are not stream-ordered return memory that no stream can be referencing.
    if (_is_stream_ordered) {
      auto const start = std::chrono::steady_clock::now();
      cudf::detail::sync_stream(stream);
      record_host_alloc_wait(
        static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                     std::chrono::steady_clock::now() - start)
                                     .count()));
    }
    return result;
  }

  /**
   * @brief Allocates storage the host may write to on return, using the resource's host-writable
   * entry point.
   *
   * Only valid when `has_host_writable()`. Storage obtained this way must be released with
   * `deallocate_host_writable`.
   *
   * @param cnt The number of objects to allocate
   * @return a \c pointer to the newly allocated objects
   */
  inline pointer allocate_host_writable(size_type cnt)
  {
    if (cnt > this->max_size()) { throw std::bad_alloc(); }  // end if
    auto const start = std::chrono::steady_clock::now();
    auto* const result =
      _host_writable->allocate_host_writable(stream, cnt * sizeof(value_type), alignof(value_type));
    record_host_alloc_wait(static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start)
        .count()));
    return static_cast<pointer>(result);
  }

  /**
   * @brief Deallocates storage obtained from `allocate_host_writable`.
   *
   * @param p A \c pointer to the previously allocated memory
   * @param cnt Number of objects that were previously allocated
   * @param device_exposed Whether device work on the stream ever accessed the storage
   */
  inline void deallocate_host_writable(pointer p, size_type cnt, bool device_exposed) noexcept
  {
    _host_writable->deallocate_host_writable(
      stream, p, cnt * sizeof(value_type), alignof(value_type), device_exposed);
  }

  /**
   * @brief Deallocates storage obtained from `allocate_async`.
   *
   * Unlike `deallocate`, this never routes to the host-writable entry point, so callers that mix
   * the two allocation paths can match each one explicitly.
   *
   * @param p A \c pointer to the previously allocated memory
   * @param cnt Number of objects that were previously allocated
   */
  inline void deallocate_async(pointer p, size_type cnt) noexcept
  {
    mr.deallocate(stream, p, cnt * sizeof(value_type), alignof(value_type));
  }

  /**
   * @brief Whether the resource offers a host-writable entry point.
   *
   * @return true if `allocate_host_writable` may be used
   */
  [[nodiscard]] bool has_host_writable() const { return _host_writable != nullptr; }

  /**
   * @brief Allocates uninitialized storage without synchronizing the stream.
   *
   * The host may not access the returned storage before synchronizing with the allocator's stream.
   *
   * @param cnt The number of objects to allocate
   * @return a \c pointer to the newly allocated objects
   */
  inline pointer allocate_async(size_type cnt)
  {
    if (cnt > this->max_size()) { throw std::bad_alloc(); }  // end if
    return static_cast<pointer>(mr.allocate(stream, cnt * sizeof(value_type), alignof(value_type)));
  }

  /**
   * @brief This method deallocates host memory previously allocated
   *  with this \c rmm_host_allocator.
   *
   *  @param p A \c pointer to the previously allocated memory.
   *  @param cnt Number of objects that were previously allocated.
   *  @note This method does not invoke \p value_type's destructor.
   *        It is the responsibility of the caller to destroy
   *        the objects stored at \p p.
   */
  inline void deallocate(pointer p, size_type cnt) noexcept
  {
    // `allocate` used the host-writable entry point, so the matching deallocate must too
    if (_host_writable != nullptr) {
      deallocate_host_writable(p, cnt, true);
      return;
    }
    mr.deallocate(stream, p, cnt * sizeof(value_type), alignof(value_type));
  }

  /**
   * @brief This method returns the maximum size of the \c cnt parameter
   *  accepted by the \p allocate() method.
   *
   *  @return The maximum number of objects that may be allocated
   *          by a single call to \p allocate().
   */
  [[nodiscard]] constexpr inline size_type max_size() const
  {
    return (std::numeric_limits<size_type>::max)() / sizeof(T);
  }

  /**
   * @brief This method tests this \p rmm_host_allocator for equality to
   *  another.
   *
   *  @param x The other \p rmm_host_allocator of interest.
   *  @return This method always returns \c true.
   */
  inline bool operator==(rmm_host_allocator const& x) const
  {
    return x.mr == mr && x.stream == stream;
  }

  /**
   * @brief This method tests this \p rmm_host_allocator for inequality
   *  to another.
   *
   *  @param x The other \p rmm_host_allocator of interest.
   *  @return This method always returns \c false.
   */
  inline bool operator!=(rmm_host_allocator const& x) const { return !operator==(x); }

  [[nodiscard]] bool is_device_accessible() const { return _is_device_accessible; }

  /**
   * @brief Whether the underlying resource orders its allocations on the stream.
   *
   * @return true if the resource is stream-ordered
   */
  [[nodiscard]] bool is_stream_ordered() const { return _is_stream_ordered; }

  /**
   * @brief The stream that this allocator's allocations are ordered on.
   *
   * @return The allocator's stream
   */
  [[nodiscard]] cuda::stream_ref get_stream() const { return stream; }

 private:
  rmm::host_async_resource_ref mr;
  cuda::stream_ref stream;
  bool _is_device_accessible;
  bool _is_stream_ordered;
  host_writable_resource* _host_writable;
};

/**
 * @brief A vector class with rmm host memory allocator
 */
template <typename T>
class host_vector : public thrust::host_vector<T, rmm_host_allocator<T>> {
 public:
  using base = thrust::host_vector<T, rmm_host_allocator<T>>;

  host_vector(rmm_host_allocator<T> const& alloc) : base(alloc) {}

  host_vector(size_t size, rmm_host_allocator<T> const& alloc) : base(size, alloc) {}

  [[nodiscard]] operator host_span<T const>() const
  {
    return host_span<T const>{
      base::data(), base::size(), base::get_allocator().is_device_accessible()};
  }

  [[nodiscard]] operator host_span<T>()
  {
    return host_span<T>{base::data(), base::size(), base::get_allocator().is_device_accessible()};
  }

  [[nodiscard]] operator std::span<T const>() const noexcept
  {
    return std::span<T const>(base::data(), base::size());
  }

  [[nodiscard]] operator std::span<T>() noexcept
  {
    return std::span<T>(base::data(), base::size());
  }
};

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
