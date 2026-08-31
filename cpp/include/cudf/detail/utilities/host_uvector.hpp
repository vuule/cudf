/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/detail/utilities/cuda.hpp>
#include <cudf/detail/utilities/host_vector.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/span.hpp>

#include <cstddef>
#include <span>
#include <type_traits>
#include <utility>

namespace CUDF_EXPORT cudf {
namespace detail {

/**
 * @brief Whether the host is going to write to a buffer right after allocating it.
 *
 * A resource that offers a host-writable entry point can make the storage safe to write with a
 * wait narrower than a stream synchronization, but only if it is told at allocation time.
 * Buffers that the device fills, and buffers the host may never touch, should stay `deferred` so
 * that they pay nothing.
 */
enum class host_write_intent : bool {
  deferred,  ///< The host will not write before synchronizing, if it writes at all
  immediate  ///< The host will write as soon as the buffer exists
};

/**
 * @brief Host storage with uninitialized elements, allocated from a host memory resource.
 *
 * The host counterpart of `rmm::device_uvector`: allocation and deallocation are ordered on the
 * buffer's stream and elements are never value-initialized.
 *
 * The contract is:
 *
 * > Allocation and deallocation are ordered on the buffer's stream. The host may not access the
 * > storage, for reading or writing, before synchronizing with that stream past the operation it
 * > depends on: the allocation for a write, the fill for a read. Device work enqueued on that
 * > stream may read the storage up to the deallocation point.
 *
 * Host access therefore goes through `host_writable()`, which synchronizes the stream on first
 * use. Handing `data()` to a copy or a kernel on the buffer's stream needs no synchronization.
 *
 * Capacity is fixed at construction: growing would copy the old contents into new storage, which
 * is a host write, and would drag in another synchronization per growth. Call sites that really
 * grow should replace the whole buffer.
 *
 * A buffer belongs to one stream. A buffer allocated on stream A and freed on stream B would free
 * against the wrong event, so the stream cannot be changed after construction.
 *
 * @tparam T The type of the elements, trivially copyable and trivially destructible
 */
template <typename T>
class host_uvector {
  static_assert(std::is_trivially_copyable_v<T> && std::is_trivially_destructible_v<T>,
                "host_uvector only supports trivially copyable, trivially destructible types");

 public:
  using value_type     = T;                      ///< The type of the elements
  using size_type      = std::size_t;            ///< The type used for sizes
  using pointer        = T*;                     ///< Pointer to an element
  using const_pointer  = T const*;               ///< Pointer to a const element
  using iterator       = T*;                     ///< Iterator type
  using const_iterator = T const*;               ///< Const iterator type
  using allocator_type = rmm_host_allocator<T>;  ///< The allocator type

  host_uvector()                               = delete;
  host_uvector(host_uvector const&)            = delete;
  host_uvector& operator=(host_uvector const&) = delete;

  /**
   * @brief Construct a buffer with the given capacity, without initializing its elements.
   *
   * The size of the buffer is set to `capacity`; use `resize_uninitialized(0)` or
   * `make_empty_host_uvector` for the incremental-fill case.
   *
   * @param capacity Number of elements to allocate storage for
   * @param alloc Allocator carrying the memory resource and the buffer's stream
   * @param intent Whether the host will write to the buffer immediately
   */
  host_uvector(size_type capacity,
               allocator_type const& alloc,
               host_write_intent intent = host_write_intent::deferred)
    : _alloc{alloc},
      _host_writable_alloc{intent == host_write_intent::immediate && _alloc.has_host_writable()},
      _data{capacity == 0          ? nullptr
            : _host_writable_alloc ? _alloc.allocate_host_writable(capacity)
                                   : _alloc.allocate_async(capacity)},
      _capacity{capacity},
      _size{capacity},
      // The host-writable path has already performed the minimal wait, and nothing can be in
      // flight on memory that no stream has seen
      _host_synchronized{_host_writable_alloc || !_alloc.is_stream_ordered()}
  {
  }

  host_uvector(host_uvector&& other) noexcept
    : _alloc{other._alloc},
      _host_writable_alloc{other._host_writable_alloc},
      _data{std::exchange(other._data, nullptr)},
      _capacity{std::exchange(other._capacity, 0)},
      _size{std::exchange(other._size, 0)},
      _host_synchronized{other._host_synchronized},
      _device_exposed{other._device_exposed}
  {
  }

  host_uvector& operator=(host_uvector&& other) noexcept
  {
    if (this != &other) {
      deallocate();
      _alloc               = other._alloc;
      _host_writable_alloc = other._host_writable_alloc;
      _data                = std::exchange(other._data, nullptr);
      _capacity            = std::exchange(other._capacity, 0);
      _size                = std::exchange(other._size, 0);
      _host_synchronized   = other._host_synchronized;
      _device_exposed      = other._device_exposed;
    }
    return *this;
  }

  ~host_uvector() { deallocate(); }

  /**
   * @brief Pointer to the underlying storage.
   *
   * No synchronization is performed. The storage may be passed to a copy or a kernel enqueued on
   * this buffer's stream; the host must not dereference it without synchronizing.
   *
   * @return Pointer to the underlying storage
   */
  [[nodiscard]] pointer data() { return _data; }

  /// @copydoc data()
  [[nodiscard]] const_pointer data() const { return _data; }

  /**
   * @brief Number of elements in the buffer
   *
   * @return Number of elements
   */
  [[nodiscard]] size_type size() const { return _size; }

  /**
   * @brief Number of elements the buffer can hold
   *
   * @return Number of elements the allocation can hold
   */
  [[nodiscard]] size_type capacity() const { return _capacity; }

  /**
   * @brief Whether the buffer is empty
   *
   * @return true if the buffer holds no elements
   */
  [[nodiscard]] bool is_empty() const { return _size == 0; }

  /**
   * @brief Change the number of elements without touching their values.
   *
   * @throw cudf::logic_error if `size` exceeds the buffer's capacity
   *
   * @param size The new number of elements, at most `capacity()`
   */
  void resize_uninitialized(size_type size)
  {
    CUDF_EXPECTS(size <= _capacity, "host_uvector cannot grow beyond its capacity");
    _size = size;
  }

  /**
   * @brief Obtain a span the host may write to, synchronizing the stream if necessary.
   *
   * Synchronizes the buffer's stream on first use and remembers that it did, so repeated calls are
   * free. Call sites that only pass the buffer to the device never call this and never
   * synchronize.
   *
   * @return A span over the buffer's elements
   */
  [[nodiscard]] host_span<T> host_writable()
  {
    synchronize();
    return {_data, _size, _alloc.is_device_accessible()};
  }

  /**
   * @brief Make the storage safe for the host to write to, synchronizing the stream if necessary.
   *
   * For callers that hand out raw pointers rather than the span from `host_writable()`.
   */
  void make_host_writable() { synchronize(); }

  /**
   * @brief Declare that the caller has synchronized the buffer's stream.
   *
   * Used by the factories that fill the buffer from the device and synchronize themselves, so that
   * the host can read the result without synchronizing again.
   */
  void mark_stream_synchronized() { _host_synchronized = true; }

  /**
   * @brief Declare that device work on the buffer's stream never accessed the storage.
   *
   * Lets the resource skip the wait for the next host writer of the recycled block entirely. Only
   * call this if the buffer was never the source or destination of a copy and was never read by a
   * kernel; the default assumption is the safe one.
   */
  void mark_host_only() { _device_exposed = false; }

  /**
   * @brief Whether the buffer's stream has been synchronized since the allocation.
   *
   * @return true if host access is known to be safe
   */
  [[nodiscard]] bool is_host_synchronized() const { return _host_synchronized; }

  /**
   * @brief Append an element, synchronizing the stream on first host access.
   *
   * @throw cudf::logic_error if the buffer is already at capacity
   *
   * @param value The value to append
   */
  void push_back(T const& value)
  {
    CUDF_EXPECTS(_size < _capacity, "host_uvector cannot grow beyond its capacity");
    synchronize();
    _data[_size++] = value;
  }

  /**
   * @brief Access an element, synchronizing the stream on first host access.
   *
   * @param index Index of the element
   * @return Reference to the element
   */
  [[nodiscard]] T& operator[](size_type index)
  {
    synchronize();
    return _data[index];
  }

  /// @copydoc operator[](size_type)
  [[nodiscard]] T const& operator[](size_type index) const
  {
    CUDF_EXPECTS(_host_synchronized, "host_uvector read before the stream was synchronized");
    return _data[index];
  }

  /**
   * @brief Iterator to the first element, synchronizing the stream on first host access.
   *
   * @return Iterator to the first element
   */
  [[nodiscard]] iterator begin() { return host_writable().begin(); }

  /**
   * @brief Iterator past the last element, synchronizing the stream on first host access.
   *
   * @return Iterator past the last element
   */
  [[nodiscard]] iterator end() { return host_writable().end(); }

  /**
   * @brief Iterator to the first element
   *
   * @return Iterator to the first element
   */
  [[nodiscard]] const_iterator begin() const { return _data; }

  /**
   * @brief Iterator past the last element
   *
   * @return Iterator past the last element
   */
  [[nodiscard]] const_iterator end() const { return _data + _size; }

  /**
   * @brief Whether the storage is accessible from the device
   *
   * @return true if the memory resource is device-accessible
   */
  [[nodiscard]] bool is_device_accessible() const { return _alloc.is_device_accessible(); }

  /**
   * @brief The stream that this buffer's allocation and deallocation are ordered on
   *
   * @return The buffer's stream
   */
  [[nodiscard]] cuda::stream_ref stream() const { return _alloc.get_stream(); }

  /**
   * @brief Conversion to a span over the buffer's elements
   *
   * @return A span over the buffer's elements
   */
  [[nodiscard]] operator host_span<T const>() const
  {
    return {_data, _size, _alloc.is_device_accessible()};
  }

  /// @copydoc operator host_span<T const>() const
  [[nodiscard]] operator host_span<T>() { return {_data, _size, _alloc.is_device_accessible()}; }

  /**
   * @brief Conversion to a `std::span` over the buffer's elements
   *
   * @return A span over the buffer's elements
   */
  [[nodiscard]] operator std::span<T const>() const noexcept { return {_data, _size}; }

  /// @copydoc operator std::span<T const>() const
  [[nodiscard]] operator std::span<T>() noexcept { return {_data, _size}; }

 private:
  void synchronize()
  {
    if (!_host_synchronized) {
      cudf::detail::sync_stream(_alloc.get_stream());
      _host_synchronized = true;
    }
  }

  void deallocate() noexcept
  {
    if (_data != nullptr) {
      // TODO: a buffer on a resource that is not stream-ordered is freed immediately, so a
      // pending H2D copy out of it becomes a use-after-free once CUDA 13 batch copies defer
      // reading the source. See .agents/notes/host-vector-types.md.
      if (_host_writable_alloc) {
        _alloc.deallocate_host_writable(_data, _capacity, _device_exposed);
      } else {
        _alloc.deallocate_async(_data, _capacity);
      }
      _data = nullptr;
    }
  }

  allocator_type _alloc;
  bool _host_writable_alloc{false};
  pointer _data{nullptr};
  size_type _capacity{0};
  size_type _size{0};
  bool _host_synchronized{false};
  bool _device_exposed{true};
};

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
