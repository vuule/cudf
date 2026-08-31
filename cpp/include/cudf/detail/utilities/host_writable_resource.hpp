/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/utilities/export.hpp>

#include <rmm/aligned.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/stream>

#include <cstddef>
#include <cstdint>

namespace CUDF_EXPORT cudf {
namespace detail {

/**
 * @brief Interface to a host memory resource that can return memory the host may write to
 * immediately.
 *
 * A pinned pool can recycle a block whose previous owner still has a copy in flight on the stream
 * it was freed on, so a host writer must not touch it until that copy has completed. Callers of
 * the ordinary `allocate` satisfy this by synchronizing the whole stream. A resource implementing
 * this interface performs the minimal wait internally instead.
 *
 * This exists because `allocate_host_writable` is not reachable through
 * `rmm::host_async_resource_ref`, which type-erases down to `allocate` and `deallocate`. Rather
 * than teaching the resource ref a new property, libcudf carries a pointer to this interface
 * alongside the erased ref.
 */
class host_writable_resource {
 public:
  host_writable_resource()                                         = default;
  host_writable_resource(host_writable_resource const&)            = delete;
  host_writable_resource& operator=(host_writable_resource const&) = delete;
  virtual ~host_writable_resource()                                = default;

  /**
   * @brief Allocates memory that the calling thread may write to on return.
   *
   * @param stream The stream in which to order this allocation
   * @param bytes The size in bytes of the allocation
   * @param alignment The alignment of the allocation
   * @return Pointer to memory the host may write to immediately
   */
  virtual void* allocate_host_writable(cuda::stream_ref stream,
                                       std::size_t bytes,
                                       std::size_t alignment) = 0;

  /**
   * @brief Deallocates memory returned by `allocate_host_writable`.
   *
   * @param stream The stream in which to order this deallocation
   * @param ptr Pointer to be deallocated
   * @param bytes The size in bytes of the allocation
   * @param alignment The alignment of the allocation
   * @param device_exposed Whether the memory was ever used by device work on `stream`. When false,
   * a later host writer needs no wait at all. True is always safe.
   */
  virtual void deallocate_host_writable(cuda::stream_ref stream,
                                        void* ptr,
                                        std::size_t bytes,
                                        std::size_t alignment,
                                        bool device_exposed) noexcept = 0;
};

/**
 * @brief Returns the host-writable interface for `mr`, or nullptr if it does not have one.
 *
 * Non-null only for libcudf's default pinned pool, and only when
 * `LIBCUDF_HOST_WRITABLE_SYNC_MODE` selects a mode other than `off`. A user-supplied pinned
 * resource, the pageable resource, and the default configuration all return nullptr, which leaves
 * the caller on the stream-synchronizing path.
 *
 * @param mr The host memory resource to query
 * @return The host-writable interface for `mr`, or nullptr
 */
CUDF_EXPORT host_writable_resource* get_host_writable_resource(rmm::host_async_resource_ref mr);

/**
 * @brief Accumulates time a host thread spent blocked making a host allocation writable.
 *
 * Measures the cost the host-writable path exists to remove, directly, rather than inferring it
 * from end-to-end timings that thread scheduling makes noisy.
 *
 * @param nanoseconds Time the calling thread was blocked
 */
CUDF_EXPORT void record_host_alloc_wait(std::uint64_t nanoseconds);

/**
 * @brief Counters describing how the host-writable path behaved, for evaluating the mechanism.
 *
 * Mirrors the resource's own counters so that this header does not depend on the resource's
 * headers.
 */
struct host_writable_stats {
  std::size_t allocations{};          ///< Host-writable allocations
  std::size_t waits{};                ///< Allocations that had to wait
  std::size_t fast_path{};            ///< Allocations that needed no wait at all
  std::size_t query_short_circuit{};  ///< Waits skipped because the event had already completed
  std::size_t event_records{};        ///< Event records made on the deallocate path
  std::size_t events_created{};       ///< CUDA events created for per-block tracking
  std::size_t pool_fallbacks{};       ///< Allocations the pool could not satisfy
};

/**
 * @brief Returns counters describing the pinned pool's host-writable allocations.
 *
 * @return Counters accumulated since the last reset, all zero if the path is disabled
 */
CUDF_EXPORT host_writable_stats get_host_writable_statistics();

/// Resets the counters returned by `get_host_writable_statistics`.
CUDF_EXPORT void reset_host_writable_statistics();

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
