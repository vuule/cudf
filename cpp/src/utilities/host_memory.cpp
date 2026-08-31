/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/utilities/getenv_or.hpp>
#include <cudf/detail/utilities/host_writable_resource.hpp>
#include <cudf/logger.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/pinned_memory.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/detail/aligned.hpp>
#include <rmm/mr/host_writable.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/stream_ref>

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_set>

namespace cudf {

namespace {

/// Size distribution of host allocations, for judging how much the allocation path can possibly be
/// worth. Buckets are powers of two.
struct size_histogram {
  static constexpr std::size_t num_buckets{40};
  std::array<std::atomic<std::size_t>, num_buckets> counts{};
  std::atomic<std::size_t> total_bytes{0};

  void record(std::size_t bytes)
  {
    auto bucket = std::size_t{0};
    while ((std::size_t{1} << bucket) < bytes && bucket + 1 < num_buckets) {
      ++bucket;
    }
    ++counts[bucket];
    total_bytes += bytes;
  }

  void print(std::FILE* out, char const* label) const
  {
    std::size_t total_count{0};
    for (auto const& count : counts) {
      total_count += count.load();
    }
    std::fprintf(
      out, "%s_allocations: count=%zu total_bytes=%zu\n", label, total_count, total_bytes.load());
    for (std::size_t bucket = 0; bucket < num_buckets; ++bucket) {
      auto const count = counts[bucket].load();
      if (count == 0) { continue; }
      std::fprintf(out, "  <=2^%-2zu %12zu\n", bucket, count);
    }
  }
};

/// Allocations made through the pageable resource, which is the default for every host allocation.
size_histogram& pageable_size_histogram()
{
  static size_histogram histogram;
  return histogram;
}

/// Time host threads spent blocked making an allocation writable, and how many times.
struct host_wait_totals {
  std::atomic<std::uint64_t> nanoseconds{0};
  std::atomic<std::uint64_t> count{0};
};

host_wait_totals& host_wait_accumulator()
{
  static host_wait_totals totals;
  return totals;
}

// Inlined from RMM internals after public MR definitions moved to source files:
// https://github.com/rapidsai/rmm/pull/2416
void* aligned_host_allocate(std::size_t bytes, std::size_t alignment)
{
  assert(rmm::is_supported_alignment(alignment));

  // allocate memory for bytes, plus potential alignment correction,
  // plus store of the correction offset
  std::size_t padded_allocation_size{bytes + alignment + sizeof(std::ptrdiff_t)};
  char* const original = static_cast<char*>(::operator new(padded_allocation_size));

  // account for storage of offset immediately prior to the aligned pointer
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  void* aligned{original + sizeof(std::ptrdiff_t)};

  // std::align modifies `aligned` to point to the first aligned location
  std::align(alignment, bytes, aligned, padded_allocation_size);

  // Compute the offset between the original and aligned pointers
  std::ptrdiff_t const offset = static_cast<char*>(aligned) - original;

  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  *(static_cast<std::ptrdiff_t*>(aligned) - 1) = offset;

  return aligned;
}

void aligned_host_deallocate(void* ptr,
                             [[maybe_unused]] std::size_t bytes,
                             [[maybe_unused]] std::size_t alignment) noexcept
{
  assert(rmm::is_supported_alignment(alignment));

  if (ptr != nullptr) {
    // Get offset from the location immediately prior to the aligned pointer
    // NOLINTNEXTLINE
    std::ptrdiff_t const offset = *(reinterpret_cast<std::ptrdiff_t*>(ptr) - 1);

    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    void* const original = static_cast<char*>(ptr) - offset;

    ::operator delete(original);
  }
}

class pinned_pool_with_fallback_memory_resource {
  using upstream_mr    = rmm::mr::pinned_host_memory_resource;
  using host_pooled_mr = rmm::mr::pool_memory_resource;

  struct fallback_state {
    mutable std::shared_mutex mutex;
    std::unordered_set<void*> allocations;
    std::atomic<std::size_t> count{0};  ///< Allocations the pool could not satisfy
  };

  void record_size(std::size_t bytes) const { sizes_->record(bytes); }

 private:
  upstream_mr upstream_mr_{};
  size_t initial_pool_size_{0};
  size_t max_pool_size_{0};
  // Raw pointer to avoid a segfault when the pool is destroyed on exit
  host_pooled_mr* pool_{nullptr};

  // Wrapped in shared_ptr so the outer class is copyable (required by any_resource)
  std::shared_ptr<fallback_state> fallback_{std::make_shared<fallback_state>()};
  std::shared_ptr<size_histogram> sizes_{std::make_shared<size_histogram>()};

 public:
  pinned_pool_with_fallback_memory_resource(size_t initial_size, size_t max_size)
    :  // rmm requires the pool size to be a multiple of 256 bytes
      initial_pool_size_{rmm::align_up(initial_size, rmm::CUDA_ALLOCATION_ALIGNMENT)},
      max_pool_size_{rmm::align_up(max_size, rmm::CUDA_ALLOCATION_ALIGNMENT)},
      pool_{new host_pooled_mr(upstream_mr_, initial_pool_size_, max_pool_size_)}
  {
    CUDF_LOG_INFO(
      "Pinned pool initial size = %zu, max size = %zu", initial_pool_size_, max_pool_size_);
  }

  // clang-tidy will complain about this function because it is completely
  // unused at runtime and only exist for tag introspection by CCCL, so we
  // ignore linting. This masks a real issue if we ever want to compile with
  // clang, though, which is that the function will actually be compiled out by
  // clang. If cudf were ever to try to support clang as a compile we would
  // need to force the compiler to emit this symbol. The same goes for the
  // other get_property definitions in this file.
  friend void get_property(pinned_pool_with_fallback_memory_resource const&,  // NOLINT
                           cuda::mr::device_accessible) noexcept
  {
  }

  friend void get_property(pinned_pool_with_fallback_memory_resource const&,  // NOLINT
                           cuda::mr::host_accessible) noexcept
  {
  }

  void* allocate_sync(std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    return allocate(cuda::stream_ref{cudaStream_t{nullptr}}, bytes, alignment);
  }

  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    deallocate(cuda::stream_ref{cudaStream_t{nullptr}}, ptr, bytes, alignment);
  }

  void* allocate(cuda::stream_ref stream,
                 std::size_t bytes,
                 std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    record_size(bytes);
    if (max_pool_size_ == 0) { return upstream_mr_.allocate(stream, bytes, alignment); }

    try {
      return pool_->allocate(stream, bytes, alignment);
    } catch (...) {
      CUDF_LOG_INFO("Pinned pool exhausted, falling back to new pinned allocation for %zu bytes",
                    bytes);
      ++fallback_->count;
      // fall back to upstream
      auto* ptr = upstream_mr_.allocate(stream, bytes, alignment);

      {
        std::unique_lock lock(fallback_->mutex);
        fallback_->allocations.insert(ptr);
      }

      return ptr;
    }
  }

  void deallocate(cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    if (max_pool_size_ == 0) {
      upstream_mr_.deallocate(stream, ptr, bytes, alignment);
      return;
    }

    bool is_fallback{false};
    {
      std::shared_lock lock(fallback_->mutex);
      is_fallback = fallback_->allocations.find(ptr) != fallback_->allocations.end();
    }

    if (is_fallback) {
      {
        std::unique_lock lock(fallback_->mutex);
        fallback_->allocations.erase(ptr);
      }
      upstream_mr_.deallocate(stream, ptr, bytes, alignment);
    } else {
      pool_->deallocate(stream, ptr, bytes, alignment);
    }
  }

  /**
   * @brief Allocates memory that the host may write to on return.
   *
   * Forwards to the pool's host-writable entry point, which waits only for work still pending on
   * the recycled block rather than for the whole stream. Memory that comes from the upstream
   * resource, either because there is no pool or because the pool is exhausted, is fresh from
   * `cudaHostAlloc` and no stream can be referencing it, so it needs no wait at all.
   */
  void* allocate_host_writable(cuda::stream_ref stream, std::size_t bytes, std::size_t alignment)
  {
    record_size(bytes);
    if (max_pool_size_ == 0) { return upstream_mr_.allocate(stream, bytes, alignment); }

    try {
      return pool_->allocate_host_writable(stream, bytes, alignment);
    } catch (...) {
      CUDF_LOG_INFO("Pinned pool exhausted, falling back to new pinned allocation for %zu bytes",
                    bytes);
      ++fallback_->count;
      auto* ptr = upstream_mr_.allocate(stream, bytes, alignment);

      {
        std::unique_lock lock(fallback_->mutex);
        fallback_->allocations.insert(ptr);
      }

      return ptr;
    }
  }

  /// Deallocates memory returned by `allocate_host_writable`.
  void deallocate_host_writable(cuda::stream_ref stream,
                                void* ptr,
                                std::size_t bytes,
                                std::size_t alignment,
                                bool device_exposed) noexcept
  {
    if (max_pool_size_ == 0) {
      upstream_mr_.deallocate(stream, ptr, bytes, alignment);
      return;
    }

    bool is_fallback{false};
    {
      std::shared_lock lock(fallback_->mutex);
      is_fallback = fallback_->allocations.find(ptr) != fallback_->allocations.end();
    }

    if (is_fallback) {
      {
        std::unique_lock lock(fallback_->mutex);
        fallback_->allocations.erase(ptr);
      }
      upstream_mr_.deallocate(stream, ptr, bytes, alignment);
    } else {
      pool_->deallocate_host_writable(stream, ptr, bytes, alignment, device_exposed);
    }
  }

  /// Selects the mechanism the pool uses to make a recycled block safe for the host to write to.
  void set_host_write_sync_mode(rmm::mr::host_write_sync_mode mode) noexcept
  {
    if (pool_ != nullptr) { pool_->set_host_write_sync_mode(mode); }
  }

  /// Returns counters describing the pool's host-writable allocations.
  [[nodiscard]] rmm::mr::host_writable_stats host_writable_statistics() const
  {
    return pool_ != nullptr ? pool_->host_writable_statistics() : rmm::mr::host_writable_stats{};
  }

  /// Resets the counters returned by `host_writable_statistics`.
  void reset_host_writable_statistics()
  {
    if (pool_ != nullptr) { pool_->reset_host_writable_statistics(); }
    fallback_->count = 0;
  }

  /// The number of allocations the pool could not satisfy, which fell back to `cudaHostAlloc`.
  [[nodiscard]] std::size_t pool_fallbacks() const { return fallback_->count; }

  /// Writes the pinned allocation size distribution to `out`.
  void print_size_histogram(std::FILE* out) const { sizes_->print(out, "pinned"); }

  bool operator==(pinned_pool_with_fallback_memory_resource const& other) const noexcept
  {
    return pool_ == other.pool_;
  }

  bool operator!=(pinned_pool_with_fallback_memory_resource const& other) const noexcept
  {
    return !(*this == other);
  }
};

static_assert(cuda::mr::resource_with<pinned_pool_with_fallback_memory_resource,
                                      cuda::mr::device_accessible,
                                      cuda::mr::host_accessible>,
              "Pinned pool mr must be accessible from both host and device");

/// The default pinned pool object, or nullptr if it has not been created yet. Needed because the
/// host-writable entry points are not reachable through a type-erased resource ref.
CUDF_EXPORT pinned_pool_with_fallback_memory_resource*& default_pinned_pool()
{
  static pinned_pool_with_fallback_memory_resource* pool{nullptr};
  return pool;
}

CUDF_EXPORT rmm::host_device_async_resource_ref& make_default_pinned_mr(
  std::optional<size_t> config_size)
{
  static pinned_pool_with_fallback_memory_resource mr = [config_size]() {
    auto const initial_size = [&config_size]() -> size_t {
      if (auto const env_val = getenv("LIBCUDF_PINNED_POOL_SIZE"); env_val != nullptr) {
        return std::atol(env_val);
      }

      if (config_size.has_value()) { return *config_size; }

      auto const total = rmm::available_device_memory().second;
      // 0.5% of the total device memory, capped at 64MB
      return std::min(total / 200, size_t{64} * 1024 * 1024);
    }();

    auto const max_size = [&initial_size]() -> size_t {
      if (auto const env_val = getenv("LIBCUDF_PINNED_POOL_MAX_SIZE"); env_val != nullptr) {
        return std::atol(env_val);
      }
      return initial_size * 16;
    }();

    return pinned_pool_with_fallback_memory_resource{initial_size, max_size};
  }();

  default_pinned_pool() = &mr;

  static rmm::host_device_async_resource_ref mr_ref{mr};
  return mr_ref;
}

CUDF_EXPORT std::mutex& host_mr_mutex()
{
  static std::mutex map_lock;
  return map_lock;
}

// Must be called with the host_mr_mutex mutex held
CUDF_EXPORT rmm::host_device_async_resource_ref& make_host_mr(
  std::optional<pinned_mr_options> const& opts, bool* did_configure = nullptr)
{
  static rmm::host_device_async_resource_ref* mr_ref = nullptr;
  bool configured                                    = false;
  if (mr_ref == nullptr) {
    configured = true;
    mr_ref     = &make_default_pinned_mr(opts ? opts->pool_size : std::nullopt);
  }

  // If the user passed an out param to detect whether this call configured a resource
  // set the result
  if (did_configure != nullptr) { *did_configure = configured; }

  return *mr_ref;
}

// Must be called with the host_mr_mutex mutex held
CUDF_EXPORT rmm::host_device_async_resource_ref& host_mr()
{
  static rmm::host_device_async_resource_ref mr_ref = make_host_mr(std::nullopt);
  return mr_ref;
}

class new_delete_memory_resource {
 public:
  void* allocate_sync(std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    pageable_size_histogram().record(bytes);
    try {
      return aligned_host_allocate(bytes, alignment);
    } catch (std::bad_alloc const& e) {
      CUDF_FAIL("Failed to allocate memory: " + std::string{e.what()}, rmm::out_of_memory);
    }
  }

  void* allocate([[maybe_unused]] cuda::stream_ref stream,
                 std::size_t bytes,
                 std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    return allocate_sync(bytes, alignment);
  }

  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    aligned_host_deallocate(ptr, bytes, alignment);
  }

  void deallocate([[maybe_unused]] cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    deallocate_sync(ptr, bytes, alignment);
  }

  bool operator==(new_delete_memory_resource const& other) const { return true; }

  bool operator!=(new_delete_memory_resource const& other) const { return !operator==(other); }

  // NOLINTBEGIN
  friend void get_property(new_delete_memory_resource const&, cuda::mr::host_accessible) noexcept {}
  // NOLINTEND
};

static_assert(cuda::mr::resource_with<new_delete_memory_resource, cuda::mr::host_accessible>,
              "Pageable pool mr must be accessible from the host");

/// Exposes the pinned pool's host-writable entry points through the interface libcudf's allocator
/// holds, since a type-erased resource ref cannot reach them.
class host_writable_pinned_adapter : public cudf::detail::host_writable_resource {
 public:
  explicit host_writable_pinned_adapter(pinned_pool_with_fallback_memory_resource& pool)
    : pool_{pool}
  {
  }

  void* allocate_host_writable(cuda::stream_ref stream,
                               std::size_t bytes,
                               std::size_t alignment) override
  {
    return pool_.allocate_host_writable(stream, bytes, alignment);
  }

  void deallocate_host_writable(cuda::stream_ref stream,
                                void* ptr,
                                std::size_t bytes,
                                std::size_t alignment,
                                bool device_exposed) noexcept override
  {
    pool_.deallocate_host_writable(stream, ptr, bytes, alignment, device_exposed);
  }

 private:
  pinned_pool_with_fallback_memory_resource& pool_;
};

/// Reads `LIBCUDF_HOST_WRITABLE_SYNC_MODE`. Nullopt means the host-writable path is disabled and
/// allocations synchronize the stream as before.
std::optional<rmm::mr::host_write_sync_mode> host_write_sync_mode_from_env()
{
  auto const value = cudf::detail::getenv_or<std::string>("LIBCUDF_HOST_WRITABLE_SYNC_MODE", "off");
  if (value == "off") { return std::nullopt; }
  if (value == "stream_sync") { return rmm::mr::host_write_sync_mode::stream_sync; }
  if (value == "stream_event") { return rmm::mr::host_write_sync_mode::stream_event; }
  if (value == "block_event") { return rmm::mr::host_write_sync_mode::block_event; }
  if (value == "clean_tracking") { return rmm::mr::host_write_sync_mode::clean_tracking; }
  CUDF_FAIL("Invalid LIBCUDF_HOST_WRITABLE_SYNC_MODE: " + value);
}

/// Prints the host-writable counters at exit when `LIBCUDF_HOST_WRITABLE_STATS` is set, so that
/// hit rates can be read off any benchmark binary without modifying it.
class host_writable_stats_printer {
 public:
  explicit host_writable_stats_printer(pinned_pool_with_fallback_memory_resource& pool)
    : pool_{pool}, enabled_{cudf::detail::get_bool_env_or("LIBCUDF_HOST_WRITABLE_STATS", false)}
  {
  }

  ~host_writable_stats_printer()
  {
    if (!enabled_) { return; }
    auto const& waits = host_wait_accumulator();
    std::fprintf(stderr,
                 "host_alloc_wait: count=%zu total_ms=%.3f mean_us=%.3f\n",
                 static_cast<std::size_t>(waits.count.load()),
                 static_cast<double>(waits.nanoseconds.load()) / 1e6,
                 waits.count.load() == 0
                   ? 0.0
                   : static_cast<double>(waits.nanoseconds.load()) / waits.count.load() / 1e3);
    pageable_size_histogram().print(stderr, "pageable");
    pool_.print_size_histogram(stderr);
    auto const stats = pool_.host_writable_statistics();
    std::fprintf(stderr,
                 "host_writable: allocations=%zu waits=%zu fast_path=%zu query_short_circuit=%zu "
                 "event_records=%zu events_created=%zu pool_fallbacks=%zu\n",
                 stats.allocations,
                 stats.waits,
                 stats.fast_path,
                 stats.query_short_circuit,
                 stats.event_records,
                 stats.events_created,
                 pool_.pool_fallbacks());
  }

 private:
  pinned_pool_with_fallback_memory_resource& pool_;
  bool enabled_;
};

/// Installs the exit-time counter printer, independently of the sync mode so that the `off`
/// baseline reports its allocation sizes too.
CUDF_EXPORT void install_stats_printer_once()
{
  [[maybe_unused]] static bool const installed = []() {
    if (!cudf::detail::get_bool_env_or("LIBCUDF_HOST_WRITABLE_STATS", false)) { return false; }
    std::scoped_lock lock{host_mr_mutex()};
    make_host_mr(std::nullopt);
    auto* pool = default_pinned_pool();
    if (pool == nullptr) { return false; }
    // Constructed after the pool, so destroyed before it
    static host_writable_stats_printer printer{*pool};
    return true;
  }();
}

/// The adapter over the default pinned pool, or nullptr when the path is disabled. Configures the
/// pool's mode on first use.
CUDF_EXPORT cudf::detail::host_writable_resource* host_writable_pinned_adapter_instance()
{
  static cudf::detail::host_writable_resource* instance =
    []() -> cudf::detail::host_writable_resource* {
    auto const mode = host_write_sync_mode_from_env();
    if (!mode.has_value()) { return nullptr; }

    std::scoped_lock lock{host_mr_mutex()};
    make_host_mr(std::nullopt);
    auto* pool = default_pinned_pool();
    // A user resource was installed before the default pool was ever created
    if (pool == nullptr) { return nullptr; }

    pool->set_host_write_sync_mode(*mode);
    static host_writable_pinned_adapter adapter{*pool};
    return &adapter;
  }();
  return instance;
}

}  // namespace

rmm::host_device_async_resource_ref set_pinned_memory_resource(
  rmm::host_device_async_resource_ref mr)
{
  std::scoped_lock lock{host_mr_mutex()};
  auto last_mr = host_mr();
  host_mr()    = mr;
  return last_mr;
}

rmm::host_device_async_resource_ref get_pinned_memory_resource()
{
  std::scoped_lock lock{host_mr_mutex()};
  return host_mr();
}

bool config_default_pinned_memory_resource(pinned_mr_options const& opts)
{
  std::scoped_lock lock{host_mr_mutex()};
  auto did_configure = false;
  make_host_mr(opts, &did_configure);
  return did_configure;
}

CUDF_EXPORT auto& kernel_pinned_copy_threshold()
{
  // use cudaMemcpyAsync for all pinned copies
  static std::atomic<size_t> threshold =
    cudf::detail::getenv_or("LIBCUDF_KERNEL_PINNED_COPY_THRESHOLD", 0);
  return threshold;
}

void set_kernel_pinned_copy_threshold(size_t threshold)
{
  kernel_pinned_copy_threshold() = threshold;
}

size_t get_kernel_pinned_copy_threshold() { return kernel_pinned_copy_threshold(); }

CUDF_EXPORT auto& allocate_host_as_pinned_threshold()
{
  // use pageable memory for all host allocations
  static std::atomic<size_t> threshold =
    cudf::detail::getenv_or("LIBCUDF_ALLOCATE_HOST_AS_PINNED_THRESHOLD", 0);
  return threshold;
}

void set_allocate_host_as_pinned_threshold(size_t threshold)
{
  allocate_host_as_pinned_threshold() = threshold;
}

size_t get_allocate_host_as_pinned_threshold() { return allocate_host_as_pinned_threshold(); }

namespace detail {

CUDF_EXPORT rmm::host_async_resource_ref get_pageable_memory_resource()
{
  static new_delete_memory_resource mr{};
  static rmm::host_async_resource_ref mr_ref{mr};
  return mr_ref;
}

CUDF_EXPORT void record_host_alloc_wait(std::uint64_t nanoseconds)
{
  auto& totals = host_wait_accumulator();
  totals.nanoseconds.fetch_add(nanoseconds, std::memory_order_relaxed);
  totals.count.fetch_add(1, std::memory_order_relaxed);
}

CUDF_EXPORT host_writable_resource* get_host_writable_resource(rmm::host_async_resource_ref mr)
{
  install_stats_printer_once();
  auto* adapter = host_writable_pinned_adapter_instance();
  if (adapter == nullptr) { return nullptr; }

  // Only the pool the adapter was built over can use it; a user-supplied pinned resource and the
  // pageable resource both stay on the stream-synchronizing path.
  static rmm::host_async_resource_ref const pool_ref{*default_pinned_pool()};
  return mr == pool_ref ? adapter : nullptr;
}

CUDF_EXPORT host_writable_stats get_host_writable_statistics()
{
  if (host_writable_pinned_adapter_instance() == nullptr) { return {}; }
  auto const stats = default_pinned_pool()->host_writable_statistics();
  return {stats.allocations,
          stats.waits,
          stats.fast_path,
          stats.query_short_circuit,
          stats.event_records,
          stats.events_created,
          default_pinned_pool()->pool_fallbacks()};
}

CUDF_EXPORT void reset_host_writable_statistics()
{
  if (host_writable_pinned_adapter_instance() == nullptr) { return; }
  default_pinned_pool()->reset_host_writable_statistics();
}

}  // namespace detail

}  // namespace cudf
