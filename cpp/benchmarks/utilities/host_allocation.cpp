/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/utilities/host_uvector.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/pinned_memory.hpp>

#include <rmm/device_uvector.hpp>

#include <nvbench/nvbench.cuh>

#include <limits>
#include <string>
#include <vector>

namespace {

using element_type = int32_t;

enum class container {
  none,
  std_vector,
  host_vector,
  host_uvector,
  // Deliberately violates the host_uvector contract by writing without synchronizing, to separate
  // the cost of the allocation-time synchronization from the cost of the pool traffic around it.
  // This is the floor a cheaper synchronization (waiting on the freeing block's own event instead
  // of the whole stream) could reach.
  host_uvector_unsynchronized,
  // No allocation at all: the cost of the synchronization primitive on its own, either the whole
  // stream or a freshly recorded event standing in for a freed block's event.
  stream_sync_only,
  event_wait_only
};

enum class resource { pageable, pinned };

/**
 * @brief Enqueues a fixed amount of device work, so that the cost of an allocation-time stream
 * synchronization is not measured against an idle stream.
 */
struct stream_load {
  rmm::device_uvector<element_type> buffer;

  stream_load(std::size_t size, cuda::stream_ref stream)
    : buffer{size, stream, cudf::get_current_device_resource_ref()}
  {
  }

  void enqueue(cuda::stream_ref stream)
  {
    if (buffer.is_empty()) { return; }
    CUDF_CUDA_TRY(
      cudaMemsetAsync(buffer.data(), 0, buffer.size() * sizeof(element_type), stream.get()));
  }
};

// Keeps the compiler from eliding the host writes
element_type volatile sink{};

cudaEvent_t reusable_event()
{
  static cudaEvent_t event = [] {
    cudaEvent_t e{};
    CUDF_CUDA_TRY(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    return e;
  }();
  return event;
}

template <container Container, resource Resource>
void allocate_and_fill(std::size_t num_elements, cuda::stream_ref stream)
{
  if constexpr (Container == container::none) {
    return;
  } else if constexpr (Container == container::std_vector) {
    auto vec = std::vector<element_type>(num_elements);
    vec[0]   = element_type{1};
    sink     = vec[0];
  } else if constexpr (Container == container::host_vector) {
    auto vec = Resource == resource::pinned
                 ? cudf::detail::make_pinned_vector_async<element_type>(num_elements, stream)
                 : cudf::detail::make_host_vector<element_type>(num_elements, stream);
    vec[0]   = element_type{1};
    sink     = vec[0];
  } else if constexpr (Container == container::host_uvector) {
    auto vec  = Resource == resource::pinned
                  ? cudf::detail::make_pinned_uvector<element_type>(num_elements, stream)
                  : cudf::detail::make_host_uvector<element_type>(num_elements, stream);
    auto span = vec.host_writable();
    span[0]   = element_type{1};
    sink      = span[0];
  } else if constexpr (Container == container::host_uvector_unsynchronized) {
    auto vec      = Resource == resource::pinned
                      ? cudf::detail::make_pinned_uvector<element_type>(num_elements, stream)
                      : cudf::detail::make_host_uvector<element_type>(num_elements, stream);
    vec.data()[0] = element_type{1};
    sink          = vec.data()[0];
  } else if constexpr (Container == container::stream_sync_only) {
    cudf::detail::sync_stream(stream);
  } else {
    CUDF_CUDA_TRY(cudaEventRecord(reusable_event(), stream.get()));
    CUDF_CUDA_TRY(cudaEventSynchronize(reusable_event()));
  }
}

}  // namespace

/**
 * @brief Compares the cost of creating a host buffer that the host then writes to.
 *
 * The two costs under study scale differently: the allocation-time stream synchronization is a
 * fixed per-allocation charge, while value-initialization is proportional to the buffer size. The
 * `stream_load` axis controls how much device work is in flight when the allocation happens, since
 * the cost of the synchronization is whatever the stream is busy with.
 */
template <container Container, resource Resource>
void bench_host_allocation(
  nvbench::state& state,
  nvbench::type_list<nvbench::enum_type<Container>, nvbench::enum_type<Resource>>)
{
  auto const num_elements = static_cast<std::size_t>(state.get_int64("num_elements"));
  auto const load_size    = static_cast<std::size_t>(state.get_int64("stream_load"));

  // Force the resource choice: the default threshold of 0 sends every make_host_vector allocation
  // to the pageable resource.
  auto const prev_threshold = cudf::get_allocate_host_as_pinned_threshold();
  cudf::set_allocate_host_as_pinned_threshold(
    Resource == resource::pinned ? std::numeric_limits<std::size_t>::max() : 0);

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));

  auto load = stream_load{load_size, stream};

  // Warm up the pool so that the measured allocations hit a populated pool
  allocate_and_fill<Container, Resource>(num_elements, stream);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    load.enqueue(stream);
    allocate_and_fill<Container, Resource>(num_elements, stream);
  });

  cudf::set_allocate_host_as_pinned_threshold(prev_threshold);
}

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  container,
  [](container c) {
    switch (c) {
      case container::none: return "none";
      case container::std_vector: return "std_vector";
      case container::host_vector: return "host_vector";
      case container::host_uvector: return "host_uvector";
      case container::host_uvector_unsynchronized: return "host_uvector_nosync";
      case container::stream_sync_only: return "stream_sync_only";
      case container::event_wait_only: return "event_wait_only";
      default: return "unknown";
    }
  },
  [](container) { return std::string{}; })

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  resource,
  [](resource r) {
    switch (r) {
      case resource::pageable: return "pageable";
      case resource::pinned: return "pinned";
      default: return "unknown";
    }
  },
  [](resource) { return std::string{}; })

using containers = nvbench::enum_type_list<container::none,
                                           container::std_vector,
                                           container::host_vector,
                                           container::host_uvector,
                                           container::host_uvector_unsynchronized,
                                           container::stream_sync_only,
                                           container::event_wait_only>;
using resources  = nvbench::enum_type_list<resource::pageable, resource::pinned>;

NVBENCH_BENCH_TYPES(bench_host_allocation, NVBENCH_TYPE_AXES(containers, resources))
  .set_name("host_allocation")
  .set_type_axes_names({"container", "resource"})
  .add_int64_power_of_two_axis("num_elements", {4, 10, 16, 22})
  .add_int64_axis("stream_load", {0, 1 << 24});
