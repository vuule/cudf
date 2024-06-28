/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cudf/detail/utilities/host_vector.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/pinned_memory.hpp>

#include <rmm/resource_ref.hpp>

#include <cstddef>

namespace cudf::detail {

CUDF_EXPORT rmm::host_async_resource_ref& get_pageable_memory_resource();

/**
 * @brief Get the rmm resource to be used for host memory allocations.
 *
 * @param size The size of the allocation
 * @return The rmm resource to be used for host memory allocations
 */
template <typename T>
rmm_host_allocator<T> get_host_allocator(std::size_t size, rmm::cuda_stream_view _stream)
{
  if (size * sizeof(T) <= get_allocate_host_as_pinned_threshold()) {
    return {get_pinned_memory_resource(), _stream};
  }
  return {get_pageable_memory_resource(), _stream};
}

}  // namespace cudf::detail