/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

#include "config_utils.hpp"

#include <cudf/utilities/error.hpp>

#include <cstdlib>
#include <string>

namespace cudf::io::detail {

namespace cufile_integration {

namespace {
/**
 * @brief Defines which cuFile usage to enable.
 */
enum class usage_policy : uint8_t { OFF, GDS, ALWAYS, KVIKIO };

/**
 * @brief Get the current usage policy.
 */
usage_policy get_env_policy()
{
  static auto const env_val = getenv_or<std::string>("LIBCUDF_CUFILE_POLICY", "KVIKIO");
  if (env_val == "OFF") return usage_policy::OFF;
  if (env_val == "GDS") return usage_policy::GDS;
  if (env_val == "ALWAYS") return usage_policy::ALWAYS;
  if (env_val == "KVIKIO") return usage_policy::KVIKIO;
  CUDF_FAIL("Invalid LIBCUDF_CUFILE_POLICY value: " + env_val);
}
}  // namespace

bool is_always_enabled() { return get_env_policy() == usage_policy::ALWAYS; }

bool is_gds_enabled() { return is_always_enabled() or get_env_policy() == usage_policy::GDS; }

bool is_kvikio_enabled() { return get_env_policy() == usage_policy::KVIKIO; }

}  // namespace cufile_integration

namespace nvcomp_integration {

namespace {
/**
 * @brief Defines which nvCOMP usage to enable.
 */
enum class usage_policy : uint8_t { OFF, STABLE, ALWAYS };

/**
 * @brief Get the current usage policy.
 */
usage_policy get_env_policy()
{
  static auto const env_val = getenv_or<std::string>("LIBCUDF_NVCOMP_POLICY", "STABLE");
  if (env_val == "OFF") return usage_policy::OFF;
  if (env_val == "STABLE") return usage_policy::STABLE;
  if (env_val == "ALWAYS") return usage_policy::ALWAYS;
  CUDF_FAIL("Invalid LIBCUDF_NVCOMP_POLICY value: " + env_val);
}
}  // namespace

bool is_all_enabled() { return get_env_policy() == usage_policy::ALWAYS; }

bool is_stable_enabled() { return is_all_enabled() or get_env_policy() == usage_policy::STABLE; }

}  // namespace nvcomp_integration

namespace io_config {

namespace {
/**
 * @brief Defines which nvCOMP usage to enable.
 */
enum class policy : uint8_t { MMAP_PAGEABLE, MMAP_PINNED, DIRECT_PAGEABLE, DIRECT_PINNED };

/**
 * @brief Get the current usage policy.
 */
policy get_env_policy()
{
  static auto const env_val = getenv_or<std::string>("LIBCUDF_IO_POLICY", "MMAP_PAGEABLE");
  if (env_val == "MMAP_PAGEABLE") return policy::MMAP_PAGEABLE;
  if (env_val == "MMAP_PINNED") return policy::MMAP_PINNED;
  if (env_val == "DIRECT_PAGEABLE") return policy::DIRECT_PAGEABLE;
  if (env_val == "DIRECT_PINNED") return policy::DIRECT_PINNED;
  CUDF_FAIL("Invalid LIBCUDF_IO_POLICY value: " + env_val);
}
}  // namespace

bool is_memory_mapping_enabled()
{
  return get_env_policy() == policy::MMAP_PAGEABLE or get_env_policy() == policy::MMAP_PINNED;
}

bool is_pinned_enabled()
{
  return get_env_policy() == policy::MMAP_PINNED or get_env_policy() == policy::DIRECT_PINNED;
}
}  // namespace io_config

}  // namespace cudf::io::detail
