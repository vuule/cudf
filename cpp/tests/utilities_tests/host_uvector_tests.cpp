/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/host_uvector.hpp>
#include <cudf/detail/utilities/host_writable_resource.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/pinned_memory.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>

#include <numeric>

using cudf::detail::host_uvector;

class HostUVectorTest : public cudf::test::BaseFixture {
  size_t prev_alloc_threshold;

 public:
  HostUVectorTest() : prev_alloc_threshold{cudf::get_allocate_host_as_pinned_threshold()} {}
  ~HostUVectorTest() override { cudf::set_allocate_host_as_pinned_threshold(prev_alloc_threshold); }
};

TEST_F(HostUVectorTest, SizeAndCapacity)
{
  auto stream = cudf::get_default_stream();

  auto vec = cudf::detail::make_host_uvector<int32_t>(100, stream);
  EXPECT_EQ(vec.size(), 100u);
  EXPECT_EQ(vec.capacity(), 100u);
  EXPECT_FALSE(vec.is_empty());

  vec.resize_uninitialized(10);
  EXPECT_EQ(vec.size(), 10u);
  EXPECT_EQ(vec.capacity(), 100u);

  EXPECT_THROW(vec.resize_uninitialized(101), cudf::logic_error);

  auto empty = cudf::detail::make_empty_host_uvector<int32_t>(100, stream);
  EXPECT_EQ(empty.size(), 0u);
  EXPECT_EQ(empty.capacity(), 100u);
  EXPECT_TRUE(empty.is_empty());
}

TEST_F(HostUVectorTest, ZeroSize)
{
  auto vec = cudf::detail::make_host_uvector<int32_t>(0, cudf::get_default_stream());
  EXPECT_EQ(vec.size(), 0u);
  EXPECT_EQ(vec.capacity(), 0u);
  EXPECT_EQ(vec.host_writable().size(), 0u);
}

TEST_F(HostUVectorTest, PushBack)
{
  auto vec = cudf::detail::make_empty_host_uvector<int32_t>(4, cudf::get_default_stream());
  for (int i = 0; i < 4; ++i) {
    vec.push_back(i);
  }
  EXPECT_EQ(vec.size(), 4u);
  EXPECT_THROW(vec.push_back(4), cudf::logic_error);

  auto const span = vec.host_writable();
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(span[i], i);
  }
}

TEST_F(HostUVectorTest, MoveConstructAndAssign)
{
  auto stream = cudf::get_default_stream();

  auto vec = cudf::detail::make_host_uvector<int32_t>(10, stream);
  std::iota(vec.host_writable().begin(), vec.host_writable().end(), 0);
  auto const* data = vec.data();

  auto moved = std::move(vec);
  EXPECT_EQ(moved.data(), data);
  EXPECT_EQ(moved.size(), 10u);
  EXPECT_EQ(vec.data(), nullptr);
  EXPECT_EQ(vec.size(), 0u);
  EXPECT_EQ(moved[9], 9);

  auto other = cudf::detail::make_host_uvector<int32_t>(3, stream);
  other      = std::move(moved);
  EXPECT_EQ(other.data(), data);
  EXPECT_EQ(other.size(), 10u);
  EXPECT_EQ(other[0], 0);
}

TEST_F(HostUVectorTest, DeviceToHost)
{
  auto stream = cudf::get_default_stream();

  auto expected = std::vector<int32_t>(1000);
  std::iota(expected.begin(), expected.end(), 0);
  auto d_vec = cudf::detail::make_device_uvector(
    cudf::host_span<int32_t const>{expected}, stream, cudf::get_current_device_resource_ref());

  // Synchronous factory: the host may read without synchronizing again
  auto const h_vec = cudf::detail::make_host_uvector(d_vec, stream);
  EXPECT_TRUE(h_vec.is_host_synchronized());
  EXPECT_EQ(h_vec.size(), expected.size());
  EXPECT_TRUE(std::equal(expected.begin(), expected.end(), h_vec.begin()));

  // Asynchronous factory: host access synchronizes on first use
  auto h_vec_async = cudf::detail::make_host_uvector_async(d_vec, stream);
  auto const span  = h_vec_async.host_writable();
  EXPECT_TRUE(h_vec_async.is_host_synchronized());
  EXPECT_TRUE(std::equal(expected.begin(), expected.end(), span.begin()));
}

TEST_F(HostUVectorTest, HostToDevice)
{
  auto stream = cudf::get_default_stream();

  auto h_vec = cudf::detail::make_host_uvector<int32_t>(1000, stream);
  std::iota(h_vec.host_writable().begin(), h_vec.host_writable().end(), 0);

  auto d_vec = cudf::detail::make_device_uvector(
    cudf::host_span<int32_t const>{h_vec}, stream, cudf::get_current_device_resource_ref());
  auto const round_trip = cudf::detail::make_std_vector(d_vec, stream);

  auto expected = std::vector<int32_t>(1000);
  std::iota(expected.begin(), expected.end(), 0);
  EXPECT_EQ(round_trip, expected);
}

TEST_F(HostUVectorTest, DeviceAccessibility)
{
  auto stream = cudf::get_default_stream();

  cudf::set_allocate_host_as_pinned_threshold(7);

  // smaller than the threshold: pinned
  {
    auto const vec = cudf::detail::make_host_uvector<char>(7, stream);
    EXPECT_TRUE(vec.is_device_accessible());
    EXPECT_TRUE(cudf::host_span<char const>{vec}.is_device_accessible());
  }

  // larger than the threshold: pageable
  {
    auto const vec = cudf::detail::make_host_uvector<char>(8, stream);
    EXPECT_FALSE(vec.is_device_accessible());
    EXPECT_FALSE(cudf::host_span<char const>{vec}.is_device_accessible());
  }

  // pinned factories always use pinned memory
  {
    auto const vec = cudf::detail::make_pinned_uvector<char>(1024, stream);
    EXPECT_TRUE(vec.is_device_accessible());
  }
}

TEST_F(HostUVectorTest, PageableBuffersNeedNoSynchronization)
{
  cudf::set_allocate_host_as_pinned_threshold(0);

  // The pageable resource is not stream-ordered, so no host synchronization is ever required
  auto vec = cudf::detail::make_host_uvector<int32_t>(10, cudf::get_default_stream());
  EXPECT_TRUE(vec.is_host_synchronized());
}

TEST_F(HostUVectorTest, PinnedBufferSynchronization)
{
  auto stream = cudf::get_default_stream();

  // With the host-writable path enabled, a buffer allocated with immediate write intent is already
  // safe to write to; otherwise the first host access synchronizes the stream.
  auto const host_writable_enabled =
    cudf::detail::get_host_writable_resource(cudf::get_pinned_memory_resource()) != nullptr;

  auto pinned = cudf::detail::make_pinned_uvector<int32_t>(10, stream);
  EXPECT_EQ(pinned.is_host_synchronized(), host_writable_enabled);
  EXPECT_EQ(pinned.host_writable().size(), 10u);
  EXPECT_TRUE(pinned.is_host_synchronized());

  // Deferred intent never waits at allocation, whether or not the path is enabled
  auto deferred = cudf::detail::make_pinned_uvector<int32_t>(
    10, stream, cudf::detail::host_write_intent::deferred);
  EXPECT_FALSE(deferred.is_host_synchronized());
  EXPECT_EQ(deferred.host_writable().size(), 10u);
  EXPECT_TRUE(deferred.is_host_synchronized());
}

TEST_F(HostUVectorTest, HostWritableStatistics)
{
  auto stream = cudf::get_default_stream();
  if (cudf::detail::get_host_writable_resource(cudf::get_pinned_memory_resource()) == nullptr) {
    GTEST_SKIP() << "host-writable path disabled";
  }

  cudf::detail::reset_host_writable_statistics();
  {
    auto vec               = cudf::detail::make_pinned_uvector<int32_t>(1024, stream);
    vec.host_writable()[0] = 1;
  }
  auto const stats = cudf::detail::get_host_writable_statistics();
  EXPECT_EQ(stats.allocations, 1u);
  EXPECT_EQ(stats.waits + stats.fast_path, stats.allocations);
}

TEST_F(HostUVectorTest, FromHostSpan)
{
  auto expected = std::vector<int32_t>(10);
  std::iota(expected.begin(), expected.end(), 0);

  auto const vec = cudf::detail::make_pinned_uvector(cudf::host_span<int32_t const>{expected},
                                                     cudf::get_default_stream());
  EXPECT_EQ(vec.size(), expected.size());
  EXPECT_TRUE(std::equal(expected.begin(), expected.end(), vec.begin()));
}

namespace {

/**
 * @brief Fills a pinned buffer, copies it to the device behind a queue of unrelated work, frees
 * it, then immediately writes a different pattern into the recycled block.
 *
 * The device must observe the first pattern. When `wait_before_overwrite` is false the wait that
 * guarantees this is skipped, which is what the host-writable path exists to avoid having to do
 * with a full stream synchronization.
 *
 * @return Whether the device received the first pattern intact
 */
bool recycled_block_keeps_first_pattern(bool wait_before_overwrite)
{
  auto stream            = cudf::get_default_stream();
  constexpr size_t count = 8 * 1024 * 1024;  // 32 MiB
  constexpr int32_t first{0x11111111};
  constexpr int32_t second{0x22222222};

  auto d_dst = rmm::device_uvector<int32_t>(count, stream, cudf::get_current_device_resource_ref());
  auto d_load =
    rmm::device_uvector<int32_t>(count, stream, cudf::get_current_device_resource_ref());

  void* recycled{};
  {
    auto h_src = cudf::detail::make_pinned_uvector<int32_t>(count, stream);
    auto s_src = h_src.host_writable();
    std::fill(s_src.begin(), s_src.end(), first);
    recycled = h_src.data();

    // Bury the copy behind enough work that it cannot have run by the time the block is recycled
    for (int i = 0; i < 64; ++i) {
      CUDF_CUDA_TRY(
        cudaMemsetAsync(d_load.data(), i, d_load.size() * sizeof(int32_t), stream.value()));
    }
    cudf::detail::cuda_memcpy_async<int32_t>(
      cudf::device_span<int32_t>{d_dst}, cudf::host_span<int32_t const>{h_src}, stream);
  }

  auto h_again = cudf::detail::make_pinned_uvector<int32_t>(
    count,
    stream,
    wait_before_overwrite ? cudf::detail::host_write_intent::immediate
                          : cudf::detail::host_write_intent::deferred);
  if (h_again.data() != recycled) { return true; }  // pool did not recycle the block; inconclusive

  if (wait_before_overwrite) {
    auto span = h_again.host_writable();
    std::fill(span.begin(), span.end(), second);
  } else {
    auto* raw = h_again.data();
    std::fill(raw, raw + count, second);
  }

  auto const result = cudf::detail::make_std_vector(d_dst, stream);
  return std::all_of(result.begin(), result.end(), [](int32_t value) { return value == first; });
}

}  // namespace

TEST_F(HostUVectorTest, RecycledBlockIsSafeToOverwrite)
{
  EXPECT_TRUE(recycled_block_keeps_first_pattern(true));
}

// Demonstrates that the test above is sensitive: without the wait, the host overwrites the block
// while the copy out of it is still queued. Disabled because it asserts on a race.
TEST_F(HostUVectorTest, DISABLED_RecycledBlockRaceWithoutWait)
{
  EXPECT_FALSE(recycled_block_keeps_first_pattern(false));
}

CUDF_TEST_PROGRAM_MAIN()
