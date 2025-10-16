/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <benchmarks/fixture/benchmark_fixture.hpp>

#include <benchmark/benchmark.h>

#include <vector>
#include <cstdint>

// Benchmark for basic vector allocation and deallocation
static void BM_vector_alloc_dealloc(benchmark::State& state)
{
  auto const size = static_cast<std::size_t>(state.range(0));
  
  for (auto _ : state) {
    std::vector<int> vec(size);
    auto* ptr = vec.data();
    benchmark::DoNotOptimize(ptr);
    benchmark::ClobberMemory();
  }
  
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * size * sizeof(int));
}

// Benchmark for vector with reserve
static void BM_vector_reserve(benchmark::State& state)
{
  auto const size = static_cast<std::size_t>(state.range(0));
  
  for (auto _ : state) {
    std::vector<int> vec;
    vec.reserve(size);
    auto* ptr = vec.data();
    benchmark::DoNotOptimize(ptr);
    benchmark::ClobberMemory();
  }
  
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * size * sizeof(int));
}

// Benchmark for vector push_back
static void BM_vector_push_back(benchmark::State& state)
{
  auto const size = static_cast<std::size_t>(state.range(0));
  
  for (auto _ : state) {
    std::vector<int> vec;
    for (std::size_t i = 0; i < size; ++i) {
      vec.push_back(static_cast<int>(i));
    }
    auto* ptr = vec.data();
    benchmark::DoNotOptimize(ptr);
    benchmark::ClobberMemory();
  }
  
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * size * sizeof(int));
}

// Benchmark for vector push_back with reserve
static void BM_vector_push_back_with_reserve(benchmark::State& state)
{
  auto const size = static_cast<std::size_t>(state.range(0));
  
  for (auto _ : state) {
    std::vector<int> vec;
    vec.reserve(size);
    for (std::size_t i = 0; i < size; ++i) {
      vec.push_back(static_cast<int>(i));
    }
    auto* ptr = vec.data();
    benchmark::DoNotOptimize(ptr);
    benchmark::ClobberMemory();
  }
  
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * size * sizeof(int));
}

// Benchmark for vector resize
static void BM_vector_resize(benchmark::State& state)
{
  auto const size = static_cast<std::size_t>(state.range(0));
  
  for (auto _ : state) {
    std::vector<int> vec;
    vec.resize(size);
    auto* ptr = vec.data();
    benchmark::DoNotOptimize(ptr);
    benchmark::ClobberMemory();
  }
  
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * size * sizeof(int));
}

// Benchmark for vector with different types
template<typename T>
static void BM_vector_alloc_typed(benchmark::State& state)
{
  auto const size = static_cast<std::size_t>(state.range(0));
  
  for (auto _ : state) {
    std::vector<T> vec(size);
    auto* ptr = vec.data();
    benchmark::DoNotOptimize(ptr);
    benchmark::ClobberMemory();
  }
  
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * size * sizeof(T));
}

// Register benchmarks with various sizes
BENCHMARK(BM_vector_alloc_dealloc)
  ->RangeMultiplier(4)
  ->Range(1 << 10, 1 << 26)  // 1K to 64M elements
  ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_vector_reserve)
  ->RangeMultiplier(4)
  ->Range(1 << 10, 1 << 26)
  ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_vector_push_back)
  ->RangeMultiplier(4)
  ->Range(1 << 8, 1 << 16)  // Smaller range as push_back is slower
  ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_vector_push_back_with_reserve)
  ->RangeMultiplier(4)
  ->Range(1 << 8, 1 << 20)
  ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_vector_resize)
  ->RangeMultiplier(4)
  ->Range(1 << 10, 1 << 26)
  ->Unit(benchmark::kMicrosecond);

// Type-specific benchmarks
BENCHMARK(BM_vector_alloc_typed<int8_t>)
  ->RangeMultiplier(4)
  ->Range(1 << 10, 1 << 26)
  ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_vector_alloc_typed<int32_t>)
  ->RangeMultiplier(4)
  ->Range(1 << 10, 1 << 26)
  ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_vector_alloc_typed<int64_t>)
  ->RangeMultiplier(4)
  ->Range(1 << 10, 1 << 26)
  ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_vector_alloc_typed<double>)
  ->RangeMultiplier(4)
  ->Range(1 << 10, 1 << 26)
  ->Unit(benchmark::kMicrosecond);

