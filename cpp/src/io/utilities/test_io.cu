/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <cudf/detail/utilities/pinned_host_vector.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/device_uvector.hpp>

#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <sys/mman.h>
#include <unistd.h>

constexpr size_t data_size_bytes = 512 * 1024 * 1024;

void fill_random_data(int* data, size_t size)
{
  // Replace this function with your data generation logic if needed.
  for (size_t i = 0; i < size / sizeof(int); ++i) {
    data[i] = rand();
  }
}

void test_io(bool use_mmap, bool use_pinned)
{
  std::cout << "test_io: use_mmap=" << use_mmap << ", use_pinned=" << use_pinned << std::endl;

  std::string file_path = "datafile.bin";
  auto stream           = cudf::get_default_stream();

  std::vector<char> data;
  data.reserve(data_size_bytes);
  fill_random_data(reinterpret_cast<int*>(data.data()), data_size_bytes);

  std::ofstream file(file_path, std::ios::binary);
  file.write(data.data(), data_size_bytes);
  file.close();

  auto const fd = open(file_path.c_str(), O_RDONLY);
  CUDF_EXPECTS(fd != -1, "Error opening the file.");

  char const* h_data = nullptr;
  char* mapped_data  = nullptr;
  std::vector<char> read_data;
  if (use_mmap) {
    mapped_data = static_cast<char*>(mmap(nullptr, data_size_bytes, PROT_READ, MAP_PRIVATE, fd, 0));
    CUDF_EXPECTS(mapped_data != MAP_FAILED, "Error mapping the file into memory.");
    h_data = mapped_data;
  } else {
    read_data.resize(data_size_bytes);
    CUDF_EXPECTS(read(fd, read_data.data(), data_size_bytes) == data_size_bytes, "read failed");
    h_data = read_data.data();
  }

  // copy to GPU
  auto const chunk_size = 64ul * 1024 * 1024;
  cudf::detail::pinned_host_vector<char> pinned_data(use_pinned ? chunk_size : 0);

  rmm::device_uvector<char> d_data{chunk_size, stream};
  for (size_t chunk = 0; chunk < (data_size_bytes + chunk_size - 1) / chunk_size; ++chunk) {
    auto const offset = chunk * chunk_size;
    auto const size   = std::min(chunk_size, data_size_bytes - offset);

    auto src = h_data + offset;
    if (use_pinned) {
      std::memcpy(pinned_data.data(), src, size);
      src = pinned_data.data();
    }

    cudaMemcpyAsync(d_data.data(), src, size, cudaMemcpyDefault, stream);
    stream.synchronize();
  }

  close(fd);
  munmap(mapped_data, data_size_bytes);
}

void test_io_all_options()
{
  test_io(false, false);
  test_io(false, true);
  test_io(true, false);
  test_io(true, true);
}
