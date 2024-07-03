/*
 *  Copyright 2024 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/resource_ref.hpp>

#include <cstddef>
#include <cstring>

namespace cudf::detail {

template <typename T>
class host_vector {
 public:
  using value_type      = T;
  using size_type       = std::size_t;
  using difference_type = std::ptrdiff_t;
  using pointer         = T*;
  using iterator        = T*;
  using const_pointer   = T const*;
  using reference       = T&;
  using const_reference = T const&;

  template <class... Properties>
  host_vector(std::size_t size,
              cuda::mr::async_resource_ref<cuda::mr::host_accessible, Properties...> mr,
              rmm::cuda_stream_view stream)
    : _size{size},
      _capacity{size},
      _is_device_accessible{
        cuda::has_property<cuda::mr::async_resource_ref<cuda::mr::host_accessible, Properties...>,
                           cuda::mr::device_accessible>},
      _mr{mr},
      _stream{stream}
  {
    if (_size != 0) {
      _data = static_cast<T*>(mr.allocate_async(_size * sizeof(T), _stream));
      _stream.synchronize();
      memset(static_cast<void*>(_data), 0, _size * sizeof(T));
    }
  }

  host_vector(host_vector const&) = delete;
  host_vector(host_vector&& other)
    : _data{other._data},
      _size{other._size},
      _capacity{other._capacity},
      _mr{other._mr},
      _stream{other._stream}
  {
    other._data     = nullptr;
    other._size     = 0;
    other._capacity = 0;
  }

  host_vector& operator=(host_vector const&) = delete;
  host_vector& operator=(host_vector&& other)
  {
    if (this != &other) {
      if (_data != nullptr) { _mr.deallocate_async(_data, _size * sizeof(T), _stream); }
      _data           = other._data;
      _size           = other._size;
      _capacity       = other._capacity;
      _mr             = other._mr;
      _stream         = other._stream;
      other._data     = nullptr;
      other._size     = 0;
      other._capacity = 0;
    }
    return *this;
  }

  ~host_vector()
  {
    if (_data != nullptr) { _mr.deallocate_async(_data, _size * sizeof(T), _stream); }
  }

  void resize(std::size_t new_size)
  {
    if (new_size > _capacity) {
      auto const new_capacity = std::max(_capacity * 2, new_size);
      auto new_data = static_cast<T*>(_mr.allocate_async(new_capacity * sizeof(T), _stream));
      _stream.synchronize();
      if (_data != nullptr) {
        memcpy(static_cast<void*>(new_data), static_cast<void*>(_data), _size * sizeof(T));
        _mr.deallocate_async(_data, _size * sizeof(T), _stream);
      }
      _data     = new_data;
      _capacity = new_capacity;
    }
    if (new_size > _size) {
      memset(static_cast<void*>(_data + _size), 0, (new_size - _size) * sizeof(T));
    }
    _size = new_size;
  }

  void reserve(std::size_t new_capacity)
  {
    if (new_capacity > _capacity) {
      auto new_data = static_cast<T*>(_mr.allocate_async(new_capacity * sizeof(T), _stream));
      _stream.synchronize();
      if (_data != nullptr) {
        std::copy(_data, _data + _size, new_data);
        _mr.deallocate_async(_data, _size * sizeof(T), _stream);
      }
      _data     = new_data;
      _capacity = new_capacity;
    }
  }

  void push_back(T const& value)
  {
    if (_size == _capacity) { reserve(_capacity == 0 ? 2 : _capacity * 2); }
    _data[_size++] = value;
  }

  void clear() { _size = 0; }

  [[nodiscard]] std::size_t size() const { return _size; }
  [[nodiscard]] std::int64_t ssize() const { return _size; }
  [[nodiscard]] bool is_empty() const { return _size == 0; }
  [[nodiscard]] std::size_t capacity() const { return _capacity; }

  [[nodiscard]] T& operator[](std::size_t idx) { return _data[idx]; }
  [[nodiscard]] T const& operator[](std::size_t idx) const { return _data[idx]; }

  [[nodiscard]] T* data() { return _data; }
  [[nodiscard]] T const* data() const { return _data; }

  [[nodiscard]] T& front() { return _data[0]; }
  [[nodiscard]] T const& front() const { return _data[0]; }

  [[nodiscard]] T& back() { return _data[_size - 1]; }
  [[nodiscard]] T const& back() const { return _data[_size - 1]; }

  [[nodiscard]] T* begin() { return _data; }
  [[nodiscard]] T const* begin() const { return _data; }

  [[nodiscard]] T* end() { return _data + _size; }
  [[nodiscard]] T const* end() const { return _data + _size; }

  [[nodiscard]] bool is_device_accessible() const { return _is_device_accessible; }

  [[nodiscard]] rmm::host_async_resource_ref memory_resource() const { return _mr; }
  [[nodiscard]] rmm::cuda_stream_view stream() const { return _stream; }

 private:
  T* _data{nullptr};
  std::size_t _size;
  std::size_t _capacity;
  bool _is_device_accessible;
  rmm::host_async_resource_ref _mr;
  rmm::cuda_stream_view _stream;
};

}  // namespace cudf::detail
