/*
 * Copyright (c) 2018-2019, NVIDIA CORPORATION.
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
#ifndef TYPES_HPP
#define TYPES_HPP

#include <algorithm>
#include <bitmask/legacy_bitmask.hpp>
#include <cassert>
#include <utilities/error_utils.hpp>
#include "cudf.h"

namespace cudf {

/**
 * @brief A wrapper for a set of gdf_columns of equal number of rows.
 *
 */
struct table {
  /**---------------------------------------------------------------------------*
   * @brief Constructs a table object from an array of `gdf_column`s
   *
   * @param cols The array of columns wrapped by the table
   * @param num_cols  The number of columns in the array
   *---------------------------------------------------------------------------**/
  table(gdf_column* cols[], gdf_size_type num_cols)
      : columns{cols}, _num_columns{num_cols} {
    CUDF_EXPECTS(nullptr != cols[0], "Null input column");

    this->_num_rows = cols[0]->size;

    std::for_each(columns, columns + _num_columns, [this](gdf_column* col) {
      CUDF_EXPECTS(nullptr != col, "Null input column");
      CUDF_EXPECTS(_num_rows == col->size, "Column size mismatch");
    });
  }

  /**---------------------------------------------------------------------------*
   * @brief Allocates and constructs a set of `gdf_column`s.
   *
   * Allocates an array of `gdf_column`s of the specified size and type.
   *
   * @note It is the caller's responsibility to free the array of gdf_columns
   *and their associated device memory.
   *
   * @note Does not support `GDF_TIMESTAMP` columns as this would require
   * passing in additional timestamp resolution information.
   *
   * @param[in] num_rows The size of each gdf_column
   * @param[in] dtypes The type of each column
   * @param[in] allocate_bitmasks If `true`, each column will be allocated an
   * appropriately sized bitmask
   *---------------------------------------------------------------------------**/
  table(gdf_size_type num_rows, std::vector<gdf_dtype> const& dtypes,
        bool allocate_bitmasks = false, cudaStream_t stream = 0)
      : _num_columns{static_cast<gdf_size_type>(dtypes.size())},
        _num_rows{num_rows} {
    columns = new gdf_column*[_num_columns];
    std::transform(
        columns, columns + _num_columns, dtypes.begin(), columns,
        [num_rows, allocate_bitmasks, stream](gdf_column*& col,
                                              gdf_dtype dtype) {
          col = new gdf_column;
          CUDF_EXPECTS(dtype != GDF_TIMESTAMP, "Timestamp unsupported.");
          col->size = num_rows;
          col->dtype = dtype;
          col->null_count = 0;
          col->valid = nullptr;

          // Timestamp currently unsupported as it would require passing in
          // additional resolution information
          gdf_dtype_extra_info extra_info;
          extra_info.time_unit = TIME_UNIT_NONE;
          col->dtype_info = extra_info;

          RMM_ALLOC(&col->data, gdf_dtype_size(dtype) * num_rows, stream);
          if (allocate_bitmasks) {
            RMM_ALLOC(
                &col->valid,
                gdf_valid_allocation_size(num_rows) * sizeof(gdf_valid_type),
                stream);
          }
          return col;
        });
  }

  /**---------------------------------------------------------------------------*
   * @brief Returns const pointer to const of the first `gdf_column` in the
   * table.
   *
   *---------------------------------------------------------------------------**/
  gdf_column const* const* begin() const { return columns; }

  /**---------------------------------------------------------------------------*
   * @brief Returns pointer to the first `gdf_column` in the table.
   *
   *---------------------------------------------------------------------------**/
  gdf_column** begin() { return columns; }

  /**---------------------------------------------------------------------------*
   * @brief Returns const pointer to const of one past the last `gdf_column` in
   * the table
   *
   *---------------------------------------------------------------------------**/
  gdf_column const* const* end() const { return columns + _num_columns; }

  /**---------------------------------------------------------------------------*
   * @brief Returns pointer to one past the last `gdf_column` in the table
   *
   *---------------------------------------------------------------------------**/
  gdf_column** end() { return columns + _num_columns; }

  /**---------------------------------------------------------------------------*
   * @brief Returns pointer to the column specified by an index.
   *
   * @param index The index of the desired column
   * @return gdf_column* Pointer to the column at `index`
   *---------------------------------------------------------------------------**/
  gdf_column* get_column(gdf_index_type index) {
    assert(index < _num_columns);
    return columns[index];
  }

  /**---------------------------------------------------------------------------*
   * @brief Returns pointer const of the column specified by an index.
   *
   * @param index The index of the desired column
   * @return gdf_column* Pointer to the column at `index`
   *---------------------------------------------------------------------------**/
  gdf_column const* get_column(gdf_index_type index) const {
    return columns[index];
  }

  /**---------------------------------------------------------------------------*
   * @brief Returns the number of columns in the table
   *
   *---------------------------------------------------------------------------**/
  gdf_size_type num_columns() const { return _num_columns; }

  /**---------------------------------------------------------------------------*
   * @brief Returns the number of rows in the table
   *
   *---------------------------------------------------------------------------**/
  gdf_size_type num_rows() const { return _num_rows; }

 private:
  gdf_column** columns;              ///< The set of gdf_columns
  gdf_size_type const _num_columns;  ///< The number of columns in the set
  gdf_size_type _num_rows;           ///< The number of elements in each column
};

}  // namespace cudf

#endif
