/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compression_common.hpp"
#include "io_test_utils.hpp"
#include "parquet_common.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/io_metadata_utilities.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf/column/column.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_metadata.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>

#include <thrust/iterator/constant_iterator.h>

#include <src/io/parquet/parquet_gpu.hpp>

#include <array>
#include <limits>
#include <memory>
#include <stdexcept>

using ParquetDecompressionTest = DecompressionTest<ParquetReaderTest>;

TEST_F(ParquetReaderTest, UserBounds)
{
  // trying to read more rows than there are should result in
  // receiving the properly capped # of rows
  {
    srand(31337);
    auto expected = create_random_fixed_table<int>(4, 4, false);

    auto filepath = temp_env->get_temp_filepath("TooManyRows.parquet");
    cudf::io::parquet_writer_options args =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, *expected);
    cudf::io::write_parquet(args);

    // attempt to read more rows than there actually are
    cudf::io::parquet_reader_options read_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath}).num_rows(16);
    auto result = cudf::io::read_parquet(read_opts);

    // we should only get back 4 rows
    EXPECT_EQ(result.tbl->view().column(0).size(), 4);
  }

  // trying to read past the end of the # of actual rows should result
  // in empty columns.
  {
    srand(31337);
    auto expected = create_random_fixed_table<int>(4, 4, false);

    auto filepath = temp_env->get_temp_filepath("PastBounds.parquet");
    cudf::io::parquet_writer_options args =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, *expected);
    cudf::io::write_parquet(args);

    // attempt to read more rows than there actually are
    cudf::io::parquet_reader_options read_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath}).skip_rows(4);
    auto result = cudf::io::read_parquet(read_opts);

    // we should get empty columns back
    EXPECT_EQ(result.tbl->view().num_columns(), 4);
    EXPECT_EQ(result.tbl->view().column(0).size(), 0);
  }

  // trying to read 0 rows should result in empty columns
  {
    srand(31337);
    auto expected = create_random_fixed_table<int>(4, 4, false);

    auto filepath = temp_env->get_temp_filepath("ZeroRows.parquet");
    cudf::io::parquet_writer_options args =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, *expected);
    cudf::io::write_parquet(args);

    // attempt to read more rows than there actually are
    cudf::io::parquet_reader_options read_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath}).num_rows(0);
    auto result = cudf::io::read_parquet(read_opts);

    EXPECT_EQ(result.tbl->view().num_columns(), 4);
    EXPECT_EQ(result.tbl->view().column(0).size(), 0);
  }

  // trying to read 0 rows past the end of the # of actual rows should result
  // in empty columns.
  {
    srand(31337);
    auto expected = create_random_fixed_table<int>(4, 4, false);

    auto filepath = temp_env->get_temp_filepath("ZeroRowsPastBounds.parquet");
    cudf::io::parquet_writer_options args =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, *expected);
    cudf::io::write_parquet(args);

    // attempt to read more rows than there actually are
    cudf::io::parquet_reader_options read_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
        .skip_rows(4)
        .num_rows(0);
    auto result = cudf::io::read_parquet(read_opts);

    // we should get empty columns back
    EXPECT_EQ(result.tbl->view().num_columns(), 4);
    EXPECT_EQ(result.tbl->view().column(0).size(), 0);
  }
}

TEST_F(ParquetReaderTest, UserBoundsWithNulls)
{
  // clang-format off
  cudf::test::fixed_width_column_wrapper<float> col{{1,1,1,1,1,1,1,1, 2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3, 4,4,4,4,4,4,4,4,  5,5,5,5,5,5,5,5, 6,6,6,6,6,6,6,6, 7,7,7,7,7,7,7,7, 8,8,8,8,8,8,8,8}
                                                   ,{true,true,true,false,false,false,true,true, true,true,true,true,true,true,true,true, false,false,false,false,false,false,false,false, true,true,true,true,true,true,false,false,  true,false,true,true,true,true,true,true, true,true,true,true,true,true,true,true, true,true,true,true,true,true,true,true, true,true,true,true,true,true,true,false}};
  // clang-format on
  cudf::table_view tbl({col});
  auto filepath = temp_env->get_temp_filepath("UserBoundsWithNulls.parquet");
  cudf::io::parquet_writer_options out_args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, tbl);
  cudf::io::write_parquet(out_args);

  // skip_rows / num_rows
  // clang-format off
  std::vector<std::pair<int, int>> params{ {-1, -1}, {1, 3}, {3, -1},
                                           {31, -1}, {32, -1}, {33, -1},
                                           {31, 5}, {32, 5}, {33, 5},
                                           {-1, 7}, {-1, 31}, {-1, 32}, {-1, 33},
                                           {62, -1}, {63, -1},
                                           {62, 2}, {63, 1}};
  // clang-format on
  for (auto p : params) {
    cudf::io::parquet_reader_options read_args =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
    if (p.first >= 0) { read_args.set_skip_rows(p.first); }
    if (p.second >= 0) { read_args.set_num_rows(p.second); }
    auto result = cudf::io::read_parquet(read_args);

    p.first  = p.first < 0 ? 0 : p.first;
    p.second = p.second < 0 ? static_cast<cudf::column_view>(col).size() - p.first : p.second;
    std::vector<cudf::size_type> slice_indices{p.first, p.first + p.second};
    auto expected = cudf::slice(col, slice_indices);

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(0), expected[0]);
  }
}

TEST_F(ParquetReaderTest, UserBoundsWithNullsMixedTypes)
{
  constexpr int num_rows = 32 * 1024;

  std::mt19937 gen(6542);
  std::bernoulli_distribution bn(0.7f);
  auto valids =
    cudf::detail::make_counting_transform_iterator(0, [&](int index) { return bn(gen); });
  auto values = thrust::make_counting_iterator(0);

  // int64
  cudf::test::fixed_width_column_wrapper<int64_t> c0(values, values + num_rows, valids);

  // list<float>
  constexpr int floats_per_row = 4;
  auto c1_offset_iter          = cudf::detail::make_counting_transform_iterator(
    0, [](cudf::size_type idx) { return idx * floats_per_row; });
  cudf::test::fixed_width_column_wrapper<cudf::size_type> c1_offsets(c1_offset_iter,
                                                                     c1_offset_iter + num_rows + 1);
  cudf::test::fixed_width_column_wrapper<float> c1_floats(
    values, values + (num_rows * floats_per_row), valids);
  auto [null_mask, null_count] = cudf::test::detail::make_null_mask(valids, valids + num_rows);

  auto _c1 = cudf::make_lists_column(
    num_rows, c1_offsets.release(), c1_floats.release(), null_count, std::move(null_mask));
  auto c1 = cudf::purge_nonempty_nulls(*_c1);

  // list<list<int>>
  auto c2 = make_parquet_list_list_col<int>(0, num_rows, 5, 8, true);

  // struct<list<string>, int, float>
  std::vector<std::string> strings{
    "abc", "x", "bananas", "gpu", "minty", "backspace", "", "cayenne", "turbine", "soft"};
  std::uniform_int_distribution<int> uni(0, strings.size() - 1);
  auto string_iter = cudf::detail::make_counting_transform_iterator(
    0, [&](cudf::size_type idx) { return strings[uni(gen)]; });
  constexpr int string_per_row  = 3;
  constexpr int num_string_rows = num_rows * string_per_row;
  cudf::test::strings_column_wrapper string_col{string_iter, string_iter + num_string_rows};
  auto offset_iter = cudf::detail::make_counting_transform_iterator(
    0, [](cudf::size_type idx) { return idx * string_per_row; });
  cudf::test::fixed_width_column_wrapper<cudf::size_type> offsets(offset_iter,
                                                                  offset_iter + num_rows + 1);

  auto _c3_valids =
    cudf::detail::make_counting_transform_iterator(0, [&](int index) { return index % 200; });
  std::vector<bool> c3_valids(num_rows);
  std::copy(_c3_valids, _c3_valids + num_rows, c3_valids.begin());
  std::tie(null_mask, null_count) = cudf::test::detail::make_null_mask(valids, valids + num_rows);
  auto _c3_list                   = cudf::make_lists_column(
    num_rows, offsets.release(), string_col.release(), null_count, std::move(null_mask));
  auto c3_list = cudf::purge_nonempty_nulls(*_c3_list);
  cudf::test::fixed_width_column_wrapper<int> c3_ints(values, values + num_rows, valids);
  cudf::test::fixed_width_column_wrapper<float> c3_floats(values, values + num_rows, valids);
  std::vector<std::unique_ptr<cudf::column>> c3_children;
  c3_children.push_back(std::move(c3_list));
  c3_children.push_back(c3_ints.release());
  c3_children.push_back(c3_floats.release());
  cudf::test::structs_column_wrapper _c3(std::move(c3_children), c3_valids);
  auto c3 = cudf::purge_nonempty_nulls(_c3);

  // write it out
  cudf::table_view tbl({c0, *c1, *c2, *c3});
  auto filepath = temp_env->get_temp_filepath("UserBoundsWithNullsMixedTypes.parquet");
  cudf::io::parquet_writer_options out_args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, tbl);
  cudf::io::write_parquet(out_args);

  // read it back
  std::vector<std::pair<int, int>> params{
    {-1, -1}, {0, num_rows}, {1, num_rows - 1}, {num_rows - 1, 1}, {517, 22000}};
  for (auto p : params) {
    cudf::io::parquet_reader_options read_args =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
    if (p.first >= 0) { read_args.set_skip_rows(p.first); }
    if (p.second >= 0) { read_args.set_num_rows(p.second); }
    auto result = cudf::io::read_parquet(read_args);

    p.first  = p.first < 0 ? 0 : p.first;
    p.second = p.second < 0 ? num_rows - p.first : p.second;
    std::vector<cudf::size_type> slice_indices{p.first, p.first + p.second};
    auto expected = cudf::slice(tbl, slice_indices);

    CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, expected[0]);
  }
}

TEST_F(ParquetReaderTest, UserBoundsWithNullsLarge)
{
  constexpr int num_rows = 30 * 10000;

  std::mt19937 gen(6747);
  std::bernoulli_distribution bn(0.7f);
  auto valids =
    cudf::detail::make_counting_transform_iterator(0, [&](int index) { return bn(gen); });
  auto values = thrust::make_counting_iterator(0);

  cudf::test::fixed_width_column_wrapper<int> col(values, values + num_rows, valids);

  // this file will have row groups of 10,000 each
  cudf::table_view tbl({col});
  auto filepath = temp_env->get_temp_filepath("UserBoundsWithNullsLarge.parquet");
  cudf::io::parquet_writer_options out_args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, tbl)
      .row_group_size_rows(10000)
      .max_page_size_rows(1000);
  cudf::io::write_parquet(out_args);

  // skip_rows / num_rows
  // clang-format off
  std::vector<std::pair<int, int>> params{ {-1, -1}, {31, -1}, {32, -1}, {33, -1}, {16130, -1}, {19999, -1},
                                           {31, 1}, {32, 1}, {33, 1},
                                           // deliberately span some row group boundaries
                                           {9900, 1001}, {9900, 2000}, {29999, 2}, {139997, -1},
                                           {167878, 3}, {229976, 31},
                                           {240031, 17}, {290001, 9899}, {299999, 1} };
  // clang-format on
  for (auto p : params) {
    cudf::io::parquet_reader_options read_args =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
    if (p.first >= 0) { read_args.set_skip_rows(p.first); }
    if (p.second >= 0) { read_args.set_num_rows(p.second); }
    auto result = cudf::io::read_parquet(read_args);

    p.first  = p.first < 0 ? 0 : p.first;
    p.second = p.second < 0 ? static_cast<cudf::column_view>(col).size() - p.first : p.second;
    std::vector<cudf::size_type> slice_indices{p.first, p.first + p.second};
    auto expected = cudf::slice(col, slice_indices);

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(0), expected[0]);
  }
}

TEST_F(ParquetReaderTest, ListUserBoundsWithNullsLarge)
{
  constexpr int num_rows = 5 * 10000;
  auto colp              = make_parquet_list_list_col<int>(0, num_rows, 5, 8, true);
  cudf::column_view col  = *colp;

  // this file will have row groups of 10,000 each
  cudf::table_view tbl({col});
  auto filepath = temp_env->get_temp_filepath("ListUserBoundsWithNullsLarge.parquet");
  cudf::io::parquet_writer_options out_args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, tbl)
      .row_group_size_rows(10000)
      .max_page_size_rows(1000);
  cudf::io::write_parquet(out_args);

  // skip_rows / num_rows
  // clang-format off
  std::vector<std::pair<int, int>> params{ {-1, -1}, {31, -1}, {32, -1}, {33, -1}, {1670, -1}, {44997, -1},
                                           {31, 1}, {32, 1}, {33, 1},
                                           // deliberately span some row group boundaries
                                           {9900, 1001}, {9900, 2000}, {29999, 2},
                                           {16567, 3}, {42976, 31},
                                           {40231, 17}, {19000, 9899}, {49999, 1} };
  // clang-format on
  for (auto p : params) {
    cudf::io::parquet_reader_options read_args =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
    if (p.first >= 0) { read_args.set_skip_rows(p.first); }
    if (p.second >= 0) { read_args.set_num_rows(p.second); }
    auto result = cudf::io::read_parquet(read_args);

    p.first  = p.first < 0 ? 0 : p.first;
    p.second = p.second < 0 ? static_cast<cudf::column_view>(col).size() - p.first : p.second;
    std::vector<cudf::size_type> slice_indices{p.first, p.first + p.second};
    auto expected = cudf::slice(col, slice_indices);

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(0), expected[0]);
  }
}

TEST_F(ParquetReaderTest, ReorderedColumns)
{
  {
    auto a = cudf::test::strings_column_wrapper{{"a", "", "c"}, {true, false, true}};
    auto b = cudf::test::fixed_width_column_wrapper<int>{1, 2, 3};

    cudf::table_view tbl{{a, b}};
    auto filepath = temp_env->get_temp_filepath("ReorderedColumns.parquet");
    cudf::io::table_input_metadata md(tbl);
    md.column_metadata[0].set_name("a");
    md.column_metadata[1].set_name("b");
    cudf::io::parquet_writer_options opts =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, tbl).metadata(md);
    cudf::io::write_parquet(opts);

    // read them out of order
    cudf::io::parquet_reader_options read_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
        .columns({"b", "a"});
    auto result = cudf::io::read_parquet(read_opts);

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(0), b);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(1), a);
  }

  {
    auto a = cudf::test::fixed_width_column_wrapper<int>{1, 2, 3};
    auto b = cudf::test::strings_column_wrapper{{"a", "", "c"}, {true, false, true}};

    cudf::table_view tbl{{a, b}};
    auto filepath = temp_env->get_temp_filepath("ReorderedColumns2.parquet");
    cudf::io::table_input_metadata md(tbl);
    md.column_metadata[0].set_name("a");
    md.column_metadata[1].set_name("b");
    cudf::io::parquet_writer_options opts =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, tbl).metadata(md);
    cudf::io::write_parquet(opts);

    // read them out of order
    cudf::io::parquet_reader_options read_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
        .columns({"b", "a"});
    auto result = cudf::io::read_parquet(read_opts);

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(0), b);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(1), a);
  }

  auto a = cudf::test::fixed_width_column_wrapper<int>{1, 2, 3, 10, 20, 30};
  auto b = cudf::test::strings_column_wrapper{{"a", "", "c", "cats", "dogs", "owls"},
                                              {true, false, true, true, false, true}};
  auto c = cudf::test::fixed_width_column_wrapper<int>{{15, 16, 17, 25, 26, 32},
                                                       {false, true, true, true, true, false}};
  auto d = cudf::test::strings_column_wrapper{"ducks", "sheep", "cows", "fish", "birds", "ants"};

  cudf::table_view tbl{{a, b, c, d}};
  auto filepath = temp_env->get_temp_filepath("ReorderedColumns3.parquet");
  cudf::io::table_input_metadata md(tbl);
  md.column_metadata[0].set_name("a");
  md.column_metadata[1].set_name("b");
  md.column_metadata[2].set_name("c");
  md.column_metadata[3].set_name("d");
  cudf::io::parquet_writer_options opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, tbl)
      .metadata(std::move(md));
  cudf::io::write_parquet(opts);

  {
    // read them out of order
    cudf::io::parquet_reader_options read_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
        .columns({"d", "a", "b", "c"});
    auto result = cudf::io::read_parquet(read_opts);

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(0), d);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(1), a);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(2), b);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(3), c);
  }

  {
    // read them out of order
    cudf::io::parquet_reader_options read_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
        .columns({"c", "d", "a", "b"});
    auto result = cudf::io::read_parquet(read_opts);

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(0), c);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(1), d);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(2), a);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(3), b);
  }

  {
    // read them out of order
    cudf::io::parquet_reader_options read_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
        .columns({"d", "c", "b", "a"});
    auto result = cudf::io::read_parquet(read_opts);

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(0), d);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(1), c);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(2), b);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(3), a);
  }
}

TEST_F(ParquetReaderTest, SelectNestedColumn)
{
  // Struct<is_human:bool,
  //        Struct<weight:float,
  //               ages:int,
  //               land_unit:List<int>>,
  //               flats:List<List<int>>
  //              >
  //       >

  auto weights_col = cudf::test::fixed_width_column_wrapper<float>{1.1, 2.4, 5.3, 8.0, 9.6, 6.9};

  auto ages_col = cudf::test::fixed_width_column_wrapper<int32_t>{
    {48, 27, 25, 31, 351, 351}, {true, true, true, true, true, false}};

  auto struct_1 = cudf::test::structs_column_wrapper{{weights_col, ages_col},
                                                     {true, true, true, true, false, true}};

  auto is_human_col = cudf::test::fixed_width_column_wrapper<bool>{
    {true, true, false, false, false, false}, {true, true, false, true, true, false}};

  auto struct_2 = cudf::test::structs_column_wrapper{{is_human_col, struct_1},
                                                     {false, true, true, true, true, true}}
                    .release();

  auto input = table_view({*struct_2});

  cudf::io::table_input_metadata input_metadata(input);
  input_metadata.column_metadata[0].set_name("being");
  input_metadata.column_metadata[0].child(0).set_name("human?");
  input_metadata.column_metadata[0].child(1).set_name("particulars");
  input_metadata.column_metadata[0].child(1).child(0).set_name("weight");
  input_metadata.column_metadata[0].child(1).child(1).set_name("age");

  auto filepath = temp_env->get_temp_filepath("SelectNestedColumn.parquet");
  cudf::io::parquet_writer_options args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, input)
      .metadata(std::move(input_metadata));
  cudf::io::write_parquet(args);

  {  // Test selecting a single leaf from the table
    cudf::io::parquet_reader_options read_args =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info(filepath))
        .columns({"being.particulars.age"});
    auto const result = cudf::io::read_parquet(read_args);

    auto expect_ages_col = cudf::test::fixed_width_column_wrapper<int32_t>{
      {48, 27, 25, 31, 351, 351}, {true, true, true, true, true, false}};
    auto expect_s_1 =
      cudf::test::structs_column_wrapper{{expect_ages_col}, {true, true, true, true, false, true}};
    auto expect_s_2 =
      cudf::test::structs_column_wrapper{{expect_s_1}, {false, true, true, true, true, true}}
        .release();
    auto expected = table_view({*expect_s_2});

    cudf::io::table_input_metadata expected_metadata(expected);
    expected_metadata.column_metadata[0].set_name("being");
    expected_metadata.column_metadata[0].child(0).set_name("particulars");
    expected_metadata.column_metadata[0].child(0).child(0).set_name("age");

    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
    cudf::test::expect_metadata_equal(expected_metadata, result.metadata);
  }

  {  // Test selecting a non-leaf and expecting all hierarchy from that node onwards
    cudf::io::parquet_reader_options read_args =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info(filepath))
        .columns({"being.particulars"});
    auto const result = cudf::io::read_parquet(read_args);

    auto expected_weights_col =
      cudf::test::fixed_width_column_wrapper<float>{1.1, 2.4, 5.3, 8.0, 9.6, 6.9};

    auto expected_ages_col = cudf::test::fixed_width_column_wrapper<int32_t>{
      {48, 27, 25, 31, 351, 351}, {true, true, true, true, true, false}};

    auto expected_s_1 = cudf::test::structs_column_wrapper{
      {expected_weights_col, expected_ages_col}, {true, true, true, true, false, true}};

    auto expect_s_2 =
      cudf::test::structs_column_wrapper{{expected_s_1}, {false, true, true, true, true, true}}
        .release();
    auto expected = table_view({*expect_s_2});

    cudf::io::table_input_metadata expected_metadata(expected);
    expected_metadata.column_metadata[0].set_name("being");
    expected_metadata.column_metadata[0].child(0).set_name("particulars");
    expected_metadata.column_metadata[0].child(0).child(0).set_name("weight");
    expected_metadata.column_metadata[0].child(0).child(1).set_name("age");

    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
    cudf::test::expect_metadata_equal(expected_metadata, result.metadata);
  }

  {  // Test selecting struct children out of order
    cudf::io::parquet_reader_options read_args =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info(filepath))
        .columns({"being.particulars.age", "being.particulars.weight", "being.human?"});
    auto const result = cudf::io::read_parquet(read_args);

    auto expected_weights_col =
      cudf::test::fixed_width_column_wrapper<float>{1.1, 2.4, 5.3, 8.0, 9.6, 6.9};

    auto expected_ages_col = cudf::test::fixed_width_column_wrapper<int32_t>{
      {48, 27, 25, 31, 351, 351}, {true, true, true, true, true, false}};

    auto expected_is_human_col = cudf::test::fixed_width_column_wrapper<bool>{
      {true, true, false, false, false, false}, {true, true, false, true, true, false}};

    auto expect_s_1 = cudf::test::structs_column_wrapper{{expected_ages_col, expected_weights_col},
                                                         {true, true, true, true, false, true}};

    auto expect_s_2 = cudf::test::structs_column_wrapper{{expect_s_1, expected_is_human_col},
                                                         {false, true, true, true, true, true}}
                        .release();

    auto expected = table_view({*expect_s_2});

    cudf::io::table_input_metadata expected_metadata(expected);
    expected_metadata.column_metadata[0].set_name("being");
    expected_metadata.column_metadata[0].child(0).set_name("particulars");
    expected_metadata.column_metadata[0].child(0).child(0).set_name("age");
    expected_metadata.column_metadata[0].child(0).child(1).set_name("weight");
    expected_metadata.column_metadata[0].child(1).set_name("human?");

    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
    cudf::test::expect_metadata_equal(expected_metadata, result.metadata);
  }
}

TEST_F(ParquetReaderTest, EmptyOutput)
{
  cudf::test::fixed_width_column_wrapper<int> c0;
  cudf::test::strings_column_wrapper c1;
  cudf::test::fixed_point_column_wrapper<int> c2({}, numeric::scale_type{2});
  cudf::test::lists_column_wrapper<float> _c3{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}};
  auto c3 = cudf::empty_like(_c3);

  cudf::test::fixed_width_column_wrapper<int> sc0;
  cudf::test::strings_column_wrapper sc1;
  cudf::test::lists_column_wrapper<int> _sc2{{1, 2}};
  std::vector<std::unique_ptr<cudf::column>> struct_children;
  struct_children.push_back(sc0.release());
  struct_children.push_back(sc1.release());
  struct_children.push_back(cudf::empty_like(_sc2));
  cudf::test::structs_column_wrapper c4(std::move(struct_children));

  table_view expected({c0, c1, c2, *c3, c4});

  // set precision on the decimal column
  cudf::io::table_input_metadata expected_metadata(expected);
  expected_metadata.column_metadata[2].set_decimal_precision(1);

  auto filepath = temp_env->get_temp_filepath("EmptyOutput.parquet");
  cudf::io::parquet_writer_options out_args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected);
  out_args.set_metadata(std::move(expected_metadata));
  cudf::io::write_parquet(out_args);

  cudf::io::parquet_reader_options read_args =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(read_args);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
}

TEST_F(ParquetReaderTest, EmptyColumnsParam)
{
  srand(31337);
  auto const expected = create_random_fixed_table<int>(2, 4, false);

  std::vector<char> out_buffer;
  cudf::io::parquet_writer_options args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{&out_buffer}, *expected);
  cudf::io::write_parquet(args);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(out_buffer.data()), out_buffer.size()}})
      .columns({});
  auto const result = cudf::io::read_parquet(read_opts);

  EXPECT_EQ(result.tbl->num_columns(), 0);
  EXPECT_EQ(result.tbl->num_rows(), 0);
}

TEST_F(ParquetReaderTest, BinaryAsStrings)
{
  std::vector<char const*> strings{
    "Monday", "Wednesday", "Friday", "Monday", "Friday", "Friday", "Friday", "Funday"};
  auto const num_rows = strings.size();

  auto seq_col0 = random_values<int>(num_rows);
  auto seq_col2 = random_values<float>(num_rows);
  auto seq_col3 = random_values<uint8_t>(num_rows);
  auto validity = cudf::test::iterators::no_nulls();

  column_wrapper<int> int_col{seq_col0.begin(), seq_col0.end(), validity};
  column_wrapper<cudf::string_view> string_col{strings.begin(), strings.end()};
  column_wrapper<float> float_col{seq_col2.begin(), seq_col2.end(), validity};
  cudf::test::lists_column_wrapper<uint8_t> list_int_col{
    {'M', 'o', 'n', 'd', 'a', 'y'},
    {'W', 'e', 'd', 'n', 'e', 's', 'd', 'a', 'y'},
    {'F', 'r', 'i', 'd', 'a', 'y'},
    {'M', 'o', 'n', 'd', 'a', 'y'},
    {'F', 'r', 'i', 'd', 'a', 'y'},
    {'F', 'r', 'i', 'd', 'a', 'y'},
    {'F', 'r', 'i', 'd', 'a', 'y'},
    {'F', 'u', 'n', 'd', 'a', 'y'}};

  auto output = table_view{{int_col, string_col, float_col, string_col, list_int_col}};
  cudf::io::table_input_metadata output_metadata(output);
  output_metadata.column_metadata[0].set_name("col_other");
  output_metadata.column_metadata[1].set_name("col_string");
  output_metadata.column_metadata[2].set_name("col_float");
  output_metadata.column_metadata[3].set_name("col_string2").set_output_as_binary(true);
  output_metadata.column_metadata[4].set_name("col_binary").set_output_as_binary(true);

  auto filepath = temp_env->get_temp_filepath("BinaryReadStrings.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, output)
      .metadata(std::move(output_metadata));
  cudf::io::write_parquet(out_opts);

  auto expected_string = table_view{{int_col, string_col, float_col, string_col, string_col}};
  auto expected_mixed  = table_view{{int_col, string_col, float_col, list_int_col, list_int_col}};

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
      .set_column_schema({{}, {}, {}, {}, {}});
  auto result = cudf::io::read_parquet(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_string, result.tbl->view());

  cudf::io::parquet_reader_options default_in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  result = cudf::io::read_parquet(default_in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_string, result.tbl->view());

  std::vector<cudf::io::reader_column_schema> md{
    {},
    {},
    {},
    cudf::io::reader_column_schema().set_convert_binary_to_strings(false),
    cudf::io::reader_column_schema().set_convert_binary_to_strings(false)};

  cudf::io::parquet_reader_options mixed_in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
      .set_column_schema(md);
  result = cudf::io::read_parquet(mixed_in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_mixed, result.tbl->view());
}

TEST_F(ParquetReaderTest, NestedByteArray)
{
  constexpr auto num_rows = 8;

  auto seq_col0       = random_values<int>(num_rows);
  auto seq_col2       = random_values<float>(num_rows);
  auto seq_col3       = random_values<uint8_t>(num_rows);
  auto const validity = cudf::test::iterators::no_nulls();

  column_wrapper<int> int_col{seq_col0.begin(), seq_col0.end(), validity};
  column_wrapper<float> float_col{seq_col2.begin(), seq_col2.end(), validity};
  cudf::test::lists_column_wrapper<uint8_t> list_list_int_col{
    {{'M', 'o', 'n', 'd', 'a', 'y'},
     {'W', 'e', 'd', 'n', 'e', 's', 'd', 'a', 'y'},
     {'F', 'r', 'i', 'd', 'a', 'y'}},
    {{'M', 'o', 'n', 'd', 'a', 'y'}, {'F', 'r', 'i', 'd', 'a', 'y'}},
    {{'M', 'o', 'n', 'd', 'a', 'y'},
     {'W', 'e', 'd', 'n', 'e', 's', 'd', 'a', 'y'},
     {'F', 'r', 'i', 'd', 'a', 'y'}},
    {{'F', 'r', 'i', 'd', 'a', 'y'},
     {'F', 'r', 'i', 'd', 'a', 'y'},
     {'F', 'u', 'n', 'd', 'a', 'y'}},
    {{'M', 'o', 'n', 'd', 'a', 'y'},
     {'W', 'e', 'd', 'n', 'e', 's', 'd', 'a', 'y'},
     {'F', 'r', 'i', 'd', 'a', 'y'}},
    {{'F', 'r', 'i', 'd', 'a', 'y'},
     {'F', 'r', 'i', 'd', 'a', 'y'},
     {'F', 'u', 'n', 'd', 'a', 'y'}},
    {{'M', 'o', 'n', 'd', 'a', 'y'},
     {'W', 'e', 'd', 'n', 'e', 's', 'd', 'a', 'y'},
     {'F', 'r', 'i', 'd', 'a', 'y'}},
    {{'M', 'o', 'n', 'd', 'a', 'y'}, {'F', 'r', 'i', 'd', 'a', 'y'}}};

  auto const expected = table_view{{int_col, float_col, list_list_int_col}};
  cudf::io::table_input_metadata output_metadata(expected);
  output_metadata.column_metadata[0].set_name("col_other");
  output_metadata.column_metadata[1].set_name("col_float");
  output_metadata.column_metadata[2].set_name("col_binary").child(1).set_output_as_binary(true);

  auto filepath = temp_env->get_temp_filepath("NestedByteArray.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .metadata(std::move(output_metadata));
  cudf::io::write_parquet(out_opts);

  auto source = cudf::io::datasource::create(filepath);
  cudf::io::parquet::FileMetaData fmd;

  read_footer(source, &fmd);
  EXPECT_EQ(fmd.schema[5].type, cudf::io::parquet::Type::BYTE_ARRAY);

  std::vector<cudf::io::reader_column_schema> md{
    {},
    {},
    cudf::io::reader_column_schema().add_child(
      cudf::io::reader_column_schema().set_convert_binary_to_strings(false))};

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
      .set_column_schema(md);
  auto result = cudf::io::read_parquet(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
}

TEST_F(ParquetReaderTest, StructByteArray)
{
  constexpr auto num_rows = 100;

  auto seq_col0       = random_values<uint8_t>(num_rows);
  auto const validity = cudf::test::iterators::no_nulls();

  column_wrapper<uint8_t> int_col{seq_col0.begin(), seq_col0.end(), validity};
  cudf::test::lists_column_wrapper<uint8_t> list_of_int{{seq_col0.begin(), seq_col0.begin() + 50},
                                                        {seq_col0.begin() + 50, seq_col0.end()}};
  auto struct_col = cudf::test::structs_column_wrapper{{list_of_int}, validity};

  auto const expected = table_view{{struct_col}};
  EXPECT_EQ(1, expected.num_columns());
  cudf::io::table_input_metadata output_metadata(expected);
  output_metadata.column_metadata[0]
    .set_name("struct_binary")
    .child(0)
    .set_name("a")
    .set_output_as_binary(true);

  auto filepath = temp_env->get_temp_filepath("StructByteArray.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .metadata(std::move(output_metadata));
  cudf::io::write_parquet(out_opts);

  std::vector<cudf::io::reader_column_schema> md{cudf::io::reader_column_schema().add_child(
    cudf::io::reader_column_schema().set_convert_binary_to_strings(false))};

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
      .set_column_schema(md);
  auto result = cudf::io::read_parquet(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
}

TEST_F(ParquetReaderTest, NestingOptimizationTest)
{
  // test nesting levels > cudf::io::parquet::detail::max_cacheable_nesting_decode_info deep.
  constexpr cudf::size_type num_nesting_levels = 16;
  static_assert(num_nesting_levels > cudf::io::parquet::detail::max_cacheable_nesting_decode_info);
  constexpr cudf::size_type rows_per_level = 2;

  constexpr cudf::size_type num_values = (1 << num_nesting_levels) * rows_per_level;
  auto value_iter                      = thrust::make_counting_iterator(0);
  auto validity =
    cudf::detail::make_counting_transform_iterator(0, [](cudf::size_type i) { return i % 2; });
  cudf::test::fixed_width_column_wrapper<int> values(value_iter, value_iter + num_values, validity);

  // ~256k values with num_nesting_levels = 16
  auto prev_col = values.release();
  for (int idx = 0; idx < num_nesting_levels; idx++) {
    auto const num_rows = (1 << (num_nesting_levels - idx));

    auto offsets_iter = cudf::detail::make_counting_transform_iterator(
      0, [](cudf::size_type i) { return i * rows_per_level; });

    cudf::test::fixed_width_column_wrapper<cudf::size_type> offsets(offsets_iter,
                                                                    offsets_iter + num_rows + 1);
    auto c   = cudf::make_lists_column(num_rows, offsets.release(), std::move(prev_col), 0, {});
    prev_col = std::move(c);
  }
  auto const& expect = prev_col;

  auto filepath = temp_env->get_temp_filepath("NestingDecodeCache.parquet");
  cudf::io::parquet_writer_options opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, table_view{{*expect}});
  cudf::io::write_parquet(opts);

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(in_opts);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expect, result.tbl->get_column(0));
}

TEST_F(ParquetReaderTest, SingleLevelLists)
{
  std::array<unsigned char, 214> list_bytes{
    0x50, 0x41, 0x52, 0x31, 0x15, 0x00, 0x15, 0x28, 0x15, 0x28, 0x15, 0xa7, 0xce, 0x91, 0x8c, 0x06,
    0x1c, 0x15, 0x04, 0x15, 0x00, 0x15, 0x06, 0x15, 0x06, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03,
    0x02, 0x02, 0x00, 0x00, 0x00, 0x03, 0x03, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x15,
    0x02, 0x19, 0x3c, 0x48, 0x0c, 0x73, 0x70, 0x61, 0x72, 0x6b, 0x5f, 0x73, 0x63, 0x68, 0x65, 0x6d,
    0x61, 0x15, 0x02, 0x00, 0x35, 0x00, 0x18, 0x01, 0x66, 0x15, 0x02, 0x15, 0x06, 0x4c, 0x3c, 0x00,
    0x00, 0x00, 0x15, 0x02, 0x25, 0x04, 0x18, 0x05, 0x61, 0x72, 0x72, 0x61, 0x79, 0x00, 0x16, 0x02,
    0x19, 0x1c, 0x19, 0x1c, 0x26, 0x08, 0x1c, 0x15, 0x02, 0x19, 0x25, 0x00, 0x06, 0x19, 0x28, 0x01,
    0x66, 0x05, 0x61, 0x72, 0x72, 0x61, 0x79, 0x15, 0x00, 0x16, 0x04, 0x16, 0x56, 0x16, 0x56, 0x26,
    0x08, 0x3c, 0x18, 0x04, 0x01, 0x00, 0x00, 0x00, 0x18, 0x04, 0x00, 0x00, 0x00, 0x00, 0x16, 0x00,
    0x28, 0x04, 0x01, 0x00, 0x00, 0x00, 0x18, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x19, 0x1c, 0x15,
    0x00, 0x15, 0x00, 0x15, 0x02, 0x00, 0x00, 0x00, 0x16, 0x56, 0x16, 0x02, 0x26, 0x08, 0x16, 0x56,
    0x14, 0x00, 0x00, 0x28, 0x13, 0x52, 0x41, 0x50, 0x49, 0x44, 0x53, 0x20, 0x53, 0x70, 0x61, 0x72,
    0x6b, 0x20, 0x50, 0x6c, 0x75, 0x67, 0x69, 0x6e, 0x19, 0x1c, 0x1c, 0x00, 0x00, 0x00, 0x9f, 0x00,
    0x00, 0x00, 0x50, 0x41, 0x52, 0x31};

  // read single level list reproducing parquet file
  cudf::io::parquet_reader_options read_opts = cudf::io::parquet_reader_options::builder(
    cudf::io::source_info{cudf::host_span<std::byte const>{
      reinterpret_cast<std::byte const*>(list_bytes.data()), list_bytes.size()}});
  auto table = cudf::io::read_parquet(read_opts);

  auto const c0 = table.tbl->get_column(0);
  EXPECT_TRUE(c0.type().id() == cudf::type_id::LIST);

  auto const lc    = cudf::lists_column_view(c0);
  auto const child = lc.child();
  EXPECT_TRUE(child.type().id() == cudf::type_id::INT32);
}

TEST_F(ParquetReaderTest, ChunkedSingleLevelLists)
{
  std::array<unsigned char, 214> list_bytes{
    0x50, 0x41, 0x52, 0x31, 0x15, 0x00, 0x15, 0x28, 0x15, 0x28, 0x15, 0xa7, 0xce, 0x91, 0x8c, 0x06,
    0x1c, 0x15, 0x04, 0x15, 0x00, 0x15, 0x06, 0x15, 0x06, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03,
    0x02, 0x02, 0x00, 0x00, 0x00, 0x03, 0x03, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x15,
    0x02, 0x19, 0x3c, 0x48, 0x0c, 0x73, 0x70, 0x61, 0x72, 0x6b, 0x5f, 0x73, 0x63, 0x68, 0x65, 0x6d,
    0x61, 0x15, 0x02, 0x00, 0x35, 0x00, 0x18, 0x01, 0x66, 0x15, 0x02, 0x15, 0x06, 0x4c, 0x3c, 0x00,
    0x00, 0x00, 0x15, 0x02, 0x25, 0x04, 0x18, 0x05, 0x61, 0x72, 0x72, 0x61, 0x79, 0x00, 0x16, 0x02,
    0x19, 0x1c, 0x19, 0x1c, 0x26, 0x08, 0x1c, 0x15, 0x02, 0x19, 0x25, 0x00, 0x06, 0x19, 0x28, 0x01,
    0x66, 0x05, 0x61, 0x72, 0x72, 0x61, 0x79, 0x15, 0x00, 0x16, 0x04, 0x16, 0x56, 0x16, 0x56, 0x26,
    0x08, 0x3c, 0x18, 0x04, 0x01, 0x00, 0x00, 0x00, 0x18, 0x04, 0x00, 0x00, 0x00, 0x00, 0x16, 0x00,
    0x28, 0x04, 0x01, 0x00, 0x00, 0x00, 0x18, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x19, 0x1c, 0x15,
    0x00, 0x15, 0x00, 0x15, 0x02, 0x00, 0x00, 0x00, 0x16, 0x56, 0x16, 0x02, 0x26, 0x08, 0x16, 0x56,
    0x14, 0x00, 0x00, 0x28, 0x13, 0x52, 0x41, 0x50, 0x49, 0x44, 0x53, 0x20, 0x53, 0x70, 0x61, 0x72,
    0x6b, 0x20, 0x50, 0x6c, 0x75, 0x67, 0x69, 0x6e, 0x19, 0x1c, 0x1c, 0x00, 0x00, 0x00, 0x9f, 0x00,
    0x00, 0x00, 0x50, 0x41, 0x52, 0x31};

  auto reader = cudf::io::chunked_parquet_reader(
    1L << 31,
    cudf::io::parquet_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(list_bytes.data()), list_bytes.size()}}));
  int iterations = 0;
  while (reader.has_next() && iterations < 10) {
    auto chunk = reader.read_chunk();
  }
  EXPECT_TRUE(iterations < 10);
}

TEST_F(ParquetReaderTest, ReorderedReadMultipleFiles)
{
  constexpr auto num_rows    = 50'000;
  constexpr auto cardinality = 20'000;

  // table 1
  auto str1 = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return "cat " + std::to_string(i % cardinality); });
  auto cols1 = cudf::test::strings_column_wrapper(str1, str1 + num_rows);

  auto int1 =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % cardinality; });
  auto coli1 = cudf::test::fixed_width_column_wrapper<int>(int1, int1 + num_rows);

  auto const expected1 = table_view{{cols1, coli1}};
  auto const swapped1  = table_view{{coli1, cols1}};

  auto const filepath1 = temp_env->get_temp_filepath("LargeReorderedRead1.parquet");
  auto out_opts1 =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath1}, expected1)
      .compression(cudf::io::compression_type::NONE);
  cudf::io::write_parquet(out_opts1);

  // table 2
  auto str2 = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return "dog " + std::to_string(i % cardinality); });
  auto cols2 = cudf::test::strings_column_wrapper(str2, str2 + num_rows);

  auto int2 = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return (i % cardinality) + cardinality; });
  auto coli2 = cudf::test::fixed_width_column_wrapper<int>(int2, int2 + num_rows);

  auto const expected2 = table_view{{cols2, coli2}};
  auto const swapped2  = table_view{{coli2, cols2}};

  auto const filepath2 = temp_env->get_temp_filepath("LargeReorderedRead2.parquet");
  auto out_opts2 =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath2}, expected2)
      .compression(cudf::io::compression_type::NONE);
  cudf::io::write_parquet(out_opts2);

  // read in both files swapping the columns
  auto read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{{filepath1, filepath2}})
      .columns({"_col1", "_col0"});
  auto result = cudf::io::read_parquet(read_opts);
  auto sliced = cudf::slice(result.tbl->view(), {0, num_rows, num_rows, 2 * num_rows});
  CUDF_TEST_EXPECT_TABLES_EQUAL(sliced[0], swapped1);
  CUDF_TEST_EXPECT_TABLES_EQUAL(sliced[1], swapped2);
}

TEST_F(ParquetReaderTest, NoFilter)
{
  srand(31337);
  auto expected = create_random_fixed_table<int>(9, 9, false);

  auto filepath = temp_env->get_temp_filepath("FilterSimple.parquet");
  cudf::io::parquet_writer_options args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, *expected);
  cudf::io::write_parquet(args);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *expected);
  EXPECT_EQ(result.metadata.num_input_row_groups, 1);
  EXPECT_FALSE(result.metadata.num_row_groups_after_stats_filter.has_value());
  EXPECT_FALSE(result.metadata.num_row_groups_after_bloom_filter.has_value());
}

TEST_F(ParquetReaderTest, FilterSimple)
{
  srand(31337);
  auto written_table = create_random_fixed_table<int>(9, 9, false);

  auto filepath = temp_env->get_temp_filepath("FilterSimple.parquet");
  cudf::io::parquet_writer_options args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, *written_table);
  cudf::io::write_parquet(args);

  // Filtering AST - table[0] < RAND_MAX/2
  auto literal_value     = cudf::numeric_scalar<decltype(RAND_MAX)>(RAND_MAX / 2);
  auto literal           = cudf::ast::literal(literal_value);
  auto col_ref_0         = cudf::ast::column_reference(0);
  auto filter_expression = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, literal);

  auto predicate = cudf::compute_column(*written_table, filter_expression);
  EXPECT_EQ(predicate->view().type().id(), cudf::type_id::BOOL8)
    << "Predicate filter should return a boolean";
  auto expected = cudf::apply_boolean_mask(*written_table, *predicate);
  // To make sure AST filters out some elements
  EXPECT_LT(expected->num_rows(), written_table->num_rows());

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
      .filter(filter_expression);
  auto result = cudf::io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *expected);
}

auto create_parquet_with_stats(std::string const& filename)
{
  auto col0 = testdata::ascending<uint32_t>();
  auto col1 = testdata::descending<int64_t>();
  auto col2 = testdata::unordered<double>();

  auto const expected = table_view{{col0, col1, col2}};

  cudf::io::table_input_metadata expected_metadata(expected);
  expected_metadata.column_metadata[0].set_name("col_uint32");
  expected_metadata.column_metadata[1].set_name("col_int64");
  expected_metadata.column_metadata[2].set_name("col_double");

  auto const filepath = temp_env->get_temp_filepath(filename);
  const cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .metadata(std::move(expected_metadata))
      .row_group_size_rows(8000)
      .stats_level(cudf::io::statistics_freq::STATISTICS_ROWGROUP);
  cudf::io::write_parquet(out_opts);

  std::vector<std::unique_ptr<column>> columns;
  columns.push_back(col0.release());
  columns.push_back(col1.release());
  columns.push_back(col2.release());

  return std::pair{cudf::table{std::move(columns)}, filepath};
}

TEST_F(ParquetReaderTest, FilterIdentity)
{
  auto [src, filepath] = create_parquet_with_stats("FilterIdentity.parquet");

  // Filtering AST - identity function, always true.
  auto literal_value     = cudf::numeric_scalar<bool>(true);
  auto literal           = cudf::ast::literal(literal_value);
  auto filter_expression = cudf::ast::operation(cudf::ast::ast_operator::IDENTITY, literal);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
      .filter(filter_expression);
  auto result = cudf::io::read_parquet(read_opts);

  cudf::io::parquet_reader_options read_opts2 =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result2 = cudf::io::read_parquet(read_opts2);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *result2.tbl);
}

TEST_F(ParquetReaderTest, FilterWithColumnProjection)
{
  // col_uint32, col_int64, col_double
  auto [src, filepath] = create_parquet_with_stats("FilterWithColumnProjection.parquet");
  auto val             = cudf::numeric_scalar<uint32_t>{10};
  auto lit             = cudf::ast::literal{val};
  auto col_ref         = cudf::ast::column_name_reference{"col_uint32"};
  auto col_index       = cudf::ast::column_reference{0};
  auto filter_expr     = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_index, lit);

  auto predicate = cudf::compute_column(src, filter_expr);

  {  // column_name_reference in parquet filter (not present in column projection)
    auto read_expr       = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref, lit);
    auto projected_table = cudf::table_view{{src.get_column(2)}};
    auto expected        = cudf::apply_boolean_mask(projected_table, *predicate);

    auto read_opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                       .columns({"col_double"})
                       .filter(read_expr);
    auto result = cudf::io::read_parquet(read_opts);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *expected);
  }

  {  // column_reference in parquet filter (indices as per order of column projection)
    auto col_index2    = cudf::ast::column_reference{1};
    auto read_ref_expr = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_index2, lit);

    auto projected_table = cudf::table_view{{src.get_column(2), src.get_column(0)}};
    auto expected        = cudf::apply_boolean_mask(projected_table, *predicate);
    auto read_opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                       .columns({"col_double", "col_uint32"})
                       .filter(read_ref_expr);
    auto result = cudf::io::read_parquet(read_opts);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *expected);
  }

  // Error cases
  {  // column_reference is not same type as literal, column_reference index is out of bounds
    for (auto const index : {0, 2}) {
      auto col_index2    = cudf::ast::column_reference{index};
      auto read_ref_expr = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_index2, lit);
      auto read_opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                         .columns({"col_double", "col_uint32"})
                         .filter(read_ref_expr);
      EXPECT_THROW(cudf::io::read_parquet(read_opts), cudf::logic_error);
    }
  }
}

TEST_F(ParquetReaderTest, FilterReferenceExpression)
{
  auto [src, filepath] = create_parquet_with_stats("FilterReferenceExpression.parquet");
  // Filtering AST - table[0] < 150
  auto literal_value     = cudf::numeric_scalar<uint32_t>(150);
  auto literal           = cudf::ast::literal(literal_value);
  auto col_ref_0         = cudf::ast::column_reference(0);
  auto filter_expression = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, literal);

  // Expected result
  auto predicate = cudf::compute_column(src, filter_expression);
  auto expected  = cudf::apply_boolean_mask(src, *predicate);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
      .filter(filter_expression);
  auto result = cudf::io::read_parquet(read_opts);
  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *expected);
}

TEST_F(ParquetReaderTest, FilterNamedExpression)
{
  auto [src, filepath] = create_parquet_with_stats("NamedExpression.parquet");
  // Filtering AST - table["col_uint32"] < 150
  auto literal_value  = cudf::numeric_scalar<uint32_t>(150);
  auto literal        = cudf::ast::literal(literal_value);
  auto col_name_0     = cudf::ast::column_name_reference("col_uint32");
  auto parquet_filter = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_name_0, literal);
  auto col_ref_0      = cudf::ast::column_reference(0);
  auto table_filter   = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, literal);

  // Expected result
  auto predicate = cudf::compute_column(src, table_filter);
  auto expected  = cudf::apply_boolean_mask(src, *predicate);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
      .filter(parquet_filter);
  auto result = cudf::io::read_parquet(read_opts);

  // tests
  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *expected);
}

TEST_F(ParquetReaderTest, FilterMultiple1)
{
  using T = cudf::string_view;

  auto const [src, filepath] = create_parquet_typed_with_stats<T>("FilterMultiple1.parquet");
  auto const written_table   = src.view();

  // Filtering AST - 10000 < table[0] < 12000
  std::string const low  = "000010000";
  std::string const high = "000012000";
  auto lov               = cudf::string_scalar(low, true);
  auto hiv               = cudf::string_scalar(high, true);
  auto filter_col        = cudf::ast::column_reference(0);
  auto lo_lit            = cudf::ast::literal(lov);
  auto hi_lit            = cudf::ast::literal(hiv);
  auto expr_1 = cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, filter_col, lo_lit);
  auto expr_2 = cudf::ast::operation(cudf::ast::ast_operator::LESS, filter_col, hi_lit);
  auto expr_3 = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, expr_1, expr_2);

  // Expected result
  auto predicate = cudf::compute_column(written_table, expr_3);
  auto expected  = cudf::apply_boolean_mask(written_table, *predicate);

  auto si                  = cudf::io::source_info(filepath);
  auto builder             = cudf::io::parquet_reader_options::builder(si).filter(expr_3);
  auto table_with_metadata = cudf::io::read_parquet(builder);
  auto result              = table_with_metadata.tbl->view();

  // tests
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), result);
}

TEST_F(ParquetReaderTest, FilterMultiple2)
{
  // multiple conditions on same column.
  using T = cudf::string_view;

  auto const [src, filepath] = create_parquet_typed_with_stats<T>("FilterMultiple2.parquet");
  auto const written_table   = src.view();
  // 0-8000, 8001-16000, 16001-20000

  // Filtering AST
  // (table[0] >= "000010000" AND table[0] < "000012000") OR
  // (table[0] >= "000017000" AND table[0] < "000019000")
  std::string const low1  = "000010000";
  std::string const high1 = "000012000";
  auto lov                = cudf::string_scalar(low1, true);
  auto hiv                = cudf::string_scalar(high1, true);
  auto filter_col         = cudf::ast::column_reference(0);
  auto lo_lit             = cudf::ast::literal(lov);
  auto hi_lit             = cudf::ast::literal(hiv);
  auto expr_1 = cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, filter_col, lo_lit);
  auto expr_2 = cudf::ast::operation(cudf::ast::ast_operator::LESS, filter_col, hi_lit);
  auto expr_3 = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, expr_1, expr_2);
  std::string const low2  = "000017000";
  std::string const high2 = "000019000";
  auto lov2               = cudf::string_scalar(low2, true);
  auto hiv2               = cudf::string_scalar(high2, true);
  auto lo_lit2            = cudf::ast::literal(lov2);
  auto hi_lit2            = cudf::ast::literal(hiv2);
  auto expr_4 = cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, filter_col, lo_lit2);
  auto expr_5 = cudf::ast::operation(cudf::ast::ast_operator::LESS, filter_col, hi_lit2);
  auto expr_6 = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, expr_4, expr_5);
  auto expr_7 = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_OR, expr_3, expr_6);

  // Expected result
  auto predicate = cudf::compute_column(written_table, expr_7);
  auto expected  = cudf::apply_boolean_mask(written_table, *predicate);

  auto si                  = cudf::io::source_info(filepath);
  auto builder             = cudf::io::parquet_reader_options::builder(si).filter(expr_7);
  auto table_with_metadata = cudf::io::read_parquet(builder);
  auto result              = table_with_metadata.tbl->view();

  // tests
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), result);
}

TEST_F(ParquetReaderTest, FilterMultiple3)
{
  // multiple conditions with reference to multiple columns.
  // index and name references mixed.
  using T                    = uint32_t;
  auto const [src, filepath] = create_parquet_typed_with_stats<T>("FilterMultiple3.parquet");
  auto const written_table   = src.view();

  // Filtering AST - (table[0] >= 70 AND table[0] < 90) OR (table[1] >= 100 AND table[1] < 120)
  // row groups min, max:
  // table[0] 0-80, 81-160, 161-200.
  // table[1] 200-121, 120-41, 40-0.
  auto filter_col1  = cudf::ast::column_reference(0);
  auto filter_col2  = cudf::ast::column_name_reference("col1");
  T constexpr low1  = 70;
  T constexpr high1 = 90;
  T constexpr low2  = 100;
  T constexpr high2 = 120;
  auto lov          = cudf::numeric_scalar(low1, true);
  auto hiv          = cudf::numeric_scalar(high1, true);
  auto lo_lit1      = cudf::ast::literal(lov);
  auto hi_lit1      = cudf::ast::literal(hiv);
  auto expr_1  = cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, filter_col1, lo_lit1);
  auto expr_2  = cudf::ast::operation(cudf::ast::ast_operator::LESS, filter_col1, hi_lit1);
  auto expr_3  = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, expr_1, expr_2);
  auto lov2    = cudf::numeric_scalar(low2, true);
  auto hiv2    = cudf::numeric_scalar(high2, true);
  auto lo_lit2 = cudf::ast::literal(lov2);
  auto hi_lit2 = cudf::ast::literal(hiv2);
  auto expr_4  = cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, filter_col2, lo_lit2);
  auto expr_5  = cudf::ast::operation(cudf::ast::ast_operator::LESS, filter_col2, hi_lit2);
  auto expr_6  = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, expr_4, expr_5);
  // expression to test
  auto expr_7 = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_OR, expr_3, expr_6);

  // Expected result
  auto filter_col2_ref = cudf::ast::column_reference(1);
  auto expr_4_ref =
    cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, filter_col2_ref, lo_lit2);
  auto expr_5_ref = cudf::ast::operation(cudf::ast::ast_operator::LESS, filter_col2_ref, hi_lit2);
  auto expr_6_ref =
    cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, expr_4_ref, expr_5_ref);
  auto expr_7_ref = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_OR, expr_3, expr_6_ref);
  auto predicate  = cudf::compute_column(written_table, expr_7_ref);
  auto expected   = cudf::apply_boolean_mask(written_table, *predicate);

  auto si                  = cudf::io::source_info(filepath);
  auto builder             = cudf::io::parquet_reader_options::builder(si).filter(expr_7);
  auto table_with_metadata = cudf::io::read_parquet(builder);
  auto result              = table_with_metadata.tbl->view();

  // tests
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), result);
}

TEST_F(ParquetReaderTest, FilterSupported)
{
  using T                    = uint32_t;
  auto const [src, filepath] = create_parquet_typed_with_stats<T>("FilterSupported.parquet");
  auto const written_table   = src.view();

  // Filtering AST - ((table[0] > 70 AND table[0] <= 90) OR (table[1] >= 100 AND table[1] < 120))
  //              AND (table[1] != 110)
  // row groups min, max:
  // table[0] 0-80, 81-160, 161-200.
  // table[1] 200-121, 120-41, 40-0.
  auto filter_col1       = cudf::ast::column_reference(0);
  auto filter_col2       = cudf::ast::column_reference(1);
  T constexpr low1       = 70;
  T constexpr high1      = 90;
  T constexpr low2       = 100;
  T constexpr high2      = 120;
  T constexpr skip_value = 110;
  auto lov               = cudf::numeric_scalar(low1, true);
  auto hiv               = cudf::numeric_scalar(high1, true);
  auto lo_lit1           = cudf::ast::literal(lov);
  auto hi_lit1           = cudf::ast::literal(hiv);
  auto expr_1  = cudf::ast::operation(cudf::ast::ast_operator::GREATER, filter_col1, lo_lit1);
  auto expr_2  = cudf::ast::operation(cudf::ast::ast_operator::LESS_EQUAL, filter_col1, hi_lit1);
  auto expr_3  = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, expr_1, expr_2);
  auto lov2    = cudf::numeric_scalar(low2, true);
  auto hiv2    = cudf::numeric_scalar(high2, true);
  auto lo_lit2 = cudf::ast::literal(lov2);
  auto hi_lit2 = cudf::ast::literal(hiv2);
  auto expr_4  = cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, filter_col2, lo_lit2);
  auto expr_5  = cudf::ast::operation(cudf::ast::ast_operator::LESS, filter_col2, hi_lit2);
  auto expr_6  = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, expr_4, expr_5);
  auto expr_7  = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_OR, expr_3, expr_6);
  auto skip_ov = cudf::numeric_scalar(skip_value, true);
  auto skip_lit = cudf::ast::literal(skip_ov);
  auto expr_8   = cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, filter_col2, skip_lit);
  auto expr_9   = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, expr_7, expr_8);

  // Expected result
  auto predicate = cudf::compute_column(written_table, expr_9);
  auto expected  = cudf::apply_boolean_mask(written_table, *predicate);

  auto si                  = cudf::io::source_info(filepath);
  auto builder             = cudf::io::parquet_reader_options::builder(si).filter(expr_9);
  auto table_with_metadata = cudf::io::read_parquet(builder);
  auto result              = table_with_metadata.tbl->view();

  // tests
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), result);
}

TEST_F(ParquetReaderTest, FilterSupported2)
{
  using T                 = uint32_t;
  constexpr auto num_rows = 4000;
  auto elements0 =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i / 2000; });
  auto elements1 =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i / 1000; });
  auto elements2 =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i / 500; });
  auto col0 = cudf::test::fixed_width_column_wrapper<T>(elements0, elements0 + num_rows);
  auto col1 = cudf::test::fixed_width_column_wrapper<T>(elements1, elements1 + num_rows);
  auto col2 = cudf::test::fixed_width_column_wrapper<T>(elements2, elements2 + num_rows);
  auto const written_table = table_view{{col0, col1, col2}};
  auto const filepath      = temp_env->get_temp_filepath("FilterSupported2.parquet");
  {
    const cudf::io::parquet_writer_options out_opts =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, written_table)
        .row_group_size_rows(1000);
    cudf::io::write_parquet(out_opts);
  }
  auto si          = cudf::io::source_info(filepath);
  auto filter_col0 = cudf::ast::column_reference(0);
  auto filter_col1 = cudf::ast::column_reference(1);
  auto filter_col2 = cudf::ast::column_reference(2);
  auto s_value     = cudf::numeric_scalar<T>(1, true);
  auto lit_value   = cudf::ast::literal(s_value);

  auto test_expr = [&](auto& expr) {
    // Expected result
    auto predicate = cudf::compute_column(written_table, expr);
    auto expected  = cudf::apply_boolean_mask(written_table, *predicate);

    // tests
    auto builder             = cudf::io::parquet_reader_options::builder(si).filter(expr);
    auto table_with_metadata = cudf::io::read_parquet(builder);
    auto result              = table_with_metadata.tbl->view();

    CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), result);
  };

  // row groups min, max:
  // table[0] 0-0, 0-0, 1-1, 1-1
  // table[1] 0-0, 1-1, 2-2, 3-3
  // table[2] 0-1, 2-3, 4-5, 6-7

  // Filtering AST -   table[i] == 1
  {
    auto expr0 = cudf::ast::operation(cudf::ast::ast_operator::EQUAL, filter_col0, lit_value);
    test_expr(expr0);

    auto expr1 = cudf::ast::operation(cudf::ast::ast_operator::EQUAL, filter_col1, lit_value);
    test_expr(expr1);

    auto expr2 = cudf::ast::operation(cudf::ast::ast_operator::EQUAL, filter_col2, lit_value);
    test_expr(expr2);
  }
  // Filtering AST -   table[i] != 1
  {
    auto expr0 = cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, filter_col0, lit_value);
    test_expr(expr0);

    auto expr1 = cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, filter_col1, lit_value);
    test_expr(expr1);

    auto expr2 = cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, filter_col2, lit_value);
    test_expr(expr2);
  }
}

// Error types - type mismatch, invalid column name, invalid literal type, invalid operator,
// non-bool filter output type.
TEST_F(ParquetReaderTest, FilterErrors)
{
  using T                    = uint32_t;
  auto const [src, filepath] = create_parquet_typed_with_stats<T>("FilterErrors.parquet");
  auto const written_table   = src.view();
  auto si                    = cudf::io::source_info(filepath);

  // Filtering AST - invalid column index
  {
    auto filter_col1 = cudf::ast::column_reference(3);
    T constexpr low  = 100;
    auto lov         = cudf::numeric_scalar(low, true);
    auto low_lot     = cudf::ast::literal(lov);
    auto expr        = cudf::ast::operation(cudf::ast::ast_operator::LESS, filter_col1, low_lot);

    auto builder = cudf::io::parquet_reader_options::builder(si).filter(expr);
    EXPECT_THROW(cudf::io::read_parquet(builder), cudf::logic_error);
  }

  // Filtering AST - invalid column name
  {
    auto filter_col1 = cudf::ast::column_name_reference("col3");
    T constexpr low  = 100;
    auto lov         = cudf::numeric_scalar(low, true);
    auto low_lot     = cudf::ast::literal(lov);
    auto expr        = cudf::ast::operation(cudf::ast::ast_operator::LESS, filter_col1, low_lot);
    auto builder     = cudf::io::parquet_reader_options::builder(si).filter(expr);
    EXPECT_THROW(cudf::io::read_parquet(builder), cudf::logic_error);
  }

  // Filtering AST - incompatible literal type
  {
    auto filter_col1      = cudf::ast::column_name_reference("col0");
    auto filter_col2      = cudf::ast::column_reference(1);
    int64_t constexpr low = 100;
    auto lov              = cudf::numeric_scalar(low, true);
    auto low_lot          = cudf::ast::literal(lov);
    auto expr1    = cudf::ast::operation(cudf::ast::ast_operator::LESS, filter_col1, low_lot);
    auto expr2    = cudf::ast::operation(cudf::ast::ast_operator::LESS, filter_col2, low_lot);
    auto builder1 = cudf::io::parquet_reader_options::builder(si).filter(expr1);
    EXPECT_THROW(cudf::io::read_parquet(builder1), cudf::logic_error);

    auto builder2 = cudf::io::parquet_reader_options::builder(si).filter(expr2);
    EXPECT_THROW(cudf::io::read_parquet(builder2), cudf::logic_error);
  }

  // Filtering AST - "table[0] + 110" is invalid filter expression
  {
    auto filter_col1      = cudf::ast::column_reference(0);
    T constexpr add_value = 110;
    auto add_v            = cudf::numeric_scalar(add_value, true);
    auto add_lit          = cudf::ast::literal(add_v);
    auto expr_8 = cudf::ast::operation(cudf::ast::ast_operator::ADD, filter_col1, add_lit);

    auto si      = cudf::io::source_info(filepath);
    auto builder = cudf::io::parquet_reader_options::builder(si).filter(expr_8);
    EXPECT_THROW(cudf::io::read_parquet(builder), cudf::logic_error);

    // Expected result throw to show that the filter expression is invalid,
    // not a limitation of the parquet predicate pushdown.
    auto predicate = cudf::compute_column(written_table, expr_8);
    EXPECT_THROW(cudf::apply_boolean_mask(written_table, *predicate), cudf::logic_error);
  }

  // Filtering AST - INT64(table[0] < 100) non-bool expression
  {
    auto filter_col1 = cudf::ast::column_reference(0);
    T constexpr low  = 100;
    auto lov         = cudf::numeric_scalar(low, true);
    auto low_lot     = cudf::ast::literal(lov);
    auto bool_expr   = cudf::ast::operation(cudf::ast::ast_operator::LESS, filter_col1, low_lot);
    auto cast        = cudf::ast::operation(cudf::ast::ast_operator::CAST_TO_INT64, bool_expr);

    auto builder = cudf::io::parquet_reader_options::builder(si).filter(cast);
    EXPECT_THROW(cudf::io::read_parquet(builder), cudf::logic_error);
    EXPECT_NO_THROW(cudf::compute_column(written_table, cast));
    auto predicate = cudf::compute_column(written_table, cast);
    EXPECT_NE(predicate->view().type().id(), cudf::type_id::BOOL8);
  }
}

// Filter without stats information in file.
TEST_F(ParquetReaderTest, FilterNoStats)
{
  using T                 = uint32_t;
  constexpr auto num_rows = 16000;
  auto elements =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i / 1000; });
  auto col0 = cudf::test::fixed_width_column_wrapper<T>(elements, elements + num_rows);
  auto const written_table = table_view{{col0}};
  auto const filepath      = temp_env->get_temp_filepath("FilterNoStats.parquet");
  {
    const cudf::io::parquet_writer_options out_opts =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, written_table)
        .row_group_size_rows(8000)
        .stats_level(cudf::io::statistics_freq::STATISTICS_NONE);
    cudf::io::write_parquet(out_opts);
  }
  auto si          = cudf::io::source_info(filepath);
  auto filter_col0 = cudf::ast::column_reference(0);
  auto s_value     = cudf::numeric_scalar<T>(1, true);
  auto lit_value   = cudf::ast::literal(s_value);

  // row groups min, max:
  // table[0] 0-0, 1-1, 2-2, 3-3
  // Filtering AST - table[0] > 1
  auto expr = cudf::ast::operation(cudf::ast::ast_operator::GREATER, filter_col0, lit_value);

  // Expected result
  auto predicate = cudf::compute_column(written_table, expr);
  auto expected  = cudf::apply_boolean_mask(written_table, *predicate);

  // tests
  auto builder             = cudf::io::parquet_reader_options::builder(si).filter(expr);
  auto table_with_metadata = cudf::io::read_parquet(builder);
  auto result              = table_with_metadata.tbl->view();

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), result);
}

// Filter for float column with NaN values
TEST_F(ParquetReaderTest, FilterFloatNAN)
{
  constexpr auto num_rows = 24000;
  auto elements           = cudf::detail::make_counting_transform_iterator(
    0, [num_rows](auto i) { return i > num_rows / 2 ? NAN : i; });
  auto col0 = cudf::test::fixed_width_column_wrapper<float>(elements, elements + num_rows);
  auto col1 = cudf::test::fixed_width_column_wrapper<double>(elements, elements + num_rows);

  auto const written_table = table_view{{col0, col1}};
  auto const filepath      = temp_env->get_temp_filepath("FilterFloatNAN.parquet");
  {
    const cudf::io::parquet_writer_options out_opts =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, written_table)
        .row_group_size_rows(8000);
    cudf::io::write_parquet(out_opts);
  }
  auto si          = cudf::io::source_info(filepath);
  auto filter_col0 = cudf::ast::column_reference(0);
  auto filter_col1 = cudf::ast::column_reference(1);
  auto s0_value    = cudf::numeric_scalar<float>(NAN, true);
  auto lit0_value  = cudf::ast::literal(s0_value);
  auto s1_value    = cudf::numeric_scalar<double>(NAN, true);
  auto lit1_value  = cudf::ast::literal(s1_value);

  // row groups min, max:
  // table[0] 0-0, 1-1, 2-2, 3-3
  // Filtering AST - table[0] == NAN, table[1] != NAN
  auto expr_eq  = cudf::ast::operation(cudf::ast::ast_operator::EQUAL, filter_col0, lit0_value);
  auto expr_neq = cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, filter_col1, lit1_value);

  // Expected result
  auto predicate0 = cudf::compute_column(written_table, expr_eq);
  auto expected0  = cudf::apply_boolean_mask(written_table, *predicate0);
  auto predicate1 = cudf::compute_column(written_table, expr_neq);
  auto expected1  = cudf::apply_boolean_mask(written_table, *predicate1);

  // tests
  auto builder0             = cudf::io::parquet_reader_options::builder(si).filter(expr_eq);
  auto table_with_metadata0 = cudf::io::read_parquet(builder0);
  auto result0              = table_with_metadata0.tbl->view();
  auto builder1             = cudf::io::parquet_reader_options::builder(si).filter(expr_neq);
  auto table_with_metadata1 = cudf::io::read_parquet(builder1);
  auto result1              = table_with_metadata1.tbl->view();

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected0->view(), result0);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected1->view(), result1);
}

TEST_F(ParquetReaderTest, RepeatedNoAnnotations)
{
  constexpr std::array<unsigned char, 662> repeated_bytes{
    0x50, 0x41, 0x52, 0x31, 0x15, 0x04, 0x15, 0x30, 0x15, 0x30, 0x4c, 0x15, 0x0c, 0x15, 0x00, 0x12,
    0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x04, 0x00,
    0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00, 0x15, 0x00, 0x15, 0x0a, 0x15, 0x0a,
    0x2c, 0x15, 0x0c, 0x15, 0x10, 0x15, 0x06, 0x15, 0x06, 0x00, 0x00, 0x03, 0x03, 0x88, 0xc6, 0x02,
    0x26, 0x80, 0x01, 0x1c, 0x15, 0x02, 0x19, 0x25, 0x00, 0x10, 0x19, 0x18, 0x02, 0x69, 0x64, 0x15,
    0x00, 0x16, 0x0c, 0x16, 0x78, 0x16, 0x78, 0x26, 0x54, 0x26, 0x08, 0x00, 0x00, 0x15, 0x04, 0x15,
    0x40, 0x15, 0x40, 0x4c, 0x15, 0x08, 0x15, 0x00, 0x12, 0x00, 0x00, 0xe3, 0x0c, 0x23, 0x4b, 0x01,
    0x00, 0x00, 0x00, 0xc7, 0x35, 0x3a, 0x42, 0x00, 0x00, 0x00, 0x00, 0x8e, 0x6b, 0x74, 0x84, 0x00,
    0x00, 0x00, 0x00, 0x55, 0xa1, 0xae, 0xc6, 0x00, 0x00, 0x00, 0x00, 0x15, 0x00, 0x15, 0x22, 0x15,
    0x22, 0x2c, 0x15, 0x10, 0x15, 0x10, 0x15, 0x06, 0x15, 0x06, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
    0x03, 0xc0, 0x03, 0x00, 0x00, 0x00, 0x03, 0x90, 0xaa, 0x02, 0x03, 0x94, 0x03, 0x26, 0xda, 0x02,
    0x1c, 0x15, 0x04, 0x19, 0x25, 0x00, 0x10, 0x19, 0x38, 0x0c, 0x70, 0x68, 0x6f, 0x6e, 0x65, 0x4e,
    0x75, 0x6d, 0x62, 0x65, 0x72, 0x73, 0x05, 0x70, 0x68, 0x6f, 0x6e, 0x65, 0x06, 0x6e, 0x75, 0x6d,
    0x62, 0x65, 0x72, 0x15, 0x00, 0x16, 0x10, 0x16, 0xa0, 0x01, 0x16, 0xa0, 0x01, 0x26, 0x96, 0x02,
    0x26, 0xba, 0x01, 0x00, 0x00, 0x15, 0x04, 0x15, 0x24, 0x15, 0x24, 0x4c, 0x15, 0x04, 0x15, 0x00,
    0x12, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x68, 0x6f, 0x6d, 0x65, 0x06, 0x00, 0x00, 0x00, 0x6d,
    0x6f, 0x62, 0x69, 0x6c, 0x65, 0x15, 0x00, 0x15, 0x20, 0x15, 0x20, 0x2c, 0x15, 0x10, 0x15, 0x10,
    0x15, 0x06, 0x15, 0x06, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03, 0xc0, 0x03, 0x00, 0x00, 0x00,
    0x03, 0x90, 0xef, 0x01, 0x03, 0x04, 0x26, 0xcc, 0x04, 0x1c, 0x15, 0x0c, 0x19, 0x25, 0x00, 0x10,
    0x19, 0x38, 0x0c, 0x70, 0x68, 0x6f, 0x6e, 0x65, 0x4e, 0x75, 0x6d, 0x62, 0x65, 0x72, 0x73, 0x05,
    0x70, 0x68, 0x6f, 0x6e, 0x65, 0x04, 0x6b, 0x69, 0x6e, 0x64, 0x15, 0x00, 0x16, 0x10, 0x16, 0x82,
    0x01, 0x16, 0x82, 0x01, 0x26, 0x8a, 0x04, 0x26, 0xca, 0x03, 0x00, 0x00, 0x15, 0x02, 0x19, 0x6c,
    0x48, 0x04, 0x75, 0x73, 0x65, 0x72, 0x15, 0x04, 0x00, 0x15, 0x02, 0x25, 0x00, 0x18, 0x02, 0x69,
    0x64, 0x00, 0x35, 0x02, 0x18, 0x0c, 0x70, 0x68, 0x6f, 0x6e, 0x65, 0x4e, 0x75, 0x6d, 0x62, 0x65,
    0x72, 0x73, 0x15, 0x02, 0x00, 0x35, 0x04, 0x18, 0x05, 0x70, 0x68, 0x6f, 0x6e, 0x65, 0x15, 0x04,
    0x00, 0x15, 0x04, 0x25, 0x00, 0x18, 0x06, 0x6e, 0x75, 0x6d, 0x62, 0x65, 0x72, 0x00, 0x15, 0x0c,
    0x25, 0x02, 0x18, 0x04, 0x6b, 0x69, 0x6e, 0x64, 0x25, 0x00, 0x00, 0x16, 0x00, 0x19, 0x1c, 0x19,
    0x3c, 0x26, 0x80, 0x01, 0x1c, 0x15, 0x02, 0x19, 0x25, 0x00, 0x10, 0x19, 0x18, 0x02, 0x69, 0x64,
    0x15, 0x00, 0x16, 0x0c, 0x16, 0x78, 0x16, 0x78, 0x26, 0x54, 0x26, 0x08, 0x00, 0x00, 0x26, 0xda,
    0x02, 0x1c, 0x15, 0x04, 0x19, 0x25, 0x00, 0x10, 0x19, 0x38, 0x0c, 0x70, 0x68, 0x6f, 0x6e, 0x65,
    0x4e, 0x75, 0x6d, 0x62, 0x65, 0x72, 0x73, 0x05, 0x70, 0x68, 0x6f, 0x6e, 0x65, 0x06, 0x6e, 0x75,
    0x6d, 0x62, 0x65, 0x72, 0x15, 0x00, 0x16, 0x10, 0x16, 0xa0, 0x01, 0x16, 0xa0, 0x01, 0x26, 0x96,
    0x02, 0x26, 0xba, 0x01, 0x00, 0x00, 0x26, 0xcc, 0x04, 0x1c, 0x15, 0x0c, 0x19, 0x25, 0x00, 0x10,
    0x19, 0x38, 0x0c, 0x70, 0x68, 0x6f, 0x6e, 0x65, 0x4e, 0x75, 0x6d, 0x62, 0x65, 0x72, 0x73, 0x05,
    0x70, 0x68, 0x6f, 0x6e, 0x65, 0x04, 0x6b, 0x69, 0x6e, 0x64, 0x15, 0x00, 0x16, 0x10, 0x16, 0x82,
    0x01, 0x16, 0x82, 0x01, 0x26, 0x8a, 0x04, 0x26, 0xca, 0x03, 0x00, 0x00, 0x16, 0x9a, 0x03, 0x16,
    0x0c, 0x00, 0x28, 0x49, 0x70, 0x61, 0x72, 0x71, 0x75, 0x65, 0x74, 0x2d, 0x72, 0x73, 0x20, 0x76,
    0x65, 0x72, 0x73, 0x69, 0x6f, 0x6e, 0x20, 0x30, 0x2e, 0x33, 0x2e, 0x30, 0x20, 0x28, 0x62, 0x75,
    0x69, 0x6c, 0x64, 0x20, 0x62, 0x34, 0x35, 0x63, 0x65, 0x37, 0x63, 0x62, 0x61, 0x32, 0x31, 0x39,
    0x39, 0x66, 0x32, 0x32, 0x64, 0x39, 0x33, 0x32, 0x36, 0x39, 0x63, 0x31, 0x35, 0x30, 0x64, 0x38,
    0x61, 0x38, 0x33, 0x39, 0x31, 0x36, 0x63, 0x36, 0x39, 0x62, 0x35, 0x65, 0x29, 0x00, 0x32, 0x01,
    0x00, 0x00, 0x50, 0x41, 0x52, 0x31};

  auto read_opts = cudf::io::parquet_reader_options::builder(
    cudf::io::source_info{cudf::host_span<std::byte const>{
      reinterpret_cast<std::byte const*>(repeated_bytes.data()), repeated_bytes.size()}});
  auto result = cudf::io::read_parquet(read_opts);

  EXPECT_EQ(result.tbl->view().column(0).size(), 6);
  EXPECT_EQ(result.tbl->view().num_columns(), 2);

  column_wrapper<int32_t> col0{1, 2, 3, 4, 5, 6};
  column_wrapper<int64_t> child0{{5555555555l, 1111111111l, 1111111111l, 2222222222l, 3333333333l}};
  cudf::test::strings_column_wrapper child1{{"-", "home", "home", "-", "mobile"},
                                            {false, true, true, false, true}};
  auto struct_col = cudf::test::structs_column_wrapper{{child0, child1}};

  auto list_offsets_column =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 0, 0, 0, 1, 2, 5}.release();
  auto num_list_rows = list_offsets_column->size() - 1;

  auto mask = cudf::create_null_mask(6, cudf::mask_state::ALL_VALID);
  cudf::set_null_mask(static_cast<cudf::bitmask_type*>(mask.data()), 0, 2, false);

  auto list_col = cudf::make_lists_column(
    num_list_rows, std::move(list_offsets_column), struct_col.release(), 2, std::move(mask));

  std::vector<std::unique_ptr<cudf::column>> struct_children;
  struct_children.push_back(std::move(list_col));

  auto outer_struct = cudf::test::structs_column_wrapper{{std::move(struct_children)},
                                                         {false, false, true, true, true, true}};
  table_view expected{{col0, outer_struct}};

  CUDF_TEST_EXPECT_TABLES_EQUAL(result.tbl->view(), expected);
}

// test that using page stats is working for full reads and various skip rows
TEST_F(ParquetReaderTest, StringsWithPageStats)
{
  constexpr int num_rows = 10'000;
  constexpr auto seed    = 21337;

  std::mt19937 engine{seed};
  auto int32_list_nulls = make_parquet_list_col<int32_t>(engine, num_rows, 5, true);
  auto int32_list       = make_parquet_list_col<int32_t>(engine, num_rows, 5, false);
  auto int64_list_nulls = make_parquet_list_col<int64_t>(engine, num_rows, 5, true);
  auto int64_list       = make_parquet_list_col<int64_t>(engine, num_rows, 5, false);
  auto int16_list_nulls = make_parquet_list_col<int16_t>(engine, num_rows, 5, true);
  auto int16_list       = make_parquet_list_col<int16_t>(engine, num_rows, 5, false);
  auto int8_list_nulls  = make_parquet_list_col<int8_t>(engine, num_rows, 5, true);
  auto int8_list        = make_parquet_list_col<int8_t>(engine, num_rows, 5, false);

  auto str_list_nulls     = make_parquet_string_list_col(engine, num_rows, 5, 32, true);
  auto str_list           = make_parquet_string_list_col(engine, num_rows, 5, 32, false);
  auto big_str_list_nulls = make_parquet_string_list_col(engine, num_rows, 5, 256, true);
  auto big_str_list       = make_parquet_string_list_col(engine, num_rows, 5, 256, false);

  auto int32_data   = random_values<int32_t>(num_rows);
  auto int64_data   = random_values<int64_t>(num_rows);
  auto int16_data   = random_values<int16_t>(num_rows);
  auto int8_data    = random_values<int8_t>(num_rows);
  auto str_data     = string_values(engine, num_rows, 32);
  auto big_str_data = string_values(engine, num_rows, 256);

  auto const validity = random_validity(engine);
  auto const no_nulls = cudf::test::iterators::no_nulls();
  column_wrapper<int32_t> int32_nulls_col{int32_data.begin(), int32_data.end(), validity};
  column_wrapper<int32_t> int32_col{int32_data.begin(), int32_data.end(), no_nulls};
  column_wrapper<int64_t> int64_nulls_col{int64_data.begin(), int64_data.end(), validity};
  column_wrapper<int64_t> int64_col{int64_data.begin(), int64_data.end(), no_nulls};

  auto str_col = cudf::test::strings_column_wrapper(str_data.begin(), str_data.end(), no_nulls);
  auto str_col_nulls = cudf::purge_nonempty_nulls(
    cudf::test::strings_column_wrapper(str_data.begin(), str_data.end(), validity));
  auto big_str_col =
    cudf::test::strings_column_wrapper(big_str_data.begin(), big_str_data.end(), no_nulls);
  auto big_str_col_nulls = cudf::purge_nonempty_nulls(
    cudf::test::strings_column_wrapper(big_str_data.begin(), big_str_data.end(), validity));

  cudf::table_view tbl({int32_col,   int32_nulls_col,    *int32_list,   *int32_list_nulls,
                        int64_col,   int64_nulls_col,    *int64_list,   *int64_list_nulls,
                        *int16_list, *int16_list_nulls,  *int8_list,    *int8_list_nulls,
                        str_col,     *str_col_nulls,     *str_list,     *str_list_nulls,
                        big_str_col, *big_str_col_nulls, *big_str_list, *big_str_list_nulls});

  auto const filepath = temp_env->get_temp_filepath("StringsWithPageStats.parquet");
  auto const out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, tbl)
      .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN)
      .max_page_size_rows(5'000)
      .build();
  cudf::io::write_parquet(out_opts);

  // skip_rows / num_rows
  // clang-format off
  std::vector<std::pair<int, int>> params{
    // skip and then read rest of file
    {-1, -1}, {1, -1}, {2, -1}, {32, -1}, {33, -1}, {128, -1}, {1'000, -1},
    // no skip but truncate
    {0, 1'000}, {0, 6'000},
    // cross page boundaries
    {3'000, 5'000}
  };

  // clang-format on
  for (auto p : params) {
    cudf::io::parquet_reader_options read_args =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
    if (p.first >= 0) { read_args.set_skip_rows(p.first); }
    if (p.second >= 0) { read_args.set_num_rows(p.second); }
    auto result = cudf::io::read_parquet(read_args);

    p.first  = p.first < 0 ? 0 : p.first;
    p.second = p.second < 0 ? num_rows - p.first : p.second;
    std::vector<cudf::size_type> slice_indices{p.first, p.first + p.second};
    std::vector<cudf::table_view> expected = cudf::slice(tbl, slice_indices);

    CUDF_TEST_EXPECT_TABLES_EQUAL(result.tbl->view(), expected[0]);
  }
}

TEST_F(ParquetReaderTest, NumRowsPerSource)
{
  int constexpr num_rows          = 10'723;  // A prime number
  int constexpr rows_in_row_group = 500;

  // Table with single col of random int64 values
  auto const int64_data = random_values<int64_t>(num_rows);
  column_wrapper<int64_t> const int64_col{
    int64_data.begin(), int64_data.end(), cudf::test::iterators::no_nulls()};
  cudf::table_view const expected({int64_col});

  // Write to Parquet
  auto const filepath = temp_env->get_temp_filepath("NumRowsPerSource.parquet");
  auto const out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .row_group_size_rows(rows_in_row_group)
      .build();
  cudf::io::write_parquet(out_opts);

  // Read single data source entirely
  {
    auto const in_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath}).build();
    auto const result = cudf::io::read_parquet(in_opts);

    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
    EXPECT_EQ(result.metadata.num_rows_per_source.size(), 1);
    EXPECT_EQ(result.metadata.num_rows_per_source[0], num_rows);
  }

  // Read rows_to_read rows skipping rows_to_skip from single data source
  {
    auto constexpr rows_to_skip = 557;  // a prime number != rows_in_row_group
    auto constexpr rows_to_read = 7'232;
    auto const in_opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                           .skip_rows(rows_to_skip)
                           .num_rows(rows_to_read)
                           .build();
    auto const result = cudf::io::read_parquet(in_opts);
    column_wrapper<int64_t> int64_col_selected{int64_data.begin() + rows_to_skip,
                                               int64_data.begin() + rows_to_skip + rows_to_read,
                                               cudf::test::iterators::no_nulls()};

    cudf::table_view const expected_selected({int64_col_selected});

    CUDF_TEST_EXPECT_TABLES_EQUAL(expected_selected, result.tbl->view());
    EXPECT_EQ(result.metadata.num_rows_per_source.size(), 1);
    EXPECT_EQ(result.metadata.num_rows_per_source[0], rows_to_read);
  }

  // Filtered read from single data source
  {
    auto constexpr max_value = 100;
    auto literal_value       = cudf::numeric_scalar<int64_t>{max_value};
    auto literal             = cudf::ast::literal{literal_value};
    auto col_ref             = cudf::ast::column_reference(0);
    auto filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::LESS_EQUAL, col_ref, literal);

    auto const in_opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                           .filter(filter_expression)
                           .build();

    std::vector<int64_t> int64_data_filtered;
    int64_data_filtered.reserve(num_rows);
    std::copy_if(
      int64_data.begin(), int64_data.end(), std::back_inserter(int64_data_filtered), [=](auto val) {
        return val <= max_value;
      });
    column_wrapper<int64_t> int64_col_filtered{
      int64_data_filtered.begin(), int64_data_filtered.end(), cudf::test::iterators::no_nulls()};

    cudf::table_view expected_filtered({int64_col_filtered});

    auto const result = cudf::io::read_parquet(in_opts);

    CUDF_TEST_EXPECT_TABLES_EQUAL(expected_filtered, result.tbl->view());
    EXPECT_EQ(result.metadata.num_rows_per_source.size(), 0);
  }

  // Read two data sources skipping the first entire file completely
  {
    auto constexpr rows_to_skip = 15'723;
    auto constexpr nsources     = 2;
    std::vector<std::string> const datasources(nsources, filepath);

    auto const in_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{datasources})
        .skip_rows(rows_to_skip)
        .build();

    auto const result = cudf::io::read_parquet(in_opts);

    column_wrapper<int64_t> int64_col_selected{int64_data.begin() + rows_to_skip - num_rows,
                                               int64_data.end(),
                                               cudf::test::iterators::no_nulls()};

    cudf::table_view const expected_selected({int64_col_selected});

    CUDF_TEST_EXPECT_TABLES_EQUAL(expected_selected, result.tbl->view());
    EXPECT_EQ(result.metadata.num_rows_per_source.size(), 2);
    EXPECT_EQ(result.metadata.num_rows_per_source[0], 0);
    EXPECT_EQ(result.metadata.num_rows_per_source[1], nsources * num_rows - rows_to_skip);
  }

  // Read ten data sources entirely
  {
    auto constexpr nsources = 10;
    std::vector<std::string> const datasources(nsources, filepath);

    auto const in_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{datasources}).build();
    auto const result = cudf::io::read_parquet(in_opts);

    // Initialize expected_counts
    std::vector<size_t> const expected_counts(nsources, num_rows);

    EXPECT_EQ(result.metadata.num_rows_per_source.size(), nsources);
    EXPECT_TRUE(std::equal(expected_counts.cbegin(),
                           expected_counts.cend(),
                           result.metadata.num_rows_per_source.cbegin()));
  }

  // Read rows_to_read rows skipping rows_to_skip (> two sources) from ten data sources
  {
    auto constexpr rows_to_skip = 25'999;
    auto constexpr rows_to_read = 10'900;

    auto constexpr nsources = 10;
    std::vector<std::string> const datasources(nsources, filepath);

    auto const in_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{datasources})
        .skip_rows(rows_to_skip)
        .num_rows(rows_to_read)
        .build();

    auto const result = cudf::io::read_parquet(in_opts);

    // Initialize expected_counts
    std::vector<size_t> expected_counts(nsources, num_rows);

    // Adjust expected_counts for rows_to_skip
    int64_t counter = 0;
    for (auto& nrows : expected_counts) {
      if (counter < rows_to_skip) {
        counter += nrows;
        nrows = (counter >= rows_to_skip) ? counter - rows_to_skip : 0;
      } else {
        break;
      }
    }

    // Reset the counter
    counter = 0;

    // Adjust expected_counts for rows_to_read
    for (auto& nrows : expected_counts) {
      if (counter < rows_to_read) {
        counter += nrows;
        nrows = (counter >= rows_to_read) ? rows_to_read - counter + nrows : nrows;
      } else if (counter > rows_to_read) {
        nrows = 0;
      }
    }

    EXPECT_EQ(result.metadata.num_rows_per_source.size(), nsources);
    EXPECT_TRUE(std::equal(expected_counts.cbegin(),
                           expected_counts.cend(),
                           result.metadata.num_rows_per_source.cbegin()));
  }
}

TEST_F(ParquetReaderTest, NumRowsPerSourceEmptyTable)
{
  auto const nsources = 10;

  column_wrapper<int64_t> const int64_empty_col{};
  cudf::table_view const expected_empty({int64_empty_col});

  // Write to Parquet
  auto const filepath_empty = temp_env->get_temp_filepath("NumRowsPerSourceEmpty.parquet");
  auto const out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath_empty}, expected_empty)
      .build();
  cudf::io::write_parquet(out_opts);

  // Read from Parquet
  std::vector<std::string> const datasources(nsources, filepath_empty);

  auto const in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{datasources}).build();
  auto const result = cudf::io::read_parquet(in_opts);

  // Initialize expected_counts
  std::vector<size_t> const expected_counts(nsources, 0);

  EXPECT_EQ(result.metadata.num_rows_per_source.size(), nsources);
  EXPECT_TRUE(std::equal(expected_counts.cbegin(),
                         expected_counts.cend(),
                         result.metadata.num_rows_per_source.cbegin()));
}

///////////////////
// metadata tests

// Test fixture for metadata tests
struct ParquetMetadataReaderTest : public cudf::test::BaseFixture {
  std::string print(cudf::io::parquet_column_schema schema, int depth = 0)
  {
    std::string child_str;
    for (auto const& child : schema.children()) {
      child_str += print(child, depth + 1);
    }
    return std::string(depth, ' ') + schema.name() + "\n" + child_str;
  }
};

TEST_F(ParquetMetadataReaderTest, TestBasic)
{
  auto const num_rows = 1200;

  auto ints   = random_values<int>(num_rows);
  auto floats = random_values<float>(num_rows);
  column_wrapper<int> int_col(ints.begin(), ints.end());
  column_wrapper<float> float_col(floats.begin(), floats.end());

  table_view expected({int_col, float_col});

  cudf::io::table_input_metadata expected_metadata(expected);
  expected_metadata.column_metadata[0].set_name("int_col");
  expected_metadata.column_metadata[1].set_name("float_col");

  auto filepath = temp_env->get_temp_filepath("MetadataTest.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .metadata(std::move(expected_metadata));
  cudf::io::write_parquet(out_opts);

  // Single file
  auto const test_parquet_metadata = [&](int num_sources) {
    auto meta =
      read_parquet_metadata(cudf::io::source_info{std::vector<std::string>(num_sources, filepath)});
    EXPECT_EQ(meta.num_rows(), num_sources * num_rows);

    auto const column_chunk_metadata = meta.columnchunk_metadata();
    // Two leaf columns
    EXPECT_EQ(column_chunk_metadata.size(), 2);
    // Check if all leaf columns have the expected number of row groups
    EXPECT_EQ(column_chunk_metadata.at("int_col").size(), meta.num_rowgroups());
    EXPECT_EQ(column_chunk_metadata.at("float_col").size(), meta.num_rowgroups());

    EXPECT_EQ(meta.num_rowgroups_per_file().size(), num_sources);
    for (auto const& num_row_groups : meta.num_rowgroups_per_file()) {
      // Each source file has only one row group
      EXPECT_EQ(num_row_groups, 1);
    }

    std::string expected_schema = R"(schema
 int_col
 float_col
)";
    EXPECT_EQ(expected_schema, print(meta.schema().root()));
    EXPECT_EQ(meta.schema().root().name(), "schema");
    EXPECT_EQ(meta.schema().root().type(), cudf::io::parquet::Type::UNDEFINED);
    ASSERT_EQ(meta.schema().root().num_children(), 2);

    EXPECT_EQ(meta.schema().root().child(0).name(), "int_col");
    EXPECT_EQ(meta.schema().root().child(1).name(), "float_col");
  };

  // Test with single file
  test_parquet_metadata(1);
  // Test with multiple files
  test_parquet_metadata(3);
}

TEST_F(ParquetMetadataReaderTest, TestNested)
{
  auto const num_rows       = 1200;
  auto const lists_per_row  = 4;
  auto const num_child_rows = num_rows * lists_per_row;

  auto keys = random_values<int>(num_child_rows);
  auto vals = random_values<float>(num_child_rows);
  column_wrapper<int> keys_col(keys.begin(), keys.end());
  column_wrapper<float> vals_col(vals.begin(), vals.end());
  auto s_col = cudf::test::structs_column_wrapper({keys_col, vals_col}).release();

  std::vector<int> row_offsets(num_rows + 1);
  for (int idx = 0; idx < num_rows + 1; ++idx) {
    row_offsets[idx] = idx * lists_per_row;
  }
  column_wrapper<int> offsets(row_offsets.begin(), row_offsets.end());

  auto list_col =
    cudf::make_lists_column(num_rows, offsets.release(), std::move(s_col), 0, rmm::device_buffer{});

  table_view expected({*list_col, *list_col});

  cudf::io::table_input_metadata expected_metadata(expected);
  expected_metadata.column_metadata[0].set_name("maps");
  expected_metadata.column_metadata[0].set_list_column_as_map();
  expected_metadata.column_metadata[1].set_name("lists");
  expected_metadata.column_metadata[1].child(1).child(0).set_name("int_field");
  expected_metadata.column_metadata[1].child(1).child(1).set_name("float_field");

  auto filepath = temp_env->get_temp_filepath("MetadataTest.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .metadata(std::move(expected_metadata));
  cudf::io::write_parquet(out_opts);

  auto meta = read_parquet_metadata(cudf::io::source_info{filepath});
  EXPECT_EQ(meta.num_rows(), num_rows);

  auto const column_chunk_metadata = meta.columnchunk_metadata();
  // Four leaf columns
  EXPECT_EQ(column_chunk_metadata.size(), 4);
  // Check if all leaf columns are present
  EXPECT_TRUE(column_chunk_metadata.find("maps.key_value.key") != column_chunk_metadata.end());
  EXPECT_TRUE(column_chunk_metadata.find("maps.key_value.value") != column_chunk_metadata.end());
  EXPECT_TRUE(column_chunk_metadata.find("lists.list.element.int_field") !=
              column_chunk_metadata.end());
  EXPECT_TRUE(column_chunk_metadata.find("lists.list.element.float_field") !=
              column_chunk_metadata.end());

  EXPECT_EQ(meta.num_rowgroups_per_file().size(), 1);
  EXPECT_EQ(meta.num_rowgroups_per_file()[0], meta.num_rowgroups());

  std::string expected_schema = R"(schema
 maps
  key_value
   key
   value
 lists
  list
   element
    int_field
    float_field
)";
  EXPECT_EQ(expected_schema, print(meta.schema().root()));

  EXPECT_EQ(meta.schema().root().name(), "schema");
  EXPECT_EQ(meta.schema().root().type(),
            cudf::io::parquet::Type::UNDEFINED);  // struct
  ASSERT_EQ(meta.schema().root().num_children(), 2);

  auto const& out_map_col = meta.schema().root().child(0);
  EXPECT_EQ(out_map_col.name(), "maps");
  EXPECT_EQ(out_map_col.type(), cudf::io::parquet::Type::UNDEFINED);  // map

  ASSERT_EQ(out_map_col.num_children(), 1);
  EXPECT_EQ(out_map_col.child(0).name(), "key_value");  // key_value (named in parquet writer)
  ASSERT_EQ(out_map_col.child(0).num_children(), 2);
  EXPECT_EQ(out_map_col.child(0).child(0).name(), "key");    // key (named in parquet writer)
  EXPECT_EQ(out_map_col.child(0).child(1).name(), "value");  // value (named in parquet writer)
  EXPECT_EQ(out_map_col.child(0).child(0).type(), cudf::io::parquet::Type::INT32);  // int
  EXPECT_EQ(out_map_col.child(0).child(1).type(),
            cudf::io::parquet::Type::FLOAT);  // float

  auto const& out_list_col = meta.schema().root().child(1);
  EXPECT_EQ(out_list_col.name(), "lists");
  EXPECT_EQ(out_list_col.type(), cudf::io::parquet::Type::UNDEFINED);  // list
  // TODO repetition type?
  ASSERT_EQ(out_list_col.num_children(), 1);
  EXPECT_EQ(out_list_col.child(0).name(), "list");  // list (named in parquet writer)
  ASSERT_EQ(out_list_col.child(0).num_children(), 1);

  auto const& out_list_struct_col = out_list_col.child(0).child(0);
  EXPECT_EQ(out_list_struct_col.name(), "element");  // elements (named in parquet writer)
  EXPECT_EQ(out_list_struct_col.type(),
            cudf::io::parquet::Type::UNDEFINED);  // struct
  ASSERT_EQ(out_list_struct_col.num_children(), 2);

  auto const& out_int_col = out_list_struct_col.child(0);
  EXPECT_EQ(out_int_col.name(), "int_field");
  EXPECT_EQ(out_int_col.type(), cudf::io::parquet::Type::INT32);

  auto const& out_float_col = out_list_struct_col.child(1);
  EXPECT_EQ(out_float_col.name(), "float_field");
  EXPECT_EQ(out_float_col.type(), cudf::io::parquet::Type::FLOAT);
}

///////////////////////
// reader source tests

template <typename T>
struct ParquetReaderSourceTest : public ParquetReaderTest {};

TYPED_TEST_SUITE(ParquetReaderSourceTest, ByteLikeTypes);

TYPED_TEST(ParquetReaderSourceTest, BufferSourceTypes)
{
  using T = TypeParam;

  srand(31337);
  auto table = create_random_fixed_table<int>(5, 5, true);

  std::vector<char> out_buffer;
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info(&out_buffer), *table);
  cudf::io::write_parquet(out_opts);

  {
    cudf::io::parquet_reader_options in_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info(
        cudf::host_span<T>(reinterpret_cast<T*>(out_buffer.data()), out_buffer.size())));
    auto const result = cudf::io::read_parquet(in_opts);

    CUDF_TEST_EXPECT_TABLES_EQUAL(*table, result.tbl->view());
  }

  {
    cudf::io::parquet_reader_options in_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info(cudf::host_span<T const>(
        reinterpret_cast<T const*>(out_buffer.data()), out_buffer.size())));
    auto const result = cudf::io::read_parquet(in_opts);

    CUDF_TEST_EXPECT_TABLES_EQUAL(*table, result.tbl->view());
  }
}

TYPED_TEST(ParquetReaderSourceTest, BufferSourceArrayTypes)
{
  using T = TypeParam;

  srand(31337);
  auto table = create_random_fixed_table<int>(5, 5, true);

  std::vector<char> out_buffer;
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info(&out_buffer), *table);
  cudf::io::write_parquet(out_opts);

  auto full_table = cudf::concatenate(std::vector<table_view>({*table, *table}));

  {
    auto spans = std::vector<cudf::host_span<T>>{
      cudf::host_span<T>(reinterpret_cast<T*>(out_buffer.data()), out_buffer.size()),
      cudf::host_span<T>(reinterpret_cast<T*>(out_buffer.data()), out_buffer.size())};
    cudf::io::parquet_reader_options in_opts = cudf::io::parquet_reader_options::builder(
      cudf::io::source_info(cudf::host_span<cudf::host_span<T>>(spans.data(), spans.size())));
    auto const result = cudf::io::read_parquet(in_opts);

    CUDF_TEST_EXPECT_TABLES_EQUAL(*full_table, result.tbl->view());
  }

  {
    auto spans = std::vector<cudf::host_span<T const>>{
      cudf::host_span<T const>(reinterpret_cast<T const*>(out_buffer.data()), out_buffer.size()),
      cudf::host_span<T const>(reinterpret_cast<T const*>(out_buffer.data()), out_buffer.size())};
    cudf::io::parquet_reader_options in_opts = cudf::io::parquet_reader_options::builder(
      cudf::io::source_info(cudf::host_span<cudf::host_span<T const>>(spans.data(), spans.size())));
    auto const result = cudf::io::read_parquet(in_opts);

    CUDF_TEST_EXPECT_TABLES_EQUAL(*full_table, result.tbl->view());
  }
}

//////////////////////////////
// predicate pushdown tests

// Test for Types - numeric, chrono, string.
template <typename T>
struct ParquetReaderPredicatePushdownTest : public ParquetReaderTest {};

TYPED_TEST_SUITE(ParquetReaderPredicatePushdownTest, SupportedTestTypes);

template <typename T, bool use_jit>
void filter_typed_test()
{
  auto const [src, filepath] = create_parquet_typed_with_stats<T>("FilterTyped.parquet");
  auto const written_table   = src.view();
  auto const col_name_0      = cudf::ast::column_name_reference("col0");
  auto const col_ref_0       = cudf::ast::column_reference(0);

  auto const test_predicate_pushdown = [&](cudf::ast::operation const& filter_expression,
                                           cudf::ast::operation const& ref_filter,
                                           cudf::size_type expected_total_row_groups,
                                           cudf::size_type expected_stats_filtered_row_groups) {
    // Expected result
    auto const predicate = cudf::compute_column(written_table, ref_filter);
    EXPECT_EQ(predicate->view().type().id(), cudf::type_id::BOOL8)
      << "Predicate filter should return a boolean";
    auto const expected = cudf::apply_boolean_mask(written_table, *predicate);

    // Reading with Predicate Pushdown
    cudf::io::parquet_reader_options read_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
        .filter(filter_expression)
        .use_jit_filter(use_jit);
    auto const result       = cudf::io::read_parquet(read_opts);
    auto const result_table = result.tbl->view();

    // Tests
    EXPECT_EQ(static_cast<int>(written_table.column(0).type().id()),
              static_cast<int>(result_table.column(0).type().id()))
      << "col0 type mismatch";

    // To make sure AST filters out some elements if row groups must be filtered
    if (expected_stats_filtered_row_groups < expected_total_row_groups) {
      EXPECT_LT(expected->num_rows(), written_table.num_rows());
    } else {
      EXPECT_LE(expected->num_rows(), written_table.num_rows());
    }
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), result_table);
    EXPECT_EQ(result.metadata.num_input_row_groups, expected_total_row_groups);
    EXPECT_TRUE(result.metadata.num_row_groups_after_stats_filter.has_value());
    EXPECT_EQ(result.metadata.num_row_groups_after_stats_filter.value(),
              expected_stats_filtered_row_groups);
    EXPECT_FALSE(result.metadata.num_row_groups_after_bloom_filter.has_value());
  };

  // The `literal_value` and stats should filter out 2 out of 4 row groups.
  {
    auto constexpr expected_total_row_groups          = 4;
    auto constexpr expected_stats_filtered_row_groups = 2;

    // Filtering AST
    auto literal_value = []() {
      if constexpr (cudf::is_timestamp<T>()) {
        // table[0] < 10000 timestamp days/seconds/milliseconds/microseconds/nanoseconds
        return cudf::timestamp_scalar<T>(T(typename T::duration(10000)));  // i (0-20,000)
      } else if constexpr (cudf::is_duration<T>()) {
        // table[0] < 10000 day/seconds/milliseconds/microseconds/nanoseconds
        return cudf::duration_scalar<T>(T(10000));  // i (0-20,000)
      } else if constexpr (std::is_same_v<T, cudf::string_view>) {
        // table[0] < "000010000"
        return cudf::string_scalar("000010000");  // i (0-20,000)
      } else {
        // table[0] < 0 or 100u
        return cudf::numeric_scalar<T>(
          (100 - 100 * std::is_signed_v<T>));  // i/100 (-100-100/ 0-200)
      }
    }();

    auto const literal = cudf::ast::literal(literal_value);
    auto const filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::LESS, col_name_0, literal);
    auto const ref_filter = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, literal);
    test_predicate_pushdown(
      filter_expression, ref_filter, expected_total_row_groups, expected_stats_filtered_row_groups);
  }

  // The `literal_value` and stats should not filter any of the 4 row groups.
  {
    auto constexpr expected_total_row_groups          = 4;
    auto constexpr expected_stats_filtered_row_groups = 4;

    // Filtering AST
    auto literal_value = []() {
      if constexpr (cudf::is_timestamp<T>()) {
        return cudf::timestamp_scalar<T>(T(typename T::duration(20000)));
      } else if constexpr (cudf::is_duration<T>()) {
        return cudf::duration_scalar<T>(T(20000));
      } else if constexpr (std::is_same_v<T, cudf::string_view>) {
        return cudf::string_scalar("000020000");
      } else {
        return cudf::numeric_scalar<T>(std::numeric_limits<T>::max());
      }
    }();

    auto const literal = cudf::ast::literal(literal_value);
    auto const filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::LESS_EQUAL, col_name_0, literal);
    auto const ref_filter =
      cudf::ast::operation(cudf::ast::ast_operator::LESS_EQUAL, col_ref_0, literal);
    test_predicate_pushdown(
      filter_expression, ref_filter, expected_total_row_groups, expected_stats_filtered_row_groups);
  }
}

template <typename T>
void filter_unary_operation_typed_test()
{
  std::mt19937 gen(0xd00dL);
  auto [src, filepath, null_count] = [&]() {
    auto constexpr num_rows            = num_ordered_rows;
    auto constexpr row_group_size_rows = num_rows / 4;
    auto _col0                         = testdata::ascending<T>().release();
    // Add nulls to col0
    [[maybe_unused]] std::bernoulli_distribution bn(0.7f);
    auto valids = cudf::detail::make_counting_transform_iterator(0, [&](int index) {
      return (index >= 2 * row_group_size_rows and index < 3 * row_group_size_rows) ? false : true;
    });
    auto [null_mask, null_count] = cudf::test::detail::make_null_mask(valids, valids + num_rows);
    _col0->set_null_mask(std::move(null_mask), null_count);
    auto col0                = cudf::purge_nonempty_nulls(_col0->view());
    auto col1                = testdata::descending<T>();
    auto col2                = testdata::unordered<T>();
    auto const written_table = table_view{{col0->view(), col1, col2}};
    auto const filepath      = temp_env->get_temp_filepath("FilterUnaryOperationTyped.parquet");
    {
      cudf::io::table_input_metadata expected_metadata(written_table);
      expected_metadata.column_metadata[0].set_name("col0");
      expected_metadata.column_metadata[1].set_name("col1");
      expected_metadata.column_metadata[2].set_name("col2");

      const cudf::io::parquet_writer_options out_opts =
        cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, written_table)
          .metadata(std::move(expected_metadata))
          .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN)
          .row_group_size_rows(row_group_size_rows);
      cudf::io::write_parquet(out_opts);
    }

    std::vector<std::unique_ptr<column>> columns;
    columns.push_back(std::move(col0));
    columns.push_back(col1.release());
    columns.push_back(col2.release());

    return std::tuple{cudf::table{std::move(columns)}, filepath, null_count};
  }();

  auto const written_table           = src.view();
  auto const test_predicate_pushdown = [&](cudf::ast::operation const& filter_expression,
                                           cudf::ast::operation const& ref_filter,
                                           cudf::size_type expected_total_row_groups,
                                           cudf::size_type expected_stats_filtered_row_groups,
                                           std::optional<cudf::size_type> expected_num_rows =
                                             std::nullopt) {
    // Expected result
    auto const predicate = cudf::compute_column(written_table, ref_filter);
    EXPECT_EQ(predicate->view().type().id(), cudf::type_id::BOOL8)
      << "Predicate filter should return a boolean";
    auto const expected = cudf::apply_boolean_mask(written_table, *predicate);

    // JIT does not support nullness-dependent operators such as IS_NULL
    // Ref: https://github.com/rapidsai/cudf/issues/20177
    auto constexpr use_jit = false;

    // Reading with Predicate Pushdown
    cudf::io::parquet_reader_options read_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
        .filter(filter_expression)
        .use_jit_filter(use_jit);
    auto const result       = cudf::io::read_parquet(read_opts);
    auto const result_table = result.tbl->view();

    // Tests
    if (expected_num_rows.has_value()) {
      EXPECT_EQ(expected->num_rows(), expected_num_rows.value());
      EXPECT_EQ(result_table.num_rows(), expected_num_rows.value());
    }
    EXPECT_EQ(static_cast<int>(written_table.column(0).type().id()),
              static_cast<int>(result_table.column(0).type().id()))
      << "col0 type mismatch";
    // To make sure AST filters out some elements if row groups must be filtered
    if (expected_stats_filtered_row_groups < expected_total_row_groups) {
      EXPECT_LT(expected->num_rows(), written_table.num_rows());
    } else {
      EXPECT_LE(expected->num_rows(), written_table.num_rows());
    }
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), result_table);
    EXPECT_EQ(result.metadata.num_input_row_groups, expected_total_row_groups);
    EXPECT_TRUE(result.metadata.num_row_groups_after_stats_filter.has_value());
    EXPECT_EQ(result.metadata.num_row_groups_after_stats_filter.value(),
              expected_stats_filtered_row_groups);
    EXPECT_FALSE(result.metadata.num_row_groups_after_bloom_filter.has_value());
  };

  auto const col_name_0 = cudf::ast::column_name_reference("col0");
  auto const col_ref_0  = cudf::ast::column_reference(0);

  // Unary operation `IS_NULL` should filter all but one row group and yield exactly `null_count`
  // rows
  {
    auto constexpr expected_total_row_groups          = 4;
    auto constexpr expected_stats_filtered_row_groups = 1;

    auto const filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::IS_NULL, col_name_0);
    auto const ref_filter = cudf::ast::operation(cudf::ast::ast_operator::IS_NULL, col_ref_0);
    test_predicate_pushdown(filter_expression,
                            ref_filter,
                            expected_total_row_groups,
                            expected_stats_filtered_row_groups,
                            null_count);
  }

  // Unary operation `NOT(IS_NULL)` should filter all but one row group and yield exactly `num_rows
  // - null_count` rows
  {
    auto constexpr expected_total_row_groups          = 4;
    auto constexpr expected_stats_filtered_row_groups = 3;

    auto const is_null_expr = cudf::ast::operation(cudf::ast::ast_operator::IS_NULL, col_name_0);
    auto const filter_expression = cudf::ast::operation(cudf::ast::ast_operator::NOT, is_null_expr);
    auto const is_null_ref_expr = cudf::ast::operation(cudf::ast::ast_operator::IS_NULL, col_ref_0);
    auto const ref_filter = cudf::ast::operation(cudf::ast::ast_operator::NOT, is_null_ref_expr);

    test_predicate_pushdown(filter_expression,
                            ref_filter,
                            expected_total_row_groups,
                            expected_stats_filtered_row_groups,
                            num_ordered_rows - null_count);
  }

  {
    auto constexpr expected_total_row_groups = 4;

    // Filtering AST
    auto literal_value = []() {
      if constexpr (cudf::is_timestamp<T>()) {
        // table[0] < 10000 timestamp days/seconds/milliseconds/microseconds/nanoseconds
        return cudf::timestamp_scalar<T>(T(typename T::duration(10000)));  // i (0-20,000)
      } else if constexpr (cudf::is_duration<T>()) {
        // table[0] < 10000 day/seconds/milliseconds/microseconds/nanoseconds
        return cudf::duration_scalar<T>(T(10000));  // i (0-20,000)
      } else if constexpr (std::is_same_v<T, cudf::string_view>) {
        // table[0] < "000010000"
        return cudf::string_scalar("000010000");  // i (0-20,000)
      } else {
        // table[0] < 0 or 100u
        return cudf::numeric_scalar<T>(
          (100 - 100 * std::is_signed_v<T>));  // i/100 (-100-100/ 0-200)
      }
    }();

    auto const literal = cudf::ast::literal(literal_value);
    auto const expr1   = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_name_0, literal);
    auto const expr2   = cudf::ast::operation(cudf::ast::ast_operator::IS_NULL, col_name_0);

    auto const ref_expr1 = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, literal);
    auto const ref_expr2 = cudf::ast::operation(cudf::ast::ast_operator::IS_NULL, col_ref_0);

    // col0 < 100 AND IS_NULL(col0)
    auto filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, expr1, expr2);
    auto ref_filter =
      cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, ref_expr1, ref_expr2);
    auto constexpr expected_filtered_row_groups_with_unary_and = 1;
    test_predicate_pushdown(filter_expression,
                            ref_filter,
                            expected_total_row_groups,
                            expected_filtered_row_groups_with_unary_and);

    // col0 < 100 OR IS_NULL(col0)
    filter_expression = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_OR, expr1, expr2);
    ref_filter = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_OR, ref_expr1, ref_expr2);
    auto constexpr expected_filtered_row_groups_with_unary_or = 3;
    test_predicate_pushdown(filter_expression,
                            ref_filter,
                            expected_total_row_groups,
                            expected_filtered_row_groups_with_unary_or);
  }
}

TYPED_TEST(ParquetReaderPredicatePushdownTest, FilterTyped)
{
  filter_typed_test<TypeParam, false>();
  filter_unary_operation_typed_test<TypeParam>();
}

TYPED_TEST(ParquetReaderPredicatePushdownTest, FilterTypedJIT)
{
  filter_typed_test<TypeParam, true>();
  // JIT does not support nullness-dependent operators such as IS_NULL so we can't call
  // `filter_unary_operation_typed_test`
  // Ref: https://github.com/rapidsai/cudf/issues/20177
}

TEST_P(ParquetDecompressionTest, RoundTripBasic)
{
  auto const compression_type = std::get<1>(GetParam());

  srand(31337);
  // Exercises multiple rowgroups
  auto expected = create_compressible_fixed_table<int>(4, 12345, 3, true);

  // Use a host buffer for faster I/O
  std::vector<char> buffer;
  cudf::io::parquet_writer_options args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{&buffer}, *expected)
      .compression(compression_type);
  cudf::io::write_parquet(args);

  cudf::io::parquet_reader_options custom_args = cudf::io::parquet_reader_options::builder(
    cudf::io::source_info{cudf::host_span<std::byte const>{
      reinterpret_cast<std::byte const*>(buffer.data()), buffer.size()}});
  auto custom_tbl = cudf::io::read_parquet(custom_args);
  CUDF_TEST_EXPECT_TABLES_EQUAL(custom_tbl.tbl->view(), expected->view());
}

INSTANTIATE_TEST_CASE_P(Nvcomp,
                        ParquetDecompressionTest,
                        ::testing::Combine(::testing::Values("NVCOMP"),
                                           ::testing::Values(cudf::io::compression_type::AUTO,
                                                             cudf::io::compression_type::SNAPPY,
                                                             cudf::io::compression_type::LZ4,
                                                             cudf::io::compression_type::ZSTD)));
//////////////////////
// wide tables tests

// The test below requires several minutes to complete with memcheck, thus it is disabled by
// default.
TEST_F(ParquetReaderTest, DISABLED_ListsWideTable)
{
  auto constexpr num_rows = 2;
  auto constexpr num_cols = 26'755;  // for slightly over 2B keys
  auto constexpr seed     = 0xceed;

  std::mt19937 engine{seed};

  auto list_list       = make_parquet_list_list_col<int32_t>(0, num_rows, 1, 1, false);
  auto list_list_nulls = make_parquet_list_list_col<int32_t>(0, num_rows, 1, 1, true);

  // switch between nullable and non-nullable
  std::vector<cudf::column_view> cols(num_cols);
  bool with_nulls = false;
  std::generate_n(cols.begin(), num_cols, [&]() {
    auto const view = with_nulls ? list_list_nulls->view() : list_list->view();
    with_nulls      = not with_nulls;
    return view;
  });

  cudf::table_view expected(cols);

  // Use a host buffer for faster I/O
  std::vector<char> buffer;
  auto const out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{&buffer}, expected).build();
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options default_in_opts = cudf::io::parquet_reader_options::builder(
    cudf::io::source_info{cudf::host_span<std::byte const>{
      reinterpret_cast<std::byte const*>(buffer.data()), buffer.size()}});
  auto const [result, _] = cudf::io::read_parquet(default_in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result->view());
}

//////////////////////////////////////////
// row bounds and predicate pushdown tests

template <bool use_jit>
void row_bounds_and_filter_test()
{
  auto constexpr num_files                = 3;
  auto constexpr num_rows_per_file        = 100'000;
  auto constexpr num_row_groups_per_table = 5;
  auto constexpr total_num_rows           = num_rows_per_file * num_files;
  auto constexpr rows_per_row_group       = num_rows_per_file / num_row_groups_per_table;

  // Table with single col of ascending int64 values
  auto int64_data = std::vector<int64_t>(num_rows_per_file);
  std::iota(int64_data.begin(), int64_data.end(), 0);
  auto const int64_col = column_wrapper<int64_t>{
    int64_data.begin(), int64_data.end(), cudf::test::iterators::no_nulls()};
  cudf::table_view const written_table({int64_col});

  // Write to parquet
  auto const filepath = temp_env->get_temp_filepath("RowBoundsAndFilter.parquet");
  {
    cudf::io::parquet_writer_options out_opts =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, written_table)
        .row_group_size_rows(rows_per_row_group)
        .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN)
        .build();
    cudf::io::write_parquet(out_opts);
  }

  // int64 data for expected table
  auto expected_int64_data = std::vector<int64_t>{};
  expected_int64_data.reserve(total_num_rows);
  for (auto i = 0; i < num_files; i++) {
    expected_int64_data.insert(expected_int64_data.end(), int64_data.cbegin(), int64_data.cend());
  }

  // Helper function to read parquet data
  auto const read_parquet_table =
    [&](auto const& filter_expression, auto rows_to_skip, auto rows_to_read) {
      auto const int64_col_row_bounded = column_wrapper<int64_t>{
        expected_int64_data.begin() + std::min(total_num_rows, rows_to_skip),
        expected_int64_data.begin() + std::min(total_num_rows, rows_to_skip + rows_to_read),
        cudf::test::iterators::no_nulls()};
      cudf::table_view const expected_row_bounded({int64_col_row_bounded});
      auto predicate = cudf::compute_column(expected_row_bounded, filter_expression);
      auto expected  = cudf::apply_boolean_mask(expected_row_bounded, *predicate);

      auto const in_opts = cudf::io::parquet_reader_options::builder(
                             cudf::io::source_info{std::vector<std::string>{num_files, filepath}})
                             .filter(filter_expression)
                             .skip_rows(rows_to_skip)
                             .num_rows(rows_to_read)
                             .use_jit_filter(use_jit)
                             .build();
      return std::tuple{cudf::io::read_parquet(in_opts), std::move(expected)};
    };

  // Filtering AST - table[0] >= 40'000
  {
    auto constexpr rows_to_skip = 30'000;
    auto constexpr rows_to_read = 40'000;

    auto literal_value = cudf::numeric_scalar<int64_t>(40'000);
    auto literal       = cudf::ast::literal(literal_value);
    auto col_ref_0     = cudf::ast::column_reference(0);
    auto filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, col_ref_0, literal);

    auto const [table_with_metadata, expected] =
      read_parquet_table(filter_expression, rows_to_skip, rows_to_read);

    EXPECT_EQ(expected->num_rows(), table_with_metadata.tbl->num_rows());
    EXPECT_EQ(expected->num_columns(), table_with_metadata.tbl->num_columns());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected->view(), table_with_metadata.tbl->view());

    auto const& metadata = table_with_metadata.metadata;
    EXPECT_EQ(metadata.num_input_row_groups, 3);  // RGs: {1,2,3},{},{}
    EXPECT_TRUE(metadata.num_row_groups_after_stats_filter.has_value() and
                metadata.num_row_groups_after_stats_filter.value() == 2);  // RGs: {2,3},{},{}
  }

  // Filtering AST - table[0] < 20'000 but skipping 30'000 rows (empty table)
  {
    auto constexpr rows_to_skip = 30'000;
    auto constexpr rows_to_read = 40'000;

    auto literal_value = cudf::numeric_scalar<int64_t>(20'000);
    auto literal       = cudf::ast::literal(literal_value);
    auto col_ref_0     = cudf::ast::column_reference(0);
    auto filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, literal);

    auto const [table_with_metadata, expected] =
      read_parquet_table(filter_expression, rows_to_skip, rows_to_read);

    CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), table_with_metadata.tbl->view());

    auto const& metadata = table_with_metadata.metadata;
    EXPECT_EQ(metadata.num_input_row_groups, 3);  // RGs: {1,2,3},{},{}
    EXPECT_TRUE(metadata.num_row_groups_after_stats_filter.has_value() and
                metadata.num_row_groups_after_stats_filter.value() == 0);  // RGs: {},{},{}
  }

  // Filtering AST - table[0] <= 100'000 but skipping 301'000 rows (empty table)
  {
    auto constexpr rows_to_skip = 301'000;
    auto constexpr rows_to_read = 1'000;

    auto literal_value = cudf::numeric_scalar<int64_t>(100'000);
    auto literal       = cudf::ast::literal(literal_value);
    auto col_ref_0     = cudf::ast::column_reference(0);
    auto filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::LESS_EQUAL, col_ref_0, literal);

    auto const [table_with_metadata, expected] =
      read_parquet_table(filter_expression, rows_to_skip, rows_to_read);

    CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), table_with_metadata.tbl->view());

    auto const& metadata = table_with_metadata.metadata;
    EXPECT_EQ(metadata.num_input_row_groups, 0);  // RGs: {},{},{}
    EXPECT_TRUE(metadata.num_row_groups_after_stats_filter.has_value() and
                metadata.num_row_groups_after_stats_filter.value() == 0);  // RGs: {},{},{}
  }

  // Filtering AST - table[0] >= 70000 and table[0] < 120000
  {
    auto constexpr rows_to_skip = 130'000;
    auto constexpr rows_to_read = 100'000;

    // Filtering AST - table[0] < 4000
    auto literal_value  = cudf::numeric_scalar<int64_t>(70'000);
    auto literal        = cudf::ast::literal(literal_value);
    auto literal_value2 = cudf::numeric_scalar<int64_t>(100'000);
    auto literal2       = cudf::ast::literal(literal_value2);

    auto col_ref_0 = cudf::ast::column_reference(0);
    auto filter_expression1 =
      cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, col_ref_0, literal);
    auto filter_expression2 =
      cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, literal2);
    auto filter_expression = cudf::ast::operation(
      cudf::ast::ast_operator::LOGICAL_AND, filter_expression1, filter_expression2);

    auto const [table_with_metadata, expected] =
      read_parquet_table(filter_expression, rows_to_skip, rows_to_read);

    EXPECT_EQ(expected->num_rows(), table_with_metadata.tbl->num_rows());
    EXPECT_EQ(expected->num_columns(), table_with_metadata.tbl->num_columns());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected->view(), table_with_metadata.tbl->view());

    auto const& metadata = table_with_metadata.metadata;
    EXPECT_EQ(metadata.num_input_row_groups, 6);  // RGs: {}, {1,2,3,4},{0,1}
    EXPECT_TRUE(metadata.num_row_groups_after_stats_filter.has_value() and
                metadata.num_row_groups_after_stats_filter.value() == 2);  // RGs: {},{4,5},{}
  }

  // Filtering AST - table[0] < 40000 or table[0] >= 80000
  {
    auto constexpr rows_to_skip = 120'000;
    auto constexpr rows_to_read = 190'000;  // Larger than the total number of rows in all files

    auto literal_value  = cudf::numeric_scalar<int64_t>(40'000);
    auto literal        = cudf::ast::literal(literal_value);
    auto literal_value2 = cudf::numeric_scalar<int64_t>(80'000);
    auto literal2       = cudf::ast::literal(literal_value2);

    auto col_ref_0 = cudf::ast::column_reference(0);
    auto filter_expression1 =
      cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, literal);
    auto filter_expression2 =
      cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, col_ref_0, literal2);
    auto filter_expression = cudf::ast::operation(
      cudf::ast::ast_operator::LOGICAL_OR, filter_expression1, filter_expression2);

    auto const [table_with_metadata, expected] =
      read_parquet_table(filter_expression, rows_to_skip, rows_to_read);

    EXPECT_EQ(expected->num_rows(), table_with_metadata.tbl->num_rows());
    EXPECT_EQ(expected->num_columns(), table_with_metadata.tbl->num_columns());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected->view(), table_with_metadata.tbl->view());

    auto const& metadata = table_with_metadata.metadata;
    EXPECT_EQ(metadata.num_input_row_groups, 9);  // RGs: {},{1,2,3,4},{0,1,2,3,4}
    EXPECT_TRUE(metadata.num_row_groups_after_stats_filter.has_value() and
                metadata.num_row_groups_after_stats_filter.value() == 5);  // RGs: {},{1,4},{0,1,4}
  }

  // Filtering AST - table[0] >= 40000 and table[0] < 80000
  {
    auto constexpr rows_to_skip = 110'000;
    auto constexpr rows_to_read = 80'000;

    auto literal_value  = cudf::numeric_scalar<int64_t>(40'000);
    auto literal        = cudf::ast::literal(literal_value);
    auto literal_value2 = cudf::numeric_scalar<int64_t>(80'000);
    auto literal2       = cudf::ast::literal(literal_value2);

    auto col_ref_0 = cudf::ast::column_reference(0);
    auto filter_expression1 =
      cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, col_ref_0, literal);
    auto filter_expression2 =
      cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, literal2);
    auto filter_expression = cudf::ast::operation(
      cudf::ast::ast_operator::LOGICAL_AND, filter_expression1, filter_expression2);

    auto const [table_with_metadata, expected] =
      read_parquet_table(filter_expression, rows_to_skip, rows_to_read);

    EXPECT_EQ(expected->num_rows(), table_with_metadata.tbl->num_rows());
    EXPECT_EQ(expected->num_columns(), table_with_metadata.tbl->num_columns());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected->view(), table_with_metadata.tbl->view());

    auto const& metadata = table_with_metadata.metadata;
    EXPECT_EQ(metadata.num_input_row_groups, 5);  // RGs: {},{0,1,2,3,4},{}
    EXPECT_TRUE(metadata.num_row_groups_after_stats_filter.has_value() and
                metadata.num_row_groups_after_stats_filter.value() == 2);  // RGs: {},{2,3},{}
  }

  // Filtering AST - table[0] < 40000 or table[0] >= 80000
  {
    auto constexpr rows_to_skip = 110'000;
    auto constexpr rows_to_read = 80'000;

    auto literal_value  = cudf::numeric_scalar<int64_t>(40'000);
    auto literal        = cudf::ast::literal(literal_value);
    auto literal_value2 = cudf::numeric_scalar<int64_t>(80'000);
    auto literal2       = cudf::ast::literal(literal_value2);

    auto col_ref_0 = cudf::ast::column_reference(0);
    auto filter_expression1 =
      cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, literal);
    auto filter_expression2 =
      cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, col_ref_0, literal2);
    auto filter_expression = cudf::ast::operation(
      cudf::ast::ast_operator::LOGICAL_OR, filter_expression1, filter_expression2);

    auto const [table_with_metadata, expected] =
      read_parquet_table(filter_expression, rows_to_skip, rows_to_read);

    EXPECT_EQ(expected->num_rows(), table_with_metadata.tbl->num_rows());
    EXPECT_EQ(expected->num_columns(), table_with_metadata.tbl->num_columns());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected->view(), table_with_metadata.tbl->view());

    auto const& metadata = table_with_metadata.metadata;
    EXPECT_EQ(metadata.num_input_row_groups, 5);  // RGs: {}, {0,1,2,3,4}, {}
    EXPECT_TRUE(metadata.num_row_groups_after_stats_filter.has_value() and
                metadata.num_row_groups_after_stats_filter.value() == 3);  // RGs: {},{0,1,4},{}
  }
}

TEST_F(ParquetReaderTest, RowBoundsAndFilter) { row_bounds_and_filter_test<false>(); }

TEST_F(ParquetReaderTest, RowBoundsAndFilterJIT) { row_bounds_and_filter_test<true>(); }

//////////////////////////////////////////
// device read async tests

TEST_F(ParquetReaderTest, DeviceReadAsyncThrows)
{
  // Create a simple parquet file in memory
  auto col0           = cudf::test::fixed_width_column_wrapper<int>{{1, 2, 3, 4, 5}};
  auto table_to_write = table_view{{col0}};

  std::vector<char> out_buffer;
  cudf::io::parquet_writer_options write_args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{&out_buffer}, table_to_write);
  cudf::io::write_parquet(write_args);

  // Create our throwing datasource
  auto throwing_source = std::make_unique<cudf::test::ThrowingDeviceReadDatasource>(out_buffer);
  cudf::io::source_info source_info(throwing_source.get());

  // Try to read the parquet file - this should either succeed or propagate AsyncException
  // from device_read_async.
  cudf::io::parquet_reader_options read_args =
    cudf::io::parquet_reader_options::builder(source_info);
  try {
    cudf::io::read_parquet(read_args);
    // Test passes if no exception is thrown
  } catch (const cudf::test::AsyncException&) {
    // Test passes if AsyncException is thrown (expected test exception)
  } catch (const std::exception& e) {
    // Test fails if any other exception is thrown
    FAIL() << "Unexpected exception thrown: " << e.what();
  }
}

TEST_F(ParquetReaderTest, DeviceWriteAsyncThrows)
{
  // Create a simple table to write
  auto col0           = cudf::test::fixed_width_column_wrapper<int>{{1, 2, 3, 4, 5}};
  auto table_to_write = table_view{{col0}};

  auto throwing_sink = std::make_unique<cudf::test::ThrowingDeviceWriteDataSink>();

  cudf::io::parquet_writer_options write_args = cudf::io::parquet_writer_options::builder(
    cudf::io::sink_info{throwing_sink.get()}, table_to_write);

  // The write_parquet call should either succeed or throw AsyncException.
  try {
    cudf::io::write_parquet(write_args);
    // Test passes if no exception is thrown
  } catch (const cudf::test::AsyncException&) {
    // Test passes if AsyncException is thrown (expected test exception)
  } catch (const std::exception& e) {
    // Test fails if any other exception is thrown
    FAIL() << "Unexpected exception thrown: " << e.what();
  }
}

//////////////////////////////////////////
// byte bounds tests

TEST_F(ParquetReaderTest, ByteBoundsOptions)
{
  using T             = cudf::string_view;
  auto const filepath = create_parquet_typed_with_stats<T>("ByteBounds.parquet").second;

  // Test options combinations

  // If skip_bytes is zero, we can set other options normally.
  EXPECT_NO_THROW(cudf::io::parquet_reader_options::builder(
                    cudf::io::source_info{std::vector<std::string>{2, filepath}})
                    .skip_bytes(0)
                    .num_rows(10)
                    .skip_rows(10)
                    .build());

  // Cannot set skip_bytes/num_bytes and row_groups together
  EXPECT_ANY_THROW(cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                     .skip_bytes(100)
                     .row_groups({{1}})
                     .build());
  EXPECT_ANY_THROW(cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                     .num_bytes(100)
                     .row_groups({{1}})
                     .build());

  // Cannot set skip_bytes/num_bytes and skip_rows/num_rows together
  EXPECT_ANY_THROW(cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                     .skip_bytes(100)
                     .num_rows(100)
                     .skip_rows(0)
                     .build());
  EXPECT_ANY_THROW(cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                     .skip_rows(100)
                     .num_bytes(100)
                     .num_rows(100)
                     .build());

  // skip_bytes/num_bytes only supported for single source case
  EXPECT_ANY_THROW(cudf::io::parquet_reader_options::builder(
                     cudf::io::source_info{std::vector<std::string>{2, filepath}})
                     .skip_bytes(100)
                     .build());
  EXPECT_ANY_THROW(cudf::io::parquet_reader_options::builder(
                     cudf::io::source_info{std::vector<std::string>{2, filepath}})
                     .num_bytes(400)
                     .build());
}

TEST_F(ParquetReaderTest, ByteBoundsOnly)
{
  using T                      = cudf::string_view;
  auto const [table, filepath] = create_parquet_typed_with_stats<T>("ByteBounds.parquet");

  // Note: Currently the row groups start at the following byte offsets: 4, 75224, 150332, 225561
  // `skip_bytes` and `num_bytes` may need to be adjusted in the future if this test suddenly starts
  // failing.

  // Only read row group 0 as only it will start in [0, 1000) byte range
  {
    auto constexpr num_bytes = 1000;
    auto const in_opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                           .num_bytes(num_bytes)
                           .build();
    auto const read = cudf::io::read_parquet(in_opts).tbl;

    auto const expected_in_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
        .row_groups({{0}})
        .build();
    auto const expected = cudf::io::read_parquet(expected_in_opts).tbl;

    CUDF_TEST_EXPECT_TABLES_EQUAL(read->view(), expected->view());
  }

  // Skip row group 0 as it won't start in [1000, inf) byte range
  {
    auto constexpr skip_bytes = 1000;
    auto const in_opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                           .skip_bytes(skip_bytes)
                           .build();
    auto const read = cudf::io::read_parquet(in_opts).tbl;

    auto const expected_in_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
        .row_groups({{1, 2, 3}})
        .build();
    auto const expected = cudf::io::read_parquet(expected_in_opts).tbl;

    CUDF_TEST_EXPECT_TABLES_EQUAL(read->view(), expected->view());
  }

  // Only read row group 1 as only it starts in [50000, 100000) byte range
  {
    auto constexpr skip_bytes = 50000;
    auto constexpr num_bytes  = 50000;
    auto const in_opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                           .skip_bytes(skip_bytes)
                           .num_bytes(num_bytes)
                           .build();
    auto const read = cudf::io::read_parquet(in_opts).tbl;

    auto const expected_in_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
        .row_groups({{1}})
        .build();
    auto const expected = cudf::io::read_parquet(expected_in_opts).tbl;

    CUDF_TEST_EXPECT_TABLES_EQUAL(read->view(), expected->view());
  }
}

TEST_F(ParquetReaderTest, ByteBoundsAndFilters)
{
  using T                      = uint64_t;
  auto const [table, filepath] = create_parquet_typed_with_stats<T>("ByteBounds.parquet");

  // Note: Currently the row groups start at the following byte offsets: 4, 2040, 4048, 6032
  // `skip_bytes` and `num_bytes` may need to be adjusted in the future if this test suddenly starts
  // failing.

  // Only read row group 0 as only it will start in [0, 1000) byte range
  {
    auto literal_value = cudf::numeric_scalar<T>(1000);
    auto literal       = cudf::ast::literal(literal_value);
    auto col_ref_0     = cudf::ast::column_reference(0);
    auto filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::LESS_EQUAL, col_ref_0, literal);

    auto constexpr num_bytes = 1000;
    auto const in_opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                           .num_bytes(num_bytes)
                           .filter(filter_expression)
                           .build();
    auto const read = cudf::io::read_parquet(in_opts).tbl;

    auto const expected_in_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
        .row_groups({{0}})
        .filter(filter_expression)
        .build();
    auto const expected = cudf::io::read_parquet(expected_in_opts).tbl;

    CUDF_TEST_EXPECT_TABLES_EQUAL(read->view(), expected->view());
  }

  // Skip row group 0 using byte range and row group 1 using the filter expression
  {
    auto literal_value = cudf::numeric_scalar<T>(12000);
    auto literal       = cudf::ast::literal(literal_value);
    auto col_ref_0     = cudf::ast::column_reference(0);
    auto filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::GREATER, col_ref_0, literal);

    auto constexpr skip_bytes = 1000;
    auto const in_opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                           .skip_bytes(skip_bytes)
                           .filter(filter_expression)
                           .build();
    auto const read = cudf::io::read_parquet(in_opts).tbl;

    auto const expected_in_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
        .row_groups({{2, 3}})
        .filter(filter_expression)
        .build();
    auto const expected = cudf::io::read_parquet(expected_in_opts).tbl;

    CUDF_TEST_EXPECT_TABLES_EQUAL(read->view(), expected->view());
  }

  // Only read row group 1 as only it starts in [1500, 3000) byte range
  {
    auto col_ref_0     = cudf::ast::column_reference(0);
    auto literal_value = cudf::numeric_scalar<T>(8000);
    auto literal       = cudf::ast::literal(literal_value);
    auto filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, literal);

    auto constexpr skip_bytes = 1500;
    auto constexpr num_bytes  = 1500;
    auto const in_opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                           .skip_bytes(skip_bytes)
                           .num_bytes(num_bytes)
                           .filter(filter_expression)
                           .build();
    auto const read = cudf::io::read_parquet(in_opts).tbl;

    auto const expected_in_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
        .row_groups({{1}})
        .filter(filter_expression)
        .build();
    auto const expected = cudf::io::read_parquet(expected_in_opts).tbl;

    CUDF_TEST_EXPECT_TABLES_EQUAL(read->view(), expected->view());
  }
}

TEST_F(ParquetReaderTest, TableTooLargeOverflows)
{
  using T                             = bool;
  constexpr int64_t per_file_num_rows = std::numeric_limits<cudf::size_type>::max() / 2 + 1000;
  static_assert(per_file_num_rows <= std::numeric_limits<cudf::size_type>::max(),
                "Number of rows per file should be less than size_type::max()");
  static_assert(2 * per_file_num_rows > std::numeric_limits<cudf::size_type>::max(),
                "Twice number of rows per file should be greather than size_type::max()");
  auto value  = thrust::make_constant_iterator(true);
  auto column = cudf::test::fixed_width_column_wrapper<T>(value, value + per_file_num_rows);

  auto filepath = temp_env->get_temp_filepath("TableTooLargeOverflows.parquet");
  {
    auto sink = cudf::io::sink_info{filepath};
    auto options =
      cudf::io::parquet_writer_options::builder(sink, cudf::table_view{{column}}).build();
    std::ignore = cudf::io::write_parquet(options);
  }
  std::vector<std::string> files{{filepath, filepath}};
  auto source                 = cudf::io::source_info(files);
  auto metadata               = cudf::io::read_parquet_metadata(source);
  auto const num_rows_to_read = metadata.num_rows() - 1000;
  EXPECT_EQ(metadata.num_rows(), per_file_num_rows * 2);
  auto options = cudf::io::parquet_reader_options::builder(source)
                   .num_rows(num_rows_to_read)
                   .skip_rows(10)
                   .build();

  EXPECT_THROW(cudf::io::read_parquet(options), std::overflow_error);
  auto reader = cudf::io::chunked_parquet_reader(0, 0, options);
  int64_t num_rows_read{0};
  while (reader.has_next()) {
    auto chunk = reader.read_chunk();
    num_rows_read += chunk.tbl->num_rows();
  }
  EXPECT_EQ(num_rows_read, num_rows_to_read);
}

TEST_F(ParquetReaderTest, LateBindSourceInfo)
{
  srand(31337);
  auto expected = create_random_fixed_table<int>(4, 4, false);

  auto filepath = temp_env->get_temp_filepath("LateBindSourceInfo.parquet");
  cudf::io::parquet_writer_options args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, *expected);
  cudf::io::write_parquet(args);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{});

  EXPECT_THROW(cudf::io::read_parquet(read_opts), cudf::logic_error);

  read_opts.set_source(cudf::io::source_info{filepath});

  auto result = cudf::io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(result.tbl->view(), expected->view());
}
