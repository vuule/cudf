# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
import pytest
from utils import assert_table_eq

from rmm.pylibrmm.stream import Stream

import pylibcudf as plc
from pylibcudf.io.experimental import read_cudftable, write_cudftable


param_pyarrow_tables = [
    pa.table({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}),
    pa.table({"a": [1, 2, 3]}),
    pa.table({"a": [1], "b": [2], "c": [3]}),
    pa.table({"a": ["a", "bb", "ccc"]}),
    pa.table({"a": [1, 2, None], "b": [None, 3, 4]}),
    pa.table({"a": pa.array([1.5, 2.5, 3.5]), "b": pa.array([True, False, True])}),
]


@pytest.mark.parametrize("arrow_tbl", param_pyarrow_tables)
def test_roundtrip_file(tmp_path, arrow_tbl):
    """Write then read a cudftable via file path."""
    path = str(tmp_path / "test.cudftable")

    plc_tbl = plc.Table.from_arrow(arrow_tbl)
    write_cudftable(plc_tbl, plc.io.SinkInfo([path]))

    result = read_cudftable(plc.io.SourceInfo([path]))
    assert_table_eq(arrow_tbl, result)


@pytest.mark.parametrize("arrow_tbl", param_pyarrow_tables)
def test_roundtrip_with_stream(tmp_path, arrow_tbl):
    """Write and read with an explicit CUDA stream."""
    path = str(tmp_path / "test.cudftable")
    stream = Stream()

    plc_tbl = plc.Table.from_arrow(arrow_tbl)
    write_cudftable(plc_tbl, plc.io.SinkInfo([path]), stream=stream)

    result = read_cudftable(plc.io.SourceInfo([path]), stream=stream)
    assert_table_eq(arrow_tbl, result)


def test_wide_table(tmp_path):
    """Round-trip a table with many columns."""
    ncols = 100
    data = {f"col_{i}": list(range(50)) for i in range(ncols)}
    arrow_tbl = pa.table(data)
    path = str(tmp_path / "wide.cudftable")

    plc_tbl = plc.Table.from_arrow(arrow_tbl)
    write_cudftable(plc_tbl, plc.io.SinkInfo([path]))

    result = read_cudftable(plc.io.SourceInfo([path]))
    assert_table_eq(arrow_tbl, result)


def test_single_row(tmp_path):
    """Round-trip a single-row table."""
    arrow_tbl = pa.table({"x": [42], "y": ["hello"]})
    path = str(tmp_path / "single.cudftable")

    plc_tbl = plc.Table.from_arrow(arrow_tbl)
    write_cudftable(plc_tbl, plc.io.SinkInfo([path]))

    result = read_cudftable(plc.io.SourceInfo([path]))
    assert_table_eq(arrow_tbl, result)


def test_all_nulls(tmp_path):
    """Round-trip a table where every value is null."""
    arrow_tbl = pa.table({
        "a": pa.array([None, None, None], type=pa.int64()),
        "b": pa.array([None, None, None], type=pa.string()),
    })
    path = str(tmp_path / "nulls.cudftable")

    plc_tbl = plc.Table.from_arrow(arrow_tbl)
    write_cudftable(plc_tbl, plc.io.SinkInfo([path]))

    result = read_cudftable(plc.io.SourceInfo([path]))
    assert_table_eq(arrow_tbl, result)


def test_large_table(tmp_path):
    """Round-trip a table with enough rows to exercise non-trivial serialization."""
    n = 100_000
    arrow_tbl = pa.table({"ints": list(range(n)), "floats": [float(i) for i in range(n)]})
    path = str(tmp_path / "large.cudftable")

    plc_tbl = plc.Table.from_arrow(arrow_tbl)
    write_cudftable(plc_tbl, plc.io.SinkInfo([path]))

    result = read_cudftable(plc.io.SourceInfo([path]))
    assert_table_eq(arrow_tbl, result)


def test_overwrite(tmp_path):
    """Writing to the same path twice should overwrite the first table."""
    path = str(tmp_path / "overwrite.cudftable")

    tbl1 = pa.table({"a": [1, 2, 3]})
    tbl2 = pa.table({"x": [10, 20], "y": [30, 40]})

    write_cudftable(plc.Table.from_arrow(tbl1), plc.io.SinkInfo([path]))
    write_cudftable(plc.Table.from_arrow(tbl2), plc.io.SinkInfo([path]))

    result = read_cudftable(plc.io.SourceInfo([path]))
    assert_table_eq(tbl2, result)
