# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport make_unique, unique_ptr
from libcpp.utility cimport move

from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream

from pylibcudf.contiguous_split cimport PackedColumns
from pylibcudf.io.types cimport SinkInfo, SourceInfo
from pylibcudf.libcudf.contiguous_split cimport packed_columns
from pylibcudf.libcudf.io.cudftable cimport packed_table
from pylibcudf.libcudf.io.types cimport sink_info, source_info
from pylibcudf.libcudf.table.table_view cimport table_view
from pylibcudf.table cimport Table
from pylibcudf.utils cimport _get_stream, _get_memory_resource

from rmm.librmm.cuda_stream_view cimport cuda_stream_view
from rmm.librmm.memory_resource cimport device_memory_resource


cdef extern from * nogil:
    """
    #include <cudf/io/experimental/cudftable.hpp>

    cudf::packed_table _call_read_cudftable(
        cudf::io::source_info const& src,
        rmm::cuda_stream_view stream,
        rmm::device_async_resource_ref mr) {
      return cudf::io::experimental::read_cudftable(
          cudf::io::experimental::cudftable_reader_options::builder(src).build(),
          stream, mr);
    }

    void _call_write_cudftable(
        cudf::io::sink_info const& sink,
        cudf::table_view const& table,
        rmm::cuda_stream_view stream) {
      cudf::io::experimental::write_cudftable(
          cudf::io::experimental::cudftable_writer_options::builder(sink, table)
              .build(),
          stream);
    }
    """
    packed_table _call_read_cudftable(
        const source_info& src,
        cuda_stream_view stream,
        device_memory_resource* mr,
    ) except +

    void _call_write_cudftable(
        const sink_info& sink,
        const table_view& table,
        cuda_stream_view stream,
    ) except +


__all__ = [
    "read_cudftable",
    "write_cudftable",
]


cpdef Table read_cudftable(
    SourceInfo source, Stream stream=None, DeviceMemoryResource mr=None
):
    """Read a table in CudfTable binary format.

    Parameters
    ----------
    source : SourceInfo
        The source to read the cudftable from.
    stream : Stream, optional
        CUDA stream used for device memory operations and kernel launches.
    mr : DeviceMemoryResource, optional
        Device memory resource used for device allocations.

    Returns
    -------
    Table
        The deserialized table.
    """
    cdef Stream s = _get_stream(stream)
    cdef DeviceMemoryResource memres = _get_memory_resource(mr)
    cdef packed_table result
    cdef table_view tv

    with nogil:
        result = move(
            _call_read_cudftable(source.c_obj, s.view(), memres.get_mr())
        )
        tv = result.table

    cdef unique_ptr[packed_columns] owned_data = make_unique[packed_columns](
        move(result.data)
    )
    cdef PackedColumns owner = PackedColumns.from_libcudf(
        move(owned_data), s, memres
    )
    return Table.from_table_view_of_arbitrary(tv, owner, s)


cpdef void write_cudftable(Table table, SinkInfo sink, Stream stream=None):
    """Write a table in CudfTable binary format.

    Parameters
    ----------
    table : Table
        The table to write.
    sink : SinkInfo
        The destination to write the cudftable to.
    stream : Stream, optional
        CUDA stream used for device memory operations and kernel launches.
    """
    cdef Stream s = _get_stream(stream)

    with nogil:
        _call_write_cudftable(sink.c_obj, table.view(), s.view())
