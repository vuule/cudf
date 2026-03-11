# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream

from pylibcudf.io.types cimport SinkInfo, SourceInfo
from pylibcudf.table cimport Table


cpdef Table read_cudftable(SourceInfo source, Stream stream=*, DeviceMemoryResource mr=*)

cpdef void write_cudftable(Table table, SinkInfo sink, Stream stream=*)
