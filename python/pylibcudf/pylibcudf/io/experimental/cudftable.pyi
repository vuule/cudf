# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from rmm.pylibrmm.memory_resource import DeviceMemoryResource
from rmm.pylibrmm.stream import Stream

from pylibcudf.io.types import SinkInfo, SourceInfo
from pylibcudf.table import Table

def read_cudftable(
    source: SourceInfo,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Table: ...

def write_cudftable(
    table: Table,
    sink: SinkInfo,
    stream: Stream | None = None,
) -> None: ...
