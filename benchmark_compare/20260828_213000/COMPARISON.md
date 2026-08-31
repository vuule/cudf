# Host-writable pinned allocation in libcudf

Port of the `host-writable-pinned-prototype` branch of `vuule/rmm` into libcudf, and measurements
of the four wait mechanisms it offers.

- libcudf built with `-DCPM_rmm_SOURCE=/home/coder/rmm` (RMM 26.10.00, commit 5714edd)
- A100 80GB PCIe, device 0, `--rmm_mode pool --cuio_host_mem pinned_pool`
- Host vectors forced pinned with `LIBCUDF_ALLOCATE_HOST_AS_PINNED_THRESHOLD=134217728`
- Mode selected with `LIBCUDF_HOST_WRITABLE_SYNC_MODE`, counters with `LIBCUDF_HOST_WRITABLE_STATS=1`

## How it is wired into libcudf

`rmm::mr::pool_memory_resource::allocate_host_writable` is not reachable through
`rmm::host_async_resource_ref`, which erases down to `allocate`/`deallocate`. libcudf therefore
carries a pointer to a small abstract interface, `cudf::detail::host_writable_resource`, alongside
the erased ref. `get_host_writable_resource(mr)` returns non-null only for libcudf's own default
pinned pool and only when the env var selects a mode; a user-supplied pinned resource and the
pageable resource keep the old behavior.

`rmm_host_allocator::allocate` uses that interface when present and skips its stream
synchronization, so `host_vector` and every cuIO call site benefits with no changes. `host_uvector`
takes a `host_write_intent`: `immediate` allocates through the host-writable path, `deferred`
(device-filled buffers) allocates with no wait at all as before.

## Micro benchmark

`HOST_ALLOCATION_NVBENCH`, pinned resource. Numbers are cost above the `none` (no allocation)
floor in the same run, i.e. the per-iteration cost of allocating and touching the buffer.

| mode | idle stream | 16 MiB memset queued |
| --- | --- | --- |
| `off` (stream sync in libcudf) | 8.77 µs | 7.52 µs |
| `stream_sync` (same wait, inside the resource) | 8.70 µs | 7.58 µs |
| `stream_event` | 1.30 µs | 0.00 µs |
| `block_event` | 2.93 µs | 0.07 µs |
| `clean_tracking` | 1.70 µs | 0.00 µs |
| no wait at all | 0.13 µs | — |

`stream_sync` reproducing `off` confirms the port is faithful. The event modes remove essentially
all of the cost, and they remove more of it when the stream is loaded: that is the case the
proposal targets, where a stream synchronize waits for work queued after the block was freed.

The `cudaEventQuery` short-circuit is what makes this work. It is not that events are cheaper than
stream synchronizes; it is that a completed event can be detected without a driver round trip,
while `cudaStreamSynchronize` pays for one unconditionally.

`stream_event` beats `block_event` because it records one event per free instead of two, and both
short-circuit at nearly the same rate.

## cuIO

Two independent repetitions of each, `off` as reference.

**Parquet compressed** (SNAPPY/ZSTD × FILEPATH/HOST_BUFFER, 512 MiB): every configuration within
±2.2%, with no consistent direction between repetitions. No measurable effect.

**JSON `json_read_io`**: FILEPATH 3.5–4.4% faster in three of four comparisons, HOST_BUFFER 0.4–1.6%
*slower* in all four, DEVICE_BUFFER unchanged.

**multibyte_split**: noise-dominated, ±2–4% between repetitions of the same mode. `file_bgzip`
appeared 10–13% slower under `block_event` in both repetitions, but rerunning that configuration
alone reversed it (165.6 ms vs 169.0 ms for `off`) with zero pool fallbacks, so the spike is
carryover from the configurations that ran before it in the same process, not the mechanism.

### Counters

Parquet run, per process:

| mode | allocations | waits | fast path | query short-circuit | event records | pool fallbacks |
| --- | --- | --- | --- | --- | --- | --- |
| `stream_event` | 109706 | 109706 | 0 | 104951 (95.7%) | 109706 | 0 |
| `block_event` | 81065 | 81062 | 3 | 79298 (97.8%) | 162130 | 0 |

## What the numbers say

The mechanism works and the micro win is real: 7.5–8.8 µs per pinned allocation becomes ~0. It does
not show up end to end because cuIO does on the order of 100 pinned allocations per iteration
against 23 ms of work, so the ceiling is a fraction of a percent to a few percent — the same
magnitude as the run-to-run noise in these benchmarks. Only JSON FILEPATH separates from noise.

Two things argue against the more elaborate options:

- **Per-block events do not pay for themselves.** They cost a second `cudaEventRecord` on every
  free and buy 2 percentage points of short-circuit rate, and they are slower than the per-stream
  event in the micro benchmark. The per-stream event needs no new machinery at all: the pool
  already records it on every deallocation. All the free-list plumbing in the prototype — the event
  on the block, the merge rule, the cross-stream re-tagging — exists only to support the mode that
  measures worse.
- **`clean_tracking` is dead in libcudf.** 3 fast-path hits out of ~100k. Every pinned buffer
  libcudf allocates is handed to the device, so `device_exposed` is true for all of them; the only
  blocks that skip the wait are the handful that come straight from upstream. `host_uvector`
  exposes `mark_host_only()` for call sites that could opt in, but there are essentially none.

If this goes upstream, `stream_event` is the mechanism to ship, as a single unconditional behavior
of `allocate_host_writable` rather than a selectable mode. The remaining question is the API shape,
since the erased resource ref cannot express it; the `host_writable_resource` interface here is one
answer, but a property on the resource concept would be cleaner.

The one result that deserves a second look before drawing conclusions is JSON HOST_BUFFER being
consistently, if slightly, slower in all four comparisons.
