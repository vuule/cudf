# Micro benchmark: host buffer allocation + first host write

- Binary: `HOST_ALLOCATION_NVBENCH -b host_allocation`, all candidates are arms of one run, so no
  main-vs-branch build was needed.
- Primary metric: CPU time per iteration, `nvbench::exec_tag::sync`. Subtract the `container=none`
  arm to get the cost of the buffer itself; the harness floor is 23.0 us on an idle stream and
  61.1 us with `stream_load=16777216` (a 64 MiB device memset in flight).
- Hardware: NVIDIA A100 80GB PCIe (device 0), 2x AMD EPYC 7642, 96 threads.
- Options: `-d 0 --min-time 0.5 --rmm_mode async --cuio_host_mem pinned_pool`. Element type
  `int32_t`, so `num_elements=2^22` is a 16 MiB buffer.

## Cost above the floor, idle stream

| num_elements | std_vector (pageable) | host_vector (pageable) | host_uvector (pageable) | host_vector (pinned) | host_uvector (pinned) |
| --- | --- | --- | --- | --- | --- |
| 2^4 = 16 | 0.3 us | 0.2 us | 0.2 us | 9.8 us | 9.8 us |
| 2^10 = 1024 | 0.2 us | 0.1 us | 0.2 us | 10.0 us | 9.8 us |
| 2^16 = 65536 | 4.5 us | 11.6 us | 0.3 us | 17.7 us | 9.8 us |
| 2^22 = 4194304 | 900 us | 1124 us | 0.2 us | 862 us | 9.8 us |

## Cost above the floor, 64 MiB memset in flight

| num_elements | host_vector (pageable) | host_uvector (pageable) | host_vector (pinned) | host_uvector (pinned) |
| --- | --- | --- | --- | --- |
| 2^4 = 16 | 0.0 us | 0.1 us | 9.4 us | 9.4 us |
| 2^10 = 1024 | 0.0 us | 0.1 us | 9.5 us | 9.4 us |
| 2^16 = 65536 | 0.1 us | 0.1 us | 17.3 us | 9.4 us |
| 2^22 = 4194304 | 1048 us | 0.1 us | 865 us | 9.4 us |

## Reading

- `host_uvector` is flat in buffer size: creating a 16 MiB buffer and writing one element costs the
  same as creating a 64 byte one. `host_vector` and `std::vector` both scale with the buffer,
  because they value-initialize it.
- The pinned arm of `host_uvector` still pays ~9.4 us for the synchronization the contract requires
  before a host write, matching the note's "what this does not fix". The pageable arm pays nothing,
  because the resource is not stream-ordered and no synchronization is needed.
- No configuration is slower than the corresponding `host_vector` one, at any size, on either
  resource, idle or loaded.
- `stream_load` changed the floor by 38 us but not the marginal cost of any container, because
  nvbench's own per-iteration synchronization already absorbs the enqueued work.

## Where the pinned cost goes, and whether a cheaper synchronization would remove it

Extra arms decompose the pinned overhead. `host_uvector_nosync` allocates and frees but writes
without synchronizing (a contract violation, for measurement only); `stream_sync_only` and
`event_wait_only` synchronize without allocating, the latter with `cudaEventRecord` plus
`cudaEventSynchronize` standing in for a wait on a freed block's own event. Cost above the floor,
`num_elements=2^10`:

| arm | idle stream | 64 MiB memset in flight |
| --- | --- | --- |
| `host_uvector_nosync` (pool alloc + stream-ordered free) | 0.19 us | 0.07 us |
| `stream_sync_only` | 7.79 us | 7.43 us |
| `event_wait_only` | 7.74 us | 7.24 us |
| `host_uvector` (all three) | 8.72 us | 8.40 us |

Pool bookkeeping is free; the whole cost is the wait. Waiting on one event costs the same as
waiting on the whole stream, so a narrower wait does not help. Standalone timings on the same
machine confirm why: on a drained stream `cudaStreamSynchronize` is 0.405 us,
`cudaEventSynchronize` on a completed event 0.302 us, `cudaEventQuery` 0.289 us, and
`cudaEventRecord` plus `cudaStreamSynchronize` 0.724 us. None of the primitives is inherently
expensive; the ~7.5 us is the host-device round trip that any of them pays when the wait has to
reach the device.

## Caveats

- The benchmark writes one element, so the large-size numbers measure the eliminated zero-init and
  its page faults in full. A call site that goes on to fill the whole buffer still pays first touch
  on every page; the saving there is one pass over the buffer, not two.
- Both arms include the conditional-synchronization change to `rmm_host_allocator`, so this run
  cannot show that effect in isolation: pageable `host_vector` already benefits from it here. A
  main-vs-branch comparison would be needed to size it, and it is only a few microseconds per
  allocation.
- The end-to-end targets in the note (`ast_jit_wide_table`, `parquet_read_chunks`,
  `multibyte_split_source`, ...) cannot move yet, since no call site has been migrated.

## Files

- `HOST_ALLOCATION.json`, `HOST_ALLOCATION.log`
