# Prototype: uninitialized host vectors and low-sync pinned allocation

This branch is an experiment, not a proposed change. It exists to answer one question: what does
libcudf actually pay for a pinned host allocation, and how much of that can be removed?

It pairs with the RMM branch `host-writable-pinned-prototype` on `https://github.com/vuule/rmm`
(commit `5714edd`, RMM 26.10.00), which adds the resource-side mechanism. Neither branch is useful
without the other.

## The two costs being attacked

A `cudf::detail::host_vector` allocation on the pinned pool pays for two things that are unrelated
to each other:

1. **A full stream synchronization.** `rmm_host_allocator::allocate` calls
   `cudf::detail::sync_stream` after every allocation. The pool can hand back a block whose
   previous owner still has a copy in flight, and a stream handle is the narrowest thing libcudf
   can wait on, so it drains the whole stream. This cost is fixed per allocation.
2. **Value-initialization.** `thrust::host_vector` constructs every element, so a 16 MiB staging
   buffer is written once before the caller touches it. This cost is proportional to size.

The first is addressed by letting the memory resource choose the wait; the second by a vector type
that leaves its storage uninitialized.

## What is in the branch

**`cpp/include/cudf/detail/utilities/host_uvector.hpp`** — new `host_uvector<T>`, a host analogue
of `rmm::device_uvector`: owning, non-resizing, uninitialized elements. Construction takes a
`host_write_intent`. `immediate` means the storage must be safe for the host to write on return;
`deferred` means the caller will only write after synchronizing itself, and skips the wait
entirely. `host_writable()` performs the wait on demand, `mark_host_only()` records that the buffer
never reached the device so the next owner can skip its wait, and the type tracks `device_exposed`
so callers do not have to assert it by hand.

**`cpp/include/cudf/detail/utilities/host_writable_resource.hpp`** — a small libcudf-side interface
over the RMM entry point. libcudf holds type-erased `rmm::host_async_resource_ref`s and cannot see
the concrete pool type, so `get_host_writable_resource` maps a resource ref to this interface, or
to null when the resource has no such path. This is also where the wait accounting lives.

**`cpp/include/cudf/detail/utilities/host_vector.hpp`** — `rmm_host_allocator` gains the
host-writable branch and, separately, only synchronizes when the resource is actually
stream-ordered. The pageable resource frees with `operator delete` and ignores the stream, so it
never needed the sync in the first place.

**`cpp/src/utilities/host_memory.cpp`** — wires the pinned pool to the RMM mechanism, reads the
environment variables below, and accumulates the counters.

**`cpp/src/io/utilities/hostdevice_vector.hpp`** — converted from `host_vector` to `host_uvector`,
which is what puts the second half of the change on a real cuIO path.

**`cpp/tests/utilities_tests/host_uvector_tests.cpp`** — behavior across sync modes, plus a
regression test for the recycling hazard: a block is freed with a copy in flight, reallocated, and
overwritten by the host, which corrupts the copy if the wait is wrong.

**`cpp/benchmarks/utilities/host_allocation.cpp`** — micro benchmark over `std::vector`,
`host_vector` and `host_uvector` on both resources, swept by buffer size.

**`benchmark_compare/*/COMPARISON.md`** — the measurement write-ups, in order. Raw nvbench JSON and
logs are left out; they are large and regenerable.

## Building and running

```bash
build-cudf-cpp -j0 -DCMAKE_CUDA_ARCHITECTURES=NATIVE -DCPM_rmm_SOURCE=/home/coder/rmm
```

The RMM override is required — `host_memory.cpp` includes `rmm/mr/host_writable.hpp`, which does
not exist in released RMM.

Everything is off by default, so an unconfigured build behaves as `main` does apart from the
`hostdevice_vector` conversion.

| Variable | Effect |
| --- | --- |
| `LIBCUDF_HOST_WRITABLE_SYNC_MODE` | `off` (default), `stream_sync`, `stream_event`, `block_event`, `clean_tracking` |
| `LIBCUDF_HOST_WRITABLE_STATS` | Print wait totals, allocation-size histograms and hit rates to stderr at exit |
| `LIBCUDF_POISON_UNINITIALIZED` | Fill every `hostdevice_vector` with `0xab` at construction, to catch reliance on the old zero-initialization |

The modes differ only in how precisely they identify the work that must finish before the host may
write: the whole stream, the pool's per-stream event, an event recorded for that specific block, or
nothing at all for a block that never reached the device.

## What was measured

Both mechanisms work, and neither moves end-to-end cuIO wall time.

The micro benchmark separates the two costs cleanly. For a 16 MiB pinned buffer, `host_vector`
costs 931 µs, almost all of it value-initialization; `host_uvector` with the event-based wait costs
2.6 µs.

Multithreaded parquet reads show the sync cost is far larger than the micro benchmark suggested,
because a real workload has a deep queue for `cudaStreamSynchronize` to drain:

| mode | mean wait per allocation | blocked time/iteration | wall |
| --- | --- | --- | --- |
| `off` | 415 µs | 9.97 ms | 13.320 ms |
| `stream_event` | 1.9 µs | 0.07 ms | 13.253 ms |

A second configuration gives 367 µs against 6.7 µs. The mechanism does what it claims: 97.7% of
waits (74,500 of 76,227) short-circuit in `cudaEventQuery` and never block.

Wall time does not follow, for two reasons. Parquet's pinned allocations are numerous and tiny —
median 8 bytes, largest 512 KiB, 46 per 26 ms iteration — so there is no large buffer whose
zero-initialization matters. And the blocked time that was removed overlapped GPU work those
threads were waiting on anyway; the host had nothing else to do with it.

## Where this would pay off

The value depends on finding a caller with host work to overlap. A concrete example is the
per-column `_stream.sync()` in PR #23710's dict transcode path: staging buffers are pageable, so
they must outlive their copies, and the loop drains the pipeline once per column instead of letting
columns pipeline against each other. Pinned buffers freed stream-ordered need no drain at all — but
today that trade is a loss, because each pinned allocation costs a full stream sync. At ~7 µs it
becomes a win.

Note also that `stream_event` gets nearly all of the benefit from `cudaEventQuery`
short-circuiting, not from event precision. `block_event` and `clean_tracking` add per-deallocation
event records for a marginally better hit rate and are slower in the micro benchmark. If ~7 µs is
good enough, the whole thing reduces to "the pinned pool waits on its own per-stream event, and
libcudf stops synchronizing" — no new RMM API, and no call-site changes anywhere in libcudf.
