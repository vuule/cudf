# hostdevice_vector on host_uvector, plus why cuIO does not move

Follow-up to `../20260828_213000/COMPARISON.md`. That round measured the host-writable allocation
path with every call site still on `thrust::host_vector`, which value-initializes. This round
converts `hostdevice_vector` — the staging buffer for parquet and orc — to `host_uvector`, so both
halves of the change are in place, and then measures what is actually left to win.

## The conversion

`hostdevice_vector` now holds `host_uvector<T>` and calls `make_host_writable()` in its
constructor. It hands out raw host pointers through `operator[]`, `host_ptr()` and `begin()`, so
the storage has to be safe to write on return rather than at first use; doing it in the constructor
reproduces exactly what `rmm_host_allocator::allocate` used to do, and picks up the cheap wait when
the host-writable path is enabled. Elements are no longer initialized.

That last part is a real semantic change, not just "the memory is no longer zero". The element
types here — `PageInfo`, `ColumnChunkDesc`, `stripe_dictionary`, `compressed_stream_info` and
others — are trivially copyable but have default member initializers, so the old code ran their
default constructors on every element. `-Wclass-memaccess` flagged them all when the poison memset
was added, which is how they were found.

To check that nothing depended on it, `LIBCUDF_POISON_UNINITIALIZED=1` fills every
`hostdevice_vector` with `0xab` at construction. `PARQUET`, `ORC`, `CSV`, `JSON` and
`MULTIBYTE_SPLIT` all pass poisoned, and all pass again with the host-writable path on.

## Size distribution of pinned allocations

`LIBCUDF_HOST_WRITABLE_STATS=1` now also prints a power-of-two histogram of every pinned
allocation. Parquet ZSTD/FILEPATH, one configuration, whole process:

```
pinned_allocations: count=24633 total_bytes=374039074   (~530 iterations)
  <=2^0    1058     <=2^10   4234     <=2^15   2116
  <=2^2    4371     <=2^11   2116     <=2^17    530
  <=2^3    5862     <=2^13    533     <=2^18    529
  <=2^4      11     <=2^14    530     <=2^19    530
  <=2^5     591
  <=2^6     531
  <=2^7    1072
```

About 46 pinned allocations per iteration, 706 KiB in total per iteration, median 8 bytes, largest
512 KiB. Nothing remotely near the 16 MiB buffer where zero-initialization costs 930 µs.

## What that predicts, and what was measured

Per 26 ms iteration:

| | predicted | measured |
| --- | --- | --- |
| removing the sync (46 × 7.5 µs) | 1.3% | 0.73% |
| removing zero-init (706 KiB memset) | 0.27% | not resolvable |

The sync measurement is six alternating runs of one configuration, `--min-time 2`:
26.454 ms ± 0.252 with the path off, 26.261 ms ± 0.203 with `block_event`. The 0.19 ms difference
is about 1.5 standard errors — the right sign and the right order of magnitude, but this benchmark
cannot resolve it. Converting `hostdevice_vector` on its own moved nothing (mean −0.07% over eight
comparisons), which matches the 0.27% prediction.

## Conclusion

Both halves work and neither matters for parquet, for the same underlying reason: its pinned
allocations are numerous and tiny. 46 allocations per iteration is enough that the fixed
per-allocation cost is the larger of the two effects — the opposite of the ordering suggested by
the size-resolved micro benchmark, where a 16 MiB buffer spends 96% of its cost on
zero-initialization. Both orderings are correct; they describe different allocation profiles.

So the value of this work depends entirely on finding a workload whose pinned buffers are large,
or one that allocates far more often. Neither is true of parquet compressed reads. Before
investing more, the useful next step is to run the histogram over the other cuIO readers and
writers and look for one with megabyte-scale pinned buffers; if none exists, the honest conclusion
is that libcudf's pinned allocation profile makes both optimizations unmeasurable in practice,
however sound they are individually.
