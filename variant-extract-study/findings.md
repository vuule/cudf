# Findings

Detail behind `README.md`. Everything was measured on one A100 with CUDA 12.9, status reporting off
unless stated, and `--min-time 0.3` for nvbench cases. "GPU time" is nvbench's cold GPU mean;
kernel-level splits come from `nsys stats --report cuda_gpu_kern_sum`, divided by instance count.

Two benchmarks are used throughout:

- **multi-field**: N sibling fields of one object, 2M rows, INT32 output. Axes for shared or disjoint
  path prefixes and for short (`f00`) or long (`attributes_field_f00`) keys.
- **workload**: 1M rows, an 85-key dictionary, a root object nesting most of its data under one key,
  and 55 paths fanning out below it at four to five steps, STRING output. Modelled on the example.

## Where the time went to begin with

At 64 fields on 2M rows the batched call sustains ~35 GB/s of input against an A100's ~1.9 TB/s, so
about 2% of bandwidth. It is latency and instruction bound, for two structural reasons:

- One thread per row, so neighbouring lanes walk unrelated blobs and nothing coalesces.
- Every probe converts a field id back to a name and compares it byte at a time.

Separately, `extract_variant_fields` materializes every path's `list<uint8>` intermediate before
casting: 5.02 GiB peak at 64 fields x 2M rows, against 80 MiB for the looped single-path API.

Baseline, before the study's prototypes:

| case | GPU time | peak memory |
|---|---|---|
| workload 1M, batched | 67.6 ms | 2.374 GiB |
| workload 1M, looped | 212.5 ms | 46.2 MiB |
| multi-field 2M, 64 fields, batched | 47.2 ms | 5.020 GiB |
| multi-field 2M, 16 fields, batched | 9.14 ms | 1.255 GiB |
| multi-field 2M, 4 fields, batched | 2.43 ms | 321 MiB |
| multi-field 2M, 1 field, batched | 0.758 ms | 80.3 MiB |

## Word-at-a-time key comparison (`use_word_compare`)

`compare_keys`, used by `find_field`. A compile-time constant rather than an environment variable so
that neither side of the A/B pays a branch.

| case | `string_view::compare` | `compare_keys` | delta |
|---|---|---|---|
| workload 1M, batched | 67.5 ms | 68.1 ms | +1% |
| workload 1M, looped | 212.9 ms | 193.7 ms | **-9%** |
| 2M x 16 fields, short keys, batched | 9.20 ms | 9.22 ms | 0 |
| 2M x 16 fields, short keys, looped | 17.58 ms | 17.07 ms | -3% |
| 2M x 16 fields, long keys, batched | 14.33 ms | 12.41 ms | **-13%** |
| 2M x 16 fields, long keys, looped | 39.70 ms | 31.88 ms | **-20%** |
| 262k x 64 fields, short keys, looped | 39.78 ms | 36.47 ms | -8% |
| 262k x 64 fields, long keys, batched | 18.98 ms | 18.27 ms | -4% |
| 2M x 64 fields, short keys, batched | 47.2 ms | 47.7 ms | +1% |
| 2M x 64 fields, short keys, looped | 180.1 ms | 147.8 ms | **-18%** |

Two separate effects, and the first was a surprise:

- Both benchmarks' original keys were shorter than eight bytes (`f00`, `item001`), so the wide path
  never ran, and every short-key number above is purely a **leaner byte loop**: one precomputed bound
  and indexed loads, against `string_view::compare`'s two bounds checks, pointer bumps and
  pointer-equality precheck per call. That alone is worth 8-18% on the single-path API, is independent
  of everything else here, and arguably belongs in `string_view::compare` itself.
- The word packing proper needed the new `key_len` axis (16-byte keys sharing their first eight
  bytes) to show up at all, and there it is worth 13-20%, including on the batched path, which the
  leaner loop did not move.
- The batched path is flat to 1% worse on short keys: with sibling merging most keys resolve in a
  single probe, so comparison is a small share of its work. Cost is 78 -> 80 registers, which does not
  change occupancy (still 3 blocks/SM).

Real variant keys (`observation`, `temperature`, `user_agent`) are mostly longer than eight bytes, so
the long-key column is the more representative one.

## Shared dictionary (`CUDF_VARIANT_SHARED_DICT=1`)

Three pieces. `metadata_lengths_uniform_kernel` and `metadata_bytes_uniform_kernel` decide whether
every row's metadata blob is byte-identical to row 0's, which is the precondition for one dictionary
to stand in for all of them; one warp per row, lanes striding the blob, so the reads coalesce and row
0's copy stays in cache. `build_shared_dictionary_kernel` then resolves, once, `rank[id]` for every
dictionary entry and `step_rank[step]` for every step token, with -1 for a token the dictionary does
not contain. `find_field_by_rank` replaces `find_field` under a template axis, threaded through
`locate_object_field_ranked`, `locate_object_fields_ranked`, `resolve_steps` and the trie kernel.

Ranks rather than field ids: an object's fields are ordered by *name*, while a dictionary need not be
sorted, so ids are not comparable but ranks order fields exactly as names would. A step token absent
from the dictionary gets rank -1, and then no field can match and the search answers "absent" without
reading a single name byte.

Walk is the locate kernel alone; detect is the two uniformity kernels plus the dictionary build:

| case | walk, names | walk, ranks | walk delta | detect | end to end |
|---|---|---|---|---|---|
| workload 1M | 45.07 ms | 13.76 ms | **-69%, 3.3x** | 2.68 ms | 68.3 -> 39.9 ms (**-42%**) |
| 2M x 64 fields, short | 23.05 ms | 16.90 ms | **-27%** | 1.83 ms | 47.7 -> 43.4 ms (-9%) |
| 2M x 16 fields, long | 6.17 ms | 2.45 ms | **-60%** | 2.46 ms | 12.46 -> 11.31 ms (-9%) |
| 2M x 16 fields, short | 2.97 ms | 2.45 ms | -18% | 0.54 ms | 9.25 -> 9.36 ms (+1%) |
| 2M x 4 fields, long | 0.985 ms | 0.555 ms | -44% | 2.42 ms | 2.68 -> 2.98 ms (+11%) |
| 2M x 4 fields, short | 0.790 ms | 0.558 ms | -29% | 0.42 ms | 2.50 -> 2.76 ms (+10%) |

- The walk wins everywhere, 1.2x to 3.3x, and wins most exactly where the current kernel is worst:
  the deep narrow workload trie, whose ~100 name steps per row each cost a dictionary parse plus a
  search over name bytes.
- It makes key length free: 16 fields costs 2.45 ms whether the keys are 3 bytes or 20. That is the
  clearest evidence that reading name bytes was the cost, not the search.
- It removes two things at once, the byte comparisons and the per-group metadata parse, since the
  ranked path never opens the blob.
- Detection decides whether any of that reaches the caller. It is a fixed 0.4-2.7 ms scan, which the
  heavy cases absorb and the 2.5 ms cases cannot.

Detection cost is an implementation artifact, not a property of the idea. Three versions span 7x: a
thread-per-row byte loop cost 6.75 ms on the workload; a flat coalesced sweep comparing byte `i`
against byte `i % blob_len` was worse still, because that is a 64-bit modulo per byte; warp-per-row is
2.50 ms. Even that is far above the streaming floor, since the workload's metadata column is ~90 MB,
so ~0.07 ms at bandwidth. Better than optimizing it is not scanning at all: if the reader hands
extraction the metadata column's Parquet dictionary encoding (one blob plus row indices, which is how
these columns are stored anyway), sharing is known for free, and a scan is needed only for
already-materialized columns.

The catch: both benchmarks and the example give every row the same blob, which is the best possible
case. A writer that emits only the keys a row actually uses gives rows with different optional fields
different blobs, and the same name a different id -- precisely the case worth batching. The prototype
detects that correctly and falls back to the name path, having paid the scan for nothing. Making it
pay on real files means grouping rows by distinct blob rather than requiring one: a hash per row, a
group-by, and a rank table per distinct dictionary.

Verdict: the only prototype here that changed the walk by more than a few percent, and the walk is
36-66% of end-to-end time, so it is worth pursuing -- but through the reader interface, not as a scan
bolted onto extraction.

### Subsumed by it: parse the metadata dictionary once per row

`parse_metadata_dictionary` is called for every name step and once per sibling group, and the
dictionary is a per-row constant, so all of those parses re-derive the same header. On the workload
trie that is on the order of 100 parses per row, against 2 in the flat 64-field case, which is why
the workload's locate kernel is 4x heavier per row-path (42.9 ns/row against 11.0).

Hoisting the parse to once per row is pure fixed-cost removal and, unlike the shared dictionary,
depends on nothing about the data. The ranked path already removes these parses, so this is worth
measuring on its own only if the shared dictionary is not taken. It was not measured here. It would
also touch the signatures of `resolve_steps` and `locate_object_field`, and it overlaps #23547, whose
search setup wants hoisting the same way. No upstream issue covers it; the nearest neighbour is
#23656, which packs `optional<size_type>` and `op_status` into one 64-bit return in the same spirit.

## Level-synchronous launches (`CUDF_VARIANT_LEVEL_SYNC=1`)

One launch per trie depth, one thread per (row, group at that depth), with the group still the unit
of work so the merged sibling sweep survives intact. The group is the grid's second dimension, so a
thread's row is 32-bit arithmetic and a block's rows are consecutive.

| case | depth-first | level-synchronous | delta |
|---|---|---|---|
| workload 1M, batched | 67.8 ms | 62.2 ms | **-8.3%** |
| workload 1M, batched, status on | 68.0 ms | 62.5 ms | **-8.1%** |
| 2M x 64 fields, shared prefix | 47.30 ms | 41.56 ms | **-12.1%** |
| 2M x 64 fields, disjoint | 46.53 ms | 41.05 ms | **-11.8%** |
| 2M x 16 fields, shared prefix | 9.14 ms | 8.91 ms | -2.5% |
| 2M x 16 fields, disjoint | 8.81 ms | 8.40 ms | -4.6% |
| 2M x 16 fields, long keys | 12.33 ms | 12.31 ms | 0 |
| 2M x 4 fields, disjoint | 2.234 ms | 2.236 ms | 0 |
| 2M x 4 fields, shared prefix | 2.46 ms | 2.62 ms | +6.5% |
| 2M x 4 fields, long keys | 2.63 ms | 3.11 ms | +18% |

On the example, batched, it is 66.9 -> 60.7 ms, a 9% win, consistent with the workload benchmark and
with this being a deep trie. Note it cannot be combined with the shared dictionary here: only the
depth-first kernel takes a resolved dictionary, and the shared dictionary is worth far more (38.1 ms),
so the two switches are alternatives rather than additive.

Peak memory is unchanged everywhere. Only *interior* slots need their value kept between levels, so
the state buffer is `interior_slots x rows x 12 B`, which is nothing next to the `list<uint8>`
intermediates, and the flat cases allocate nothing at all.

What it buys is not more parallelism but less local memory:

- The depth-first kernel is `REG:80 STACK:1552`, and that 1.5 KB is the group state array it must keep
  alive while the walk is anywhere inside a group's subtrees, touched at every slot. Level-synchronous
  consumes each group's results as they are produced, so it holds one value at a time: `REG:98
  STACK:16`.
- Occupancy went *down*, not up: 98 registers is 2 blocks/SM against the depth-first kernel's 3. It
  wins anyway.
- Confirmed by trying to have it both ways: `__launch_bounds__(block_size, 3)` caps the kernel at 85
  registers, which spills (`STACK:80`) and costs far more than the extra block buys -- 64 fields goes
  41.6 -> 60.6 ms, the workload 62.2 -> 74.3 ms.

The narrow-trie regression resisted removal, and a block-size sweep says why: the best configuration
depends on trie shape, so no single one wins everywhere (GPU ms, shared prefix,
`CUDF_VARIANT_LEVEL_BLOCK`):

| case | depth-first | level 64 | level 128 | level 256 |
|---|---|---|---|---|
| 2M x 4, short | **2.46** | 2.52 | 2.51 | 2.65 |
| 2M x 4, long | **2.63** | 2.95 | 2.94 | 3.13 |
| 2M x 16, short | 9.14 | 8.93 | **8.83** | 9.00 |
| 2M x 16, long | 12.33 | 12.25 | **12.08** | 12.41 |
| 2M x 64, short | 47.30 | 45.60 | 45.86 | **41.89** |
| workload 1M | 67.8 | 65.3 | 65.4 | **62.6** |

- Specializing the kernel on "this trie keeps no state at all", removing the per-key state lookup
  entirely, did not move the 4-field number (2.619 against 2.615 ms), so the state indirection is not
  the cost.
- Smaller blocks divide the register file more finely and recover part of it (4 short, 2.65 -> 2.51),
  but cost the wide cases more than they buy (64 fields, 41.9 -> 45.9). And 4 long stays ~12% down at
  every block size, so occupancy is not the whole story either.
- What is left is that the level kernel inlines the slot's remaining steps and its output writes into
  the sweep's emit callback, where the depth-first walk does them in a separate loop. That inflates
  the sweep's loop body, 80 -> 95 registers, which a 4-key sweep has no length to amortize over.

Worth taking as a **replacement** for the depth-first walk rather than a second kernel behind a shape
dispatch. Keeping both means maintaining the status and malformed-blob rules twice, in the part of
this code that was hardest to get right, with a suite that exercises only one path per run. The
replacement is close to free in code size, because most of what the depth-first walk carries exists
only to bound its walk state:

- Gone: `max_local_walk_state` and its 128-slot cap, `max_global_scratch_blocks`, the `d_scratch`
  allocation and the grid cap that comes with it, `walk_state_size`, the `UseLocalScratch` template
  axis, `trie.state_base`, and the `MergeSiblings` axis (level-synchronous always merges) -- four
  kernel instantiations down to one.
- Added: the group-member CSR (which subsumes `group_first`), `slot_state`, `group_parent_state` and
  the groups-by-depth buckets, all plain CSR arrays built once on the host.
- The real saving is conceptual. The depth-first design's hardest invariant is that a group's resolved
  children stay live for as long as the walk is anywhere inside their subtrees; that is what
  `state_base`'s widest-group-per-depth reservation implements, and what the slot cap and global
  fallback exist to bound. Level-synchronous deletes the invariant: a value is written once and read
  once, at the next level.

The price is a few percent on tries narrower than ~16 keys, worst measured at +18% on a 2.6 ms case.
Cheap next to a permanent second implementation, and revisitable by splitting the output writes back
out of the sweep's emit. Worth keeping in perspective: that regression is against the other batched
kernel, not against what a caller would otherwise use, since the looped API is 17.6 ms on the same
4-key case against either kernel's 2.5.

## Warp per row (`CUDF_VARIANT_WARP_ROW=1`)

One warp per row, walk state in dynamic shared memory (one slice per warp, 8 warps x 128 slots x 12 B
= 12 KiB per block), with the lanes of a warp splitting a sibling group's keys. A group's key slots
are resolved end to end in that parallel phase, so the slot loop keeps serial work only for index
steps and groups with nothing to merge.

| case | thread per row | warp per row | delta |
|---|---|---|---|
| workload 1M, batched | 68.2 ms | 116.0 ms | **1.7x slower** |
| 2M x 1 field | 0.771 ms | 0.769 ms | 0 (falls back to the single-path API) |
| 2M x 4 fields, short keys | 2.49 ms | 16.14 ms | **6.5x slower** |
| 2M x 16 fields, short keys | 9.23 ms | 28.91 ms | **3.1x slower** |
| 2M x 64 fields, short keys | 47.6 ms | 83.4 ms | **1.8x slower** |
| 2M x 4 fields, long keys | 2.66 ms | 19.02 ms | 7.1x slower |
| 2M x 16 fields, long keys | 12.46 ms | 35.47 ms | 2.8x slower |

The shape of the loss says why the premise was wrong:

- The kernel is `REG:95`, so 2 blocks/SM: 16 rows in flight per SM against 768 for thread-per-row
  (`REG:80`, 3 blocks/SM). Row-level parallelism was what hid the probe latency, and one warp per row
  throws away ~48x of it to buy at most 32x within a row, and only when a group is 32 keys wide. At 4
  keys it buys 4x, which is exactly where the loss is worst.
- Splitting keys across lanes also gives up the merged sweep. A lane does not know where the key
  before it landed, so each key is a full `log2(num_fields)` search again (~6 probes at 64 fields
  against ~1), and every lane re-parses the metadata and object headers the sweep parsed once per
  group. Work per row grows ~7x; the trend across 4 -> 16 -> 64 keys (6.5x -> 3.1x -> 1.8x) is the
  widening parallelism slowly clawing that back, never reaching break-even.
- The coalescing the idea was after is real but small: it applies only to the field id and dictionary
  offset reads, which the merged sweep had already cut to about one per key.

Useful conclusion: parallelism *inside* a row is not the lever while the kernel is at 2% of bandwidth
with 768 rows resident per SM. The one variant not refuted by this run is a merge join in the other
direction -- lanes scan contiguous chunks of the object's *field ids*, fully coalesced, and binary
search the small sorted key array for each, so work stays O(num_fields) per group rather than
O(num_keys x log num_fields). It still pays the occupancy loss, so it needs groups much wider than 32
to have a chance.

## Sizing the fused cast (not implemented)

Casting straight from the located (row, offset, size) triples, with no `list<uint8>` intermediate.
Per-iteration kernel breakdowns, from `nsys`:

| 2M rows x 64 fields -> INT32 | per iteration | share of 47.6 ms |
|---|---|---|
| locate kernel | 23.01 ms | 48% |
| `BatchMemcpyKernel` (bytes -> `list<uint8>`) | 11.43 ms | **24%** |
| `cast_variant_primitive_kernel<int>` (64x) | 1.90 ms | 4% |
| `DeviceScanKernel`, sizes -> offsets (64x) | 1.63 ms | 3% |
| `count_set_bits` (128x) | 0.45 ms | 1% |
| scan init, multi-block memcpy, D2D copies | 0.28 ms | 1% |
| unaccounted (allocation, launch gaps) | ~8.9 ms | 19% |

| 1M rows x 50 paths -> STRING (workload) | per iteration | share of 67.8 ms |
|---|---|---|
| locate kernel | 45.00 ms | 66% |
| `BatchMemcpyKernel` | 6.89 ms | **10%** |
| `strings_children_kernel` (2 passes/path) | 3.12 ms | 5% |
| `DeviceScanKernel` | 1.85 ms | 3% |
| `count_set_bits`, scan init | 0.57 ms | 1% |
| unaccounted | ~10.4 ms | 15% |

So it is cheap on strings and not cheap at all on wide fixed-width extraction: the intermediate copy
alone is 24% there, and with the offsets scans a fixed-width output would not need either, ~28%. That
is more than level-synchronous won, for a change that removes code rather than adding a kernel. Some
of the unaccounted time should also go, since it is dominated by allocating and freeing the per-path
intermediates.

Memory is the larger prize. At 64 fields the 5.02 GiB peak is mostly bookkeeping for the
intermediates: `d_sizes` and `d_src_offsets` are 2 x 64 x 2M x 4 B = 1.07 GiB, the per-path offsets
columns another 512 MiB, plus the located bytes themselves, against 512 MiB of actual INT32 output.

Design note, from what the warp and level prototypes taught: cast from the triples in a separate
kernel rather than fusing the decode into the locate kernel. Fusing all the way would make the located
bytes free to decode, since they are already in registers, but it needs per-path target-type dispatch
inside the locate kernel, and register pressure there is worth 10-20% either way. A separate typed
pass re-reads a few bytes per row scattered instead of the whole compacted intermediate, which should
still be far cheaper than writing and reading it. For strings the sizing pass can come from `d_sizes`,
which the locate pass already produced, so only the copy pass remains.

On the example, in the fastest measured configuration (batched, shared dictionary), the budget is:

| | per iteration | share of 38.5 ms |
|---|---|---|
| locate trie kernel | 14.04 ms | 36% |
| `BatchMemcpyKernel`, bytes -> `list<uint8>` | 6.35 ms | **16%** |
| `metadata_bytes_uniform_kernel` (detection) | 2.50 ms | 6% |
| `strings_children_kernel`, 2 passes per path | 2.34 ms | 6% |
| `DeviceScanKernel`, sizes -> offsets | 1.81 ms | 5% |
| single-path kernel (columns A and B, one path each) | 0.47 ms | 1% |
| `count_set_bits`, scan init, small memcpys, dict build | 0.96 ms | 2% |
| not on the GPU: allocation, launch gaps | ~10 ms | 26% |

Projected onto that, the fused cast should land the example at ~29-31 ms: the intermediate copy goes
outright (6.35 ms), so do the intermediates' own offsets scans and null masks (~1.1 ms);
`strings_children` stays but reads the variant blob scattered rather than a compacted intermediate,
call it +0.5 ms, which is the softest number here; and some of the ~10 ms off-GPU goes with the
per-path allocations. With a reader-side sharing signal deleting the detection scan too, ~26-28 ms.

## Shredded-column projection (assessed, not implemented)

Where a field is shredded, extraction can project the typed column directly instead of parsing blobs,
with the unshredded rows falling back to the paths above.

Shredding has its own waste: a shredded column is materialized at full width whatever its null count,
so a rarely-present INT64 field costs 8 B plus a validity bit per row (~16 MB at 2M rows) even though
the nulls cost almost nothing on disk as RLE definition levels. It stays manageable because shredding
is per field and partial shredding leaves the rest in the `value` blob, and because an unrequested
shredded column is never read, unlike the `metadata` column that every extraction touches. The real
limit is column explosion in the footer for wide sparse schemas.

## Gotchas worth keeping

- Metadata is a **per-row** field: every row carries its own dictionary and they may all differ. Rows
  sharing a dictionary is a property of the data, not a guarantee of the format. Sharing must also be
  byte-exact, since a writer that emits only the keys a row uses gives the same name a different id in
  different rows.
- Field ids inside an object are ordered by **name**, not by id, so an id-based binary search is only
  valid when the dictionary is sorted (then rank == id). Hence the rank table.
- Both extraction benchmarks, and the example, hand every row the same metadata blob. That is the best
  possible case for anything dictionary-related; an axis that varies dictionaries across rows is
  needed before those numbers mean much.
- A fat per-row dictionary can overflow `size_type`: 64 keys of 20 bytes is a ~1.4 kB blob, and at 2M
  rows the metadata child column exceeds 2^31 bytes and the column factory throws. Hit while adding
  the `key_len` axis, so it is a real ceiling on any "union dictionary in every row" scheme.
- Local memory in the depth-first walk is **not** a locality problem: CUDA interleaves local memory
  across lanes, and the index into the walk state is warp-uniform (it depends on slot and depth, not
  row), so those accesses already coalesce like an explicit `[slot][row]` array and get L1 reuse on
  top. What hurt was the volume, not the layout.
- Neither restructuring prototype was limited by parallelism. Warp-per-row added intra-row parallelism
  and lost 1.7-6.5x; level-synchronous cut occupancy from 3 blocks/SM to 2 and still won 8-12%. With
  768 rows resident per SM there is no shortage of independent work to hide latency behind, so the
  levers that pay are the ones that remove work or memory traffic per row.

## If the writer is under our control

- `sorted_strings = 1` with unique sorted keys makes dictionary rank equal field id, and since the
  spec already requires an object's field ids to appear in lexicographic order of their names, the ids
  inside an object are then strictly ascending: probe by integer binary search, no dictionary reads
  and no rank table, and it needs no sharing at all. This is #23638.
- A canonical dictionary per row group, byte-identical in every row, makes ids stable across rows so a
  name resolves once for the whole group. The encoding does not require a row's dictionary to be
  minimal, only that every name its value uses is present, and both Spark's and the Rust
  `parquet-variant` builders can seed a builder with a known dictionary.
- But a union dictionary inflates per-row metadata: ~700 B for 85 keys x 2M rows is ~1.4 GB for the
  kernel to read, against ~60 MB of minimal per-row dictionaries. It is nearly free on disk, where the
  metadata column dictionary-encodes to one blob plus indices, and expensive only because cudf expands
  it per row.
- So the writer-side win needs a reader-side counterpart: keep the metadata column's Parquet
  dictionary encoding and hand extraction that pair. That works for any file whose metadata column is
  dictionary-encoded, not only ones we wrote, and it is the interface the shared dictionary prototype
  should have modelled, with a GPU-side hash-and-group fallback for already-materialized columns.
