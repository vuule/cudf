# Parquet VARIANT extraction: optimization study

This branch measures where time goes in Parquet VARIANT field extraction and which optimizations pay.
It is a study branch, not a merge candidate: each prototype sits behind its own switch so that one
binary can A/B all of them, and two of them are not worth shipping in the form they are in here.

`findings.md` next to this file has the per-prototype detail, the numbers, and the reasoning. This
file is orientation: what is on the branch, what the headline result is, and how to reproduce it.

## What is on the branch

Below the study commits, the branch carries work that was in flight when the study was done:

- Batched multi-field extraction, `get_variant_fields` and `extract_variant_fields` (issue #22897),
  which resolve many paths per row in one kernel over a trie of the requested paths.
- Sibling lookup merging: paths sharing a parent object have their first steps resolved in one sweep
  over that object's field ids rather than one search per path.
- Binary search on the metadata dictionary and on object field lookups, from #23547, #23638 and
  #23657, merged in locally because they were still open.

On top of that, four prototypes, one commit each:

| switch | prototype | verdict |
|---|---|---|
| `use_word_compare` (compile-time constant, on) | compare dictionary keys a machine word at a time | **take**, and consider moving into `string_view::compare` |
| `CUDF_VARIANT_SHARED_DICT=1` | resolve each row's dictionary once, then probe objects by integer rank | **strongest result**, but input-dependent; wants a reader-side signal |
| `CUDF_VARIANT_LEVEL_SYNC=1` (`CUDF_VARIANT_LEVEL_BLOCK` sets block size) | one kernel per trie depth instead of a per-thread depth-first walk | worth taking as a *replacement*, not as a second kernel |
| `CUDF_VARIANT_WARP_ROW=1` | one warp per row, walk state in shared memory, keys split across lanes | **rejected**, 1.7-6.5x slower; kept for what it rules out |

`CUDF_VARIANT_MERGE_SIBLINGS=0` turns off sibling merging, which is base-branch behaviour rather than
a study prototype.

Also on the branch: `cpp/examples/variant_workload`, from the closed feature branch of #22434 and
brought up to the current API, plus a batched mode. It is the most realistic workload available here
and every number labelled "example" comes from it.

## Headline

The example, 1M rows, 57 paths four to five steps deep, cast to STRING. A100, CUDA 12.9, mean of five
iterations after a warm-up. The `main` row is measured, not extrapolated: the branch touches only
four libcudf files, so checking those out at `origin/main` and rebuilding gives main's number on the
same fixture and GPU.

| configuration | mean / iter | vs main | peak memory |
|---|---|---|---|
| `origin/main`, looped `extract_variant_field` | 804.7 ms | -- | 2011 MiB |
| this branch, looped (binary searches + word compare) | 194.2 ms | **4.1x** | 2011 MiB |
| batched `extract_variant_fields` | 67.3 ms | **12.0x** | 4307 MiB |
| batched + shared dictionary | 38.5 ms | **20.9x** | 4307 MiB |
| + fused cast (projected, not implemented) | ~29-31 ms | ~26-28x | ~2 GiB |

Two things worth reading off this table before anything else:

- The single largest factor is not on this branch at all. Replacing main's linear scan of the
  metadata dictionary with binary search is 4.1x by itself, on a dictionary of only 85 keys. Whatever
  else happens, land #23547, #23638 and #23657.
- Batching buys 2.9x of time for 2.1x of peak memory, because every path materializes a `list<uint8>`
  intermediate before its cast. Casting straight from the located spans is the one unexplored change
  that is both a time win and a memory win, and it depends on nothing about the data. It is sized in
  `findings.md` but not implemented.

## Reproducing

Build for the native architecture, and check `nvidia-smi` for a free device first.

```bash
build-cudf-cpp -j0 -DCMAKE_CUDA_ARCHITECTURES=NATIVE
cd cpp/build/latest && ninja VARIANT_EXTRACT_TEST VARIANT_NVBENCH
```

Correctness gate. Every prototype is meant to be behaviour-preserving, so the suite has to pass with
each switch set, not just on the default path:

```bash
for cfg in "" "CUDF_VARIANT_SHARED_DICT=1" "CUDF_VARIANT_WARP_ROW=1" "CUDF_VARIANT_LEVEL_SYNC=1"; do
  env CUDA_VISIBLE_DEVICES=0 $cfg ./gtests/VARIANT_EXTRACT_TEST
done
```

These tests are a real cross-check rather than a tautology: they compare batched output against the
looped single-path API, which none of the prototypes touch.

The example, and the batched-versus-looped comparison:

```bash
cd cpp/examples && ./build.sh                      # or configure variant_workload alone
cd variant_workload/build
CUDA_VISIBLE_DEVICES=0 ./variant_workload_example 1048576 5                      # looped
CUDA_VISIBLE_DEVICES=0 VARIANT_WORKLOAD_BATCHED=1 ./variant_workload_example 1048576 5
CUDA_VISIBLE_DEVICES=0 VARIANT_WORKLOAD_BATCHED=1 CUDF_VARIANT_SHARED_DICT=1 \
  ./variant_workload_example 1048576 5
```

To reproduce the `main` row, check out `origin/main`'s copies of the four libcudf files the branch
touches, rebuild, and build the example with `-DVARIANT_LOOPED_ONLY` (main has no batched API):

```bash
git checkout origin/main -- cpp/include/cudf/io/experimental/variant.hpp \
  cpp/src/io/parquet/experimental/variant_extract.cu \
  cpp/src/io/parquet/experimental/variant_path.{cpp,hpp}
```

The benchmarks, whose axes the study leans on (`key_len` distinguishes keys longer than a machine
word that share a prefix; `prefix` picks a shared or disjoint path set):

```bash
CUDA_VISIBLE_DEVICES=0 ./benchmarks/VARIANT_NVBENCH -d 0 \
  -b bench_variant_extract_multi_field \
  -a num_rows=2097152 -a num_fields=[4,16,64] -a prefix=shared -a key_len=[short,long] \
  -a api=batched -a status=off --min-time 0.3
CUDA_VISIBLE_DEVICES=0 ./benchmarks/VARIANT_NVBENCH -d 0 \
  -b bench_variant_extract_workload -a num_rows=1048576 -a api=batched -a status=off --min-time 0.3
```

Kernel-level splits in `findings.md` come from `nsys`, since `ncu` cannot read counters in this
container:

```bash
nsys profile --force-overwrite true -o /tmp/p --trace cuda --sample none <command>
nsys stats --report cuda_gpu_kern_sum /tmp/p.nsys-rep
```

## Where the code is

Everything except the example is in `cpp/src/io/parquet/experimental/variant_extract.cu`. Comment
blocks marked `PROTOTYPE SCAFFOLDING` delimit what belongs to the study rather than to the base
work. The host-side trie, including sibling grouping, is built in `variant_path.cpp`.

## If someone picks this up

In this order, on the reasoning in `findings.md`:

1. Land the binary search PRs. Largest single factor, already reviewed.
2. Implement the fused cast. 24-28% of wide fixed-width extraction and most of the peak memory, with
   no dependence on the data. Do it as a separate typed pass over the located (row, offset, size)
   triples, not fused into the locate kernel, where register pressure is worth 10-20% either way.
3. Take word compare, or better, take it into `string_view::compare`, where the leaner byte loop
   would benefit every caller.
4. Replace the depth-first walk with the level-synchronous one, accepting a few percent on narrow
   tries in exchange for deleting the walk-state machinery it makes unnecessary.
5. Revisit the shared dictionary only together with a reader-side signal that the metadata column was
   dictionary-encoded. As a scan bolted onto extraction it costs more than it is worth on small
   workloads.
