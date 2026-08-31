# Proposal: host vector types for libcudf

## Problem

`cudf::detail::host_vector` is `thrust::host_vector` with `rmm_host_allocator`, and it has two costs
that neither of its uses needs:

- Every allocation blocks the host. `rmm_host_allocator::allocate` calls `sync_stream(stream)`
  unconditionally, which costs 8.7 us against 2.9 us for an equivalent `std::vector` when building a
  small array. Nearly all of the difference is the synchronization: dropping it, and nothing else,
  brings the pinned allocation to 2.9 us. It is charged even when the resource is not stream-ordered
  (see below), where it buys nothing at all.
- Sized construction value-initializes every element, then the caller immediately overwrites it.
  For the staging buffers in `cpp/src/column/column_device_view.cu:73`,
  `cpp/src/table/table_device_view.cu:60`, `cpp/src/io/text/data_chunk_source_factories.cpp:69` and
  `cpp/src/io/parquet/io_utils/parquet_io_utils.cpp:523` that is a memset of megabytes, milliseconds
  of host time before the real data arrives.

There is also a latent correctness problem in the opposite direction, described under
[Deallocation](#deallocation-must-follow-the-resource).

## Uses to support

1. **Device to host.** Create, then fill by copying from the device. No initialization needed. No
   synchronization needed at allocation, because the fill happens on the stream; the host must
   synchronize before *reading* the result, which the existing `make_host_vector(device_span, ...)`
   already does.
2. **Host to device.** Create, fill on the host, copy to the device. No initialization needed. A
   synchronization before the host writes is required, not merely tolerated: the pinned pool can
   hand back a block whose previous owner still has a copy in flight
   (`stream_ordered_memory_resource::get_block` only inserts a `cudaStreamWaitEvent` for blocks
   taken from *other* streams). What matters beyond that is that deallocation is stream-ordered, so
   the buffer outlives the H2D copy.
3. **Host only.** No device transfer at any point, e.g. the regex scratch buffer in
   `cpp/src/strings/regex/gkexec.cpp:68` and much of the host-side metadata built with
   `make_empty_host_vector`. Wants no initialization, no synchronization, and no pinned pool.
4. **Long-lived reusable staging.** Allocated once and reused across many transfers, growing by
   replacement, e.g. `cpp/src/io/text/data_chunk_source_factories.cpp` and the bgzip source. Wants
   to amortize pool traffic; the per-allocation question mostly disappears.

## Contract

One rule covers cases 1 through 3:

> Allocation and deallocation are ordered on the buffer's stream. The host may not access the
> storage, for reading or writing, before synchronizing with that stream past the operation it
> depends on: the allocation for a write, the fill for a read. Device work enqueued on that stream
> may read the storage up to the deallocation point.

This closes the block-recycling hazard without any event tracking of our own. For same-stream reuse
the required synchronization drains the previous owner's pending work; for cross-stream reuse the
pool has already made our stream wait on the donor's event, so synchronizing our stream covers it.

Two consequences worth stating explicitly in the type's documentation:

- **Capacity is fixed at construction.** Growth would copy old contents into new storage, which is a
  host write, and would therefore drag in another synchronization per growth. The call sites that
  really grow (case 4) replace the whole buffer instead.
- **A buffer belongs to one stream.** cuIO passes buffers between stream-pool streams; a buffer
  allocated on stream A and freed on stream B frees against the wrong event. Either forbid it or
  offer an explicit rebind that records on A and waits on B.

## Deallocation must follow the resource

`get_host_allocator` picks the resource by size, and with `LIBCUDF_ALLOCATE_HOST_AS_PINNED_THRESHOLD`
defaulting to 0 every `make_host_vector` allocation goes to `new_delete_memory_resource`, which is
not stream-ordered in either direction:

```276:296:cpp/src/utilities/host_memory.cpp
  void* allocate([[maybe_unused]] cuda::stream_ref stream,
                 std::size_t bytes,
                 std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    return allocate_sync(bytes, alignment);
  }
  ...
  void deallocate([[maybe_unused]] cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    deallocate_sync(ptr, bytes, alignment);
  }
```

So the allocation synchronization is pure waste there, and the free is immediate rather than
stream-ordered. A case-2 buffer on this resource is freed while its H2D may still be pending. Today
that is survivable because a pageable `cudaMemcpyAsync` stages the data before returning, but
`cuda_memcpy.hpp` documents that the CUDA 13 batch path with `cudaMemcpySrcAccessOrderStream` reads
the source when the stream reaches the copy, and `rapidsai/rmm#2521` is the same bug seen from RMM's
side. The type must therefore know whether its resource is stream-ordered and either synchronize on
destruction or reject case-2 use when it is not.

## Proposed types

A single container plus factories that encode the use case, rather than a family of containers. The
container is `rmm::device_uvector`'s host counterpart:

```cpp
namespace cudf::detail {

/**
 * @brief Host storage with uninitialized elements, allocated from a host memory resource.
 *
 * Allocation and deallocation are ordered on `stream`. The host must synchronize with `stream`
 * before accessing the storage: past the allocation to write, past the fill to read.
 */
template <typename T>
class host_uvector {
  static_assert(std::is_trivially_copyable_v<T> && std::is_trivially_destructible_v<T>);

 public:
  using value_type = T;

  template <typename ResourceType>
  host_uvector(std::size_t capacity, ResourceType mr, cuda::stream_ref stream);

  host_uvector(host_uvector&&);
  host_uvector& operator=(host_uvector&&);
  ~host_uvector();  // stream-ordered deallocation

  // Storage, no synchronization; safe to hand to a copy or a kernel on this buffer's stream
  [[nodiscard]] T* data();
  [[nodiscard]] std::size_t size() const;
  [[nodiscard]] std::size_t capacity() const;
  void resize_uninitialized(std::size_t size);  // <= capacity, does not touch elements

  // Host access; synchronizes the stream on first use, then remembers it
  [[nodiscard]] host_span<T> host_writable();

  [[nodiscard]] operator host_span<T const>() const;  // carries is_device_accessible
  [[nodiscard]] operator host_span<T>();

  [[nodiscard]] bool is_device_accessible() const;
};

}  // namespace cudf::detail
```

The `host_writable()` accessor is what turns the contract from a comment into an API: case 1 never
calls it and therefore never synchronizes, case 2 calls it once, and an illegal early write is hard
to write by accident. It also enables batching, since one synchronization can cover several buffers
allocated back to back, where today each allocation pays its own.

`push_back` and `operator[]` are conveniences on top of `host_writable()` for the call sites that
fill incrementally; they must not be reachable without having synchronized.

## Mapping the existing factories

Call sites in `cpp/src`, counted by spelling (`<T>(` for an explicit element type, which is almost
always the sized case-2 form, and `(` for a deduced one, which is almost always the case-1 form that
copies from a device container):

| Factory | Call sites in `cpp/src` | Case | Proposed replacement |
| --- | --- | --- | --- |
| `make_host_vector(device_container, stream)` | 28 | 1 | `make_host_uvector(device_span, stream)`: uninitialized, copy, sync |
| `make_host_vector<T>(size, stream)` | 39 | 2 | `make_host_uvector<T>(size, stream)` plus `host_writable()` |
| `make_host_vector_async(...)` | 24 | 1 | same as case 1 without the sync |
| `make_empty_host_vector<T>(capacity, stream)` | 27 | 2, 3 | `make_host_uvector<T>(capacity, stream)` plus `host_writable()` |
| `make_pinned_vector(...)` | 28 | 1, 2 | pinned resource, uninitialized |
| `make_pinned_vector_async(...)` | 32 | 1, 2 | pinned resource, uninitialized, no sync |
| `make_empty_pinned_vector<T>(capacity, stream)` | 6 | 2 | pinned resource, `host_writable()` |

The resource stays a parameter of the factory, not a property of the type, so the size-based policy
in `get_host_allocator` and the `set_allocate_host_as_pinned_threshold` knob keep working unchanged.

Element types are not an obstacle: every type used with these factories in `cpp/src` is trivially
copyable (`uint8_t`, `char`, `bool`, `size_type`, and PODs such as `statistics_merge_group`,
`codec_exec_result`, `page_index_info`, `column_parse::flags`). The only non-trivial uses are
`thrust::host_vector<std::string>` in test utilities, which do not go through these factories and
can stay on `thrust::host_vector`.

## What this does not fix

Case 2 still pays one synchronization per buffer (or per batch), because a host write into
pool-recycled pinned memory genuinely has to wait for the previous owner. Making that wait cheap
means waiting on the freeing block's own event instead of the whole stream, which is an RMM change;
`rapidsai/rmm#1995` is the closest existing thread, and `rapidsai/rmm#2053` plus
`rapidsai/rmm#2054` may make it moot by moving pinned allocation onto CUDA's async pools. That is
out of scope here.

Nor does it make pinned memory worth using for small host-filled arrays. Removing the allocation
synchronization brings the allocation itself down to `std::vector` cost, but the stream-ordered free
still records an event, and a few hundred bytes of H2D is too little for the pinned path to win it
back; call sites that stage kilobytes rather than megabytes should keep using `std::vector`. The
wins to expect from this type are the eliminated zero-init on large buffers and the eliminated
synchronization on cases 1 and 3.

## How to tell whether it helps

The two costs scale differently, so they need different experiments. The synchronization is a fixed
per-allocation charge of a few microseconds, visible only where allocations are frequent and the
surrounding work is small. The zero-init is proportional to the buffer, and it is the only one that
can show up as milliseconds.

Micro level, to confirm the type itself is not a regression: an nvbench with one arm per candidate
(`std::vector`, `host_vector`, `host_uvector`) over the pinned and the pageable resource, sweeping
element count from a handful to megabytes, timed with `nvbench::exec_tag::sync`. Include an arm that
enqueues work on the stream before allocating, since the cost of the allocation sync is whatever is
in flight, not a constant, and an arm that does nothing, to establish the harness floor.

End to end, at the call sites where the costs are large enough to survive being averaged into a real
operation:

| Target | Benchmark | What should move |
| --- | --- | --- |
| `column_device_view` / `table_device_view` staging, sized by column count | `TRANSFORM_NVBENCH -b ast_jit_wide_table`, `PARQUET_READER_WIDE_NVBENCH -b parquet_read_wide_tables` | zero-init and one sync per view creation, both proportional to width |
| Parquet reader host buffers | `PARQUET_READER_CHUNKS_NVBENCH -b parquet_read_chunks` | zero-init of the large staging buffers |
| `multibyte_split` chunk source | `MULTIBYTE_SPLIT_NVBENCH -b multibyte_split_source` | zero-init per chunk, plus the sync on the reusable buffer |
| Decompression and nvcomp host fallback | `PARQUET_READER_COMPRESSED_NVBENCH -b parquet_read_io_compression` | many small allocations, so mostly sync |
| JSON reader host-side metadata | `JSON_READER_NVBENCH -b json_read_io` | many small allocations |

Method: build both variants, run them interleaved rather than one suite after the other, and compare
with `nvbench_compare.py`. Fix the resources explicitly, `--rmm_mode pool` or `async` and
`--cuio_host_mem pinned_pool`, because the pinned pool's state affects allocation cost, and set
`LIBCUDF_ALLOCATE_HOST_AS_PINNED_THRESHOLD` deliberately rather than inheriting the default of 0,
which sends every `make_host_vector` to the pageable resource. Raise `--min-time` until the spread
is below the effect being looked for; a 5 us change per allocation is inside the noise of a default
run. For attribution rather than magnitude, `nsys` with NVTX gives the clearer answer: count
`cudaStreamSynchronize` calls and total host time in the allocator before and after, which
distinguishes "the sync went away" from "the benchmark got faster for some other reason".

Thresholds worth agreeing on before running: case 1 and case 3 sites should be strictly no slower,
since nothing about them gets more expensive, and any site with a buffer above roughly a megabyte
should show a measurable improvement or the zero-init argument does not hold there. A site that only
moves inside noise is an argument for leaving it on `std::vector`, not for reworking the type.

## Suggested order of work

1. Add `host_uvector` plus factories, no call-site changes.
2. Make the allocation synchronization conditional on the resource being stream-ordered, which is a
   standalone win for every `make_host_vector*` call site, since they all land on the pageable
   `new_delete_memory_resource` at the default threshold of 0.
3. Migrate case 1 (`make_host_vector(device_span, ...)`, `make_host_vector_async`), the largest group
   and the one that drops both costs.
4. Migrate the large sized case-2 buffers, where zero-init dominates.
5. Decide on the stream-ordered-free requirement for pageable case-2 buffers before CUDA 13 batch
   copies become the default path.
