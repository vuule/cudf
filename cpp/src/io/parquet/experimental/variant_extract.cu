/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "io/utilities/block_utils.cuh"
#include "variant_path.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/detail/utilities/batched_memcpy.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/experimental/variant.hpp>
#include <cudf/io/experimental/variant_spec.hpp>
#include <cudf/lists/lists_column_device_view.cuh>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/numeric>
#include <cuda/std/array>
#include <cuda/std/cstring>
#include <cuda/std/limits>
#include <cuda/std/optional>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/stream>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <optional>
#include <string_view>
#include <vector>

namespace cudf {
namespace io::parquet::experimental {
namespace {

constexpr int variant_version_v1 = 1;

// Bytes consumed by the leading metadata byte common to every Variant value.
constexpr size_type variant_header_bytes = 1;

// Low 2 bits of a value's metadata byte: the basic type.
using basic_type = variant_basic_type;

// For a primitive value, the value_header is the physical type id of the payload.
using primitive_type = variant_primitive_type;

// The status of a VARIANT operation.
using op_status = variant_operation_status;

__device__ cuda::std::optional<uint64_t> read_uint64(device_span<uint8_t const> data,
                                                     size_type pos,
                                                     int width)
{
  if (cuda::std::cmp_greater(pos + width, data.size())) { return cuda::std::nullopt; }
  uint64_t v = 0;
  cuda::std::memcpy(&v, data.data() + pos, width);
  return v;
}

__device__ cuda::std::optional<size_type> narrow_cast(cuda::std::optional<uint64_t> value)
{
  if (!value.has_value() ||
      cuda::std::cmp_greater(value.value(), cuda::std::numeric_limits<size_type>::max())) {
    return cuda::std::nullopt;
  }
  return static_cast<size_type>(value.value());
}

__device__ basic_type decode_basic_type(uint8_t value_metadata)
{
  return static_cast<basic_type>(value_metadata & 0x03);
}

__device__ uint8_t variant_value_header(uint8_t value_metadata)
{
  return (value_metadata >> 2) & 0x3F;
}

struct object_array_header {
  int field_offset_size;  // bytes per field_offset entry
  int field_id_size;      // bytes per field_id (0 for arrays)
  int num_elements_size;  // bytes holding num_elements
};

/**
 * @brief Decode the size fields packed into an object/array value header.
 *
 * For object and array values, the 6-bit value header (bits 2..7 of the value metadata byte)
 * encodes the widths used by the rest of the value. The layout differs between the two:
 *
 *   object value_header bits:  | is_large (1) | field_id_size-1 (2) | field_offset_size-1 (2) |
 *   array  value_header bits:  |          is_large (1)             | field_offset_size-1 (2) |
 *
 * where each `*_size-1` field stores (width in bytes - 1), so the decoded width is the field + 1
 * (1..4 bytes), and `is_large` selects the width of the `num_elements` field: 4 bytes if set,
 * else 1 byte. Arrays have no field ids, so `field_id_size` is 0.
 *
 * @param value_header The 6-bit value header (see variant_value_header)
 * @param is_object True for object values, false for array values
 * @return The decoded byte widths
 */
__device__ object_array_header decode_object_array_header(uint8_t value_header, bool is_object)
{
  auto const large_bit = is_object ? 4 : 2;
  bool const is_large  = (value_header >> large_bit) & 0x01;

  return {.field_offset_size = (value_header & 0x03) + 1,
          .field_id_size     = is_object ? ((value_header >> 2) & 0x03) + 1 : 0,
          .num_elements_size = is_large ? 4 : 1};
}

/**
 * @brief Compute the total encoded byte length of a single VARIANT value.
 *
 * Every value starts with a 1-byte value metadata header (`basic_type` in bits 0..1, `value_header`
 * in bits 2..7); the bytes that follow depend on the basic type:
 *
 *   - primitive (0): header + a fixed payload keyed by the primitive type id. Binary/long_string
 *     carry a 4-byte little-endian length prefix followed by that many payload bytes.
 *   - short_string (1): header + `value_header` payload bytes (the header is the string length).
 *   - object/array (2/3): header + num_elements + field-id list + field-offset list + values; the
 *     total values-region size is read from the trailing field_offset (the "sentinel" at index
 *     num_elements). See decode_object_array_header / locate_object_field for the sub-layout.
 *
 * @param enc The encoded value bytes (must begin at the value metadata byte)
 * @return The total length in bytes of the value, or nullopt if `enc` is empty/malformed or the
 *         type id is unrecognized
 */
__device__ cuda::std::optional<uint64_t> variant_value_length(device_span<uint8_t const> enc)
{
  if (enc.size() < 1) { return cuda::std::nullopt; }
  auto const value_metadata = enc[0];
  auto const btype          = decode_basic_type(value_metadata);
  auto const value_header   = variant_value_header(value_metadata);

  if (btype == basic_type::PRIMITIVE) {
    uint64_t payload = 0;
    switch (static_cast<primitive_type>(value_header)) {
      case primitive_type::NULLVAL:
      case primitive_type::BOOLEAN_TRUE:
      case primitive_type::BOOLEAN_FALSE: break;  // no payload
      case primitive_type::INT8: payload = 1; break;
      case primitive_type::INT16: payload = 2; break;
      case primitive_type::INT32:
      case primitive_type::DATE:
      case primitive_type::FLOAT32: payload = 4; break;
      case primitive_type::INT64:
      case primitive_type::FLOAT64:
      case primitive_type::TIMESTAMP_MICROS:
      case primitive_type::TIMESTAMP_NTZ_MICROS:
      case primitive_type::TIME_NTZ_MICROS:
      case primitive_type::TIMESTAMP_NANOS:
      case primitive_type::TIMESTAMP_NTZ_NANOS: payload = 8; break;
      case primitive_type::DECIMAL4: payload = 1 + 4; break;    // scale + int32
      case primitive_type::DECIMAL8: payload = 1 + 8; break;    // scale + int64
      case primitive_type::DECIMAL16: payload = 1 + 16; break;  // scale + int128
      case primitive_type::UUID: payload = 16; break;
      case primitive_type::BINARY:
      case primitive_type::LONG_STRING: {
        constexpr int length_prefix_bytes = 4;
        auto const len = read_uint64(enc, variant_header_bytes, length_prefix_bytes);
        if (!len.has_value()) { return cuda::std::nullopt; }
        payload = length_prefix_bytes + len.value();
        break;
      }
      default: return cuda::std::nullopt;
    }
    return variant_header_bytes + payload;
  }

  if (btype == basic_type::SHORT_STRING) {
    // The value header is the payload length, following the header byte.
    return variant_header_bytes + static_cast<uint64_t>(value_header);
  }

  // Object / array: the encoded size is the header bytes (metadata byte, element count, optional
  // field-id list, and offset list)
  bool const is_object = btype == basic_type::OBJECT;
  auto const [offset_size, id_size, num_elements_size] =
    decode_object_array_header(value_header, is_object);

  auto const num_elements = read_uint64(enc, variant_header_bytes, num_elements_size);
  if (!num_elements.has_value()) { return cuda::std::nullopt; }
  auto const n = num_elements.value();

  auto const offsets_start = variant_header_bytes + num_elements_size + n * id_size;
  auto const values_base   = offsets_start + (n + 1) * offset_size;
  // Sentinel offset (entry n) holds the total size of the values region.
  auto const sentinel_pos = narrow_cast(offsets_start + n * offset_size);
  if (!sentinel_pos.has_value()) { return cuda::std::nullopt; }
  auto const sentinel = read_uint64(enc, sentinel_pos.value(), offset_size);
  if (!sentinel.has_value()) { return cuda::std::nullopt; }
  return values_base + sentinel.value();
}

/*
 * Locating fields within an object value.
 *
 * Object value layout, following the 1-byte value metadata header (basic_type=object in the low 2
 * bits; value_header in the high 6 bits, see decode_object_array_header):
 *
 *   bytes 1..:     num_elements   (num_elements_size bytes) = number of fields N
 *   next N*field_id_size bytes:        field_ids[0..N-1]   (sorted by field name)
 *   next (N+1)*field_offset_size bytes: field_offsets[0..N] (relative to values_base)
 *   remaining bytes (values_base..):   the concatenated field values
 *
 * `num_elements_size`, `field_id_size`, and `field_offset_size` come from the value header (see
 * decode_object_array_header). The trailing offset `field_offsets[N]` is the total size of the
 * values region.
 *
 * Per the spec, `field_ids[0..N-1]` are ordered by the corresponding field name
 * (lexicographically), not by the numeric id value, and the values themselves may be in any order,
 * so `field_offsets` are not necessarily monotonic -- hence the value length is taken from each
 * field's own header rather than from offset deltas.
 *
 * Because field_ids are name-ordered rather than id-ordered, a field id cannot be binary searched
 * directly. Instead, each probe turns `field_ids[mid]` into its dictionary name via an O(1) lookup
 * in `meta` -- the metadata offset table is indexed directly by id, independent of whether the
 * dictionary strings themselves are sorted -- and compares that name against the target's,
 * giving O(log N) unconditionally.
 */

// The metadata dictionary's offset table, parsed once so that field ids resolve to names in O(1).
// `status` is non-success for a malformed or unsupported blob, and every lookup on it then fails.
struct metadata_dictionary {
  device_span<uint8_t const> meta;
  size_type num_entries;
  size_type offsets_start;
  size_type strings_base;
  size_type strings_declared;  ///< Length of string_data as declared by the terminal offset
  int offset_size;
  op_status status;
};

__device__ metadata_dictionary parse_metadata_dictionary(device_span<uint8_t const> meta)
{
  auto const malformed = metadata_dictionary{{}, 0, 0, 0, 0, 0, op_status::MALFORMED_VARIANT};

  auto const meta_len = static_cast<size_type>(meta.size());
  if (meta_len < 1) { return malformed; }
  auto const header = meta[0];
  if ((header & 0x0F) != variant_version_v1) { return malformed; }
  int const offset_size = ((header >> 6) & 0x03) + 1;

  size_type pos          = 1;
  auto const num_entries = narrow_cast(read_uint64(meta, pos, offset_size));
  if (!num_entries.has_value()) { return malformed; }
  pos += offset_size;

  auto const offsets_start = pos;
  auto const offsets_bytes = (static_cast<uint64_t>(num_entries.value()) + 1) * offset_size;
  if (cuda::std::cmp_greater(offsets_bytes, meta_len - offsets_start)) { return malformed; }
  auto const strings_base   = offsets_start + static_cast<size_type>(offsets_bytes);
  auto const strings_extent = meta_len - strings_base;

  // The spec requires offsets[0] == 0, and offsets[N] is the declared length of string_data. Keys
  // are bounded by that declared length rather than by the physical remainder of the blob, so a key
  // reaching past it is rejected instead of being read out of a region the blob never claimed.
  auto const first_off = read_uint64(meta, offsets_start, offset_size);
  if (!first_off.has_value() || first_off.value() != 0) { return malformed; }
  auto const terminal_off =
    read_uint64(meta, offsets_start + num_entries.value() * offset_size, offset_size);
  if (!terminal_off.has_value() || cuda::std::cmp_greater(terminal_off.value(), strings_extent)) {
    return malformed;
  }

  return {meta,
          num_entries.value(),
          offsets_start,
          strings_base,
          static_cast<size_type>(terminal_off.value()),
          offset_size,
          op_status::SUCCESS};
}

// O(1) name lookup by field id: two offset reads into the metadata table. Returns nullopt when the
// id is outside the dictionary or its offsets are inconsistent, both of which are malformed.
__device__ cuda::std::optional<cudf::string_view> name_for_id(metadata_dictionary const& dict,
                                                              size_type field_id)
{
  if (field_id >= dict.num_entries) { return cuda::std::nullopt; }
  auto const s =
    read_uint64(dict.meta, dict.offsets_start + field_id * dict.offset_size, dict.offset_size);
  auto const e = read_uint64(
    dict.meta, dict.offsets_start + (field_id + 1) * dict.offset_size, dict.offset_size);
  if (!s.has_value() || !e.has_value()) { return cuda::std::nullopt; }
  if (e.value() < s.value() || cuda::std::cmp_greater(e.value(), dict.strings_declared)) {
    return cuda::std::nullopt;
  }
  return cudf::string_view{
    reinterpret_cast<char const*>(dict.meta.data() + dict.strings_base + s.value()),
    static_cast<size_type>(e.value() - s.value())};
}

// An object value's header, parsed once so that several of its fields can be located in one pass.
// `status` is non-success when `val` is not an object or is truncated, and every lookup on it then
// fails.
struct object_fields {
  device_span<uint8_t const> val;
  size_type num_fields;
  size_type ids_start;
  size_type offsets_start;
  size_type values_base;
  size_type values_region;  ///< Size of the values region as declared by the terminal offset
  int id_size;
  int offset_size;
  op_status status;
};

__device__ object_fields parse_object_fields(device_span<uint8_t const> val)
{
  auto const fail = [](op_status status) { return object_fields{{}, 0, 0, 0, 0, 0, 0, 0, status}; };

  auto const val_len = static_cast<size_type>(val.size());
  if (val_len < 1) { return fail(op_status::MALFORMED_VARIANT); }
  auto const value_metadata = val[0];
  if (decode_basic_type(value_metadata) != basic_type::OBJECT) {
    return fail(op_status::MISSING_PATH);
  }

  auto const [offset_size, id_size, num_elements_size] =
    decode_object_array_header(variant_value_header(value_metadata), true);

  size_type pos         = 1;
  auto const num_fields = narrow_cast(read_uint64(val, pos, num_elements_size));
  if (!num_fields.has_value()) { return fail(op_status::MALFORMED_VARIANT); }
  pos += num_elements_size;

  auto const ids_start = pos;
  auto const ids_bytes = static_cast<uint64_t>(num_fields.value()) * id_size;
  if (cuda::std::cmp_greater(ids_bytes, val_len - ids_start)) {
    return fail(op_status::MALFORMED_VARIANT);
  }

  auto const offsets_start = ids_start + static_cast<size_type>(ids_bytes);
  auto const offsets_bytes = (static_cast<uint64_t>(num_fields.value()) + 1) * offset_size;
  if (cuda::std::cmp_greater(offsets_bytes, val_len - offsets_start)) {
    return fail(op_status::MALFORMED_VARIANT);
  }

  auto const values_base   = offsets_start + static_cast<size_type>(offsets_bytes);
  auto const values_extent = val_len - values_base;

  // The terminal offset (entry num_fields) is the authoritative end of the values region. Using
  // the physical remainder instead would let a malformed object reference bytes past it.
  auto const terminal_off =
    read_uint64(val, offsets_start + num_fields.value() * offset_size, offset_size);
  if (!terminal_off.has_value() || cuda::std::cmp_greater(terminal_off.value(), values_extent)) {
    return fail(op_status::MALFORMED_VARIANT);
  }

  return {val,
          num_fields.value(),
          ids_start,
          offsets_start,
          values_base,
          static_cast<size_type>(terminal_off.value()),
          id_size,
          offset_size,
          op_status::SUCCESS};
}

// The name of the field at position `index` of the object, via its field id.
__device__ cuda::std::optional<cudf::string_view> field_name_at(object_fields const& obj,
                                                                metadata_dictionary const& dict,
                                                                size_type index)
{
  auto const id =
    narrow_cast(read_uint64(obj.val, obj.ids_start + index * obj.id_size, obj.id_size));
  if (!id.has_value()) { return cuda::std::nullopt; }
  return name_for_id(dict, id.value());
}

// The encoded bytes of the field at position `index`. Field offsets are not monotonic, so the
// length comes from the value's own header rather than from an offset delta. A failure here is a
// malformed blob rather than an absent field.
__device__ cuda::std::pair<device_span<uint8_t const>, op_status> field_value_at(
  object_fields const& obj, size_type index)
{
  auto const start =
    read_uint64(obj.val, obj.offsets_start + index * obj.offset_size, obj.offset_size);
  if (!start.has_value() || cuda::std::cmp_greater(start.value(), obj.values_region)) {
    return {{}, op_status::MALFORMED_VARIANT};
  }
  auto const value_len = variant_value_length(obj.val.subspan(obj.values_base + start.value()));
  if (!value_len.has_value()) { return {{}, op_status::MALFORMED_VARIANT}; }
  if (cuda::std::cmp_greater(start.value() + value_len.value(), obj.values_region)) {
    return {{}, op_status::MALFORMED_VARIANT};
  }
  return {obj.val.subspan(obj.values_base + start.value(), value_len.value()), op_status::SUCCESS};
}

// Where a search for one key landed among the object's name-ordered fields: `index` is the position
// of the match, or the lower bound of the key when `found` is false. A non-success `status` is a
// malformed blob, which invalidates the whole object rather than just this key.
struct field_search {
  size_type index;
  bool found;
  op_status status;
};

// PROTOTYPE SCAFFOLDING: flip to false to measure against cudf::string_view::compare.
constexpr bool use_word_compare = true;

// Eight bytes packed big-endian, so that comparing the words as unsigned integers orders the bytes
// exactly as an unsigned byte-by-byte comparison would.
__device__ uint64_t pack_eight(uint8_t const* p)
{
  uint64_t w = 0;
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    w = (w << 8) | p[k];
  }
  return w;
}

// Lexicographic byte comparison, taking eight bytes at a time. Dictionary strings sit at arbitrary
// offsets inside the metadata blob, so the bytes still have to be loaded one at a time; what this
// buys over `cudf::string_view::compare` is that a block's eight loads are independent of each
// other and resolve in one comparison, rather than forming a serial chain of load, compare and
// branch per byte. Keys sharing a prefix are the case that pays.
__device__ int compare_keys(cudf::string_view lhs, cudf::string_view rhs)
{
  if constexpr (!use_word_compare) { return lhs.compare(rhs); }

  auto const* a      = reinterpret_cast<uint8_t const*>(lhs.data());
  auto const* b      = reinterpret_cast<uint8_t const*>(rhs.data());
  auto const overlap = cuda::std::min(lhs.size_bytes(), rhs.size_bytes());

  size_type i = 0;
  for (; i + 8 <= overlap; i += 8) {
    auto const wa = pack_eight(a + i);
    auto const wb = pack_eight(b + i);
    if (wa != wb) { return wa < wb ? -1 : 1; }
  }
  for (; i < overlap; ++i) {
    if (a[i] != b[i]) { return a[i] < b[i] ? -1 : 1; }
  }
  return lhs.size_bytes() - rhs.size_bytes();
}

// Search `[begin, num_fields)` with `compare_at`, which orders the field at a position against the
// caller's target and returns nullopt when the entry cannot be read. `peek` first tests position
// `begin`, which resolves the key in one probe when the caller is walking keys that are dense in
// the object.
template <typename CompareAt>
__device__ field_search
find_field_with(object_fields const& obj, size_type begin, bool peek, CompareAt compare_at)
{
  size_type lo = begin;
  size_type hi = obj.num_fields;

  if (peek && lo < hi) {
    auto const cmp = compare_at(lo);
    if (!cmp.has_value()) { return {lo, false, op_status::MALFORMED_VARIANT}; }
    if (cmp.value() == 0) { return {lo, true, op_status::SUCCESS}; }
    // The fields are name-ordered, so a greater name here puts the target before this position.
    if (cmp.value() > 0) { return {lo, false, op_status::SUCCESS}; }
    ++lo;
  }

  // Not using thrust::lower_bound since it does not propagate entry read failures.
  while (lo < hi) {
    size_type const mid = lo + (hi - lo) / 2;
    auto const cmp      = compare_at(mid);
    if (!cmp.has_value()) { return {mid, false, op_status::MALFORMED_VARIANT}; }
    if (cmp.value() == 0) { return {mid, true, op_status::SUCCESS}; }
    if (cmp.value() < 0) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return {lo, false, op_status::SUCCESS};
}

__device__ field_search find_field(object_fields const& obj,
                                   metadata_dictionary const& dict,
                                   cudf::string_view target,
                                   size_type begin,
                                   bool peek)
{
  return find_field_with(obj, begin, peek, [&](size_type index) -> cuda::std::optional<int> {
    auto const name = field_name_at(obj, dict, index);
    if (!name.has_value()) { return cuda::std::nullopt; }
    return compare_keys(name.value(), target);
  });
}

// PROTOTYPE SCAFFOLDING (shared dictionary), through to the end of locate_object_fields_ranked.
//
// Every row's dictionary resolved once up front, so that a probe compares two small integers
// instead of reading and comparing name bytes out of the row's metadata blob.
//
// A field id cannot be compared directly: an object's ids are ordered by *name*, and a dictionary
// need not be sorted. `rank[id]` is the position id's name takes in sorted order, which is the
// order the object's fields are in, so comparing ranks orders fields exactly as comparing names
// would. `step_rank[step]` is the rank each step token resolves to, or -1 when the dictionary does
// not contain it -- then no field can match, and the search answers "absent" without a single name
// byte.
struct shared_dictionary_view {
  device_span<size_type const> rank;       ///< rank[id], one entry per dictionary entry
  device_span<size_type const> step_rank;  ///< Rank of each step token, -1 if absent
};

__device__ field_search find_field_by_rank(object_fields const& obj,
                                           shared_dictionary_view const& shared,
                                           size_type target_rank,
                                           size_type begin,
                                           bool peek)
{
  auto const num_entries = static_cast<size_type>(shared.rank.size());
  return find_field_with(obj, begin, peek, [&](size_type index) -> cuda::std::optional<int> {
    auto const id =
      narrow_cast(read_uint64(obj.val, obj.ids_start + index * obj.id_size, obj.id_size));
    // An id outside the dictionary is malformed, exactly as a failed name lookup would be.
    if (!id.has_value() || id.value() >= num_entries) { return cuda::std::nullopt; }
    auto const rank = shared.rank[id.value()];
    return rank == target_rank ? 0 : (rank < target_rank ? -1 : 1);
  });
}

// The encoded bytes of the field named `target_name`, or an empty span and the reason it failed if
// `val` is not an object, the field is absent, or either blob is malformed.
__device__ cuda::std::pair<device_span<uint8_t const>, op_status> locate_object_field(
  device_span<uint8_t const> meta, device_span<uint8_t const> val, cudf::string_view target_name)
{
  auto const dict = parse_metadata_dictionary(meta);
  if (dict.status != op_status::SUCCESS) { return {{}, dict.status}; }
  auto const obj = parse_object_fields(val);
  if (obj.status != op_status::SUCCESS) { return {{}, obj.status}; }

  auto const match = find_field(obj, dict, target_name, 0, false);
  if (match.status != op_status::SUCCESS) { return {{}, match.status}; }
  if (!match.found) { return {{}, op_status::MISSING_PATH}; }
  return field_value_at(obj, match.index);
}

/**
 * @brief Locate several fields of one object in a single pass over its field ids.
 *
 * The object's field ids are ordered by name, so a sorted list of keys can be resolved in one
 * forward sweep: each key is searched only in the fields left after the previous key's match. Two
 * probes are saved on top of that, since the metadata and object headers are parsed once for the
 * whole group rather than once per key.
 *
 * Where the keys are dense in what remains of the object -- the case worth batching, e.g. asking
 * for most of its fields -- the next field is likely the next key, and testing it costs one probe
 * against the `log2(num_fields)` of a search. Where they are sparse, that test is wasted, so it is
 * only made when the remaining keys can be expected to land within a few fields of each other.
 *
 * Keys must be sorted ascending by name, as `variant_path_trie` guarantees for a group's keys.
 *
 * @param emit Receives each key's index, the encoded bytes of its field (empty unless the status is
 *        success), and the status of that key's lookup
 */
template <typename Emit>
__device__ void locate_object_fields(device_span<uint8_t const> meta,
                                     device_span<uint8_t const> val,
                                     column_device_view steps,
                                     device_span<size_type const> key_steps,
                                     Emit emit)
{
  auto const num_keys = static_cast<size_type>(key_steps.size());
  auto const dict     = parse_metadata_dictionary(meta);
  auto const obj      = parse_object_fields(val);

  // A malformed blob fails the object as a whole, as does one that is not an object at all; either
  // way every key of the group carries that one reason.
  auto group_status = dict.status != op_status::SUCCESS ? dict.status : obj.status;

  size_type key = 0;
  if (group_status == op_status::SUCCESS) {
    size_type at = 0;
    for (; key < num_keys; ++key) {
      auto const target = steps.element<cudf::string_view>(key_steps[key]);
      // Four fields per key is where a peek that misses still costs less than the search it
      // replaces two keys out of three.
      bool const dense = (obj.num_fields - at) <= 4 * (num_keys - key);
      auto const match = find_field(obj, dict, target, at, dense);
      if (match.status != op_status::SUCCESS) {
        group_status = match.status;
        break;
      }
      if (!match.found) {
        emit(key, device_span<uint8_t const>{}, op_status::MISSING_PATH);
        at = match.index;
        continue;
      }
      auto const [field, status] = field_value_at(obj, match.index);
      emit(key, field, status);
      at = match.index + 1;
    }
  }
  for (; key < num_keys; ++key) {
    emit(key, device_span<uint8_t const>{}, group_status);
  }
}

// locate_object_field against a pre-resolved dictionary: the metadata blob is never read, so a
// missing rank settles the lookup without touching the object at all.
__device__ cuda::std::pair<device_span<uint8_t const>, op_status> locate_object_field_ranked(
  shared_dictionary_view const& shared, device_span<uint8_t const> val, size_type step)
{
  auto const obj = parse_object_fields(val);
  if (obj.status != op_status::SUCCESS) { return {{}, obj.status}; }

  auto const match = find_field_by_rank(obj, shared, shared.step_rank[step], 0, false);
  if (match.status != op_status::SUCCESS) { return {{}, match.status}; }
  if (!match.found) { return {{}, op_status::MISSING_PATH}; }
  return field_value_at(obj, match.index);
}

// locate_object_fields against a pre-resolved dictionary. Same sweep, ranks instead of names, and
// no metadata parse for the group.
template <typename Emit>
__device__ void locate_object_fields_ranked(shared_dictionary_view const& shared,
                                            device_span<uint8_t const> val,
                                            device_span<size_type const> key_steps,
                                            Emit emit)
{
  auto const num_keys = static_cast<size_type>(key_steps.size());
  auto const obj      = parse_object_fields(val);

  auto group_status = obj.status;

  size_type key = 0;
  if (group_status == op_status::SUCCESS) {
    size_type at = 0;
    for (; key < num_keys; ++key) {
      bool const dense = (obj.num_fields - at) <= 4 * (num_keys - key);
      auto const match =
        find_field_by_rank(obj, shared, shared.step_rank[key_steps[key]], at, dense);
      if (match.status != op_status::SUCCESS) {
        group_status = match.status;
        break;
      }
      if (!match.found) {
        emit(key, device_span<uint8_t const>{}, op_status::MISSING_PATH);
        at = match.index;
        continue;
      }
      auto const [field, status] = field_value_at(obj, match.index);
      emit(key, field, status);
      at = match.index + 1;
    }
  }
  for (; key < num_keys; ++key) {
    emit(key, device_span<uint8_t const>{}, group_status);
  }
}

// Parse an array value header and return the sub-span of the element at `index` (0-based) within
// `val`. Returns an empty span if `val` is not an array (`basic_type != array`), if `index` is out
// of bounds, or if the encoded data is truncated.
//
// Array layout per the Variant spec:
//   byte 0: header (basic_type=array in low 2 bits; value_header in high 6 bits)
//     value_header bits: (offset_size - 1) in bits 0-1, is_large in bit 2, bits 3-5 unused
//   num_elements: 1 byte if !is_large else 4 bytes (little-endian)
//   offsets:      (num_elements + 1) entries, each `offset_size` bytes, relative to the end of
//                 offsets
//   values:       concatenated element blobs
//
// Array element offsets are monotonically increasing, so the element length is taken directly from
// the offset delta (o1 - o0) rather than from the element's own header.
__device__ cuda::std::pair<device_span<uint8_t const>, op_status> locate_array_element(
  device_span<uint8_t const> value, size_type index)
{
  if (index < 0) { return {{}, op_status::MISSING_PATH}; }

  auto const value_size = static_cast<size_type>(value.size());
  if (value_size < 1) { return {{}, op_status::MALFORMED_VARIANT}; }
  uint8_t const value_metadata = value[0];
  if (decode_basic_type(value_metadata) != basic_type::ARRAY) {
    return {{}, op_status::MISSING_PATH};
  }

  int const value_header = variant_value_header(value_metadata);
  [[maybe_unused]] auto const [offset_size, _, num_elements_size] =
    decode_object_array_header(value_header, false);

  size_type position            = 1;
  auto const num_elements_value = narrow_cast(read_uint64(value, position, num_elements_size));
  if (!num_elements_value.has_value()) { return {{}, op_status::MALFORMED_VARIANT}; }
  auto const num_elements = num_elements_value.value();
  if (index >= num_elements) { return {{}, op_status::MISSING_PATH}; }
  position += num_elements_size;

  size_type const offsets_start = position;

  // Computed in 64-bit because (num_elements + 1) * offset_size can exceed the signed `size_type`
  // range (which would be UB); the check below then rejects any array that overruns the value blob.
  auto const offsets_bytes = (static_cast<uint64_t>(num_elements) + 1) * offset_size;
  if (cuda::std::cmp_greater(offsets_bytes, value_size - offsets_start)) {
    return {{}, op_status::MALFORMED_VARIANT};
  }
  size_type const values_base = offsets_start + static_cast<size_type>(offsets_bytes);
  auto const values_extent    = value_size - values_base;
  // Read the terminal offset offsets[num_elements]; it is the spec-declared bound on the
  // values region and must be used instead of the physical extent so that an element whose
  // offset escapes the declared boundary is caught as malformed even when physical bytes
  // are present beyond it.
  auto const terminal_off_pos = offsets_start + static_cast<uint64_t>(num_elements) * offset_size;
  auto const terminal_off     = read_uint64(value, terminal_off_pos, offset_size);
  if (!terminal_off.has_value() || cuda::std::cmp_greater(*terminal_off, values_extent)) {
    return {{}, op_status::MALFORMED_VARIANT};
  }
  // The spec requires offsets[0] == 0; a nonzero first offset silently skips leading
  // value bytes and can return a plausible result from a malformed array.
  auto const first_off = read_uint64(value, offsets_start, offset_size);
  if (!first_off.has_value() || *first_off != 0) { return {{}, op_status::MALFORMED_VARIANT}; }

  auto const start_offset_pos = offsets_start + static_cast<uint64_t>(index) * offset_size;
  auto const end_offset_pos   = offsets_start + (static_cast<uint64_t>(index) + 1) * offset_size;
  if (cuda::std::cmp_greater(end_offset_pos + offset_size, value_size)) {
    return {{}, op_status::MALFORMED_VARIANT};
  }
  auto const start_offset = read_uint64(value, start_offset_pos, offset_size);
  auto const end_offset   = read_uint64(value, end_offset_pos, offset_size);
  if (!start_offset.has_value() || !end_offset.has_value()) {
    return {{}, op_status::MALFORMED_VARIANT};
  }
  auto const element_start = *start_offset;
  auto const element_end   = *end_offset;
  if (element_end < element_start || cuda::std::cmp_greater(element_end, *terminal_off)) {
    return {{}, op_status::MALFORMED_VARIANT};
  }
  return {value.subspan(values_base + element_start, element_end - element_start),
          op_status::SUCCESS};
}

__device__ bool is_variant_null(device_span<uint8_t const> enc)
{
  if (enc.empty()) { return false; }
  auto const vm = enc[0];
  return decode_basic_type(vm) == basic_type::PRIMITIVE &&
         variant_value_header(vm) == static_cast<uint8_t>(primitive_type::NULLVAL);
}

// The fixed-width signed integers a VARIANT value can be cast to: INT{8,16,32,64}.  Matches the
// exact width types (not e.g. __int128) since those are the only variant primitive int headers.
template <typename T>
constexpr bool is_variant_int =
  cudf::is_integral_not_bool<T>() && cudf::is_signed<T>() && !cuda::std::is_same_v<T, __int128_t>;

// The fixed-width primitive types (signed integers and floats) a VARIANT value can be decoded into.
template <typename T>
constexpr bool is_variant_numerical = is_variant_int<T> || cudf::is_floating_point<T>();

// The output types a VARIANT value can be cast to: the fixed-width signed integers, floats, bool,
// and strings.
template <typename T>
constexpr bool is_variant_castable = is_variant_numerical<T> || cuda::std::is_same_v<T, bool> ||
                                     cuda::std::is_same_v<T, cudf::string_view>;

// Maps a fixed-width output type to the VARIANT primitive type header id that encodes it.
template <typename T>
  requires(is_variant_numerical<T>)
__device__ constexpr primitive_type primitive_type_for()
{
  if constexpr (cuda::std::is_same_v<T, int8_t>) {
    return primitive_type::INT8;
  } else if constexpr (cuda::std::is_same_v<T, int16_t>) {
    return primitive_type::INT16;
  } else if constexpr (cuda::std::is_same_v<T, int32_t>) {
    return primitive_type::INT32;
  } else if constexpr (cuda::std::is_same_v<T, int64_t>) {
    return primitive_type::INT64;
  } else if constexpr (cuda::std::is_same_v<T, float>) {
    return primitive_type::FLOAT32;
  } else if constexpr (cuda::std::is_same_v<T, double>) {
    return primitive_type::FLOAT64;
  } else {
    CUDF_UNREACHABLE("primitive_type_for: T is not a supported variant primitive type");
    return primitive_type::NULLVAL;
  }
}

/**
 * @brief Decode a single VARIANT value blob into a fixed-width primitive of type `T`.
 *
 * Requires `basic_type == primitive` and a value header whose physical type id matches `T` exactly.
 */
template <typename T>
__device__ inline cuda::std::optional<T> decode_primitive(device_span<uint8_t const> enc)
{
  if (cuda::std::cmp_less(enc.size(), 1 + sizeof(T))) { return cuda::std::nullopt; }

  uint8_t const value_metadata = enc[0];
  if (decode_basic_type(value_metadata) != basic_type::PRIMITIVE ||
      variant_value_header(value_metadata) != static_cast<uint8_t>(primitive_type_for<T>())) {
    return cuda::std::nullopt;
  }
  return cudf::io::unaligned_load<T>(enc.data() + 1);
}

/**
 * @brief Decode a single VARIANT value blob into a bool.
 *
 * Boolean values carry no payload: the distinction between true and false is encoded entirely in
 * the primitive type header (`boolean_true` vs `boolean_false`).
 */
__device__ inline cuda::std::optional<bool> decode_bool(device_span<uint8_t const> enc)
{
  if (enc.empty()) { return cuda::std::nullopt; }
  uint8_t const value_metadata = enc[0];
  if (decode_basic_type(value_metadata) != basic_type::PRIMITIVE) { return cuda::std::nullopt; }
  auto const value_header = variant_value_header(value_metadata);
  if (value_header == static_cast<uint8_t>(primitive_type::BOOLEAN_TRUE)) { return true; }
  if (value_header == static_cast<uint8_t>(primitive_type::BOOLEAN_FALSE)) { return false; }
  return cuda::std::nullopt;
}

// Parse an array-index step token of the form "[<N>]" into its zero-based index. Returns nullopt
// for any malformed token or an index that does not fit in `size_type` (such an index is out of
// range for any array, so the caller treats it as a missing element).
__device__ cuda::std::optional<size_type> parse_index_step(cudf::string_view step)
{
  auto const step_size  = step.size_bytes();
  auto const* step_data = step.data();
  if (step_size < 3 || step_data[0] != '[' || step_data[step_size - 1] != ']') {
    return cuda::std::nullopt;
  }

  // Accumulate directly in `size_type`; the checked-arithmetic helpers reject the token if the
  // running value overflows, which means the index is out of range for any array and the caller
  // treats it as a missing element.
  size_type index = 0;
  for (size_type k = 1; k < step_size - 1; ++k) {
    char const c = step_data[k];
    if (c < '0' || c > '9') { return cuda::std::nullopt; }
    if (cuda::mul_overflow(index, index, size_type{10}) ||
        cuda::add_overflow(index, index, static_cast<size_type>(c - '0'))) {
      return cuda::std::nullopt;
    }
  }
  return index;
}

// Walk the steps `path[step_begin : step_end]` level by level starting at `val` and return the span
// of the final value (a subspan of `val`) with the status of the walk. Returns an empty span and
// the reason on failure.
//
// The last step of the range is treated as terminal: a VARIANT null there is reported as
// `VARIANT_NULL` with its bytes, while one reached mid-range is `MISSING_PATH`. A caller that
// resolves a path in several ranges keeps that distinction, since continuing from a VARIANT null
// fails the next step as a non-object or non-array, which is `MISSING_PATH` too.
//
// Each path step is encoded in the `path` strings column as either:
//   - "<name>"  -> descend into an object by dictionary key, or
//   - "[<N>]"   -> descend into an array by zero-based integer index.
// The step kind is inferred from the first byte (`'['` means index).
// `SharedDict` resolves name steps by rank against `shared` instead of by name against `meta`.
template <bool SharedDict = false>
__device__ cuda::std::pair<device_span<uint8_t const>, op_status> resolve_steps(
  device_span<uint8_t const> meta,
  device_span<uint8_t const> val,
  column_device_view path,
  size_type step_begin,
  size_type step_end,
  shared_dictionary_view shared = {})
{
  // An empty starting value cannot resolve anything, and checking up front keeps a failed shared
  // prefix from paying for a metadata lookup once per path below it.
  if (val.empty()) { return {{}, op_status::MALFORMED_VARIANT}; }

  device_span<uint8_t const> sub_val = val;
  for (size_type i = step_begin; i < step_end; ++i) {
    auto const step = path.element<cudf::string_view>(i);
    if (step.size_bytes() >= 1 && step.data()[0] == '[') {
      auto const index = parse_index_step(step);
      if (!index.has_value()) { return {{}, op_status::MISSING_PATH}; }
      auto const [span, st] = locate_array_element(sub_val, index.value());
      if (st != op_status::SUCCESS) { return {{}, st}; }
      sub_val = span;
    } else {
      auto const [span, st] = [&] {
        if constexpr (SharedDict) {
          return locate_object_field_ranked(shared, sub_val, i);
        } else {
          return locate_object_field(meta, sub_val, step);
        }
      }();
      if (st != op_status::SUCCESS) { return {{}, st}; }
      sub_val = span;
    }

    // VARIANT null before the end of the path is missing_path per spec.
    if (i + 1 < step_end && is_variant_null(sub_val)) { return {{}, op_status::MISSING_PATH}; }
    // A zero-length resolved value is not decodable; the value-only path drops the row.
    if (sub_val.empty()) { return {{}, op_status::MALFORMED_VARIANT}; }
  }

  // Terminal VARIANT null: return the bytes with variant_null status.
  if (is_variant_null(sub_val)) { return {sub_val, op_status::VARIANT_NULL}; }
  return {sub_val, op_status::SUCCESS};
}

// Walk every step of `path` starting at `val`.
__device__ cuda::std::pair<device_span<uint8_t const>, op_status> resolve_path(
  device_span<uint8_t const> meta, device_span<uint8_t const> val, column_device_view path)
{
  return resolve_steps(meta, val, path, 0, path.size());
}

__device__ cuda::std::optional<device_span<uint8_t const>> decode_string(
  device_span<uint8_t const> enc)
{
  auto const len = enc.size();
  if (len < 1) { return cuda::std::nullopt; }
  uint8_t const value_metadata = enc[0];
  auto const btype             = decode_basic_type(value_metadata);
  auto const value_header      = variant_value_header(value_metadata);

  if (btype == basic_type::SHORT_STRING) {
    // Short string: value_header = length
    std::size_t const str_len = value_header;
    if (1 + str_len > len) { return cuda::std::nullopt; }
    return enc.subspan(1, str_len);
  }
  if (btype == basic_type::PRIMITIVE &&
      value_header == static_cast<uint8_t>(primitive_type::LONG_STRING)) {
    // Long string: 1-byte header + 4-byte LE length + char bytes
    constexpr std::size_t long_string_prefix_bytes = 1 + sizeof(uint32_t);
    if (len < long_string_prefix_bytes) { return cuda::std::nullopt; }
    auto const str_len = cudf::io::unaligned_load<uint32_t>(enc.data() + 1);
    // Encoded length claims more char bytes than the buffer holds: truncated/malformed blob
    if (long_string_prefix_bytes + str_len > len) { return cuda::std::nullopt; }
    return enc.subspan(long_string_prefix_bytes, str_len);
  }
  return cuda::std::nullopt;
}

__device__ device_span<uint8_t const> list_row_span(cudf::lists_column_device_view const& col,
                                                    size_type row)
{
  auto const begin = col.offset_at(row);
  auto const end   = col.offset_at(row + 1);
  return {col.child().data<uint8_t>() + begin, static_cast<std::size_t>(end - begin)};
}

__device__ cuda::std::pair<device_span<uint8_t const>, device_span<uint8_t const>>
metadata_and_value_at(cudf::lists_column_device_view const& metadata,
                      cudf::lists_column_device_view const& values,
                      size_type row)
{
  return {list_row_span(metadata, row), list_row_span(values, row)};
}

constexpr int block_size = 256;

/**
 * @brief Resolves `path` in each VARIANT row and record the located field's size and source offset.
 *
 * For each non-null row, walks `path` to the target value and writes its byte length to
 * `d_sizes[row]` and its offset within the row's value blob to `d_src_offsets[row]`. Rows that are
 * null, or whose path does not resolve, are marked null in `d_null_mask` with a size of 0.
 */
CUDF_KERNEL __launch_bounds__(block_size) void locate_variant_fields_kernel(
  cudf::lists_column_device_view metadata,
  cudf::lists_column_device_view values,
  column_device_view path,
  device_span<size_type> d_sizes,
  device_span<size_type> d_src_offsets,
  bitmask_type* d_null_mask,
  device_span<op_status> d_status)  // empty when no status was requested
{
  auto const num_rows = static_cast<size_type>(d_sizes.size());
  auto const tid      = cudf::detail::grid_1d::global_thread_id<block_size>();
  auto const stride   = cudf::detail::grid_1d::grid_stride<block_size>();

  for (auto row = tid; row < num_rows; row += stride) {
    if (!cudf::bit_is_set(d_null_mask, row)) {
      d_sizes[row]       = 0;
      d_src_offsets[row] = 0;
      if (!d_status.empty()) { d_status[row] = op_status::ROW_NULL; }
      continue;
    }

    auto const [meta, val] = metadata_and_value_at(metadata, values, row);
    auto const [field, st] = resolve_path(meta, val, path);

    if (!d_status.empty()) { d_status[row] = st; }

    if (field.empty()) {
      d_sizes[row]       = 0;
      d_src_offsets[row] = 0;
      cudf::clear_bit(d_null_mask, row);
    } else {
      d_sizes[row]       = static_cast<size_type>(field.size());
      d_src_offsets[row] = static_cast<size_type>(field.data() - val.data());
    }
  }
}

// Returns true for every primitive_type ID that variant_value_length maps to a known payload
// size in its `basic_type::PRIMITIVE` switch, i.e. every ID other than its `default` case.
__device__ bool is_recognized_primitive_type(primitive_type ptype)
{
  switch (ptype) {
    case primitive_type::NULLVAL:
    case primitive_type::BOOLEAN_TRUE:
    case primitive_type::BOOLEAN_FALSE:
    case primitive_type::INT8:
    case primitive_type::INT16:
    case primitive_type::INT32:
    case primitive_type::INT64:
    case primitive_type::FLOAT64:
    case primitive_type::DECIMAL4:
    case primitive_type::DECIMAL8:
    case primitive_type::DECIMAL16:
    case primitive_type::DATE:
    case primitive_type::TIMESTAMP_MICROS:
    case primitive_type::TIMESTAMP_NTZ_MICROS:
    case primitive_type::FLOAT32:
    case primitive_type::BINARY:
    case primitive_type::LONG_STRING:
    case primitive_type::TIME_NTZ_MICROS:
    case primitive_type::TIMESTAMP_NANOS:
    case primitive_type::TIMESTAMP_NTZ_NANOS:
    case primitive_type::UUID: return true;
    default: return false;
  }
}

/**
 * @brief Status helper for fixed-width primitive targets: classifies why `decode_primitive<T>`
 * failed to decode `val`, per `variant_operation_status` semantics.
 */
template <typename T>
  requires(is_variant_numerical<T>)
__device__ op_status cast_status_for_primitive(device_span<uint8_t const> val)
{
  if (val.empty()) { return op_status::MALFORMED_VARIANT; }
  if (is_variant_null(val)) { return op_status::VARIANT_NULL; }
  if (decode_primitive<T>(val).has_value()) { return op_status::SUCCESS; }
  if (decode_basic_type(val[0]) != basic_type::PRIMITIVE) { return op_status::TYPE_MISMATCH; }
  auto const ptype = static_cast<primitive_type>(variant_value_header(val[0]));
  if (ptype == primitive_type_for<T>()) { return op_status::MALFORMED_VARIANT; }
  return is_recognized_primitive_type(ptype) ? op_status::TYPE_MISMATCH
                                             : op_status::MALFORMED_VARIANT;
}

/**
 * @brief Where one trie slot's value was found within a row's value blob, and how the walk to it
 * ended.
 *
 * Validity lives entirely in `size`: a slot whose steps did not resolve for this row has
 * `invalid_slot_size`, and resolving anything below it fails immediately because its span is empty.
 * `status` is why, and a slot below an unresolved one reports the same reason rather than one made
 * up from probing an empty value. A resolved VARIANT null is valid and carries `VARIANT_NULL`.
 */
struct slot_result {
  size_type src_offset;
  size_type size;
  op_status status;
};

constexpr size_type invalid_slot_size = -1;

__device__ slot_result make_slot_result(device_span<uint8_t const> field,
                                        uint8_t const* val_base,
                                        op_status status)
{
  if (field.empty()) { return {0, invalid_slot_size, status}; }
  return {
    static_cast<size_type>(field.data() - val_base), static_cast<size_type>(field.size()), status};
}

__device__ bool slot_is_valid(slot_result const& result)
{
  return result.size != invalid_slot_size;
}

// The span a slot located, or an empty span if it did not resolve.
__device__ device_span<uint8_t const> slot_span(device_span<uint8_t const> val,
                                                slot_result const& result)
{
  if (!slot_is_valid(result)) { return {}; }
  return val.subspan(result.src_offset, result.size);
}

// Walks needing at most this many located values keep them in a per-thread array the compiler can
// place in registers or local memory; a larger one uses a global scratch allocation instead. One
// value per depth is enough without sibling merging, but merging holds a group's resolved children
// while the walk is inside any of their subtrees.
constexpr size_type max_local_walk_state = 128;

// Global scratch is allocated per thread, so the grid has to be capped for the allocation to stay
// independent of the row count. This many blocks still saturates the walk.
constexpr int max_global_scratch_blocks = 256;

// The trie arrays a walk reads, gathered to keep the kernel's parameter list manageable.
struct trie_device_view {
  column_device_view steps;
  device_span<size_type const> slot_steps;
  device_span<size_type const> slot_depth;
  device_span<size_type const> output_offsets;
  device_span<size_type const> output_paths;
  device_span<size_type const> slot_group;
  device_span<size_type const> slot_group_pos;
  device_span<size_type const> group_keys;
  device_span<size_type const> group_key_step;
  device_span<size_type const> group_first;
  device_span<size_type const> group_parent;
  device_span<size_type const> state_base;
};

/**
 * @brief Resolves a whole trie of VARIANT paths in each row, recording each path's result.
 *
 * Slots are visited in index order, which is depth-first pre-order, so a slot's parent is always
 * still in the walk's state from when it descended. A shared prefix is therefore resolved once per
 * row and reused by every path below it, and a prefix that fails leaves an empty span behind that
 * makes its whole subtree fail at its first step.
 *
 * Without `MergeSiblings`, each slot's first step is searched for on its own and the walk only has
 * to remember one located value per depth. With it, the first steps of all the slots sharing a
 * parent are resolved together in one pass over that object (see locate_object_fields), which
 * trades holding a group's resolved children for the whole of their subtrees against searching that
 * object once per key instead of once per group.
 *
 * For each path `p` and row, the located field's byte length is written to
 * `d_sizes[p * num_rows + row]` and its offset within the row's value blob to `d_src_offsets`, so
 * each path's outputs are contiguous. Rows that are null in `d_row_valid`, or whose path does not
 * resolve, get a size of 0 and are marked null in that path's mask in `d_null_masks`. When
 * `d_statuses` is not empty it holds one buffer per path, each receiving that path's per-row
 * status.
 *
 * @tparam UseLocalScratch Keep the per-thread located values in local memory rather than
 * `d_scratch`
 * @tparam MergeSiblings Resolve the first steps of sibling slots in one pass per object
 * @tparam SharedDict Probe by rank against a dictionary resolved once for every row, rather than by
 * name against each row's own metadata blob
 */
template <bool UseLocalScratch, bool MergeSiblings, bool SharedDict = false>
CUDF_KERNEL __launch_bounds__(block_size) void locate_variant_field_trie_kernel(
  cudf::lists_column_device_view metadata,
  cudf::lists_column_device_view values,
  trie_device_view trie,
  bitmask_type const* d_row_valid,
  size_type num_rows,
  size_type walk_state_size,
  device_span<size_type> d_sizes,
  device_span<size_type> d_src_offsets,
  device_span<bitmask_type* const> d_null_masks,
  device_span<op_status* const> d_statuses,  // empty when no status was requested
  device_span<slot_result> d_scratch,
  shared_dictionary_view shared = {})
{
  auto const num_slots = static_cast<size_type>(trie.slot_depth.size());
  auto const num_paths = static_cast<size_type>(d_null_masks.size());
  auto const tid       = cudf::detail::grid_1d::global_thread_id<block_size>();
  auto const stride    = cudf::detail::grid_1d::grid_stride<block_size>();

  [[maybe_unused]] cuda::std::array<slot_result, UseLocalScratch ? max_local_walk_state : 1> local;
  auto* const located = [&]() -> slot_result* {
    if constexpr (UseLocalScratch) {
      return local.data();
    } else {
      return d_scratch.data() + tid * walk_state_size;
    }
  }();

  // Where a slot's located value lives: one entry per depth, or one per live group member when
  // sibling merging needs a whole group's children at once.
  auto const state_of = [&](size_type slot, size_type depth) {
    if constexpr (MergeSiblings) {
      return trie.state_base[depth] + trie.slot_group_pos[slot];
    } else {
      return depth;
    }
  };

  for (auto row = tid; row < num_rows; row += stride) {
    bool const row_valid = d_row_valid == nullptr || cudf::bit_is_set(d_row_valid, row);

    if (!row_valid) {
      for (size_type path = 0; path < num_paths; ++path) {
        auto const out     = path * num_rows + static_cast<size_type>(row);
        d_sizes[out]       = 0;
        d_src_offsets[out] = 0;
        cudf::clear_bit(d_null_masks[path], row);
        if (!d_statuses.empty()) { d_statuses[path][row] = op_status::ROW_NULL; }
      }
      continue;
    }

    auto const [meta, val] = metadata_and_value_at(metadata, values, row);

    for (size_type slot = 0; slot < num_slots; ++slot) {
      auto const depth = trie.slot_depth[slot];
      auto const me    = state_of(slot, depth);
      auto step_begin  = trie.slot_steps[slot];

      auto parent = val;
      // An empty value blob resolves nothing, and a slot under one that did not resolve reports
      // its parent's reason rather than what probing an empty value would say.
      auto parent_status = val.empty() ? op_status::MALFORMED_VARIANT : op_status::SUCCESS;
      if (depth > 0) {
        auto const& parent_state = [&]() -> slot_result const& {
          if constexpr (MergeSiblings) {
            auto const group = trie.slot_group[slot];
            return located[trie.state_base[depth - 1] + trie.group_parent[group]];
          } else {
            return located[depth - 1];
          }
        }();
        parent        = slot_span(val, parent_state);
        parent_status = parent_state.status;
      }

      if constexpr (MergeSiblings) {
        auto const group     = trie.slot_group[slot];
        auto const key_begin = trie.group_keys[group];
        auto const num_keys  = trie.group_keys[group + 1] - key_begin;
        auto const base      = trie.state_base[depth];

        // The group's first slot resolves every mergeable key of the group, so that the siblings
        // behind it find their first step already applied.
        if (num_keys > 0 && slot == trie.group_first[group]) {
          if (parent.empty()) {
            for (size_type key = 0; key < num_keys; ++key) {
              located[base + key] = {0, invalid_slot_size, parent_status};
            }
          } else {
            auto const emit = [&](
                                size_type key, device_span<uint8_t const> field, op_status status) {
              located[base + key] = make_slot_result(field, val.data(), status);
            };
            auto const keys = trie.group_key_step.subspan(key_begin, num_keys);
            if constexpr (SharedDict) {
              locate_object_fields_ranked(shared, parent, keys, emit);
            } else {
              locate_object_fields(meta, parent, trie.steps, keys, emit);
            }
          }
        }
        // Keys lead their group, so a slot within that prefix starts from its merged result.
        if (trie.slot_group_pos[slot] < num_keys) {
          parent        = slot_span(val, located[me]);
          parent_status = located[me].status;
          ++step_begin;
        }
      }

      if (parent.empty()) {
        located[me] = {0, invalid_slot_size, parent_status};
      } else {
        auto const [field, status] = resolve_steps<SharedDict>(
          meta, parent, trie.steps, step_begin, trie.slot_steps[slot + 1], shared);
        located[me] = make_slot_result(field, val.data(), status);
      }

      for (auto out_idx = trie.output_offsets[slot]; out_idx < trie.output_offsets[slot + 1];
           ++out_idx) {
        auto const path = trie.output_paths[out_idx];
        auto const out  = path * num_rows + static_cast<size_type>(row);
        if (!d_statuses.empty()) { d_statuses[path][row] = located[me].status; }
        if (slot_is_valid(located[me])) {
          d_sizes[out]       = located[me].size;
          d_src_offsets[out] = located[me].src_offset;
        } else {
          d_sizes[out]       = 0;
          d_src_offsets[out] = 0;
          cudf::clear_bit(d_null_masks[path], row);
        }
      }
    }
  }
}

/**
 * @brief Per-row kernel: decode each VARIANT value blob into a fixed-width primitive of type `T`.
 *
 * Writes the decoded value to `d_output[row]` for non-null rows whose blob is a variant primitive
 * whose physical type id matches `T` exactly (e.g. an int16 value does not decode into an int32
 * output, and a float32 value does not decode into a float64 output; there is no widening). Rows
 * that are null, or whose value is not an exact-width match for `T`, are marked null in
 * `d_null_mask` with an output of 0.
 */
// `d_status`, when present, is an in-out buffer: it is read as incoming status from a prior
// `get_variant_field` call (rows already marked non-success are propagated without decoding), then
// overwritten in place with the final per-row status. Callers with no real incoming status must
// pre-fill every row with `op_status::SUCCESS` before calling.
template <typename T>
CUDF_KERNEL __launch_bounds__(block_size) void cast_variant_primitive_kernel(
  cudf::lists_column_device_view values,
  device_span<T> d_output,
  bitmask_type* d_null_mask,
  op_status* d_status)  // nullptr when no status was requested
{
  auto const num_rows = static_cast<size_type>(d_output.size());
  auto const tid      = cudf::detail::grid_1d::global_thread_id<block_size>();
  auto const stride   = cudf::detail::grid_1d::grid_stride<block_size>();

  for (auto row = tid; row < num_rows; row += stride) {
    if (d_status != nullptr) {
      // Status column is always non-nullable; row_null replaces the null bit.
      auto const s = d_status[row];
      if (s != op_status::SUCCESS) {
        d_output[row] = T{};
        if (cudf::bit_is_set(d_null_mask, row)) { cudf::clear_bit(d_null_mask, row); }
        continue;
      }
      if (!cudf::bit_is_set(d_null_mask, row)) {
        d_output[row] = T{};
        d_status[row] = op_status::ROW_NULL;
        continue;
      }
    } else {
      if (!cudf::bit_is_set(d_null_mask, row)) {
        d_output[row] = T{};
        continue;
      }
    }

    auto const val     = list_row_span(values, row);
    auto const decoded = decode_primitive<T>(val);
    if (decoded.has_value()) {
      d_output[row] = *decoded;
      if (d_status != nullptr) { d_status[row] = op_status::SUCCESS; }
    } else {
      d_output[row] = T{};
      cudf::clear_bit(d_null_mask, row);
      if (d_status != nullptr) { d_status[row] = cast_status_for_primitive<T>(val); }
    }
  }
}

__device__ op_status cast_status_for_bool(device_span<uint8_t const> val)
{
  if (val.empty()) { return op_status::MALFORMED_VARIANT; }
  if (is_variant_null(val)) { return op_status::VARIANT_NULL; }
  if (decode_bool(val).has_value()) { return op_status::SUCCESS; }
  if (decode_basic_type(val[0]) != basic_type::PRIMITIVE) { return op_status::TYPE_MISMATCH; }
  // Boolean values carry no payload, so a BOOLEAN_TRUE/FALSE header can never be truncated;
  // decode_bool would have succeeded above.  Any remaining primitive ID is a type mismatch when
  // recognised, or malformed when not.
  auto const ptype = static_cast<primitive_type>(variant_value_header(val[0]));
  return is_recognized_primitive_type(ptype) ? op_status::TYPE_MISMATCH
                                             : op_status::MALFORMED_VARIANT;
}

__device__ op_status cast_status_for_string(device_span<uint8_t const> val)
{
  if (val.empty()) { return op_status::MALFORMED_VARIANT; }
  if (is_variant_null(val)) { return op_status::VARIANT_NULL; }
  if (decode_string(val).has_value()) { return op_status::SUCCESS; }
  auto const btype = decode_basic_type(val[0]);
  if (btype == basic_type::SHORT_STRING) { return op_status::MALFORMED_VARIANT; }
  if (btype == basic_type::PRIMITIVE) {
    auto const ptype = static_cast<primitive_type>(variant_value_header(val[0]));
    // LONG_STRING is a recognized string type whose payload was truncated.
    if (ptype == primitive_type::LONG_STRING) { return op_status::MALFORMED_VARIANT; }
    return is_recognized_primitive_type(ptype) ? op_status::TYPE_MISMATCH
                                               : op_status::MALFORMED_VARIANT;
  }
  // OBJECT, ARRAY, or other non-primitive basic types: well-formed, just not a string.
  return op_status::TYPE_MISMATCH;
}

/**
 * @brief Strings-children functor: decode each VARIANT value blob into a string.
 *
 * Used with `make_strings_children`, so it runs in two passes. On the sizing pass (`d_chars ==
 * nullptr`) it writes each decoded string's length to `d_sizes[row]`; on the write pass it copies
 * the decoded bytes to `d_chars` at `d_offsets[row]`. Rows that are null, or whose value does not
 * decode to a string, are marked null in `d_null_mask` with size 0.
 */
struct cast_variant_string_fn {
  cudf::lists_column_device_view d_values;
  bitmask_type* d_null_mask;
  size_type* d_sizes;
  char* d_chars;
  cudf::detail::input_offsetalator d_offsets;
  // In-out status tracking (optional: d_status non-null to enable; status is always non-nullable).
  // Read as incoming status from a prior `get_variant_field` call, then overwritten in place with
  // the final status on the sizing pass.
  op_status* d_status{nullptr};

  __device__ void operator()(size_type row)
  {
    // Status is only written on the sizing pass (d_chars == nullptr). On the writing pass the
    // null mask may already be cleared from the sizing pass, so we must not re-inspect it to
    // write status (that would misidentify a decode-failed row as a SQL-null row).
    bool const is_sizing_pass = (d_chars == nullptr);

    if (d_status) {
      // Status column is always non-nullable; row_null replaces the null bit.
      auto const s = d_status[row];
      if (s != op_status::SUCCESS) {
        if (is_sizing_pass) { d_sizes[row] = 0; }
        if (cudf::bit_is_set(d_null_mask, row)) { cudf::clear_bit(d_null_mask, row); }
        return;
      }
      if (!cudf::bit_is_set(d_null_mask, row)) {
        if (is_sizing_pass) {
          d_sizes[row]  = 0;
          d_status[row] = op_status::ROW_NULL;
        }
        return;
      }
    } else {
      if (!cudf::bit_is_set(d_null_mask, row)) {
        if (is_sizing_pass) { d_sizes[row] = 0; }
        return;
      }
    }

    auto const val = list_row_span(d_values, row);

    auto const str = decode_string(val);
    if (!str) {
      if (is_sizing_pass) { d_sizes[row] = 0; }
      cudf::clear_bit(d_null_mask, row);
      if (is_sizing_pass && d_status) { d_status[row] = cast_status_for_string(val); }
      return;
    }

    if (is_sizing_pass) {
      d_sizes[row] = str->size();
    } else {
      cuda::std::memcpy(d_chars + d_offsets[row], str->data(), str->size());
    }
    if (is_sizing_pass && d_status) { d_status[row] = op_status::SUCCESS; }
  }
};

// An empty `list<uint8>` column: the shape a VARIANT field extraction produces for an empty input.
std::unique_ptr<column> make_empty_variant_value_column()
{
  return cudf::make_lists_column(
    0, make_empty_column(type_id::INT32), make_empty_column(type_id::UINT8), 0, {});
}

void validate_variant_child(column_view const& child)
{
  CUDF_EXPECTS(child.type().id() == type_id::LIST,
               "VARIANT metadata/value column must be a list",
               std::invalid_argument);
  CUDF_EXPECTS(lists_column_view{child}.child().type().id() == type_id::UINT8,
               "VARIANT metadata/value column must be list<uint8>",
               std::invalid_argument);
}

// Statuses are written one per row of the input, in place, so the buffer has to line up with it.
// `input_name` names the column the row count must match, for the error message.
void validate_status_column(std::optional<mutable_column_view> const& status,
                            size_type num_rows,
                            std::string const& input_name)
{
  if (!status.has_value()) { return; }
  CUDF_EXPECTS(!status->nullable(),
               "status column must not be nullable; use row_null for SQL-null rows",
               std::invalid_argument);
  CUDF_EXPECTS(
    status->type().id() == type_id::UINT8, "status column must be UINT8", std::invalid_argument);
  CUDF_EXPECTS(status->size() == num_rows,
               "status column must have the same number of rows as " + input_name,
               std::invalid_argument);
}

// The device view of a status buffer, empty when the caller asked for no status.
device_span<op_status> status_span(std::optional<mutable_column_view> const& status)
{
  if (!status.has_value()) { return {}; }
  return {reinterpret_cast<op_status*>(status->data<uint8_t>()),
          static_cast<std::size_t>(status->size())};
}

struct cast_variant_fn {
  cudf::lists_column_device_view values;
  size_type num_rows;
  data_type desired_type;
  bitmask_type* d_null_mask;
  rmm::device_buffer null_mask;
  cuda::stream_ref stream;
  rmm::device_async_resource_ref mr;
  // In-out status tracking; null when no status was requested.
  op_status* d_status{nullptr};

  template <typename T>
  std::unique_ptr<column> operator()()
    requires(is_variant_numerical<T>)
  {
    rmm::device_buffer data{num_rows * sizeof(T), stream, mr};
    auto const grid = cudf::detail::grid_1d{num_rows, block_size};
    auto const d_out =
      device_span<T>{static_cast<T*>(data.data()), static_cast<std::size_t>(num_rows)};
    cast_variant_primitive_kernel<T>
      <<<grid.num_blocks, block_size, 0, stream.get()>>>(values, d_out, d_null_mask, d_status);
    CUDF_CUDA_TRY(cudaGetLastError());

    auto const null_count =
      num_rows - cudf::detail::count_set_bits(d_null_mask, 0, num_rows, stream);
    return std::make_unique<column>(desired_type,
                                    num_rows,
                                    std::move(data),
                                    null_count > 0 ? std::move(null_mask) : rmm::device_buffer{},
                                    null_count);
  }

  template <typename T>
  std::unique_ptr<column> operator()()
    requires(cuda::std::is_same_v<T, bool>)
  {
    rmm::device_buffer data{num_rows * sizeof(bool), stream, mr};

    auto* dp_s = d_status;

    thrust::for_each(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                     cuda::counting_iterator<size_type>(0),
                     cuda::counting_iterator<size_type>(num_rows),
                     [vals  = this->values,
                      d_out = static_cast<bool*>(data.data()),
                      dnm   = this->d_null_mask,
                      dp_s] __device__(size_type row) {
                       auto const fail = [&](op_status s) {
                         d_out[row] = false;
                         if (cudf::bit_is_set(dnm, row)) { cudf::clear_bit(dnm, row); }
                         if (dp_s) { dp_s[row] = s; }
                       };
                       if (dp_s and dp_s[row] != op_status::SUCCESS) { return fail(dp_s[row]); }
                       // Status column is always non-nullable; ROW_NULL replaces the null bit.
                       if (!cudf::bit_is_set(dnm, row)) { return fail(op_status::ROW_NULL); }
                       auto const val     = list_row_span(vals, row);
                       auto const decoded = decode_bool(val);
                       if (!decoded) { return fail(cast_status_for_bool(val)); }
                       d_out[row] = *decoded;
                       if (dp_s) { dp_s[row] = op_status::SUCCESS; }
                     });

    auto const null_count =
      num_rows - cudf::detail::count_set_bits(d_null_mask, 0, num_rows, stream);
    return std::make_unique<column>(desired_type,
                                    num_rows,
                                    std::move(data),
                                    null_count > 0 ? std::move(null_mask) : rmm::device_buffer{},
                                    null_count);
  }

  template <typename T>
  std::unique_ptr<column> operator()()
    requires(cuda::std::is_same_v<T, cudf::string_view>)
  {
    cast_variant_string_fn fn{values, d_null_mask, nullptr, nullptr, {}, d_status};
    auto [offsets_column, chars] =
      cudf::strings::detail::make_strings_children(fn, num_rows, stream, mr);

    auto const null_count =
      num_rows - cudf::detail::count_set_bits(d_null_mask, 0, num_rows, stream);
    return make_strings_column(num_rows,
                               std::move(offsets_column),
                               chars.release(),
                               null_count,
                               null_count > 0 ? std::move(null_mask) : rmm::device_buffer{});
  }

  template <typename T>
  std::unique_ptr<column> operator()()
    requires(not is_variant_castable<T>)
  {
    CUDF_FAIL("unsupported type for variant cast", std::invalid_argument);
  }
};

/**
 * @brief Classifies only the first (value_metadata) byte of enc; does not validate the remaining
 * payload. A recognized header returns its logical type even when the payload is truncated. Returns
 * nullopt for an empty blob or an unrecognized primitive type ID.
 */
__device__ cuda::std::optional<variant_logical_type> logical_type_of(device_span<uint8_t const> enc)
{
  if (enc.empty()) { return cuda::std::nullopt; }
  auto const value_metadata = enc[0];
  auto const btype          = decode_basic_type(value_metadata);

  if (btype == basic_type::SHORT_STRING) { return variant_logical_type::STRING; }
  if (btype == basic_type::OBJECT) { return variant_logical_type::OBJECT; }
  if (btype == basic_type::ARRAY) { return variant_logical_type::ARRAY; }

  switch (static_cast<primitive_type>(variant_value_header(value_metadata))) {
    case primitive_type::NULLVAL: return variant_logical_type::NULL_VALUE;
    case primitive_type::BOOLEAN_TRUE:
    case primitive_type::BOOLEAN_FALSE: return variant_logical_type::BOOLEAN;
    case primitive_type::INT8:
    case primitive_type::INT16:
    case primitive_type::INT32:
    case primitive_type::INT64: return variant_logical_type::LONG_VALUE;
    case primitive_type::FLOAT64: return variant_logical_type::DOUBLE_VALUE;
    case primitive_type::DECIMAL4:
    case primitive_type::DECIMAL8:
    case primitive_type::DECIMAL16: return variant_logical_type::DECIMAL;
    case primitive_type::DATE: return variant_logical_type::DATE;
    case primitive_type::TIMESTAMP_MICROS:
    case primitive_type::TIMESTAMP_NANOS: return variant_logical_type::TIMESTAMP;
    case primitive_type::TIMESTAMP_NTZ_MICROS:
    case primitive_type::TIMESTAMP_NTZ_NANOS: return variant_logical_type::TIMESTAMP_NTZ;
    case primitive_type::FLOAT32: return variant_logical_type::FLOAT_VALUE;
    case primitive_type::BINARY: return variant_logical_type::BINARY;
    case primitive_type::LONG_STRING: return variant_logical_type::STRING;
    case primitive_type::TIME_NTZ_MICROS: return variant_logical_type::TIME_NTZ;
    case primitive_type::UUID: return variant_logical_type::UUID;
    default: return cuda::std::nullopt;
  }
}

std::unique_ptr<column> build_path_column(cudf::host_span<std::string const> steps,
                                          cuda::stream_ref stream,
                                          rmm::device_async_resource_ref mr)
{
  auto const depth = steps.size();

  std::string host_chars;
  std::vector<size_type> host_offsets(depth + 1);
  for (size_t i = 0; i < depth; ++i) {
    host_offsets[i] = static_cast<size_type>(host_chars.size());
    host_chars.append(steps[i]);
  }
  host_offsets[depth] = host_chars.size();

  auto d_offsets   = cudf::detail::make_device_uvector_async(host_offsets, stream, mr);
  auto offsets_col = std::make_unique<column>(data_type{type_id::INT32},
                                              static_cast<size_type>(host_offsets.size()),
                                              d_offsets.release(),
                                              rmm::device_buffer{},
                                              0);

  auto d_chars = cudf::detail::make_device_uvector(
    host_span<char const>{host_chars.data(), host_chars.size()}, stream, mr);
  return cudf::make_strings_column(
    depth, std::move(offsets_col), d_chars.release(), 0, rmm::device_buffer{});
}

// Prototype switch, so that both lookup schemes can be measured from one build. Set
// CUDF_VARIANT_MERGE_SIBLINGS=0 to search each sibling's first step on its own instead.
// PROTOTYPE SCAFFOLDING (shared dictionary) below, through to shared_dictionary_plan.

// True when every row's metadata blob is byte-identical to row 0's, which is what lets one resolved
// dictionary stand in for all of them. Reads the whole metadata column, and is the price of not
// being able to assume anything about it: the format makes the dictionary a per-row field.
//
// Two passes so that both are coalesced. The lengths have to match first, and then a row's byte `i`
// has to equal byte `i % blob_len` of the first row -- which, once the lengths are equal, is the
// same as comparing row against row, but with consecutive threads reading consecutive bytes of the
// child column rather than each thread walking a blob of its own.
CUDF_KERNEL __launch_bounds__(block_size) void metadata_lengths_uniform_kernel(
  cudf::lists_column_device_view metadata, size_type num_rows, int* d_uniform)
{
  auto const tid      = cudf::detail::grid_1d::global_thread_id<block_size>();
  auto const stride   = cudf::detail::grid_1d::grid_stride<block_size>();
  auto const blob_len = metadata.offset_at(1) - metadata.offset_at(0);
  for (auto row = tid; row < num_rows; row += stride) {
    if (metadata.offset_at(row + 1) - metadata.offset_at(row) != blob_len) {
      atomicAnd(d_uniform, 0);
      return;
    }
  }
}

// One warp per row, lanes striding the blob, so the reads coalesce and row 0's copy stays in cache.
// A warp rather than a block because a metadata blob is tens of bytes, and a block would leave most
// of its threads with nothing to compare.
CUDF_KERNEL __launch_bounds__(block_size) void metadata_bytes_uniform_kernel(
  cudf::lists_column_device_view metadata, size_type num_rows, int* d_uniform)
{
  auto const* child   = metadata.child().data<uint8_t>();
  auto const first    = child + metadata.offset_at(0);
  auto const blob_len = metadata.offset_at(1) - metadata.offset_at(0);
  if (blob_len <= 0) {
    if (blockIdx.x == 0 && threadIdx.x == 0) { atomicAnd(d_uniform, 0); }
    return;
  }

  auto const warp = static_cast<size_type>(cudf::detail::grid_1d::global_thread_id<block_size>() /
                                           cudf::detail::warp_size);
  auto const num_warps = static_cast<size_type>(cudf::detail::grid_1d::grid_stride<block_size>() /
                                                cudf::detail::warp_size);
  auto const lane      = static_cast<size_type>(threadIdx.x % cudf::detail::warp_size);

  for (auto row = warp; row < num_rows; row += num_warps) {
    auto const* blob = child + metadata.offset_at(row);
    for (auto i = lane; i < blob_len; i += cudf::detail::warp_size) {
      if (blob[i] != first[i]) {
        atomicAnd(d_uniform, 0);
        return;
      }
    }
  }
}

// Resolves row 0's dictionary once: the sorted-order rank of every entry, and the rank each step
// token of the trie maps to. One block, since both loops are quadratic in sizes that are small
// here; a real implementation would sort rather than count.
CUDF_KERNEL __launch_bounds__(block_size) void build_shared_dictionary_kernel(
  cudf::lists_column_device_view metadata,
  column_device_view steps,
  device_span<size_type> d_rank,
  device_span<size_type> d_step_rank,
  int* d_ok)
{
  auto const dict = parse_metadata_dictionary(list_row_span(metadata, 0));
  if (dict.status != op_status::SUCCESS || dict.num_entries != d_rank.size()) {
    if (threadIdx.x == 0) { *d_ok = 0; }
    return;
  }

  // rank[id] is how many entries sort before it. Unique names, per the spec, so this is a
  // permutation.
  for (auto id = static_cast<size_type>(threadIdx.x); id < dict.num_entries;
       id += static_cast<size_type>(block_size)) {
    auto const name = name_for_id(dict, id);
    if (!name.has_value()) {
      *d_ok = 0;
      return;
    }
    size_type rank = 0;
    for (size_type other = 0; other < dict.num_entries; ++other) {
      auto const other_name = name_for_id(dict, other);
      if (!other_name.has_value()) {
        *d_ok = 0;
        return;
      }
      if (compare_keys(other_name.value(), name.value()) < 0) { ++rank; }
    }
    d_rank[id] = rank;
  }

  // A step token's rank is where it would sort among the entries, and -1 when it is absent, which
  // includes every index step. Computed the same way, so it does not have to wait for d_rank.
  auto const num_steps = static_cast<size_type>(d_step_rank.size());
  for (auto step = static_cast<size_type>(threadIdx.x); step < num_steps;
       step += static_cast<size_type>(block_size)) {
    auto const token = steps.element<cudf::string_view>(step);
    size_type rank   = 0;
    bool present     = false;
    for (size_type other = 0; other < dict.num_entries; ++other) {
      auto const other_name = name_for_id(dict, other);
      if (!other_name.has_value()) {
        *d_ok = 0;
        return;
      }
      auto const cmp = compare_keys(other_name.value(), token);
      if (cmp < 0) { ++rank; }
      if (cmp == 0) { present = true; }
    }
    d_step_rank[step] = present ? rank : -1;
  }
}

// Prototype switch: set CUDF_VARIANT_SHARED_DICT=1 to resolve the dictionary once for all rows and
// probe by rank, when every row's metadata blob turns out to be identical.
bool shared_dictionary_enabled()
{
  static bool const enabled = [] {
    auto const* setting = std::getenv("CUDF_VARIANT_SHARED_DICT");
    return setting != nullptr && std::string_view{setting} == "1";
  }();
  return enabled;
}

bool merge_sibling_lookups()
{
  static bool const enabled = [] {
    auto const* setting = std::getenv("CUDF_VARIANT_MERGE_SIBLINGS");
    return setting == nullptr || std::string_view{setting} != "0";
  }();
  return enabled;
}

}  // namespace

namespace detail {

std::unique_ptr<column> get_variant_field(column_view const& variant_column,
                                          std::string_view path,
                                          std::optional<mutable_column_view> status,
                                          cuda::stream_ref stream,
                                          rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(variant_column.type().id() == type_id::STRUCT,
               "VARIANT column must be struct type",
               std::invalid_argument);
  CUDF_EXPECTS(variant_column.num_children() >= 2,
               "VARIANT struct must have at least two children",
               std::invalid_argument);
  validate_variant_child(variant_column.child(0));
  validate_variant_child(variant_column.child(1));

  // Validate the path even for empty input columns
  auto const steps = parse_variant_path(path);

  auto const num_rows = variant_column.size();
  validate_status_column(status, num_rows, "variant_column");

  if (num_rows == 0) { return make_empty_variant_value_column(); }

  auto const temp_mr = cudf::get_current_device_resource_ref();

  auto path_column      = build_path_column(steps, stream, temp_mr);
  auto path_device_view = column_device_view::create(path_column->view(), stream);

  // Resolve children with respect to any slice/offset on the parent struct
  structs_column_view const variant_struct{variant_column};
  auto const meta_view = variant_struct.get_sliced_child(0, stream);
  auto const val_view  = variant_struct.get_sliced_child(1, stream);

  auto meta_device_view = column_device_view::create(meta_view, stream);
  auto val_device_view  = column_device_view::create(val_view, stream);
  cudf::lists_column_device_view meta_lists_device_view(*meta_device_view);
  cudf::lists_column_device_view val_lists_device_view(*val_device_view);

  rmm::device_uvector<size_type> d_sizes(num_rows, stream, temp_mr);
  // Caches the per-row intra-value byte offset
  rmm::device_uvector<size_type> d_src_offsets(num_rows, stream, temp_mr);
  auto null_mask =
    variant_column.nullable()
      ? cudf::detail::copy_bitmask(variant_column, stream, mr)
      : cudf::create_null_mask(variant_column.size(), mask_state::ALL_VALID, stream, mr);
  auto* d_null_mask = static_cast<bitmask_type*>(null_mask.data());

  auto grid = cudf::detail::grid_1d{num_rows, block_size};

  auto const d_status = status_span(status);
  locate_variant_fields_kernel<<<grid.num_blocks, block_size, 0, stream.get()>>>(
    meta_lists_device_view,
    val_lists_device_view,
    *path_device_view,
    d_sizes,
    d_src_offsets,
    d_null_mask,
    d_status);
  CUDF_CUDA_TRY(cudaGetLastError());

  auto [offsets_column, total_bytes] =
    cudf::strings::detail::make_offsets_child_column(d_sizes, stream, mr);
  CUDF_EXPECTS(total_bytes <= std::numeric_limits<size_type>::max(),
               "VARIANT extracted bytes exceed cudf size_type limit",
               std::overflow_error);
  device_span<size_type const> d_offsets{offsets_column->view().data<size_type>(),
                                         static_cast<std::size_t>(num_rows + 1)};

  auto val_child = make_numeric_column(
    data_type{type_id::UINT8}, total_bytes, mask_state::UNALLOCATED, stream, mr);
  if (total_bytes > 0) {
    auto const out_base = val_child->mutable_view().data<uint8_t>();
    auto src_iter       = cudf::detail::make_counting_transform_iterator(
      size_type{0},
      cuda::proclaim_return_type<uint8_t const*>(
        [vlv   = val_lists_device_view,
         d_src = d_src_offsets.data()] __device__(size_type row) -> uint8_t const* {
          return vlv.child().template data<uint8_t>() + vlv.offset_at(row) + d_src[row];
        }));
    auto dst_iter = cudf::detail::make_counting_transform_iterator(
      size_type{0},
      cuda::proclaim_return_type<uint8_t*>(
        [out_base, d_off = d_offsets.data()] __device__(size_type row) -> uint8_t* {
          return out_base + d_off[row];
        }));
    cudf::detail::batched_memcpy_async(src_iter, dst_iter, d_sizes.begin(), num_rows, stream);
  }

  auto const null_count = num_rows - cudf::detail::count_set_bits(d_null_mask, 0, num_rows, stream);
  return make_lists_column(num_rows,
                           std::move(offsets_column),
                           std::move(val_child),
                           null_count,
                           null_count > 0 ? std::move(null_mask) : rmm::device_buffer{});
}

std::unique_ptr<table> get_variant_fields(column_view const& variant_column,
                                          host_span<std::string_view const> paths,
                                          host_span<mutable_column_view const> statuses,
                                          cuda::stream_ref stream,
                                          rmm::device_async_resource_ref mr)
{
  // Validate the variant column
  CUDF_EXPECTS(variant_column.type().id() == type_id::STRUCT,
               "VARIANT column must be struct type",
               std::invalid_argument);
  CUDF_EXPECTS(variant_column.num_children() >= 2,
               "VARIANT struct must have at least two children",
               std::invalid_argument);
  validate_variant_child(variant_column.child(0));
  validate_variant_child(variant_column.child(1));

  auto const num_paths = static_cast<size_type>(paths.size());
  auto const num_rows  = variant_column.size();

  // Validate the status columns even for empty inputs, so that a caller always hears about a
  // malformed one
  auto const want_status = !statuses.empty();
  CUDF_EXPECTS(!want_status || statuses.size() == paths.size(),
               "status columns must be empty or one per path",
               std::invalid_argument);
  for (auto const& status : statuses) {
    validate_status_column(status, num_rows, "variant_column");
  }

  std::vector<std::unique_ptr<column>> output;
  output.reserve(num_paths);
  if (num_paths == 0) { return std::make_unique<table>(std::move(output)); }

  // A single path has no prefixes to share, so it is exactly the single-path entry point; the
  // batched setup would only add fixed overhead.
  if (num_paths == 1) {
    output.push_back(get_variant_field(variant_column,
                                       paths.front(),
                                       want_status ? std::optional{statuses.front()} : std::nullopt,
                                       stream,
                                       mr));
    return std::make_unique<table>(std::move(output));
  }

  // Validate and merge the paths even for empty input columns
  auto const trie = build_variant_path_trie(paths);

  if (num_rows == 0) {
    std::generate_n(
      std::back_inserter(output), num_paths, [] { return make_empty_variant_value_column(); });
    return std::make_unique<table>(std::move(output));
  }

  auto const temp_mr = cudf::get_current_device_resource_ref();

  auto steps_column      = build_path_column(trie.steps, stream, temp_mr);
  auto steps_device_view = column_device_view::create(steps_column->view(), stream);
  auto const upload      = [&](auto const& host_array) {
    return cudf::detail::make_device_uvector_async(host_array, stream, temp_mr);
  };
  auto const d_slot_steps     = upload(trie.slot_steps);
  auto const d_slot_depth     = upload(trie.slot_depth);
  auto const d_output_offsets = upload(trie.output_offsets);
  auto const d_output_paths   = upload(trie.output_paths);
  auto const d_slot_group     = upload(trie.slot_group);
  auto const d_slot_group_pos = upload(trie.slot_group_pos);
  auto const d_group_keys     = upload(trie.group_keys);
  auto const d_group_key_step = upload(trie.group_key_step);
  auto const d_group_first    = upload(trie.group_first);
  auto const d_group_parent   = upload(trie.group_parent);
  auto const d_state_base     = upload(trie.state_base);

  // Resolve children with respect to any slice/offset on the parent struct
  structs_column_view const variant_struct{variant_column};
  auto const meta_view = variant_struct.get_sliced_child(0, stream);
  auto const val_view  = variant_struct.get_sliced_child(1, stream);

  auto meta_device_view = column_device_view::create(meta_view, stream);
  auto val_device_view  = column_device_view::create(val_view, stream);
  cudf::lists_column_device_view meta_lists_device_view(*meta_device_view);
  cudf::lists_column_device_view val_lists_device_view(*val_device_view);

  // Input row validity, copied so that it is indexable by row regardless of any slice offset
  auto const row_mask     = variant_column.nullable()
                              ? cudf::detail::copy_bitmask(variant_column, stream, temp_mr)
                              : rmm::device_buffer{};
  auto const* d_row_valid = static_cast<bitmask_type const*>(row_mask.data());

  // Per-path outputs are contiguous, so each path's sizes can be scanned on their own
  CUDF_EXPECTS(static_cast<int64_t>(num_paths) * num_rows <= std::numeric_limits<size_type>::max(),
               "VARIANT paths times rows exceeds cudf size_type limit",
               std::overflow_error);
  auto const num_outputs = num_paths * num_rows;
  rmm::device_uvector<size_type> d_sizes(num_outputs, stream, temp_mr);
  rmm::device_uvector<size_type> d_src_offsets(num_outputs, stream, temp_mr);

  // One null mask per output column, narrowed from all-valid by the walk
  std::vector<rmm::device_buffer> null_masks;
  null_masks.reserve(num_paths);
  std::vector<bitmask_type*> h_null_masks(num_paths);
  for (size_type p = 0; p < num_paths; ++p) {
    null_masks.push_back(cudf::create_null_mask(num_rows, mask_state::ALL_VALID, stream, mr));
    h_null_masks[p] = static_cast<bitmask_type*>(null_masks.back().data());
  }
  auto const d_null_masks = cudf::detail::make_device_uvector_async(h_null_masks, stream, temp_mr);

  // One caller-owned status buffer per path, written by the walk
  std::vector<op_status*> h_statuses(want_status ? num_paths : 0);
  for (size_type p = 0; p < static_cast<size_type>(h_statuses.size()); ++p) {
    h_statuses[p] = reinterpret_cast<op_status*>(statuses[p].data<uint8_t>());
  }
  auto const d_statuses = cudf::detail::make_device_uvector_async(h_statuses, stream, temp_mr);

  // Resolve the whole trie per row and compute the output sizes. Without merging, the walk keeps
  // one located value per trie level, so only a pathologically deep trie needs scratch outside the
  // thread; merging instead keeps a group's resolved children live across their subtrees.
  bool const merge_siblings = merge_sibling_lookups();
  auto const trie_depth     = 1 + *std::max_element(trie.slot_depth.begin(), trie.slot_depth.end());
  auto const walk_state_size   = merge_siblings ? trie.state_base.back() : trie_depth;
  bool const use_local_scratch = walk_state_size <= max_local_walk_state;
  auto const grid              = cudf::detail::grid_1d{num_rows, block_size};
  auto const num_blocks =
    use_local_scratch
      ? grid.num_blocks
      : std::min(grid.num_blocks, static_cast<thread_index_type>(max_global_scratch_blocks));
  rmm::device_uvector<slot_result> d_scratch(
    use_local_scratch ? 0 : static_cast<std::size_t>(num_blocks) * block_size * walk_state_size,
    stream,
    temp_mr);

  trie_device_view const d_trie{*steps_device_view,
                                d_slot_steps,
                                d_slot_depth,
                                d_output_offsets,
                                d_output_paths,
                                d_slot_group,
                                d_slot_group_pos,
                                d_group_keys,
                                d_group_key_step,
                                d_group_first,
                                d_group_parent,
                                d_state_base};

  // The shared dictionary path needs every row's metadata blob to be identical, which is a property
  // of the data rather than a guarantee of the format, so it has to be checked. The check reads the
  // whole metadata column and then stalls the stream to bring one flag back.
  rmm::device_uvector<size_type> d_rank(0, stream, temp_mr);
  rmm::device_uvector<size_type> d_step_rank(0, stream, temp_mr);
  bool use_shared_dict = false;
  if (shared_dictionary_enabled()) {
    rmm::device_scalar<int> d_flags(1, stream, temp_mr);
    metadata_lengths_uniform_kernel<<<grid.num_blocks, block_size, 0, stream.get()>>>(
      meta_lists_device_view, num_rows, d_flags.data());
    CUDF_CUDA_TRY(cudaGetLastError());
    // One warp per row, capped so a tall column does not launch an unbounded grid.
    auto constexpr warps_per_block = block_size / cudf::detail::warp_size;
    auto const byte_blocks =
      std::min((num_rows + warps_per_block - 1) / warps_per_block, 8 * max_global_scratch_blocks);
    metadata_bytes_uniform_kernel<<<byte_blocks, block_size, 0, stream.get()>>>(
      meta_lists_device_view, num_rows, d_flags.data());
    CUDF_CUDA_TRY(cudaGetLastError());
    use_shared_dict = d_flags.value(stream) == 1;

    if (use_shared_dict) {
      auto const num_entries = [&] {
        // The entry count comes from the same header the device parse reads, so ask the device.
        rmm::device_scalar<size_type> d_count(0, stream, temp_mr);
        thrust::for_each(
          rmm::exec_policy_nosync(stream, temp_mr),
          cuda::counting_iterator<size_type>(0),
          cuda::counting_iterator<size_type>(1),
          [meta = meta_lists_device_view, count = d_count.data()] __device__(size_type) {
            auto const dict = parse_metadata_dictionary(list_row_span(meta, 0));
            *count          = dict.status == op_status::SUCCESS ? dict.num_entries : 0;
          });
        return d_count.value(stream);
      }();

      if (num_entries > 0) {
        d_rank.resize(num_entries, stream);
        d_step_rank.resize(steps_column->size(), stream);
        int const ok = 1;
        d_flags.set_value_async(ok, stream);
        build_shared_dictionary_kernel<<<1, block_size, 0, stream.get()>>>(
          meta_lists_device_view, *steps_device_view, d_rank, d_step_rank, d_flags.data());
        CUDF_CUDA_TRY(cudaGetLastError());
        use_shared_dict = d_flags.value(stream) == 1;
      } else {
        use_shared_dict = false;
      }
    }
  }
  shared_dictionary_view const d_shared{d_rank, d_step_rank};

  auto const launch = [&](auto use_local, auto merge, auto shared_dict) {
    locate_variant_field_trie_kernel<decltype(use_local)::value,
                                     decltype(merge)::value,
                                     decltype(shared_dict)::value>
      <<<num_blocks, block_size, 0, stream.get()>>>(meta_lists_device_view,
                                                    val_lists_device_view,
                                                    d_trie,
                                                    d_row_valid,
                                                    num_rows,
                                                    walk_state_size,
                                                    d_sizes,
                                                    d_src_offsets,
                                                    d_null_masks,
                                                    d_statuses,
                                                    d_scratch,
                                                    d_shared);
  };
  auto const dispatch = [&](auto merge, auto shared_dict) {
    if (use_local_scratch) {
      launch(cuda::std::true_type{}, merge, shared_dict);
    } else {
      launch(cuda::std::false_type{}, merge, shared_dict);
    }
  };
  auto const dispatch_merge = [&](auto shared_dict) {
    if (merge_siblings) {
      dispatch(cuda::std::true_type{}, shared_dict);
    } else {
      dispatch(cuda::std::false_type{}, shared_dict);
    }
  };
  if (use_shared_dict) {
    dispatch_merge(cuda::std::true_type{});
  } else {
    dispatch_merge(cuda::std::false_type{});
  }
  CUDF_CUDA_TRY(cudaGetLastError());

  // Convert each path's sizes to offsets and allocate its output bytes
  std::vector<std::unique_ptr<column>> offsets_columns;
  std::vector<std::unique_ptr<column>> value_children;
  offsets_columns.reserve(num_paths);
  value_children.reserve(num_paths);
  std::vector<size_type const*> h_offsets(num_paths);
  std::vector<uint8_t*> h_out(num_paths);
  int64_t all_paths_bytes = 0;
  for (size_type p = 0; p < num_paths; ++p) {
    device_span<size_type const> const path_sizes{
      d_sizes.data() + static_cast<std::size_t>(p) * num_rows, static_cast<std::size_t>(num_rows)};
    auto [offsets_column, total_bytes] =
      cudf::strings::detail::make_offsets_child_column(path_sizes, stream, mr);
    CUDF_EXPECTS(total_bytes <= std::numeric_limits<size_type>::max(),
                 "VARIANT extracted bytes exceed cudf size_type limit",
                 std::overflow_error);

    auto value_child = make_numeric_column(data_type{type_id::UINT8},
                                           static_cast<size_type>(total_bytes),
                                           mask_state::UNALLOCATED,
                                           stream,
                                           mr);
    h_offsets[p]     = offsets_column->view().data<size_type>();
    h_out[p]         = value_child->mutable_view().data<uint8_t>();
    all_paths_bytes += total_bytes;
    offsets_columns.push_back(std::move(offsets_column));
    value_children.push_back(std::move(value_child));
  }

  // Copy the located values of every (path, row) pair in one pass
  if (all_paths_bytes > 0) {
    auto const d_offsets = cudf::detail::make_device_uvector_async(h_offsets, stream, temp_mr);
    auto const d_out     = cudf::detail::make_device_uvector_async(h_out, stream, temp_mr);

    auto src_iter = cudf::detail::make_counting_transform_iterator(
      size_type{0},
      cuda::proclaim_return_type<uint8_t const*>(
        [vlv = val_lists_device_view, d_src = d_src_offsets.data(), num_rows] __device__(
          size_type i) -> uint8_t const* {
          auto const row = i % num_rows;
          return vlv.child().template data<uint8_t>() + vlv.offset_at(row) + d_src[i];
        }));
    auto dst_iter = cudf::detail::make_counting_transform_iterator(
      size_type{0},
      cuda::proclaim_return_type<uint8_t*>([d_off = d_offsets.data(),
                                            d_dst = d_out.data(),
                                            num_rows] __device__(size_type i) -> uint8_t* {
        return d_dst[i / num_rows] + d_off[i / num_rows][i % num_rows];
      }));
    cudf::detail::batched_memcpy_async(src_iter, dst_iter, d_sizes.begin(), num_outputs, stream);
  }

  for (size_type p = 0; p < num_paths; ++p) {
    auto const null_count =
      num_rows - cudf::detail::count_set_bits(h_null_masks[p], 0, num_rows, stream);
    output.push_back(
      make_lists_column(num_rows,
                        std::move(offsets_columns[p]),
                        std::move(value_children[p]),
                        null_count,
                        null_count > 0 ? std::move(null_masks[p]) : rmm::device_buffer{}));
  }

  return std::make_unique<table>(std::move(output));
}

std::unique_ptr<column> cast_variant(column_view const& values,
                                     data_type desired_type,
                                     std::optional<mutable_column_view> status,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
{
  validate_variant_child(values);

  switch (desired_type.id()) {
    case type_id::INT8:
    case type_id::INT16:
    case type_id::INT32:
    case type_id::INT64:
    case type_id::FLOAT32:
    case type_id::FLOAT64:
    case type_id::BOOL8:
    case type_id::STRING: break;
    default: CUDF_FAIL("unsupported type for variant cast", std::invalid_argument);
  }

  size_type const num_rows = values.size();

  // Validate status before the empty-values fast path so callers always get
  // std::invalid_argument for a malformed status column, even when values is empty.
  validate_status_column(status, num_rows, "the values column");

  if (num_rows == 0) { return make_empty_column(desired_type); }

  auto val_device_view = column_device_view::create(values, stream);
  cudf::lists_column_device_view val_lists_device_view(*val_device_view);

  auto null_mask    = values.nullable()
                        ? cudf::detail::copy_bitmask(values, stream, mr)
                        : cudf::create_null_mask(num_rows, mask_state::ALL_VALID, stream, mr);
  auto* d_null_mask = static_cast<bitmask_type*>(null_mask.data());

  return cudf::type_dispatcher(
    desired_type,
    cast_variant_fn{
      val_lists_device_view,
      num_rows,
      desired_type,
      d_null_mask,
      std::move(null_mask),
      stream,
      mr,
      status.has_value() ? reinterpret_cast<op_status*>(status->data<uint8_t>()) : nullptr});
}

std::unique_ptr<table> extract_variant_fields(column_view const& variant_column,
                                              host_span<std::string_view const> paths,
                                              host_span<data_type const> desired_types,
                                              host_span<mutable_column_view const> statuses,
                                              cuda::stream_ref stream,
                                              rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(paths.size() == desired_types.size(),
               "VARIANT paths and desired types must have the same size",
               std::invalid_argument);

  auto const temp_mr     = cudf::get_current_device_resource_ref();
  auto const want_status = !statuses.empty();
  // Each path's status is filled by the walk, then read back by that path's cast as incoming status
  // and overwritten in place with the final status.
  auto const values = get_variant_fields(variant_column, paths, statuses, stream, temp_mr);

  std::vector<std::unique_ptr<column>> output;
  output.reserve(paths.size());
  for (size_type p = 0; p < values->num_columns(); ++p) {
    output.push_back(cast_variant(values->get_column(p).view(),
                                  desired_types[p],
                                  want_status ? std::optional{statuses[p]} : std::nullopt,
                                  stream,
                                  mr));
  }
  return std::make_unique<table>(std::move(output));
}

std::unique_ptr<column> get_variant_type_id(column_view const& values,
                                            cuda::stream_ref stream,
                                            rmm::device_async_resource_ref mr)
{
  validate_variant_child(values);
  size_type const num_rows = values.size();
  if (num_rows == 0) { return make_empty_column(data_type{type_id::UINT8}); }

  auto val_device_view = column_device_view::create(values, stream);
  cudf::lists_column_device_view val_lists_device_view(*val_device_view);

  auto null_mask    = values.nullable()
                        ? cudf::detail::copy_bitmask(values, stream, mr)
                        : cudf::create_null_mask(num_rows, mask_state::ALL_VALID, stream, mr);
  auto* d_null_mask = static_cast<bitmask_type*>(null_mask.data());

  rmm::device_buffer data{static_cast<std::size_t>(num_rows) * sizeof(uint8_t), stream, mr};

  thrust::transform(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    cuda::counting_iterator<size_type>(0),
    cuda::counting_iterator<size_type>(num_rows),
    static_cast<uint8_t*>(data.data()),
    [values = val_lists_device_view, d_null_mask] __device__(size_type row) -> uint8_t {
      if (!cudf::bit_is_set(d_null_mask, row)) { return 0; }
      auto const ltype = logical_type_of(list_row_span(values, row));
      if (ltype.has_value()) { return static_cast<uint8_t>(ltype.value()); }
      cudf::clear_bit(d_null_mask, row);
      return 0;
    });

  auto const null_count = num_rows - cudf::detail::count_set_bits(d_null_mask, 0, num_rows, stream);
  return std::make_unique<column>(data_type{type_id::UINT8},
                                  num_rows,
                                  std::move(data),
                                  null_count > 0 ? std::move(null_mask) : rmm::device_buffer{},
                                  null_count);
}

}  // namespace detail

std::unique_ptr<column> get_variant_field(column_view const& variant_column,
                                          std::string_view path,
                                          std::optional<mutable_column_view> status,
                                          cuda::stream_ref stream,
                                          rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::get_variant_field(variant_column, path, status, stream, mr);
}

std::unique_ptr<column> cast_variant(column_view const& values,
                                     data_type desired_type,
                                     std::optional<mutable_column_view> status,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::cast_variant(values, desired_type, status, stream, mr);
}

std::unique_ptr<column> get_variant_type_id(column_view const& values,
                                            cuda::stream_ref stream,
                                            rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::get_variant_type_id(values, stream, mr);
}

std::unique_ptr<column> extract_variant_field(column_view const& variant_column,
                                              std::string_view path,
                                              data_type desired_type,
                                              std::optional<mutable_column_view> status,
                                              cuda::stream_ref stream,
                                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  auto const temp_mr = cudf::get_current_device_resource_ref();

  if (status.has_value()) {
    // `status` is filled by `get_variant_field`, then read back by `cast_variant` as incoming
    // status and overwritten in place with the final per-row status.
    auto value = detail::get_variant_field(variant_column, path, status, stream, temp_mr);
    return detail::cast_variant(value->view(), desired_type, status, stream, mr);
  }

  auto value = detail::get_variant_field(variant_column, path, std::nullopt, stream, temp_mr);
  return detail::cast_variant(value->view(), desired_type, std::nullopt, stream, mr);
}

std::unique_ptr<table> get_variant_fields(column_view const& variant_column,
                                          host_span<std::string_view const> paths,
                                          host_span<mutable_column_view const> statuses,
                                          cuda::stream_ref stream,
                                          rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::get_variant_fields(variant_column, paths, statuses, stream, mr);
}

std::unique_ptr<table> extract_variant_fields(column_view const& variant_column,
                                              host_span<std::string_view const> paths,
                                              host_span<data_type const> desired_types,
                                              host_span<mutable_column_view const> statuses,
                                              cuda::stream_ref stream,
                                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::extract_variant_fields(variant_column, paths, desired_types, statuses, stream, mr);
}

}  // namespace io::parquet::experimental
}  // namespace cudf
