/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "io/json/nested_json.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/getenv_or.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/io/detail/tokenize_json.hpp>
#include <cudf/io/experimental/variant.hpp>
#include <cudf/io/experimental/variant_spec.hpp>
#include <cudf/io/json.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/atomic>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/cstring>
#include <cuda/std/limits>
#include <cuda/std/tuple>
#include <cuda/stream>
#include <math_constants.h>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/gather.h>
#include <thrust/logical.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

namespace cudf {
namespace io::parquet::experimental {
namespace {

using cudf::io::json::NodeIndexT;
using cudf::io::json::NodeT;
using cudf::io::json::SymbolOffsetT;
using cudf::io::json::TreeDepthT;
using op_status = variant_operation_status;

constexpr int block_size = 256;

// Rows are joined with this byte so the tokenizer can treat each row as a JSON line. Strict JSON
// rejects any raw control character before the end of the first value, so input bytes equal to the
// delimiter are rewritten to another control character, which keeps the per-row verdict unchanged.
constexpr char row_delimiter        = '\x1e';
constexpr char delimiter_substitute = '\x01';
// Each row is followed by a space and the delimiter: in recovery mode the tokenizer rejects a root
// scalar that ends directly at the delimiter
constexpr int row_padding = 2;

// Limits enforced by Spark's VariantBuilder (`VariantUtil.SIZE_LIMIT`) and by the default Jackson
// `StreamReadConstraints` that Spark's `parse_json` runs under.
constexpr int64_t variant_size_limit   = int64_t{128} * 1024 * 1024;
constexpr int max_nesting_depth        = 1000;
constexpr int max_number_length        = 1000;
constexpr int64_t max_string_length    = 20'000'000;
constexpr int64_t max_name_length      = 50'000;
constexpr int64_t max_short_string_len = 63;

// Deepest tree level the JSON tree builder can represent. Each object contributes two levels
// (the object and its field name), each array one.
constexpr int max_tree_level = cuda::std::numeric_limits<TreeDepthT>::max();

constexpr std::size_t default_batch_size = std::size_t{256} * 1024 * 1024;

CUDF_HOST_DEVICE constexpr uint8_t status_value(op_status s) { return static_cast<uint8_t>(s); }

__device__ constexpr bool is_json_whitespace(char c)
{
  return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

__device__ constexpr int integer_size(int64_t v)
{
  return v <= 0xFF ? 1 : (v <= 0xFFFF ? 2 : (v <= 0xFFFFFF ? 3 : 4));
}

__device__ inline void write_le(uint8_t* out, uint64_t v, int num_bytes)
{
  for (int i = 0; i < num_bytes; ++i) {
    out[i] = static_cast<uint8_t>(v >> (8 * i));
  }
}

__device__ constexpr int64_t array_size(int64_t num_elements, int64_t data_size)
{
  return 1 + (num_elements > 0xFF ? 4 : 1) + (num_elements + 1) * integer_size(data_size) +
         data_size;
}

__device__ constexpr int64_t object_size(int64_t num_fields, int64_t max_id, int64_t data_size)
{
  return 1 + (num_fields > 0xFF ? 4 : 1) + num_fields * integer_size(max_id) +
         (num_fields + 1) * integer_size(data_size) + data_size;
}

/**
 * @brief Finds the first position in `[begin, len)` that satisfies `pred`, cooperatively across a
 * warp. Every lane receives the result, which is `len` if no position matches.
 */
template <typename Predicate>
__device__ int64_t warp_find_first(int64_t begin, int64_t len, int lane, Predicate pred)
{
  for (auto base = begin; base < len; base += cudf::detail::warp_size) {
    auto const i     = base + lane;
    auto const found = __ballot_sync(0xffff'ffffu, i < len && pred(i));
    if (found != 0) { return base + __ffs(found) - 1; }
  }
  return len;
}

/**
 * @brief Determines whether the first JSON value of a row nests too deeply for the tokenizer.
 *
 * The warp scans the row 32 bytes at a time. Ballots locate escaped characters and string
 * boundaries so that brackets inside strings are ignored, and a warp scan tracks the nesting depth
 * and the tree builder's level count (lists once, objects twice) at every byte. Characters after
 * the first complete value are never parsed by Spark, so the scan stops there. Escapes are tracked
 * outside strings as well, which only changes the verdict for rows the tokenizer rejects anyway.
 *
 * @return INVALID_JSON if the value nests deeper than Jackson permits, UNSUPPORTED_INPUT if it
 *         exceeds the tree builder's level range, SUCCESS otherwise. Every lane gets the result.
 */
__device__ op_status first_value_depth_status(char const* row, int64_t len, int lane)
{
  auto constexpr full_mask = 0xffff'ffffu;
  auto constexpr warp_size = cudf::detail::warp_size;

  auto const first =
    warp_find_first(0, len, lane, [&](int64_t i) { return !is_json_whitespace(row[i]); });
  if (first == len || (row[first] != '[' && row[first] != '{')) { return op_status::SUCCESS; }

  auto const lanes_below = (1u << lane) - 1;
  int depth              = 0;      // containers open before this chunk
  int level              = 0;      // tree levels open before this chunk
  bool in_string         = false;  // whether this chunk starts inside a string
  bool odd_backslashes   = false;  // whether the previous chunk ends in an odd backslash run
  bool exceeds_level     = false;
  for (auto base = first; base < len; base += warp_size) {
    auto const i = base + lane;
    auto const c = i < len ? row[i] : ' ';

    // A character is escaped when the run of backslashes right before it has odd length
    auto const backslashes = __ballot_sync(full_mask, c == '\\');
    auto const others      = ~backslashes & lanes_below;
    auto const escaped     = others != 0 ? ((lane - (32 - __clz(static_cast<int>(others)))) & 1)
                                         : ((lane & 1) != odd_backslashes);
    auto const quotes      = __ballot_sync(full_mask, c == '"' && !escaped);
    auto const quoted      = in_string != (__popc(quotes & lanes_below) & 1);

    int depth_delta = 0;
    int level_delta = 0;
    if (!quoted) {
      switch (c) {
        case '[': depth_delta = 1, level_delta = 1; break;
        case '{': depth_delta = 1, level_delta = 2; break;
        case ']': depth_delta = -1, level_delta = -1; break;
        case '}': depth_delta = -1, level_delta = -2; break;
        default: break;
      }
    }
    for (int offset = 1; offset < warp_size; offset *= 2) {
      auto const d = __shfl_up_sync(full_mask, depth_delta, offset);
      auto const l = __shfl_up_sync(full_mask, level_delta, offset);
      if (lane >= offset) {
        depth_delta += d;
        level_delta += l;
      }
    }
    auto const current_depth = depth + depth_delta;
    auto const current_level = level + level_delta;

    // The first value ends where its depth returns to zero
    auto const closed = __ballot_sync(full_mask, current_depth <= 0);
    auto const last   = closed != 0 ? __ffs(closed) - 1 : warp_size - 1;
    auto max_depth    = lane <= last ? current_depth : 0;
    auto max_level    = lane <= last ? current_level : 0;
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
      max_depth = cuda::std::max(max_depth, __shfl_xor_sync(full_mask, max_depth, offset));
      max_level = cuda::std::max(max_level, __shfl_xor_sync(full_mask, max_level, offset));
    }
    if (max_depth > max_nesting_depth) { return op_status::INVALID_JSON; }
    exceeds_level |= max_level > max_tree_level;
    if (closed != 0) { break; }

    depth     = __shfl_sync(full_mask, current_depth, warp_size - 1);
    level     = __shfl_sync(full_mask, current_level, warp_size - 1);
    in_string = in_string != (__popc(quotes) & 1);
    // A chunk of only backslashes extends the run by an even length, keeping its parity
    if (backslashes != full_mask) { odd_backslashes = __clz(static_cast<int>(~backslashes)) & 1; }
  }
  return exceeds_level ? op_status::UNSUPPORTED_INPUT : op_status::SUCCESS;
}

/**
 * @brief Finds where the first value of a row ends when that value is a scalar.
 *
 * Spark parses only the first value of a document and ignores what follows it. The tokenizer does
 * not handle content after a root scalar reliably, so the caller blanks it. A string ends at its
 * closing quote. A number or literal ends at the first whitespace; anything else directly after it
 * is left in place so that the tokenizer (or the root-literal check) can judge it as Jackson does.
 *
 * @return The length of the row prefix to keep, or `len` if the first value is a container
 */
__device__ int64_t scalar_root_end(char const* row, int64_t len, int lane)
{
  auto const first =
    warp_find_first(0, len, lane, [&](int64_t i) { return !is_json_whitespace(row[i]); });
  if (first == len || row[first] == '{' || row[first] == '[') { return len; }
  if (row[first] == '"') {
    auto const close = warp_find_first(first + 1, len, lane, [&](int64_t i) {
      if (row[i] != '"') { return false; }
      int64_t backslashes = 0;
      while (i - 1 - backslashes > first && row[i - 1 - backslashes] == '\\') {
        ++backslashes;
      }
      return backslashes % 2 == 0;
    });
    return close == len ? len : close + 1;
  }
  return warp_find_first(
    first + 1, len, lane, [&](int64_t i) { return is_json_whitespace(row[i]); });
}

/**
 * @brief Copies each row into the tokenizer buffer, followed by the row padding.
 *
 * One warp per row. Null rows and rows that nest too deeply are blanked so the tokenizer sees an
 * empty line; their status records why. Content after a root scalar is blanked as well.
 */
CUDF_KERNEL __launch_bounds__(block_size) void prepare_rows_kernel(
  char const* chars,
  cudf::detail::input_offsetalator offsets,
  bitmask_type const* null_mask,
  size_type mask_offset,
  size_type num_rows,
  char* buffer,
  uint8_t* row_status)
{
  auto constexpr warp_size = cudf::detail::warp_size;
  auto const lane          = static_cast<int>(threadIdx.x % warp_size);
  auto const num_warps     = cudf::detail::grid_1d::grid_stride<block_size>() / warp_size;
  auto const base          = offsets[0];
  for (auto row = cudf::detail::grid_1d::global_thread_id<block_size>() / warp_size; row < num_rows;
       row += num_warps) {
    auto const begin   = offsets[row];
    auto const len     = offsets[row + 1] - begin;
    auto const out     = buffer + 1 + (begin - base) + int64_t{row} * row_padding;
    auto const is_null = null_mask != nullptr && !bit_is_set(null_mask, row + mask_offset);
    auto const keep    = is_null ? 0 : scalar_root_end(chars + begin, len, lane);

    int64_t weight = 0;
    for (int64_t i = lane; i < len; i += warp_size) {
      auto c = i < keep ? chars[begin + i] : ' ';
      if (c == row_delimiter) { c = delimiter_substitute; }
      out[i] = c;
      weight += (c == '[') + 2 * (c == '{');
    }
    for (int i = warp_size / 2; i > 0; i /= 2) {
      weight += __shfl_down_sync(0xffff'ffffu, weight, i);
    }
    weight = __shfl_sync(0xffff'ffffu, weight, 0);

    auto status = is_null ? op_status::ROW_NULL : op_status::SUCCESS;
    if (!is_null && weight > max_tree_level) {
      __syncwarp();
      status = first_value_depth_status(out, len, lane);
      if (status != op_status::SUCCESS) {
        for (int64_t i = lane; i < len; i += warp_size) {
          out[i] = ' ';
        }
      }
    }
    if (lane == 0) {
      out[len]        = ' ';
      out[len + 1]    = row_delimiter;
      row_status[row] = status_value(status);
    }
  }
}

struct string_stats {
  int64_t utf8_bytes;
  int64_t utf16_units;
};

__device__ inline int hex_digit(char c)
{
  if (c >= '0' && c <= '9') { return c - '0'; }
  if (c >= 'a' && c <= 'f') { return c - 'a' + 10; }
  return c - 'A' + 10;
}

__device__ inline uint32_t read_hex4(char const* s)
{
  return (hex_digit(s[0]) << 12) | (hex_digit(s[1]) << 8) | (hex_digit(s[2]) << 4) |
         hex_digit(s[3]);
}

/**
 * @brief Decodes the escapes of a validated JSON string body into UTF-8.
 *
 * Matches Java semantics: a surrogate escape that is not part of a valid pair becomes '?' when the
 * decoded `String` is encoded as UTF-8.
 *
 * @tparam write Whether to write the decoded bytes to `out`
 */
template <bool write>
__device__ string_stats decode_json_string(char const* begin, char const* end, uint8_t* out)
{
  int64_t num_bytes = 0;
  int64_t num_units = 0;
  auto emit         = [&](uint8_t b) {
    if constexpr (write) { out[num_bytes] = b; }
    ++num_bytes;
  };
  auto s = begin;
  while (s < end) {
    auto const c = static_cast<uint8_t>(*s);
    if (c != '\\') {
      emit(c);
      if ((c & 0xC0) != 0x80) { num_units += ((c & 0xF8) == 0xF0) ? 2 : 1; }
      ++s;
      continue;
    }
    auto const esc = s[1];
    s += 2;
    ++num_units;
    switch (esc) {
      case 'b': emit('\b'); break;
      case 'f': emit('\f'); break;
      case 'n': emit('\n'); break;
      case 'r': emit('\r'); break;
      case 't': emit('\t'); break;
      case 'u': {
        uint32_t cp = read_hex4(s);
        s += 4;
        if (cp >= 0xD800 && cp <= 0xDFFF) {
          bool paired = false;
          if (cp <= 0xDBFF && end - s >= 6 && s[0] == '\\' && s[1] == 'u') {
            auto const low = read_hex4(s + 2);
            if (low >= 0xDC00 && low <= 0xDFFF) {
              cp     = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
              paired = true;
              s += 6;
              ++num_units;
            }
          }
          if (!paired) {
            emit('?');
            break;
          }
        }
        if (cp < 0x80) {
          emit(static_cast<uint8_t>(cp));
        } else if (cp < 0x800) {
          emit(0xC0 | (cp >> 6));
          emit(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
          emit(0xE0 | (cp >> 12));
          emit(0x80 | ((cp >> 6) & 0x3F));
          emit(0x80 | (cp & 0x3F));
        } else {
          emit(0xF0 | (cp >> 18));
          emit(0x80 | ((cp >> 12) & 0x3F));
          emit(0x80 | ((cp >> 6) & 0x3F));
          emit(0x80 | (cp & 0x3F));
        }
        break;
      }
      default: emit(static_cast<uint8_t>(esc)); break;
    }
  }
  return {num_bytes, num_units};
}

/**
 * @brief Iterates over the UTF-16 code units of a UTF-8 string.
 */
struct utf16_reader {
  uint8_t const* data;
  int64_t size;
  int64_t pos      = 0;
  int32_t low_unit = -1;

  /// Returns the next code unit, or -1 at the end of the string
  __device__ int32_t next()
  {
    if (low_unit >= 0) {
      auto const unit = low_unit;
      low_unit        = -1;
      return unit;
    }
    if (pos >= size) { return -1; }
    uint32_t const c = data[pos];
    uint32_t cp;
    if (c < 0x80) {
      cp = c;
      pos += 1;
    } else if (c < 0xE0) {
      cp = ((c & 0x1F) << 6) | (data[pos + 1] & 0x3F);
      pos += 2;
    } else if (c < 0xF0) {
      cp = ((c & 0x0F) << 12) | ((data[pos + 1] & 0x3F) << 6) | (data[pos + 2] & 0x3F);
      pos += 3;
    } else {
      cp = ((c & 0x07) << 18) | ((data[pos + 1] & 0x3F) << 12) | ((data[pos + 2] & 0x3F) << 6) |
           (data[pos + 3] & 0x3F);
      pos += 4;
    }
    if (cp < 0x10000) { return static_cast<int32_t>(cp); }
    cp -= 0x10000;
    low_unit = static_cast<int32_t>(0xDC00 + (cp & 0x3FF));
    return static_cast<int32_t>(0xD800 + (cp >> 10));
  }
};

/**
 * @brief Three-way comparison in Java `String.compareTo` order (UTF-16 code units).
 */
__device__ int utf16_compare(uint8_t const* a, int64_t a_size, uint8_t const* b, int64_t b_size)
{
  int64_t i = 0;
  while (i < a_size && i < b_size && a[i] == b[i]) {
    ++i;
  }
  if (i == a_size || i == b_size) { return (a_size > i) - (b_size > i); }
  // Restart at the code point containing the first differing byte
  while (i > 0 && (a[i] & 0xC0) == 0x80) {
    --i;
  }
  utf16_reader ra{a + i, a_size - i};
  utf16_reader rb{b + i, b_size - i};
  while (true) {
    auto const ua = ra.next();
    auto const ub = rb.next();
    if (ua != ub) { return ua < ub ? -1 : 1; }
    if (ua < 0) { return 0; }
  }
}

constexpr uint64_t long_key_code = cuda::std::numeric_limits<uint64_t>::max();

/**
 * @brief Encodes a key of at most 8 bytes as an integer that orders like `utf16_compare`.
 *
 * UTF-8 byte order matches UTF-16 order once the leads of U+E000-U+FFFF move above the
 * supplementary leads. Zero padding requires keys without NUL bytes, and lead 0xED is excluded
 * because lone surrogates do not follow this order. Other keys return `long_key_code`.
 */
__device__ uint64_t short_key_code(uint8_t const* key, int64_t len)
{
  if (len > 8) { return long_key_code; }
  uint64_t code = 0;
  for (int64_t i = 0; i < 8; ++i) {
    uint8_t b = 0;
    if (i < len) {
      b = key[i];
      if (b == 0 || b == 0xED) { return long_key_code; }
      if (b == 0xEE || b == 0xEF) { b += 0xF5 - 0xEE; }
    }
    code = (code << 8) | b;
  }
  return code;
}

/// Lexical components of a JSON number that passed strict validation
struct number_parts {
  bool negative;
  char const* int_digits;
  int int_len;
  bool has_fraction;
  char const* frac_digits;
  int frac_len;
  bool has_exponent;
  int exp_len;
  int64_t exponent;  // saturated to +/- 2^40
};

__device__ number_parts split_number(char const* s, int len)
{
  number_parts p{};
  int i      = 0;
  p.negative = s[0] == '-';
  if (p.negative) { ++i; }
  p.int_digits = s + i;
  while (i < len && s[i] >= '0' && s[i] <= '9') {
    ++i;
    ++p.int_len;
  }
  if (i < len && s[i] == '.') {
    p.has_fraction = true;
    ++i;
    p.frac_digits = s + i;
    while (i < len && s[i] >= '0' && s[i] <= '9') {
      ++i;
      ++p.frac_len;
    }
  }
  if (i < len && (s[i] == 'e' || s[i] == 'E')) {
    p.has_exponent = true;
    ++i;
    bool exp_negative = false;
    if (s[i] == '+' || s[i] == '-') {
      exp_negative = s[i] == '-';
      ++i;
    }
    int64_t e = 0;
    while (i < len) {
      e = cuda::std::min<int64_t>(e * 10 + (s[i] - '0'), int64_t{1} << 40);
      ++i;
      ++p.exp_len;
    }
    p.exponent = exp_negative ? -e : e;
  }
  return p;
}

enum class scalar_kind : uint8_t {
  INVALID,
  NULL_VALUE,
  TRUE_VALUE,
  FALSE_VALUE,
  INT,
  DECIMAL,
  DOUBLE
};

struct scalar_class {
  scalar_kind kind;
  int width;  // payload bytes for INT and DECIMAL
  int scale;
  __int128_t unscaled;  // integer value for INT, unscaled value for DECIMAL
};

__device__ inline bool matches(char const* s, int len, char const* literal, int literal_len)
{
  if (len != literal_len) { return false; }
  for (int i = 0; i < len; ++i) {
    if (s[i] != literal[i]) { return false; }
  }
  return true;
}

/**
 * @brief Classifies a validated non-string JSON scalar the way Spark's VariantBuilder does.
 *
 * Integers that fit in int64 use the narrowest integer type. Otherwise, numbers written without an
 * exponent whose precision and scale are at most 38 become decimals; everything else is a double.
 */
__device__ scalar_class classify_scalar(char const* s, int len)
{
  if (matches(s, len, "null", 4)) { return {scalar_kind::NULL_VALUE, 0, 0, 0}; }
  if (matches(s, len, "true", 4)) { return {scalar_kind::TRUE_VALUE, 0, 0, 0}; }
  if (matches(s, len, "false", 5)) { return {scalar_kind::FALSE_VALUE, 0, 0, 0}; }
  if (len == 0 || !(s[0] == '-' || (s[0] >= '0' && s[0] <= '9'))) {
    return {scalar_kind::INVALID, 0, 0, 0};
  }
  auto const p = split_number(s, len);
  if (!p.has_fraction && !p.has_exponent && p.int_len <= 19) {
    uint64_t magnitude = 0;
    bool overflow      = false;
    for (int i = 0; i < p.int_len; ++i) {
      auto const d = static_cast<uint64_t>(p.int_digits[i] - '0');
      if (magnitude > (cuda::std::numeric_limits<uint64_t>::max() - d) / 10) { overflow = true; }
      magnitude = magnitude * 10 + d;
    }
    auto const limit = static_cast<uint64_t>(cuda::std::numeric_limits<int64_t>::max()) +
                       static_cast<uint64_t>(p.negative);
    if (!overflow && magnitude <= limit) {
      auto const v =
        p.negative ? static_cast<int64_t>(0 - magnitude) : static_cast<int64_t>(magnitude);
      auto const width = (v == static_cast<int8_t>(v))    ? 1
                         : (v == static_cast<int16_t>(v)) ? 2
                         : (v == static_cast<int32_t>(v)) ? 4
                                                          : 8;
      return {scalar_kind::INT, width, 0, v};
    }
  }
  if (!p.has_exponent) {
    // BigDecimal precision counts digits of the unscaled value without leading zeros (minimum 1)
    int total       = p.int_len + p.frac_len;
    int first_digit = total;
    for (int i = 0; i < total; ++i) {
      auto const c = i < p.int_len ? p.int_digits[i] : p.frac_digits[i - p.int_len];
      if (c != '0') {
        first_digit = i;
        break;
      }
    }
    auto const precision = cuda::std::max(1, total - first_digit);
    auto const scale     = p.frac_len;
    if (precision <= 38 && scale <= 38) {
      __int128_t unscaled = 0;
      for (int i = first_digit; i < total; ++i) {
        auto const c = i < p.int_len ? p.int_digits[i] : p.frac_digits[i - p.int_len];
        unscaled     = unscaled * 10 + (c - '0');
      }
      if (p.negative) { unscaled = -unscaled; }
      auto const width = (scale <= 9 && precision <= 9)     ? 4
                         : (scale <= 18 && precision <= 18) ? 8
                                                            : 16;
      return {scalar_kind::DECIMAL, width, scale, unscaled};
    }
  }
  return {scalar_kind::DOUBLE, 8, 0, 0};
}

__device__ constexpr int64_t scalar_encoded_size(scalar_class const& c)
{
  switch (c.kind) {
    case scalar_kind::INT: return 1 + c.width;
    case scalar_kind::DECIMAL: return 2 + c.width;
    case scalar_kind::DOUBLE: return 9;
    default: return 1;
  }
}

/**
 * @brief Checks Jackson's number-length constraint.
 *
 * Jackson's slow path, taken when a number starts with '0' or runs to the end of the input, counts
 * a missing fraction or exponent as -1 instead of 0.
 */
__device__ bool number_length_ok(number_parts const& p, bool at_input_end)
{
  if (!p.has_fraction && !p.has_exponent) { return p.int_len <= max_number_length; }
  auto const slow_path = at_input_end || p.int_digits[0] == '0';
  auto const absent    = slow_path ? -1 : 0;
  auto const length =
    p.int_len + (p.has_fraction ? p.frac_len : absent) + (p.has_exponent ? p.exp_len : absent);
  return length <= max_number_length;
}

/**
 * @brief Fixed-capacity unsigned big integer used to settle double rounding exactly.
 */
struct bigint {
  static constexpr int capacity = 112;
  uint32_t limbs[capacity];
  int size = 0;

  __device__ void set(uint64_t v)
  {
    size = 0;
    while (v != 0) {
      limbs[size++] = static_cast<uint32_t>(v);
      v >>= 32;
    }
  }

  __device__ void mul_add(uint32_t m, uint32_t a)
  {
    uint64_t carry = a;
    for (int i = 0; i < size; ++i) {
      auto const prod = static_cast<uint64_t>(limbs[i]) * m + carry;
      limbs[i]        = static_cast<uint32_t>(prod);
      carry           = prod >> 32;
    }
    if (carry != 0) { limbs[size++] = static_cast<uint32_t>(carry); }
  }

  __device__ void mul_pow5(int e)
  {
    constexpr uint32_t pow5_13 = 1'220'703'125u;
    while (e >= 13) {
      mul_add(pow5_13, 0);
      e -= 13;
    }
    uint32_t m = 1;
    while (e-- > 0) {
      m *= 5;
    }
    if (m != 1) { mul_add(m, 0); }
  }

  [[nodiscard]] __device__ int64_t bit_length() const
  {
    if (size == 0) { return 0; }
    return int64_t{32} * (size - 1) + (32 - __clz(static_cast<int>(limbs[size - 1])));
  }

  __device__ void shift_left(int64_t bits)
  {
    if (size == 0 || bits == 0) { return; }
    auto const words    = static_cast<int>(bits / 32);
    auto const rem      = static_cast<int>(bits % 32);
    int new_size        = size + words + 1;
    limbs[new_size - 1] = 0;
    for (int i = size - 1; i >= 0; --i) {
      auto const v = limbs[i];
      if (rem != 0) {
        limbs[i + words + 1] |= v >> (32 - rem);
        limbs[i + words] = v << rem;
      } else {
        limbs[i + words] = v;
      }
    }
    for (int i = 0; i < words; ++i) {
      limbs[i] = 0;
    }
    while (new_size > 0 && limbs[new_size - 1] == 0) {
      --new_size;
    }
    size = new_size;
  }
};

__device__ int compare(bigint const& a, bigint const& b)
{
  if (a.size != b.size) { return a.size < b.size ? -1 : 1; }
  for (int i = a.size - 1; i >= 0; --i) {
    if (a.limbs[i] != b.limbs[i]) { return a.limbs[i] < b.limbs[i] ? -1 : 1; }
  }
  return 0;
}

/// Compares `a * 2^a_shift` with `b * 2^b_shift`; both shifts are non-negative
__device__ int compare_shifted(bigint& a, int64_t a_shift, bigint& b, int64_t b_shift)
{
  auto const a_bits = a.size == 0 ? 0 : a.bit_length() + a_shift;
  auto const b_bits = b.size == 0 ? 0 : b.bit_length() + b_shift;
  if (a_bits != b_bits) { return a_bits < b_bits ? -1 : 1; }
  auto const common = cuda::std::min(a_shift, b_shift);
  a.shift_left(a_shift - common);
  b.shift_left(b_shift - common);
  return compare(a, b);
}

// Significant digits beyond this count cannot change which double a decimal rounds to, as long as
// a sticky flag records whether any of the dropped digits were nonzero.
constexpr int max_significant_digits = 800;

struct decimal_digits {
  number_parts const* parts;
  int first;  // index of the first nonzero digit
  int count;  // number of significant digits kept
  bool sticky;
  int64_t exponent;  // value = digits * 10^exponent

  [[nodiscard]] __device__ char digit(int i) const
  {
    auto const idx = first + i;
    return idx < parts->int_len ? parts->int_digits[idx] : parts->frac_digits[idx - parts->int_len];
  }
};

/// Sign of (digits * 10^exponent) - (m * 2^k)
__device__ int compare_decimal_to_binary(decimal_digits const& d, uint64_t m, int64_t k)
{
  bigint lhs;
  bigint rhs;
  lhs.set(0);
  for (int i = 0; i < d.count;) {
    auto const chunk = cuda::std::min(9, d.count - i);
    uint32_t value   = 0;
    uint32_t scale   = 1;
    for (int j = 0; j < chunk; ++j) {
      value = value * 10 + (d.digit(i + j) - '0');
      scale *= 10;
    }
    if (lhs.size == 0) {
      lhs.set(value);
    } else {
      lhs.mul_add(scale, value);
    }
    i += chunk;
  }
  rhs.set(m);
  int64_t lhs_shift = 0;
  int64_t rhs_shift = 0;
  if (d.exponent >= 0) {
    lhs.mul_pow5(static_cast<int>(d.exponent));
    lhs_shift = d.exponent;
  } else {
    rhs.mul_pow5(static_cast<int>(-d.exponent));
    rhs_shift = -d.exponent;
  }
  rhs_shift += k;
  auto const common = cuda::std::min(lhs_shift, rhs_shift);
  auto const c      = compare_shifted(lhs, lhs_shift - common, rhs, rhs_shift - common);
  return (c == 0 && d.sticky) ? 1 : c;
}

__device__ double pow10_exact(int e)
{
  constexpr double table[] = {1e0,  1e1,  1e2,  1e3,  1e4,  1e5,  1e6,  1e7,
                              1e8,  1e9,  1e10, 1e11, 1e12, 1e13, 1e14, 1e15,
                              1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22};
  return table[e];
}

/**
 * @brief Parses a validated JSON number into the nearest double, rounding half to even.
 *
 * Matches Java's `Double.parseDouble`. Small inputs take Clinger's exact fast path; all others
 * start from an estimate and step one ulp at a time, deciding each step with exact big-integer
 * comparisons against the rounding midpoints.
 */
__device__ double parse_double(char const* s, int len)
{
  auto const p     = split_number(s, len);
  auto const neg   = p.negative;
  auto const total = p.int_len + p.frac_len;
  auto digit_at    = [&](int i) {
    return i < p.int_len ? p.int_digits[i] : p.frac_digits[i - p.int_len];
  };
  int first = 0;
  while (first < total && digit_at(first) == '0') {
    ++first;
  }
  if (first == total) { return neg ? -0.0 : 0.0; }
  int last = total - 1;
  while (digit_at(last) == '0') {
    --last;
  }
  auto const num_digits = last - first + 1;
  int64_t exponent      = p.exponent - p.frac_len + (total - 1 - last);

  if (num_digits + exponent > 310) { return neg ? -CUDART_INF : CUDART_INF; }
  if (num_digits + exponent < -324) { return neg ? -0.0 : 0.0; }

  uint64_t leading       = 0;
  auto const num_leading = cuda::std::min(num_digits, 19);
  for (int i = 0; i < num_leading; ++i) {
    leading = leading * 10 + (digit_at(first + i) - '0');
  }
  if (num_digits <= 15 && exponent >= -22 && exponent <= 22) {
    auto const w = static_cast<double>(leading);
    auto const r = exponent >= 0 ? w * pow10_exact(static_cast<int>(exponent))
                                 : w / pow10_exact(static_cast<int>(-exponent));
    return neg ? -r : r;
  }

  decimal_digits d{&p, first, num_digits, false, exponent};
  if (num_digits > max_significant_digits) {
    d.count  = max_significant_digits;
    d.sticky = true;
    d.exponent += num_digits - max_significant_digits;
  }

  auto const est_exp = static_cast<int>(exponent + (num_digits - num_leading));
  auto const half    = est_exp / 2;
  double x           = static_cast<double>(leading) * exp10(static_cast<double>(half)) *
             exp10(static_cast<double>(est_exp - half));
  if (!(x < CUDART_INF)) { x = cuda::std::numeric_limits<double>::max(); }

  for (int iter = 0; iter < 64; ++iter) {
    auto const bits = __double_as_longlong(x);
    if (bits == 0x7FF0'0000'0000'0000LL) { break; }
    auto const exp_bits = static_cast<int>((bits >> 52) & 0x7FF);
    auto const frac     = static_cast<uint64_t>(bits) & ((uint64_t{1} << 52) - 1);
    uint64_t m          = exp_bits == 0 ? frac : (frac | (uint64_t{1} << 52));
    int64_t e           = exp_bits == 0 ? -1074 : exp_bits - 1075;
    bool const m_odd    = (m & 1) != 0;
    // Midpoint between x and the next double up: (2m + 1) * 2^(e - 1)
    auto const up = compare_decimal_to_binary(d, 2 * m + 1, e - 1);
    if (up > 0 || (up == 0 && m_odd)) {
      x = __longlong_as_double(bits + 1);
      continue;
    }
    if (m == 0) { break; }
    // Midpoint between x and the next double down; below a power of two the spacing halves
    auto const down = (frac == 0 && exp_bits > 1) ? compare_decimal_to_binary(d, 4 * m - 1, e - 2)
                                                  : compare_decimal_to_binary(d, 2 * m - 1, e - 1);
    if (down < 0 || (down == 0 && m_odd)) {
      x = __longlong_as_double(bits - 1);
      continue;
    }
    break;
  }
  return neg ? -x : x;
}

/// Per-node data derived from the node's text
struct node_text_view {
  char const* buffer;
  NodeT const* categories;
  TreeDepthT const* levels;
  SymbolOffsetT const* range_begin;
  SymbolOffsetT const* range_end;
  size_type const* rows;
  SymbolOffsetT const* row_starts;
};

/**
 * @brief Computes the encoded size of each leaf, the decoded length of each field name, and the
 * validations that the tokenizer does not perform.
 */
CUDF_KERNEL __launch_bounds__(block_size) void classify_nodes_kernel(node_text_view nodes,
                                                                     NodeIndexT num_nodes,
                                                                     int64_t* node_size,
                                                                     int32_t* key_length,
                                                                     uint8_t* row_invalid)
{
  auto const stride = cudf::detail::grid_1d::grid_stride<block_size>();
  for (auto n = cudf::detail::grid_1d::global_thread_id<block_size>(); n < num_nodes; n += stride) {
    auto const begin = nodes.range_begin[n];
    auto const end   = nodes.range_end[n];
    auto const text  = nodes.buffer + begin;
    bool invalid     = false;
    int64_t size     = 0;
    switch (nodes.categories[n]) {
      case cudf::io::json::NC_STR: {
        auto const stats = decode_json_string<false>(text + 1, nodes.buffer + end - 1, nullptr);
        invalid          = stats.utf16_units > max_string_length;
        size             = 1 + (stats.utf8_bytes > max_short_string_len ? 4 : 0) + stats.utf8_bytes;
        break;
      }
      case cudf::io::json::NC_FN: {
        auto const stats = decode_json_string<false>(text, nodes.buffer + end, nullptr);
        invalid          = stats.utf16_units > max_name_length;
        key_length[n]    = static_cast<int32_t>(stats.utf8_bytes);
        break;
      }
      case cudf::io::json::NC_VAL: {
        auto const len = static_cast<int>(end - begin);
        auto const cls = classify_scalar(text, len);
        invalid        = cls.kind == scalar_kind::INVALID;
        if (cls.kind == scalar_kind::INT || cls.kind == scalar_kind::DECIMAL ||
            cls.kind == scalar_kind::DOUBLE) {
          auto const at_input_end = end + row_padding == nodes.row_starts[nodes.rows[n] + 1];
          auto const parts        = split_number(text, len);
          // The tokenizer accepts a decimal point followed directly by an exponent; Jackson does
          // not
          invalid |= parts.has_fraction && parts.frac_len == 0;
          invalid |= !number_length_ok(parts, at_input_end);
          // Jackson requires whitespace (or the end of input) after a root-level number
          invalid |= nodes.levels[n] == 0 && !is_json_whitespace(nodes.buffer[end]);
        }
        size = scalar_encoded_size(cls);
        break;
      }
      case cudf::io::json::NC_STRUCT:
      case cudf::io::json::NC_LIST: {
        // The tokenizer accepts a trailing comma before the closing bracket; Jackson does not
        auto i = static_cast<int64_t>(end) - 2;
        while (is_json_whitespace(nodes.buffer[i])) {
          --i;
        }
        invalid = nodes.buffer[i] == ',';
        break;
      }
      default: invalid = true;
    }
    node_size[n] = size;
    if (invalid) { row_invalid[nodes.rows[n]] = 1; }
  }
}

CUDF_KERNEL __launch_bounds__(block_size) void decode_keys_kernel(char const* buffer,
                                                                  NodeIndexT const* fn_nodes,
                                                                  SymbolOffsetT const* range_begin,
                                                                  SymbolOffsetT const* range_end,
                                                                  int64_t const* key_offsets,
                                                                  size_type num_keys,
                                                                  uint8_t* key_chars)
{
  auto const stride = cudf::detail::grid_1d::grid_stride<block_size>();
  for (auto i = cudf::detail::grid_1d::global_thread_id<block_size>(); i < num_keys; i += stride) {
    auto const n = fn_nodes[i];
    decode_json_string<true>(
      buffer + range_begin[n], buffer + range_end[n], key_chars + key_offsets[i]);
  }
}

/// Tree-shaped inputs and per-node encoding state shared by the size and write kernels
struct encode_state {
  NodeT const* categories;
  NodeIndexT const* parents;
  uint8_t* kept;           // field names only: whether this field survives deduplication
  int32_t const* dict_id;  // field names only
  int32_t* child_count;    // arrays: elements; objects: kept fields
  int32_t* max_id;         // objects: largest field id among their fields
  int64_t* data_size;      // containers and field names: total size of their encoded children
  int64_t* node_size;      // full encoded size of the node
  int64_t* rel_offset;     // offset of a member within its parent's data section
  int32_t* member_index;   // index of a member within its parent's offset list
  int64_t* position;       // absolute position within the value buffer
  uint8_t* live;           // whether the node is written
};

CUDF_KERNEL __launch_bounds__(block_size) void count_children_kernel(encode_state st,
                                                                     NodeIndexT num_nodes)
{
  auto const stride = cudf::detail::grid_1d::grid_stride<block_size>();
  for (auto n = cudf::detail::grid_1d::global_thread_id<block_size>(); n < num_nodes; n += stride) {
    auto const p = st.parents[n];
    if (p < 0) { continue; }
    auto const parent_category = st.categories[p];
    if (parent_category == cudf::io::json::NC_LIST) {
      cuda::atomic_ref<int32_t, cuda::thread_scope_device>{st.child_count[p]}.fetch_add(
        1, cuda::std::memory_order_relaxed);
    } else if (parent_category == cudf::io::json::NC_STRUCT) {
      if (st.kept[n]) {
        cuda::atomic_ref<int32_t, cuda::thread_scope_device>{st.child_count[p]}.fetch_add(
          1, cuda::std::memory_order_relaxed);
      }
      cuda::atomic_ref<int32_t, cuda::thread_scope_device>{st.max_id[p]}.fetch_max(
        st.dict_id[n], cuda::std::memory_order_relaxed);
    }
  }
}

/// Finalizes the sizes of one tree level and adds them to the parents' data sizes
CUDF_KERNEL __launch_bounds__(block_size) void size_level_kernel(encode_state st,
                                                                 NodeIndexT const* level_nodes,
                                                                 NodeIndexT count)
{
  auto const stride = cudf::detail::grid_1d::grid_stride<block_size>();
  for (auto i = cudf::detail::grid_1d::global_thread_id<block_size>(); i < count; i += stride) {
    auto const n = level_nodes[i];
    int64_t size;
    switch (st.categories[n]) {
      case cudf::io::json::NC_LIST: size = array_size(st.child_count[n], st.data_size[n]); break;
      case cudf::io::json::NC_STRUCT:
        size = object_size(st.child_count[n], st.max_id[n], st.data_size[n]);
        break;
      case cudf::io::json::NC_FN: size = st.data_size[n]; break;
      default: size = st.node_size[n];
    }
    st.node_size[n] = size;
    auto const p    = st.parents[n];
    if (p >= 0 && (st.categories[p] != cudf::io::json::NC_STRUCT || st.kept[n])) {
      cuda::atomic_ref<int64_t, cuda::thread_scope_device>{st.data_size[p]}.fetch_add(
        size, cuda::std::memory_order_relaxed);
    }
  }
}

/// Places the nodes of one tree level below their (already placed) parents
CUDF_KERNEL __launch_bounds__(block_size) void place_level_kernel(encode_state st,
                                                                  NodeIndexT const* level_nodes,
                                                                  NodeIndexT count)
{
  auto const stride = cudf::detail::grid_1d::grid_stride<block_size>();
  for (auto i = cudf::detail::grid_1d::global_thread_id<block_size>(); i < count; i += stride) {
    auto const n = level_nodes[i];
    auto const p = st.parents[n];
    if (st.categories[p] == cudf::io::json::NC_FN) {
      st.position[n] = st.position[p];
      st.live[n]     = st.live[p];
    } else {
      auto const header = st.node_size[p] - st.data_size[p];
      st.position[n]    = st.position[p] + header + st.rel_offset[n];
      st.live[n] = st.live[p] && (st.categories[p] != cudf::io::json::NC_STRUCT || st.kept[n]);
    }
  }
}

__device__ void write_scalar(char const* text, int len, uint8_t* out)
{
  auto const cls = classify_scalar(text, len);
  switch (cls.kind) {
    case scalar_kind::NULL_VALUE: out[0] = 0; break;
    case scalar_kind::TRUE_VALUE:
      out[0] = static_cast<uint8_t>(variant_primitive_type::BOOLEAN_TRUE) << 2;
      break;
    case scalar_kind::FALSE_VALUE:
      out[0] = static_cast<uint8_t>(variant_primitive_type::BOOLEAN_FALSE) << 2;
      break;
    case scalar_kind::INT: {
      auto const type = cls.width == 1   ? variant_primitive_type::INT8
                        : cls.width == 2 ? variant_primitive_type::INT16
                        : cls.width == 4 ? variant_primitive_type::INT32
                                         : variant_primitive_type::INT64;
      out[0]          = static_cast<uint8_t>(type) << 2;
      write_le(out + 1, static_cast<uint64_t>(static_cast<int64_t>(cls.unscaled)), cls.width);
      break;
    }
    case scalar_kind::DECIMAL: {
      auto const type = cls.width == 4   ? variant_primitive_type::DECIMAL4
                        : cls.width == 8 ? variant_primitive_type::DECIMAL8
                                         : variant_primitive_type::DECIMAL16;
      out[0]          = static_cast<uint8_t>(type) << 2;
      out[1]          = static_cast<uint8_t>(cls.scale);
      auto const v    = static_cast<__uint128_t>(cls.unscaled);
      write_le(out + 2, static_cast<uint64_t>(v), cuda::std::min(cls.width, 8));
      if (cls.width == 16) { write_le(out + 10, static_cast<uint64_t>(v >> 64), 8); }
      break;
    }
    case scalar_kind::DOUBLE: {
      out[0] = static_cast<uint8_t>(variant_primitive_type::FLOAT64) << 2;
      write_le(out + 1, static_cast<uint64_t>(__double_as_longlong(parse_double(text, len))), 8);
      break;
    }
    default: break;
  }
}

CUDF_KERNEL __launch_bounds__(block_size) void write_values_kernel(encode_state st,
                                                                   node_text_view nodes,
                                                                   NodeIndexT num_nodes,
                                                                   uint8_t* values)
{
  auto const stride = cudf::detail::grid_1d::grid_stride<block_size>();
  for (auto n = cudf::detail::grid_1d::global_thread_id<block_size>(); n < num_nodes; n += stride) {
    if (!st.live[n]) { continue; }
    auto const category = st.categories[n];
    auto const out      = values + st.position[n];
    auto const begin    = nodes.range_begin[n];
    auto const end      = nodes.range_end[n];
    switch (category) {
      case cudf::io::json::NC_STR: {
        auto const header = st.node_size[n] > 1 + max_short_string_len ? 5 : 1;
        auto const len    = st.node_size[n] - header;
        if (header == 1) {
          out[0] =
            static_cast<uint8_t>(len << 2) | static_cast<uint8_t>(variant_basic_type::SHORT_STRING);
        } else {
          out[0] = static_cast<uint8_t>(variant_primitive_type::LONG_STRING) << 2;
          write_le(out + 1, static_cast<uint64_t>(len), 4);
        }
        decode_json_string<true>(nodes.buffer + begin + 1, nodes.buffer + end - 1, out + header);
        break;
      }
      case cudf::io::json::NC_VAL:
        write_scalar(nodes.buffer + begin, static_cast<int>(end - begin), out);
        break;
      case cudf::io::json::NC_LIST: {
        auto const count = st.child_count[n];
        auto const data  = st.data_size[n];
        auto const large = count > 0xFF;
        auto const width = integer_size(data);
        auto const nb    = large ? 4 : 1;
        out[0]           = static_cast<uint8_t>((large << 4) | ((width - 1) << 2)) |
                 static_cast<uint8_t>(variant_basic_type::ARRAY);
        write_le(out + 1, count, nb);
        write_le(out + 1 + nb + int64_t{count} * width, data, width);
        break;
      }
      case cudf::io::json::NC_STRUCT: {
        auto const count    = st.child_count[n];
        auto const data     = st.data_size[n];
        auto const large    = count > 0xFF;
        auto const id_width = integer_size(st.max_id[n]);
        auto const width    = integer_size(data);
        auto const nb       = large ? 4 : 1;
        out[0] = static_cast<uint8_t>((large << 6) | ((id_width - 1) << 4) | ((width - 1) << 2)) |
                 static_cast<uint8_t>(variant_basic_type::OBJECT);
        write_le(out + 1, count, nb);
        write_le(out + 1 + nb + int64_t{count} * (id_width + width), data, width);
        break;
      }
      default: break;
    }

    // Members write their own entries into the parent's id and offset lists
    auto const p = st.parents[n];
    if (p < 0) { continue; }
    auto const parent_category = st.categories[p];
    if (parent_category != cudf::io::json::NC_LIST &&
        parent_category != cudf::io::json::NC_STRUCT) {
      continue;
    }
    auto const parent_out = values + st.position[p];
    auto const count      = st.child_count[p];
    auto const nb         = count > 0xFF ? 4 : 1;
    auto const width      = integer_size(st.data_size[p]);
    auto const index      = st.member_index[n];
    if (parent_category == cudf::io::json::NC_LIST) {
      write_le(parent_out + 1 + nb + int64_t{index} * width, st.rel_offset[n], width);
    } else {
      auto const id_width = integer_size(st.max_id[p]);
      write_le(parent_out + 1 + nb + int64_t{index} * id_width, st.dict_id[n], id_width);
      write_le(parent_out + 1 + nb + int64_t{count} * id_width + int64_t{index} * width,
               st.rel_offset[n],
               width);
    }
  }
}

/**
 * @brief Resolves a row the tokenizer rejected but Spark accepts: a root literal followed directly
 * by a character Jackson does not treat as part of the token, such as `true]`.
 *
 * @return The value's header byte, or -1 if the row stays invalid, or -2 if the verdict depends on
 *         Unicode identifier rules that are not implemented
 */
__device__ int root_literal_header(char const* row, int64_t len)
{
  int64_t i = 0;
  while (i < len && is_json_whitespace(row[i])) {
    ++i;
  }
  struct literal {
    char const* text;
    int len;
    int header;
  };
  literal const literals[] = {
    {"null", 4, 0},
    {"true", 4, static_cast<int>(variant_primitive_type::BOOLEAN_TRUE) << 2},
    {"false", 5, static_cast<int>(variant_primitive_type::BOOLEAN_FALSE) << 2}};
  for (auto const& lit : literals) {
    if (len - i <= lit.len || !matches(row + i, lit.len, lit.text, lit.len)) { continue; }
    auto const c = static_cast<uint8_t>(row[i + lit.len]);
    if (c < '0' || c == ']' || c == '}') { return lit.header; }
    if (c >= 0x80) { return -2; }
    bool const identifier_part = (c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') ||
                                 (c >= 'a' && c <= 'z') || c == '_' || c == 0x7F;
    return identifier_part ? -1 : lit.header;
  }
  return -1;
}

struct row_state {
  uint8_t* status;
  uint8_t const* invalid;
  uint8_t const* duplicate;
  NodeIndexT const* root;
  int32_t const* num_keys;
  int64_t const* dict_bytes;
  int16_t* literal_header;
  size_type* value_size;
  size_type* metadata_size;
};

CUDF_KERNEL __launch_bounds__(block_size) void finalize_rows_kernel(row_state rows,
                                                                    size_type num_rows,
                                                                    char const* buffer,
                                                                    SymbolOffsetT const* row_starts,
                                                                    int64_t const* node_size,
                                                                    bool allow_duplicate_keys)
{
  auto const stride = cudf::detail::grid_1d::grid_stride<block_size>();
  for (auto r = cudf::detail::grid_1d::global_thread_id<block_size>(); r < num_rows; r += stride) {
    auto status         = static_cast<op_status>(rows.status[r]);
    int16_t literal     = -1;
    int64_t value_bytes = 0;
    int64_t meta_bytes  = 0;
    if (status == op_status::SUCCESS) {
      auto const root = rows.root[r];
      if (root < 0) {
        auto const header = root_literal_header(buffer + row_starts[r],
                                                row_starts[r + 1] - row_starts[r] - row_padding);
        status            = header >= 0    ? op_status::SUCCESS
                            : header == -2 ? op_status::UNSUPPORTED_INPUT
                                           : op_status::INVALID_JSON;
        literal           = static_cast<int16_t>(header);
        value_bytes       = 1;
        meta_bytes        = 3;
      } else if (rows.invalid[r]) {
        status = op_status::INVALID_JSON;
      } else if (rows.duplicate[r] && !allow_duplicate_keys) {
        status = op_status::DUPLICATE_KEY;
      } else {
        auto const num_keys   = rows.num_keys[r];
        auto const dict_bytes = rows.dict_bytes[r];
        auto const max_size   = cuda::std::max<int64_t>(dict_bytes, num_keys);
        auto const width      = integer_size(max_size);
        value_bytes           = node_size[root];
        meta_bytes            = 1 + width + int64_t{num_keys + 1} * width + dict_bytes;
        if (value_bytes > variant_size_limit || max_size > variant_size_limit ||
            meta_bytes > variant_size_limit) {
          status = op_status::SIZE_LIMIT;
        }
      }
    }
    if (status != op_status::SUCCESS) {
      value_bytes = 0;
      meta_bytes  = 0;
    }
    rows.status[r]         = status_value(status);
    rows.literal_header[r] = status == op_status::SUCCESS ? literal : int16_t{-1};
    rows.value_size[r]     = static_cast<size_type>(value_bytes);
    rows.metadata_size[r]  = static_cast<size_type>(meta_bytes);
  }
}

CUDF_KERNEL __launch_bounds__(block_size) void write_row_headers_kernel(
  row_state rows,
  size_type num_rows,
  size_type const* value_offsets,
  size_type const* meta_offsets,
  uint8_t* values,
  uint8_t* metadata)
{
  auto const stride = cudf::detail::grid_1d::grid_stride<block_size>();
  for (auto r = cudf::detail::grid_1d::global_thread_id<block_size>(); r < num_rows; r += stride) {
    if (rows.status[r] != status_value(op_status::SUCCESS)) { continue; }
    auto const out = metadata + meta_offsets[r];
    if (rows.literal_header[r] >= 0) {
      values[value_offsets[r]] = static_cast<uint8_t>(rows.literal_header[r]);
      out[0]                   = 1;
      out[1]                   = 0;
      out[2]                   = 0;
      continue;
    }
    auto const num_keys   = rows.num_keys[r];
    auto const dict_bytes = rows.dict_bytes[r];
    auto const width      = integer_size(cuda::std::max<int64_t>(dict_bytes, num_keys));
    out[0]                = static_cast<uint8_t>(1 | ((width - 1) << 6));
    write_le(out + 1, num_keys, width);
    write_le(out + 1 + width + int64_t{num_keys} * width, dict_bytes, width);
  }
}

struct dictionary_keys {
  int32_t const* group_row;
  int32_t const* group_first_key;  // index into the field-name arrays of the key's first use
  int32_t const* dict_order;       // groups ordered by dictionary id
  int64_t const* key_offset_in_row;
  int64_t const* key_offsets;
  uint8_t const* key_chars;
};

CUDF_KERNEL __launch_bounds__(block_size) void write_dictionary_kernel(
  dictionary_keys dict,
  int32_t num_groups,
  row_state rows,
  size_type const* meta_offsets,
  uint8_t* metadata)
{
  auto const stride = cudf::detail::grid_1d::grid_stride<block_size>();
  for (auto j = cudf::detail::grid_1d::global_thread_id<block_size>(); j < num_groups;
       j += stride) {
    auto const g   = dict.dict_order[j];
    auto const row = dict.group_row[g];
    if (rows.status[row] != status_value(op_status::SUCCESS)) { continue; }
    auto const num_keys   = rows.num_keys[row];
    auto const dict_bytes = rows.dict_bytes[row];
    auto const width      = integer_size(cuda::std::max<int64_t>(dict_bytes, num_keys));
    // Dictionary ids are dense per row and follow `dict_order`
    int64_t first_of_row = 0;
    for (int64_t hi = j; first_of_row < hi;) {
      auto const mid = (first_of_row + hi) / 2;
      if (dict.group_row[dict.dict_order[mid]] < row) {
        first_of_row = mid + 1;
      } else {
        hi = mid;
      }
    }
    auto const id         = j - first_of_row;
    auto const out        = metadata + meta_offsets[row];
    auto const key_offset = dict.key_offset_in_row[j];
    write_le(out + 1 + width + id * width, key_offset, width);
    auto const key   = dict.group_first_key[g];
    auto const begin = dict.key_offsets[key];
    auto const len   = dict.key_offsets[key + 1] - begin;
    cuda::std::memcpy(
      out + 1 + width + int64_t{num_keys + 1} * width + key_offset, dict.key_chars + begin, len);
  }
}

std::unique_ptr<column> make_byte_lists(size_type num_rows,
                                        rmm::device_uvector<size_type> const& sizes,
                                        cuda::stream_ref stream,
                                        rmm::device_async_resource_ref mr,
                                        uint8_t** data)
{
  auto [offsets, total] =
    cudf::strings::detail::make_offsets_child_column(sizes.begin(), sizes.end(), stream, mr);
  CUDF_EXPECTS(total <= std::numeric_limits<size_type>::max(),
               "VARIANT output exceeds the cudf size_type limit; split the input into smaller "
               "batches",
               std::overflow_error);
  auto child = make_numeric_column(
    data_type{type_id::UINT8}, static_cast<size_type>(total), mask_state::UNALLOCATED, stream, mr);
  *data = child->mutable_view().data<uint8_t>();
  return make_lists_column(num_rows,
                           std::move(offsets),
                           std::move(child),
                           0,
                           cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));
}

struct batch_result {
  std::unique_ptr<column> variant;
  rmm::device_uvector<uint8_t> status;
};

/**
 * @brief Converts one batch of JSON documents; the batch's joined size must fit the tokenizer.
 */
batch_result encode_batch(strings_column_view const& input,
                          bool allow_duplicate_keys,
                          cuda::stream_ref stream,
                          rmm::device_async_resource_ref mr)
{
  auto const num_rows = input.size();
  auto const temp_mr  = cudf::get_current_device_resource_ref();
  auto const policy   = rmm::exec_policy_nosync(stream, temp_mr);
  auto const offsets =
    cudf::detail::offsetalator_factory::make_input_iterator(input.offsets(), input.offset());

  // Join the rows into one JSON Lines buffer: a leading space, then each row and a delimiter.
  // The leading byte guarantees no real token sits at index 0, which marks rejected lines below.
  auto const offset_bounds = [&] {
    rmm::device_uvector<int64_t> bounds(2, stream, temp_mr);
    thrust::transform(
      policy,
      cuda::counting_iterator<int>{0},
      cuda::counting_iterator<int>{2},
      bounds.begin(),
      [offsets, num_rows] __device__(int i) -> int64_t { return offsets[i == 0 ? 0 : num_rows]; });
    return cudf::detail::make_std_vector(bounds, stream);
  }();
  auto const total_chars = offset_bounds[1] - offset_bounds[0];
  auto const buffer_size =
    static_cast<std::size_t>(1 + total_chars + int64_t{num_rows} * row_padding);

  rmm::device_uvector<char> buffer(buffer_size, stream, temp_mr);
  rmm::device_uvector<uint8_t> row_status(num_rows, stream, temp_mr);
  CUDF_CUDA_TRY(cudaMemsetAsync(buffer.data(), ' ', 1, stream.get()));
  {
    auto const grid = cudf::detail::grid_1d{
      static_cast<thread_index_type>(num_rows) * cudf::detail::warp_size, block_size};
    prepare_rows_kernel<<<grid.num_blocks, block_size, 0, stream.get()>>>(input.chars_begin(stream),
                                                                          offsets,
                                                                          input.null_mask(),
                                                                          input.offset(),
                                                                          num_rows,
                                                                          buffer.data(),
                                                                          row_status.data());
    CUDF_CUDA_TRY(cudaGetLastError());
  }

  rmm::device_uvector<SymbolOffsetT> row_starts(num_rows + 1, stream, temp_mr);
  thrust::transform(
    policy,
    cuda::counting_iterator<size_type>{0},
    cuda::counting_iterator<size_type>{num_rows + 1},
    row_starts.begin(),
    [offsets, base = offset_bounds[0]] __device__(size_type r) -> SymbolOffsetT {
      return static_cast<SymbolOffsetT>(1 + (offsets[r] - base) + int64_t{r} * row_padding);
    });

  cudf::io::json_reader_options options{};
  options.enable_lines(true);
  options.set_delimiter(row_delimiter);
  options.set_recovery_mode(cudf::io::json_recovery_mode_t::RECOVER_WITH_NULL);
  options.set_strict_validation(true);
  options.allow_numeric_leading_zeros(false);
  options.allow_nonnumeric_numbers(false);
  options.allow_unquoted_control_chars(false);

  auto [tokens, token_indices] =
    cudf::io::json::detail::get_token_stream(buffer, options, stream, temp_mr);
  {
    // Rejected lines come back as a `{}` token pair at index 0; drop them
    auto zipped = cuda::make_zip_iterator(tokens.begin(), token_indices.begin());
    auto const nd =
      thrust::remove_if(policy,
                        zipped,
                        zipped + tokens.size(),
                        token_indices.begin(),
                        [] __device__(SymbolOffsetT idx) -> bool { return idx == 0; }) -
      zipped;
    tokens.resize(nd, stream);
    token_indices.resize(nd, stream);
  }

  auto tree = [&] {
    if (tokens.is_empty()) {
      return cudf::io::json::tree_meta_t{rmm::device_uvector<NodeT>(0, stream, temp_mr),
                                         rmm::device_uvector<NodeIndexT>(0, stream, temp_mr),
                                         rmm::device_uvector<TreeDepthT>(0, stream, temp_mr),
                                         rmm::device_uvector<SymbolOffsetT>(0, stream, temp_mr),
                                         rmm::device_uvector<SymbolOffsetT>(0, stream, temp_mr)};
    }
    return cudf::io::json::detail::get_tree_representation(
      tokens, token_indices, true, stream, temp_mr);
  }();
  tokens.release();
  token_indices.release();

  auto const num_nodes = static_cast<NodeIndexT>(tree.node_categories.size());

  // The tree builder reports literals and numbers as strings; only real strings start with a quote
  thrust::transform(policy,
                    tree.node_categories.begin(),
                    tree.node_categories.end(),
                    tree.node_range_begin.begin(),
                    tree.node_categories.begin(),
                    [buf = buffer.data()] __device__(NodeT category, SymbolOffsetT begin) -> NodeT {
                      return (category == cudf::io::json::NC_STR && buf[begin] != '"')
                               ? cudf::io::json::NC_VAL
                               : category;
                    });

  rmm::device_uvector<size_type> node_row(num_nodes, stream, temp_mr);
  thrust::transform(
    policy,
    tree.node_range_begin.begin(),
    tree.node_range_begin.end(),
    node_row.begin(),
    [starts = row_starts.data(), num_rows] __device__(SymbolOffsetT pos) -> size_type {
      return static_cast<size_type>(
        thrust::upper_bound(thrust::seq, starts, starts + num_rows + 1, pos) - starts - 1);
    });

  node_text_view const nodes{buffer.data(),
                             tree.node_categories.data(),
                             tree.node_levels.data(),
                             tree.node_range_begin.data(),
                             tree.node_range_end.data(),
                             node_row.data(),
                             row_starts.data()};

  auto row_invalid =
    cudf::detail::make_zeroed_device_uvector_async<uint8_t>(num_rows, stream, temp_mr);
  auto row_duplicate =
    cudf::detail::make_zeroed_device_uvector_async<uint8_t>(num_rows, stream, temp_mr);
  auto row_num_keys =
    cudf::detail::make_zeroed_device_uvector_async<int32_t>(num_rows, stream, temp_mr);
  auto row_dict_bytes =
    cudf::detail::make_zeroed_device_uvector_async<int64_t>(num_rows, stream, temp_mr);
  rmm::device_uvector<NodeIndexT> row_root(num_rows, stream, temp_mr);
  thrust::fill(policy, row_root.begin(), row_root.end(), NodeIndexT{-1});

  auto node_size =
    cudf::detail::make_zeroed_device_uvector_async<int64_t>(num_nodes, stream, temp_mr);
  auto key_length =
    cudf::detail::make_zeroed_device_uvector_async<int32_t>(num_nodes, stream, temp_mr);
  auto node_kept =
    cudf::detail::make_zeroed_device_uvector_async<uint8_t>(num_nodes, stream, temp_mr);
  auto node_dict_id =
    cudf::detail::make_zeroed_device_uvector_async<int32_t>(num_nodes, stream, temp_mr);
  auto child_count =
    cudf::detail::make_zeroed_device_uvector_async<int32_t>(num_nodes, stream, temp_mr);
  auto max_id = cudf::detail::make_zeroed_device_uvector_async<int32_t>(num_nodes, stream, temp_mr);
  auto data_size =
    cudf::detail::make_zeroed_device_uvector_async<int64_t>(num_nodes, stream, temp_mr);
  auto rel_offset =
    cudf::detail::make_zeroed_device_uvector_async<int64_t>(num_nodes, stream, temp_mr);
  auto member_index =
    cudf::detail::make_zeroed_device_uvector_async<int32_t>(num_nodes, stream, temp_mr);
  auto position =
    cudf::detail::make_zeroed_device_uvector_async<int64_t>(num_nodes, stream, temp_mr);
  auto live = cudf::detail::make_zeroed_device_uvector_async<uint8_t>(num_nodes, stream, temp_mr);

  encode_state const st{tree.node_categories.data(),
                        tree.parent_node_ids.data(),
                        node_kept.data(),
                        node_dict_id.data(),
                        child_count.data(),
                        max_id.data(),
                        data_size.data(),
                        node_size.data(),
                        rel_offset.data(),
                        member_index.data(),
                        position.data(),
                        live.data()};

  // Dictionary state, kept alive until the metadata is written
  rmm::device_uvector<int32_t> group_row(0, stream, temp_mr);
  rmm::device_uvector<int32_t> group_first_key(0, stream, temp_mr);
  rmm::device_uvector<int32_t> dict_order(0, stream, temp_mr);
  rmm::device_uvector<int64_t> key_offset_in_row(0, stream, temp_mr);
  rmm::device_uvector<int64_t> key_offsets(0, stream, temp_mr);
  rmm::device_uvector<uint8_t> key_chars(0, stream, temp_mr);
  int32_t num_groups = 0;

  // Host copy of the level boundaries of `level_nodes`
  rmm::device_uvector<NodeIndexT> level_nodes(0, stream, temp_mr);
  std::vector<NodeIndexT> level_offsets{0};

  if (num_nodes > 0) {
    auto const grid = cudf::detail::grid_1d{num_nodes, block_size};
    classify_nodes_kernel<<<grid.num_blocks, block_size, 0, stream.get()>>>(
      nodes, num_nodes, node_size.data(), key_length.data(), row_invalid.data());
    CUDF_CUDA_TRY(cudaGetLastError());

    thrust::for_each_n(policy,
                       cuda::counting_iterator<NodeIndexT>{0},
                       num_nodes,
                       [levels = tree.node_levels.data(),
                        rows   = node_row.data(),
                        roots  = row_root.data()] __device__(NodeIndexT n) {
                         if (levels[n] == 0) { roots[rows[n]] = n; }
                       });

    // Field names, in node (parse) order
    auto const is_field = [cats = tree.node_categories.data()] __device__(NodeIndexT n) -> bool {
      return cats[n] == cudf::io::json::NC_FN;
    };
    auto const num_keys =
      static_cast<size_type>(thrust::count_if(policy,
                                              cuda::counting_iterator<NodeIndexT>{0},
                                              cuda::counting_iterator<NodeIndexT>{num_nodes},
                                              is_field));
    rmm::device_uvector<NodeIndexT> fn_nodes(num_keys, stream, temp_mr);
    thrust::copy_if(policy,
                    cuda::counting_iterator<NodeIndexT>{0},
                    cuda::counting_iterator<NodeIndexT>{num_nodes},
                    fn_nodes.begin(),
                    is_field);

    if (num_keys > 0) {
      key_offsets.resize(num_keys + 1, stream);
      auto const key_len_it = cuda::make_transform_iterator(
        cuda::counting_iterator<int32_t>{0},
        [lens = key_length.data(), fn = fn_nodes.data(), num_keys] __device__(
          int32_t i) -> int64_t { return i < num_keys ? lens[fn[i]] : 0; });
      thrust::exclusive_scan(
        policy, key_len_it, key_len_it + num_keys + 1, key_offsets.begin(), int64_t{0});
      auto const total_key_bytes = key_offsets.element(num_keys, stream);
      key_chars.resize(total_key_bytes, stream);
      auto const kgrid = cudf::detail::grid_1d{num_keys, block_size};
      decode_keys_kernel<<<kgrid.num_blocks, block_size, 0, stream.get()>>>(
        buffer.data(),
        fn_nodes.data(),
        tree.node_range_begin.data(),
        tree.node_range_end.data(),
        key_offsets.data(),
        num_keys,
        key_chars.data());
      CUDF_CUDA_TRY(cudaGetLastError());

      // Group equal keys within each row, ordered by key in Java String order
      rmm::device_uvector<int32_t> fn_rows(num_keys, stream, temp_mr);
      thrust::gather(policy, fn_nodes.begin(), fn_nodes.end(), node_row.begin(), fn_rows.begin());
      rmm::device_uvector<int32_t> by_key(num_keys, stream, temp_mr);
      thrust::sequence(policy, by_key.begin(), by_key.end());
      rmm::device_uvector<uint64_t> short_key(num_keys, stream, temp_mr);
      thrust::transform(
        policy,
        cuda::counting_iterator<int32_t>{0},
        cuda::counting_iterator<int32_t>{num_keys},
        short_key.begin(),
        [offs = key_offsets.data(), chars = key_chars.data()] __device__(int32_t i) -> uint64_t {
          return short_key_code(chars + offs[i], offs[i + 1] - offs[i]);
        });
      bool const all_short =
        thrust::none_of(policy, short_key.begin(), short_key.end(), [] __device__(uint64_t code) {
          return code == long_key_code;
        });
      if (all_short) {
        // Two stable radix passes order by (row, key) and keep parse order among equal keys
        thrust::stable_sort_by_key(policy, short_key.begin(), short_key.end(), by_key.begin());
        short_key = rmm::device_uvector<uint64_t>(0, stream, temp_mr);
        rmm::device_uvector<int32_t> sorted_rows(num_keys, stream, temp_mr);
        thrust::gather(policy, by_key.begin(), by_key.end(), fn_rows.begin(), sorted_rows.begin());
        thrust::stable_sort_by_key(policy, sorted_rows.begin(), sorted_rows.end(), by_key.begin());
      } else {
        auto const key_less = [rows  = fn_rows.data(),
                               offs  = key_offsets.data(),
                               chars = key_chars.data()] __device__(int32_t a, int32_t b) -> bool {
          if (rows[a] != rows[b]) { return rows[a] < rows[b]; }
          auto const c = utf16_compare(
            chars + offs[a], offs[a + 1] - offs[a], chars + offs[b], offs[b + 1] - offs[b]);
          return c != 0 ? c < 0 : a < b;
        };
        thrust::sort(policy, by_key.begin(), by_key.end(), key_less);
      }

      rmm::device_uvector<int32_t> group_of(
        num_keys, stream, temp_mr);  // indexed by position in by_key
      thrust::transform(policy,
                        cuda::counting_iterator<int32_t>{0},
                        cuda::counting_iterator<int32_t>{num_keys},
                        group_of.begin(),
                        [order = by_key.data(),
                         rows  = fn_rows.data(),
                         offs  = key_offsets.data(),
                         chars = key_chars.data()] __device__(int32_t i) -> int32_t {
                          if (i == 0) { return 1; }
                          auto const a = order[i - 1];
                          auto const b = order[i];
                          if (rows[a] != rows[b]) { return 1; }
                          auto const len = offs[a + 1] - offs[a];
                          if (len != offs[b + 1] - offs[b]) { return 1; }
                          for (int64_t k = 0; k < len; ++k) {
                            if (chars[offs[a] + k] != chars[offs[b] + k]) { return 1; }
                          }
                          return 0;
                        });
      thrust::inclusive_scan(policy, group_of.begin(), group_of.end(), group_of.begin());
      num_groups = group_of.element(num_keys - 1, stream);
      thrust::transform(policy,
                        group_of.begin(),
                        group_of.end(),
                        group_of.begin(),
                        [] __device__(int32_t g) -> int32_t { return g - 1; });

      // First use of each key: the group's first entry, since ties sort by parse order
      group_row.resize(num_groups, stream);
      group_first_key.resize(num_groups, stream);
      thrust::for_each_n(policy,
                         cuda::counting_iterator<int32_t>{0},
                         num_keys,
                         [order   = by_key.data(),
                          groups  = group_of.data(),
                          rows    = fn_rows.data(),
                          g_row   = group_row.data(),
                          g_first = group_first_key.data()] __device__(int32_t i) {
                           if (i == 0 || groups[i] != groups[i - 1]) {
                             g_row[groups[i]]   = rows[order[i]];
                             g_first[groups[i]] = order[i];
                           }
                         });

      // Dictionary ids follow first use within the row
      dict_order.resize(num_groups, stream);
      thrust::sequence(policy, dict_order.begin(), dict_order.end());
      {
        rmm::device_uvector<int32_t> first_use(num_groups, stream, temp_mr);
        thrust::copy(policy, group_first_key.begin(), group_first_key.end(), first_use.begin());
        thrust::sort_by_key(policy, first_use.begin(), first_use.end(), dict_order.begin());
      }
      rmm::device_uvector<int32_t> group_dict_id(num_groups, stream, temp_mr);
      key_offset_in_row.resize(num_groups, stream);
      {
        auto const dict_rows =
          cuda::make_permutation_iterator(group_row.begin(), dict_order.begin());
        rmm::device_uvector<int32_t> ranks(num_groups, stream, temp_mr);
        thrust::exclusive_scan_by_key(policy,
                                      dict_rows,
                                      dict_rows + num_groups,
                                      cuda::constant_iterator<int32_t>{1},
                                      ranks.begin());
        thrust::scatter(
          policy, ranks.begin(), ranks.end(), dict_order.begin(), group_dict_id.begin());
        auto const dict_lengths = cuda::make_transform_iterator(
          dict_order.begin(),
          [first = group_first_key.data(), offs = key_offsets.data()] __device__(
            int32_t g) -> int64_t { return offs[first[g] + 1] - offs[first[g]]; });
        thrust::exclusive_scan_by_key(
          policy, dict_rows, dict_rows + num_groups, dict_lengths, key_offset_in_row.begin());
      }

      // Rank of each key in Java String order within its row
      rmm::device_uvector<int32_t> group_rank(num_groups, stream, temp_mr);
      thrust::exclusive_scan_by_key(policy,
                                    group_row.begin(),
                                    group_row.end(),
                                    cuda::constant_iterator<int32_t>{1},
                                    group_rank.begin());

      thrust::for_each_n(
        policy,
        cuda::counting_iterator<int32_t>{0},
        num_groups,
        [g_row      = group_row.data(),
         g_first    = group_first_key.data(),
         offs       = key_offsets.data(),
         num_keys_r = row_num_keys.data(),
         bytes_r    = row_dict_bytes.data()] __device__(int32_t g) {
          auto const row = g_row[g];
          cuda::atomic_ref<int32_t, cuda::thread_scope_device>{num_keys_r[row]}.fetch_add(
            1, cuda::std::memory_order_relaxed);
          cuda::atomic_ref<int64_t, cuda::thread_scope_device>{bytes_r[row]}.fetch_add(
            offs[g_first[g] + 1] - offs[g_first[g]], cuda::std::memory_order_relaxed);
        });

      // Fields of each object in key order; equal keys stay in parse order, and the last wins
      rmm::device_uvector<int32_t> fn_rank(num_keys, stream, temp_mr);
      thrust::for_each_n(policy,
                         cuda::counting_iterator<int32_t>{0},
                         num_keys,
                         [order   = by_key.data(),
                          groups  = group_of.data(),
                          ranks   = group_rank.data(),
                          dict_id = group_dict_id.data(),
                          fn      = fn_nodes.data(),
                          fn_rank = fn_rank.data(),
                          node_id = node_dict_id.data()] __device__(int32_t i) {
                           auto const k   = order[i];
                           fn_rank[k]     = ranks[groups[i]];
                           node_id[fn[k]] = dict_id[groups[i]];
                         });
      rmm::device_uvector<uint64_t> field_keys(num_keys, stream, temp_mr);
      thrust::transform(policy,
                        cuda::counting_iterator<int32_t>{0},
                        cuda::counting_iterator<int32_t>{num_keys},
                        field_keys.begin(),
                        [fn      = fn_nodes.data(),
                         parents = tree.parent_node_ids.data(),
                         rank    = fn_rank.data()] __device__(int32_t k) -> uint64_t {
                          return (static_cast<uint64_t>(parents[fn[k]]) << 32) |
                                 static_cast<uint32_t>(rank[k]);
                        });
      rmm::device_uvector<NodeIndexT> sorted_fields(num_keys, stream, temp_mr);
      thrust::copy(policy, fn_nodes.begin(), fn_nodes.end(), sorted_fields.begin());
      thrust::stable_sort_by_key(
        policy, field_keys.begin(), field_keys.end(), sorted_fields.begin());

      rmm::device_uvector<int32_t> kept_flags(num_keys, stream, temp_mr);
      thrust::transform(policy,
                        cuda::counting_iterator<int32_t>{0},
                        cuda::counting_iterator<int32_t>{num_keys},
                        kept_flags.begin(),
                        [keys = field_keys.data(), num_keys] __device__(int32_t i) -> int32_t {
                          return i + 1 == num_keys || keys[i + 1] != keys[i];
                        });
      rmm::device_uvector<int32_t> field_index(num_keys, stream, temp_mr);
      auto const parent_keys = cuda::make_transform_iterator(
        field_keys.begin(),
        [] __device__(uint64_t k) -> uint32_t { return static_cast<uint32_t>(k >> 32); });
      thrust::exclusive_scan_by_key(
        policy, parent_keys, parent_keys + num_keys, kept_flags.begin(), field_index.begin());
      thrust::for_each_n(policy,
                         cuda::counting_iterator<int32_t>{0},
                         num_keys,
                         [fields = sorted_fields.data(),
                          kept_f = kept_flags.data(),
                          index  = field_index.data(),
                          rows   = node_row.data(),
                          dup    = row_duplicate.data(),
                          st] __device__(int32_t i) {
                           auto const n       = fields[i];
                           st.kept[n]         = static_cast<uint8_t>(kept_f[i]);
                           st.member_index[n] = index[i];
                           if (!kept_f[i]) { dup[rows[n]] = 1; }
                         });
    }

    count_children_kernel<<<grid.num_blocks, block_size, 0, stream.get()>>>(st, num_nodes);
    CUDF_CUDA_TRY(cudaGetLastError());

    // Level slices, deepest last
    level_nodes.resize(num_nodes, stream);
    thrust::sequence(policy, level_nodes.begin(), level_nodes.end());
    {
      rmm::device_uvector<TreeDepthT> sorted_levels(num_nodes, stream, temp_mr);
      thrust::copy(policy, tree.node_levels.begin(), tree.node_levels.end(), sorted_levels.begin());
      thrust::stable_sort_by_key(
        policy, sorted_levels.begin(), sorted_levels.end(), level_nodes.begin());
      auto const max_level = sorted_levels.element(num_nodes - 1, stream);
      rmm::device_uvector<NodeIndexT> bounds(max_level + 2, stream, temp_mr);
      thrust::lower_bound(policy,
                          sorted_levels.begin(),
                          sorted_levels.end(),
                          cuda::counting_iterator<int>{0},
                          cuda::counting_iterator<int>{max_level + 2},
                          bounds.begin());
      auto const h_bounds = cudf::detail::make_std_vector(bounds, stream);
      level_offsets.assign(h_bounds.begin(), h_bounds.end());
    }
    auto const num_levels = static_cast<int>(level_offsets.size()) - 1;
    for (int level = num_levels - 1; level >= 0; --level) {
      auto const count = level_offsets[level + 1] - level_offsets[level];
      if (count == 0) { continue; }
      auto const lgrid = cudf::detail::grid_1d{count, block_size};
      size_level_kernel<<<lgrid.num_blocks, block_size, 0, stream.get()>>>(
        st, level_nodes.data() + level_offsets[level], count);
      CUDF_CUDA_TRY(cudaGetLastError());
    }

    // Offsets of members within their parent's data section, in parse order
    auto const is_member = [cats = tree.node_categories.data(),
                            parents =
                              tree.parent_node_ids.data()] __device__(NodeIndexT n) -> bool {
      auto const p = parents[n];
      return p >= 0 && (cats[p] == cudf::io::json::NC_LIST || cats[p] == cudf::io::json::NC_STRUCT);
    };
    auto const num_members =
      static_cast<NodeIndexT>(thrust::count_if(policy,
                                               cuda::counting_iterator<NodeIndexT>{0},
                                               cuda::counting_iterator<NodeIndexT>{num_nodes},
                                               is_member));
    if (num_members > 0) {
      rmm::device_uvector<NodeIndexT> members(num_members, stream, temp_mr);
      thrust::copy_if(policy,
                      cuda::counting_iterator<NodeIndexT>{0},
                      cuda::counting_iterator<NodeIndexT>{num_nodes},
                      members.begin(),
                      is_member);
      rmm::device_uvector<NodeIndexT> member_parent(num_members, stream, temp_mr);
      thrust::gather(policy,
                     members.begin(),
                     members.end(),
                     tree.parent_node_ids.begin(),
                     member_parent.begin());
      thrust::stable_sort_by_key(
        policy, member_parent.begin(), member_parent.end(), members.begin());
      auto const contribution =
        cuda::make_transform_iterator(members.begin(), [st] __device__(NodeIndexT n) -> int64_t {
          auto const p = st.parents[n];
          return (st.categories[p] == cudf::io::json::NC_STRUCT && !st.kept[n]) ? 0
                                                                                : st.node_size[n];
        });
      rmm::device_uvector<int64_t> offsets_in_parent(num_members, stream, temp_mr);
      thrust::exclusive_scan_by_key(policy,
                                    member_parent.begin(),
                                    member_parent.end(),
                                    contribution,
                                    offsets_in_parent.begin());
      thrust::scatter(policy,
                      offsets_in_parent.begin(),
                      offsets_in_parent.end(),
                      members.begin(),
                      rel_offset.begin());
      rmm::device_uvector<int32_t> element_index(num_members, stream, temp_mr);
      thrust::exclusive_scan_by_key(policy,
                                    member_parent.begin(),
                                    member_parent.end(),
                                    cuda::constant_iterator<int32_t>{1},
                                    element_index.begin());
      thrust::for_each_n(
        policy,
        cuda::counting_iterator<NodeIndexT>{0},
        num_members,
        [members = members.data(), index = element_index.data(), st] __device__(NodeIndexT i) {
          auto const n = members[i];
          if (st.categories[st.parents[n]] == cudf::io::json::NC_LIST) {
            st.member_index[n] = index[i];
          }
        });
    }
  }

  // Row verdicts and output sizes
  rmm::device_uvector<int16_t> literal_header(num_rows, stream, temp_mr);
  rmm::device_uvector<size_type> value_sizes(num_rows, stream, temp_mr);
  rmm::device_uvector<size_type> meta_sizes(num_rows, stream, temp_mr);
  row_state const rows{row_status.data(),
                       row_invalid.data(),
                       row_duplicate.data(),
                       row_root.data(),
                       row_num_keys.data(),
                       row_dict_bytes.data(),
                       literal_header.data(),
                       value_sizes.data(),
                       meta_sizes.data()};
  auto const row_grid = cudf::detail::grid_1d{num_rows, block_size};
  finalize_rows_kernel<<<row_grid.num_blocks, block_size, 0, stream.get()>>>(
    rows, num_rows, buffer.data(), row_starts.data(), node_size.data(), allow_duplicate_keys);
  CUDF_CUDA_TRY(cudaGetLastError());

  uint8_t* d_values   = nullptr;
  uint8_t* d_metadata = nullptr;
  auto metadata_col   = make_byte_lists(num_rows, meta_sizes, stream, mr, &d_metadata);
  auto value_col      = make_byte_lists(num_rows, value_sizes, stream, mr, &d_values);
  auto const value_offsets =
    lists_column_view(value_col->view()).offsets().template data<size_type>();
  auto const meta_offsets =
    lists_column_view(metadata_col->view()).offsets().template data<size_type>();

  if (num_nodes > 0) {
    // Roots sit at their row's offset; descendants are placed level by level
    auto const root_count = level_offsets[1] - level_offsets[0];
    thrust::for_each_n(
      policy,
      level_nodes.begin(),
      root_count,
      [st, rows = node_row.data(), status = row_status.data(), value_offsets] __device__(
        NodeIndexT n) {
        auto const r   = rows[n];
        st.position[n] = value_offsets[r];
        st.live[n]     = status[r] == status_value(op_status::SUCCESS);
      });
    auto const num_levels = static_cast<int>(level_offsets.size()) - 1;
    for (int level = 1; level < num_levels; ++level) {
      auto const count = level_offsets[level + 1] - level_offsets[level];
      if (count == 0) { continue; }
      auto const lgrid = cudf::detail::grid_1d{count, block_size};
      place_level_kernel<<<lgrid.num_blocks, block_size, 0, stream.get()>>>(
        st, level_nodes.data() + level_offsets[level], count);
      CUDF_CUDA_TRY(cudaGetLastError());
    }
    auto const grid = cudf::detail::grid_1d{num_nodes, block_size};
    write_values_kernel<<<grid.num_blocks, block_size, 0, stream.get()>>>(
      st, nodes, num_nodes, d_values);
    CUDF_CUDA_TRY(cudaGetLastError());
  }

  write_row_headers_kernel<<<row_grid.num_blocks, block_size, 0, stream.get()>>>(
    rows, num_rows, value_offsets, meta_offsets, d_values, d_metadata);
  CUDF_CUDA_TRY(cudaGetLastError());
  if (num_groups > 0) {
    dictionary_keys const dict{group_row.data(),
                               group_first_key.data(),
                               dict_order.data(),
                               key_offset_in_row.data(),
                               key_offsets.data(),
                               key_chars.data()};
    auto const dgrid = cudf::detail::grid_1d{num_groups, block_size};
    write_dictionary_kernel<<<dgrid.num_blocks, block_size, 0, stream.get()>>>(
      dict, num_groups, rows, meta_offsets, d_metadata);
    CUDF_CUDA_TRY(cudaGetLastError());
  }

  auto [null_mask, null_count] = cudf::detail::valid_if(
    row_status.begin(),
    row_status.end(),
    [] __device__(uint8_t s) -> bool { return s == status_value(op_status::SUCCESS); },
    stream,
    mr);
  std::vector<std::unique_ptr<column>> children;
  children.push_back(std::move(metadata_col));
  children.push_back(std::move(value_col));
  auto variant = make_structs_column(
    num_rows,
    std::move(children),
    null_count,
    null_count > 0 ? std::move(null_mask) : cudf::create_null_mask(0, mask_state::UNALLOCATED),
    stream,
    mr);
  return {std::move(variant), std::move(row_status)};
}

std::unique_ptr<column> make_empty_variant_column(cuda::stream_ref stream,
                                                  rmm::device_async_resource_ref mr)
{
  std::vector<std::unique_ptr<column>> children;
  children.push_back(make_empty_lists_column(data_type{type_id::UINT8}));
  children.push_back(make_empty_lists_column(data_type{type_id::UINT8}));
  return make_structs_column(
    0, std::move(children), 0, cudf::create_null_mask(0, mask_state::UNALLOCATED), stream, mr);
}

}  // namespace

std::unique_ptr<column> parse_json_to_variant(strings_column_view const& input,
                                              bool allow_duplicate_keys,
                                              std::optional<mutable_column_view> status,
                                              cuda::stream_ref stream,
                                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  auto const num_rows = input.size();
  if (status.has_value()) {
    CUDF_EXPECTS(!status->nullable(), "status column must not be nullable", std::invalid_argument);
    CUDF_EXPECTS(
      status->type().id() == type_id::UINT8, "status column must be UINT8", std::invalid_argument);
    CUDF_EXPECTS(status->size() == num_rows,
                 "status column must have the same number of rows as the input",
                 std::invalid_argument);
  }
  if (num_rows == 0) { return make_empty_variant_column(stream, mr); }
  auto const temp_mr = cudf::get_current_device_resource_ref();

  // Split the rows into batches whose joined size stays below the batch limit. The joined buffer,
  // including its leading byte, must stay within the tokenizer's 2^31-byte input limit.
  auto const batch_limit = std::min<std::size_t>(
    cudf::detail::getenv_or<std::size_t>("LIBCUDF_JSON_TO_VARIANT_BATCH_SIZE", default_batch_size),
    std::numeric_limits<int32_t>::max());
  auto const offsets =
    cudf::detail::offsetalator_factory::make_input_iterator(input.offsets(), input.offset());
  auto const total_bytes =
    static_cast<std::size_t>(
      cudf::strings::detail::get_offset_value(input.offsets(), input.offset() + num_rows, stream) -
      cudf::strings::detail::get_offset_value(input.offsets(), input.offset(), stream)) +
    std::size_t{row_padding} * num_rows;
  std::vector<size_type> batch_bounds{0};
  if (total_bytes > batch_limit) {
    rmm::device_uvector<int64_t> d_offsets(num_rows + 1, stream, temp_mr);
    thrust::copy(
      rmm::exec_policy_nosync(stream, temp_mr), offsets, offsets + num_rows + 1, d_offsets.begin());
    auto const h_offsets    = cudf::detail::make_std_vector(d_offsets, stream);
    std::size_t batch_bytes = 0;
    for (size_type r = 0; r < num_rows; ++r) {
      auto const row_bytes =
        static_cast<std::size_t>(h_offsets[r + 1] - h_offsets[r]) + row_padding;
      if (batch_bytes > 0 && batch_bytes + row_bytes > batch_limit) {
        batch_bounds.push_back(r);
        batch_bytes = 0;
      }
      batch_bytes += row_bytes;
    }
  }
  batch_bounds.push_back(num_rows);

  auto const num_batches = static_cast<int>(batch_bounds.size()) - 1;
  if (num_batches == 1) {
    auto result = encode_batch(input, allow_duplicate_keys, stream, mr);
    if (status.has_value()) {
      thrust::copy(rmm::exec_policy_nosync(stream, temp_mr),
                   result.status.begin(),
                   result.status.end(),
                   status->data<uint8_t>());
    }
    return std::move(result.variant);
  }

  std::vector<std::unique_ptr<column>> parts;
  for (int b = 0; b < num_batches; ++b) {
    auto const slice =
      cudf::slice(input.parent(), {batch_bounds[b], batch_bounds[b + 1]}, stream).front();
    auto result = encode_batch(strings_column_view{slice}, allow_duplicate_keys, stream, temp_mr);
    if (status.has_value()) {
      thrust::copy(rmm::exec_policy_nosync(stream, temp_mr),
                   result.status.begin(),
                   result.status.end(),
                   status->data<uint8_t>() + batch_bounds[b]);
    }
    parts.push_back(std::move(result.variant));
  }
  std::vector<column_view> views;
  views.reserve(parts.size());
  for (auto const& p : parts) {
    views.push_back(p->view());
  }
  return cudf::concatenate(views, stream, mr);
}

}  // namespace io::parquet::experimental
}  // namespace cudf
