// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0

#pragma once

#include "tracegen/common/types.cuh"
#include "tracegen/precompiles/fp_op.cuh"
#include "tracegen/precompiles/fp_lt.cuh"
#include "tracegen/precompiles/curves.cuh"
#include "tracegen/riscv/memory_read_write.cuh"
#include "tracegen/riscv/syscall_addr.cuh"

namespace pico_gpu::curve_op {
  using curves::AffinePoint;

  /// Write a u16 addr as 3 field elements (addr low 16, bits 16-31, bits 32-47).
  template <class F>
  __device__ inline void addr_to_u16_limbs(F dst[3], uint32_t addr) {
    dst[0] = F::from_canonical_u16(static_cast<uint16_t>(addr & 0xFFFF));
    dst[1] = F::from_canonical_u16(static_cast<uint16_t>((addr >> 16) & 0xFFFF));
    dst[2] = F::zero(); // high limb is 0 for 32-bit addresses
  }

  /// Populate a MemoryWriteColsU8 from an FfiEcMemoryWriteRecord.
  /// rec.value = current (written) value, rec.prev_value = previous value.
  template <class F>
  __device__ inline void populate_write_cols_u8(
    MemoryWriteColsU8<F>& col, const FfiEcMemoryWriteRecord& rec, F* byte_trace)
  {
    uint64_t cur_value = rec.value;
    uint64_t prev_value = rec.prev_value;

    // Write prev_value word (4 u16 limbs).
    memory_read_write_u64::write_word_from_u64(col.inner.prev_value, prev_value);

    // Populate access columns with u64 values.
    memory_read_write_u64::populate_access_u64(col.inner.access,
      rec.chunk, rec.timestamp, cur_value,
      rec.prev_chunk, rec.prev_timestamp, prev_value,
      byte_trace);

    // Populate U16ToU8Gadget: extract the low byte from each u16 limb of prev_value.
    col.prev_value_u8.low_bytes[0] = F::from_canonical_u8(static_cast<uint8_t>(prev_value & 0xFF));
    col.prev_value_u8.low_bytes[1] = F::from_canonical_u8(static_cast<uint8_t>((prev_value >> 16) & 0xFF));
    col.prev_value_u8.low_bytes[2] = F::from_canonical_u8(static_cast<uint8_t>((prev_value >> 32) & 0xFF));
    col.prev_value_u8.low_bytes[3] = F::from_canonical_u8(static_cast<uint8_t>((prev_value >> 48) & 0xFF));

    // Add u8 range checks for [low_byte, high_byte] pairs per u16 limb.
    if (byte_trace) {
      // Limb 0: bits [0..15]
      byte::add_u8_range_check<F>(byte_trace,
        static_cast<uint8_t>(prev_value & 0xFF),
        static_cast<uint8_t>((prev_value >> 8) & 0xFF));
      // Limb 1: bits [16..31]
      byte::add_u8_range_check<F>(byte_trace,
        static_cast<uint8_t>((prev_value >> 16) & 0xFF),
        static_cast<uint8_t>((prev_value >> 24) & 0xFF));
      // Limb 2: bits [32..47]
      byte::add_u8_range_check<F>(byte_trace,
        static_cast<uint8_t>((prev_value >> 32) & 0xFF),
        static_cast<uint8_t>((prev_value >> 40) & 0xFF));
      // Limb 3: bits [48..63]
      byte::add_u8_range_check<F>(byte_trace,
        static_cast<uint8_t>((prev_value >> 48) & 0xFF),
        static_cast<uint8_t>((prev_value >> 56) & 0xFF));
    }
  }

  /// Populate a MemoryReadColsU8 from an FfiEcMemoryReadRecord.
  template <class F>
  __device__ inline void populate_read_cols_u8(
    MemoryReadColsU8<F>& col, const FfiEcMemoryReadRecord& rec, F* byte_trace)
  {
    uint64_t value = rec.value;

    // Populate access columns with u64 values.
    // For reads, prev_value == current_value.
    memory_read_write_u64::populate_access_u64(col.inner.access,
      rec.chunk, rec.timestamp, value,
      rec.prev_chunk, rec.prev_timestamp, value,
      byte_trace);

    // Populate U16ToU8Gadget: extract the low byte from each u16 limb of value.
    col.prev_value_u8.low_bytes[0] = F::from_canonical_u8(static_cast<uint8_t>(value & 0xFF));
    col.prev_value_u8.low_bytes[1] = F::from_canonical_u8(static_cast<uint8_t>((value >> 16) & 0xFF));
    col.prev_value_u8.low_bytes[2] = F::from_canonical_u8(static_cast<uint8_t>((value >> 32) & 0xFF));
    col.prev_value_u8.low_bytes[3] = F::from_canonical_u8(static_cast<uint8_t>((value >> 48) & 0xFF));

    if (byte_trace) {
      // Limb 0: bits [0..15]
      byte::add_u8_range_check<F>(byte_trace,
        static_cast<uint8_t>(value & 0xFF),
        static_cast<uint8_t>((value >> 8) & 0xFF));
      // Limb 1: bits [16..31]
      byte::add_u8_range_check<F>(byte_trace,
        static_cast<uint8_t>((value >> 16) & 0xFF),
        static_cast<uint8_t>((value >> 24) & 0xFF));
      // Limb 2: bits [32..47]
      byte::add_u8_range_check<F>(byte_trace,
        static_cast<uint8_t>((value >> 32) & 0xFF),
        static_cast<uint8_t>((value >> 40) & 0xFF));
      // Limb 3: bits [48..63]
      byte::add_u8_range_check<F>(byte_trace,
        static_cast<uint8_t>((value >> 48) & 0xFF),
        static_cast<uint8_t>((value >> 56) & 0xFF));
    }
  }

  /// Populate field operations for Weierstrass curve point addition (u64 version).
  /// Computes: result = p + q using the formula:
  /// slope = (q.y - p.y) / (q.x - p.x)
  /// x3 = slope^2 - p.x - q.x
  /// y3 = slope * (p.x - x3) - p.y
  ///
  /// NumWords here is the number of u32 words per field element (= Limbs/4).
  template <
    class F,
    const int NumWords,
    const int NumLimbs,
    const int NumWitnesses,
    const int NumBitsPerLimb,
    const int WitnessOffset,
    bool UseScratch = false>
  __device__ void populate_field_ops(
    WeierstrassAddAssignCols<F, NumWords, NumLimbs, NumWitnesses>& col,
    const AffinePoint<NumWords>& p,
    const AffinePoint<NumWords>& q,
    const uint32_t modulus[NumWords],
    const uint32_t mod_minus_two[NumWords],
    const uint32_t rr[NumWords],
    uint32_t m0,
    F* byte_trace,
    fp_op::FpScratch<NumLimbs, NumWitnesses>* scratch = nullptr)
  {
    // slope_numerator = q.y - p.y
    uint32_t slope_numerator[NumWords];
    fp_op::populate_sub<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
      col.slope_numerator, q.y, p.y, slope_numerator, modulus, false, byte_trace);

    // slope_denominator = q.x - p.x
    uint32_t slope_denominator[NumWords];
    fp_op::populate_sub<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
      col.slope_denominator, q.x, p.x, slope_denominator, modulus, false, byte_trace);

    // inverse_check = 1 / slope_denominator (verifies denominator != 0)
    // For padding rows (all zeros), denominator is 0 — use 0/0=0.
    bool denom_is_zero = true;
    for (int i = 0; i < NumWords; i++) {
      if (slope_denominator[i] != 0) { denom_is_zero = false; break; }
    }
    uint32_t inverse_numerator[NumWords] = {0};
    if (!denom_is_zero) inverse_numerator[0] = 1;
    uint32_t inverse_result[NumWords];
    fp_op::populate_div_mont<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset, UseScratch>(
      col.inverse_check, inverse_numerator, slope_denominator, inverse_result, modulus, mod_minus_two, rr, m0, byte_trace, scratch);

    // slope = slope_numerator / slope_denominator
    uint32_t slope[NumWords];
    fp_op::populate_div_mont<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset, UseScratch>(
      col.slope, slope_numerator, slope_denominator, slope, modulus, mod_minus_two, rr, m0, byte_trace, scratch);

    // slope_squared = slope * slope
    uint32_t slope_squared[NumWords];
    fp_op::populate_mul<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset, UseScratch>(
      col.slope_squared, slope, slope, slope_squared, modulus, false, byte_trace, scratch);

    // p_x_plus_q_x = p.x + q.x
    uint32_t p_x_plus_q_x[NumWords];
    fp_op::populate_add<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
      col.p_x_plus_q_x, p.x, q.x, p_x_plus_q_x, modulus, false, byte_trace);

    // x3 = slope^2 - (p.x + q.x)
    uint32_t x3_ins[NumWords];
    fp_op::populate_sub<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
      col.x3_ins, slope_squared, p_x_plus_q_x, x3_ins, modulus, false, byte_trace);

    // x3_range: verify x3 < modulus
    fp_lt::populate<F, NumLimbs, NumWitnesses>(col.x3_range, x3_ins, modulus, byte_trace);

    // p_x_minus_x = p.x - x3
    uint32_t p_x_minus_x[NumWords];
    fp_op::populate_sub<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
      col.p_x_minus_x, p.x, x3_ins, p_x_minus_x, modulus, false, byte_trace);

    // slope_times_p_x_minus_x = slope * (p.x - x3)
    uint32_t slope_times_p_x_minus_x[NumWords];
    fp_op::populate_mul<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset, UseScratch>(
      col.slope_times_p_x_minus_x, slope, p_x_minus_x, slope_times_p_x_minus_x, modulus, false, byte_trace, scratch);

    // y3 = slope * (p.x - x3) - p.y
    uint32_t y3_ins[NumWords];
    fp_op::populate_sub<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
      col.y3_ins, slope_times_p_x_minus_x, p.y, y3_ins, modulus, false, byte_trace);

    // y3_range: verify y3 < modulus
    fp_lt::populate<F, NumLimbs, NumWitnesses>(col.y3_range, y3_ins, modulus, byte_trace);
  }

  /// Convert elliptic curve addition event to trace row (u64 version).
  /// NumWords = NumWordsDword (= WordsCurvePoint = Limbs/4).
  /// Each u64 dword in event.p/q is reinterpreted as 2 u32 words for AffinePoint.
  template <
    class F,
    const int NumWords,
    const int NumLimbs,
    const int NumWitnesses,
    const int NumBitsPerLimb,
    const int WitnessOffset,
    bool UseScratch = false>
  __device__ inline void add_event_to_row(
    const WeierstrassAddEvent<NumWords>* events,
    WeierstrassAddAssignCols<F, NumWords, NumLimbs, NumWitnesses>& col,
    const size_t idx,
    const size_t count,
    const uint32_t modulus[NumWords],
    const uint32_t mod_minus_two[NumWords],
    const uint32_t rr[NumWords],
    uint32_t m0,
    F* byte_trace,
    fp_op::FpScratch<NumLimbs, NumWitnesses>* scratch = nullptr)
  {
    if (idx < count) {
      const auto& event = events[idx];

      // Reinterpret u64 dwords as u32 words in LE order.
      const uint32_t* p_words = reinterpret_cast<const uint32_t*>(event.p);
      const uint32_t* q_words = reinterpret_cast<const uint32_t*>(event.q);
      AffinePoint<NumWords> p(AffinePoint<NumWords>::from_words_le(p_words));
      AffinePoint<NumWords> q(AffinePoint<NumWords>::from_words_le(q_words));

      // Populate basic columns.
      col.is_real = F::one();
      col.chunk = F::from_canonical_u32(event.chunk);
      col.clk = F::from_canonical_u32(event.clk);

      // Populate SyscallAddrGadget for p_ptr and q_ptr.
      constexpr int num_limbs = NumWords;  // NumWordsDword = WordsCurvePoint
      col.p_ptr.populate(byte_trace, event.p_ptr, num_limbs * 2);
      col.q_ptr.populate(byte_trace, event.q_ptr, num_limbs * 2);

      // Perform Weierstrass addition field operations.
      populate_field_ops<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset, UseScratch>(
        col, p, q, modulus, mod_minus_two, rr, m0, byte_trace, scratch);

      // Populate memory access columns and address gadgets (one per u64 dword).
      for (size_t i = 0; i < NumWords; i++) {
        populate_write_cols_u8<F>(col.p_access[i], event.p_memory_records[i], byte_trace);
        populate_read_cols_u8<F>(col.q_access[i], event.q_memory_records[i], byte_trace);
        col.p_addrs[i].populate(byte_trace, event.p_ptr, 8 * i);
        col.q_addrs[i].populate(byte_trace, event.q_ptr, 8 * i);
      }
    } else {
      // Padding row: q_x=1, q_y=1 so field_ops constraints (not gated by is_real) are satisfied.
      // Half of the dword count is for one field element.
      constexpr int num_words_field_element = NumWords / 2;

      // Set q_access[0] value to 1 (q_x first dword = 1).
      col.q_access[0].inner.access.value._0[0] = F::ONE;
      col.q_access[0].prev_value_u8.low_bytes[0] = F::ONE;
      // Set q_access[num_words_field_element] value to 1 (q_y first dword = 1).
      col.q_access[num_words_field_element].inner.access.value._0[0] = F::ONE;
      col.q_access[num_words_field_element].prev_value_u8.low_bytes[0] = F::ONE;

      // Build AffinePoints with q=(1,1).
      uint32_t q_words_pad[2 * NumWords] = {0};
      q_words_pad[0] = 1;  // q_x low word = 1
      q_words_pad[NumWords] = 1;  // q_y low word = 1

      AffinePoint<NumWords> p_zero = AffinePoint<NumWords>::zero();
      AffinePoint<NumWords> q_pad = AffinePoint<NumWords>::from_words_le(q_words_pad);

      populate_field_ops<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset, UseScratch>(
        col, p_zero, q_pad, modulus, mod_minus_two, rr, m0, NULL, scratch);
    }
  }

  /// Populate field operations for Weierstrass curve point doubling.
  /// Computes: result = 2 * p using the formula:
  /// slope = (3 * p.x^2 + a) / (2 * p.y)
  /// x3 = slope^2 - 2 * p.x
  /// y3 = slope * (p.x - x3) - p.y
  template <
    class F,
    const int NumWords,
    const int NumLimbs,
    const int NumWitnesses,
    const int NumBitsPerLimb,
    const int WitnessOffset,
    bool UseScratch = false>
  __PICO_HOSTDEV__ void populate_double_field_ops(
    WeierstrassDoubleAssignCols<F, NumWords, NumLimbs, NumWitnesses>& col,
    const AffinePoint<NumWords>& p,
    const uint32_t a[NumWords],
    const uint32_t modulus[NumWords],
    const uint32_t mod_minus_two[NumWords],
    const uint32_t rr[NumWords],
    uint32_t m0,
    F* byte_trace,
    fp_op::FpScratch<NumLimbs, NumWitnesses>* scratch = nullptr)
  {
    uint32_t three[NumWords] = {0};
    three[0] = 3;
    uint32_t two[NumWords] = {0};
    two[0] = 2;

    // Calculate slope = (3 * p.x^2 + a) / (2 * p.y).
    uint32_t p_x_squared[NumWords];
    fp_op::populate_mul<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset, UseScratch>(
      col.p_x_squared, p.x, p.x, p_x_squared, modulus, false, byte_trace, scratch);

    uint32_t p_x_squared_times_3[NumWords];
    fp_op::populate_mul<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset, UseScratch>(
      col.p_x_squared_times_3, p_x_squared, three, p_x_squared_times_3, modulus, false, byte_trace, scratch);

    uint32_t slope_numerator[NumWords];
    fp_op::populate_add<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
      col.slope_numerator, a, p_x_squared_times_3, slope_numerator, modulus, false, byte_trace);

    uint32_t slope_denominator[NumWords];
    fp_op::populate_mul<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset, UseScratch>(
      col.slope_denominator, two, p.y, slope_denominator, modulus, false, byte_trace, scratch);

    // inverse_check = 1 / slope_denominator (verifies denominator != 0)
    // For padding rows (all zeros), denominator is 0 — use 0/0=0.
    bool denom_is_zero = true;
    for (int i = 0; i < NumWords; i++) {
      if (slope_denominator[i] != 0) { denom_is_zero = false; break; }
    }
    uint32_t inverse_numerator[NumWords] = {0};
    if (!denom_is_zero) inverse_numerator[0] = 1;
    uint32_t inverse_result[NumWords];
    fp_op::populate_div_mont<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset, UseScratch>(
      col.inverse_check, inverse_numerator, slope_denominator, inverse_result, modulus, mod_minus_two, rr, m0, byte_trace, scratch);

    uint32_t slope[NumWords];
    fp_op::populate_div_mont<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset, UseScratch>(
      col.slope, slope_numerator, slope_denominator, slope, modulus, mod_minus_two, rr, m0, byte_trace, scratch);

    // Calculate x3 = slope^2 - 2 * p.x.
    uint32_t slope_squared[NumWords];
    fp_op::populate_mul<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset, UseScratch>(
      col.slope_squared, slope, slope, slope_squared, modulus, false, byte_trace, scratch);

    uint32_t p_x_plus_p_x[NumWords];
    fp_op::populate_add<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
      col.p_x_plus_p_x, p.x, p.x, p_x_plus_p_x, modulus, false, byte_trace);

    uint32_t x[NumWords];
    fp_op::populate_sub<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
      col.x3_ins, slope_squared, p_x_plus_p_x, x, modulus, false, byte_trace);

    // Calculate y3 = slope * (p.x - x3) - p.y.
    uint32_t p_x_minus_x[NumWords];
    fp_op::populate_sub<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
      col.p_x_minus_x, p.x, x, p_x_minus_x, modulus, false, byte_trace);

    uint32_t slope_times_p_x_minus_x[NumWords];
    fp_op::populate_mul<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset, UseScratch>(
      col.slope_times_p_x_minus_x, slope, p_x_minus_x, slope_times_p_x_minus_x, modulus, false, byte_trace, scratch);

    uint32_t y[NumWords];
    fp_op::populate_sub<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
      col.y3_ins, slope_times_p_x_minus_x, p.y, y, modulus, false, byte_trace);

    // Range check x3 and y3 against the field modulus.
    fp_lt::populate<F, NumLimbs, NumWitnesses>(col.x3_range, x, modulus, byte_trace);
    fp_lt::populate<F, NumLimbs, NumWitnesses>(col.y3_range, y, modulus, byte_trace);
  }

  /// Convert Weierstrass doubling event to trace row (u64 branch).
  template <
    class F,
    const int NumWordsDword,
    const int NumLimbs,
    const int NumWitnesses,
    const int NumBitsPerLimb,
    const int WitnessOffset,
    bool UseScratch = false>
  __device__ inline void double_event_to_row(
    const WeierstrassDoubleEvent<NumWordsDword>* events,
    WeierstrassDoubleAssignCols<F, NumWordsDword, NumLimbs, NumWitnesses>& col,
    const size_t idx,
    const size_t count,
    const uint32_t a[NumWordsDword],
    const uint32_t modulus[NumWordsDword],
    const uint32_t mod_minus_two[NumWordsDword],
    const uint32_t rr[NumWordsDword],
    uint32_t m0,
    F* byte_trace,
    fp_op::FpScratch<NumLimbs, NumWitnesses>* scratch = nullptr)
  {
    // NumWordsDword is the number of u64 dwords per curve point.
    // The old NumWords (u32 word count) = NumWordsDword (since we reinterpret u64[] as u32[2*N]).
    // AffinePoint still uses NumWordsDword as template param (u32 word count for internal math).
    constexpr int NumWords = NumWordsDword;

    if (idx < count) {
      const auto& event = events[idx];

      // Reinterpret u64 dwords as u32 words for AffinePoint construction.
      const uint32_t* p_as_u32 = reinterpret_cast<const uint32_t*>(event.p);
      uint32_t p_words[2 * NumWords];
      for (int i = 0; i < 2 * NumWords; i++) p_words[i] = p_as_u32[i];

      AffinePoint<NumWords> p(AffinePoint<NumWords>::from_words_le(p_words));

      col.is_real = F::one();
      col.chunk = F::from_canonical_u32(event.chunk);
      col.clk = F::from_canonical_u32(event.clk);

      // Populate SyscallAddrGadget for p_ptr.
      uint64_t len = static_cast<uint64_t>(NumWordsDword) * 2;
      col.p_ptr.populate(byte_trace, event.p_ptr, len);

      // Perform Weierstrass doubling field operations (includes x3/y3 range checks).
      populate_double_field_ops<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset, UseScratch>(
        col, p, a, modulus, mod_minus_two, rr, m0, byte_trace, scratch);

      // Populate memory access columns and address gadgets using u64-aware helpers.
      for (size_t i = 0; i < NumWordsDword; i++) {
        populate_write_cols_u8(col.p_access[i], event.p_memory_records[i], byte_trace);
        col.p_addrs[i].populate(byte_trace, event.p_ptr, 8 * i);
      }
    } else {
      // Padding row: use point (0, 1) to avoid division-by-zero in slope denominator (2*p_y).
      // For curves with a != 0 (e.g., secp256r1), using (0,0) would cause
      // slope_numerator = a != 0 divided by slope_denominator = 0.
      AffinePoint<NumWords> pad_point = AffinePoint<NumWords>::zero();
      pad_point.y[0] = 1;  // Set y = 1
      populate_double_field_ops<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset, UseScratch>(
        col, pad_point, a, modulus, mod_minus_two, rr, m0, NULL, scratch);

      // Set p_access prev_value to match the padding point (0, 1).
      // Constraints read p_y from p_access.inner.prev_value via ungated field op
      // polynomial identities, so prev_value must be consistent with field ops.
      constexpr int num_dwords_per_field = NumWordsDword / 2;
      col.p_access[num_dwords_per_field].inner.prev_value._0[0] = F::one();
      col.p_access[num_dwords_per_field].prev_value_u8.low_bytes[0] = F::one();
    }
  }

  /// Populate field operations for point decompression.
  /// Given x-coordinate, computes y = sqrt(x^3 + ax + b).
  template <
    class F,
    const int NumWords,
    const int NumLimbs,
    const int NumWitnesses,
    const int NumBitsPerLimb,
    const int WitnessOffset,
    bool UseScratch = false>
  __PICO_HOSTDEV__ void populate_decompress_field_ops(
    WeierstrassDecompressCols<F, NumWords, NumLimbs, NumWitnesses>& col,
    const uint32_t x[NumWords],
    const uint32_t modulus[NumWords],
    const uint32_t mod_minus_two[NumWords],
    const uint32_t sqrt_exp[NumWords],
    const uint32_t weierstrass_a[NumWords],
    const uint32_t weierstrass_b[NumWords],
    const uint32_t rr[NumWords],
    uint32_t m0,
    F* byte_trace,
    fp_op::FpScratch<NumLimbs, NumWitnesses>* scratch = nullptr)
  {
    fp_lt::populate<F, NumLimbs, NumWitnesses>(col.range_x, x, modulus, byte_trace);

    // Calculate x^2 and x^3.
    uint32_t x_2[NumWords];
    fp_op::populate_mul<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset, UseScratch>(
      col.x_2, x, x, x_2, modulus, false, byte_trace, scratch);

    uint32_t x_3[NumWords];
    fp_op::populate_mul<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset, UseScratch>(
      col.x_3, x_2, x, x_3, modulus, false, byte_trace, scratch);

    // Calculate ax + b via inner product: [a, b] · [x, 1].
    uint32_t one[NumWords] = {0};
    one[0] = 1;
    uint32_t ax_plus_b[NumWords];
    const uint32_t* ip_a[2] = {weierstrass_a, weierstrass_b};
    const uint32_t* ip_b[2] = {x, one};
    fp_op::populate_inner_prod<F, 2, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
      col.ax_plus_b, ip_a, ip_b, ax_plus_b, modulus, byte_trace);

    // Calculate x^3 + ax + b.
    uint32_t x_3_plus_ax_plus_b[NumWords];
    fp_op::populate_add<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
      col.x_3_plus_ax_plus_b, x_3, ax_plus_b, x_3_plus_ax_plus_b, modulus, false, byte_trace);

    // Compute square root: y = (x^3 + ax + b)^sqrt_exp.
    uint32_t zero[NumWords] = {0};
    uint32_t sqrt[NumWords];
#ifdef __CUDA_ARCH__
    // GPU path: Montgomery-accelerated modpow for sqrt
    fast_mont::mont_modpow<NumWords>(x_3_plus_ax_plus_b, sqrt_exp, modulus, rr, m0, sqrt);
#else
    // CPU path: naive modpow
    fp_op::modpow_words<NumWords, NumLimbs>(x_3_plus_ax_plus_b, sqrt_exp, modulus, sqrt);
#endif

    // Populate y and -y columns.
    uint32_t neg_y[NumWords];
    fp_op::populate_mul<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset, UseScratch>(
      col.y.multiplication, sqrt, sqrt, neg_y, modulus, false, byte_trace, scratch);

    // Rewrite result with the square root.
    fp_op::words_to_t_bytes_le<F, NumWords, NumLimbs>(sqrt, col.y.multiplication.result, byte_trace);
    fp_lt::populate<F, NumLimbs, NumWitnesses>(col.y.range, sqrt, modulus, byte_trace);
    col.y.lsb = sqrt[0] & 1 ? F::one() : F::zero();

    fp_op::populate_sub<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
      col.neg_y, zero, sqrt, neg_y, modulus, false, byte_trace);

    // Range check neg_y < modulus.
    fp_lt::populate<F, NumLimbs, NumWitnesses>(col.neg_y_range_check, neg_y, modulus, byte_trace);

    // Add byte lookup for LSB extraction.
    if (byte_trace) { byte::handle_byte_lookup_event(byte_trace, ByteOpcode::AND, sqrt[0], 1); }
  }

  /// Convert point decompression event to trace row.
  template <
    class F,
    const int NumWords,
    const int NumLimbs,
    const int NumWitnesses,
    const int NumBitsPerLimb,
    const int WitnessOffset,
    bool UseScratch = false>
  __PICO_HOSTDEV__ inline void decompress_event_to_row(
    const EllipticCurveDecompressEventFFI<NumWords, NumLimbs>* events,
    WeierstrassDecompressCols<F, NumWords, NumLimbs, NumWitnesses>& col,
    const size_t idx,
    const size_t count,
    const uint32_t dummy_x[NumWords],
    const uint32_t sqrt_exp[NumWords],
    const uint32_t a[NumWords],
    const uint32_t b[NumWords],
    const uint32_t modulus[NumWords],
    const uint32_t mod_minus_two[NumWords],
    const uint32_t rr[NumWords],
    uint32_t m0,
    F* byte_trace,
    fp_op::FpScratch<NumLimbs, NumWitnesses>* scratch = nullptr)
  {
    if (idx < count) {
      const auto& event = events[idx];

      col.is_real = F::one();
      col.chunk = F::from_canonical_u32(event.chunk);
      col.clk = F::from_canonical_u32(event.clk);

      // Populate SyscallAddrGadget for ptr.
      constexpr int num_limbs_val = NumLimbs;  // BaseField::Limbs
      col.ptr.populate(byte_trace, event.ptr, num_limbs_val * 2);
      col.sign_bit = event.sign_bit ? F::one() : F::zero();

      // Reconstruct x from bytes.
      uint32_t x[NumWords] = {0};
      for (int i = 0; i < NumLimbs; i++) {
        x[i / 4] |= event.x_bytes[i] << (8 * (i % 4));
      }

      // Perform Weierstrass decompression.
      populate_decompress_field_ops<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset, UseScratch>(
        col, x, modulus, mod_minus_two, sqrt_exp, a, b, rr, m0, byte_trace, scratch);

      // Populate memory access columns and address gadgets (u64 value, u8 cols).
      // NumWords/2 entries because each u64 dword = 2 u32 words.
      constexpr int NumDwords = NumWords / 2;
      constexpr int num_limbs_dwords = NumDwords;  // = NumWords/2 = WordsFieldElement
      for (size_t i = 0; i < NumDwords; i++) {
        populate_read_cols_u8<F>(col.x_access[i], event.x_memory_records[i], byte_trace);
        populate_write_cols_u8<F>(col.y_access[i], event.y_memory_records[i], byte_trace);
        // x_addrs: ptr + num_limbs_dwords*8 + 8*i  (x is at offset num_limbs from ptr)
        col.x_addrs[i].populate(byte_trace, event.ptr + num_limbs_dwords * 8, 8 * i);
        // y_addrs: ptr + 8*i
        col.y_addrs[i].populate(byte_trace, event.ptr, 8 * i);
      }
    } else {
      // Padding row with dummy x.
      populate_decompress_field_ops<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset, UseScratch>(
        col, dummy_x, modulus, mod_minus_two, sqrt_exp, a, b, rr, m0, NULL, scratch);

      // Populate dummy x access columns: combine pairs of u32 words into u64 values.
      constexpr int NumDwords = NumWords / 2;
      for (size_t i = 0; i < NumDwords; i++) {
        // Combine two u32 words into one u64.
        uint64_t val = (uint64_t)dummy_x[2*i] | ((uint64_t)dummy_x[2*i+1] << 32);
        // Set value as 4 u16 limbs.
        col.x_access[i].inner.access.value._0[0] = F::from_canonical_u16(val & 0xffff);
        col.x_access[i].inner.access.value._0[1] = F::from_canonical_u16((val >> 16) & 0xffff);
        col.x_access[i].inner.access.value._0[2] = F::from_canonical_u16((val >> 32) & 0xffff);
        col.x_access[i].inner.access.value._0[3] = F::from_canonical_u16((val >> 48) & 0xffff);

        // Set prev_value_u8 low bytes (low byte of each u16 limb).
        col.x_access[i].prev_value_u8.low_bytes[0] = F::from_canonical_u8(val & 0xff);
        col.x_access[i].prev_value_u8.low_bytes[1] = F::from_canonical_u8((val >> 16) & 0xff);
        col.x_access[i].prev_value_u8.low_bytes[2] = F::from_canonical_u8((val >> 32) & 0xff);
        col.x_access[i].prev_value_u8.low_bytes[3] = F::from_canonical_u8((val >> 48) & 0xff);
      }
    }
  }

  /// Populate lexicographic choice columns for decompression.
  /// Chooses between y and -y based on sign bit and lexicographic ordering.
  template <
    class F,
    const int NumWords,
    const int NumLimbs,
    const int NumWitnesses,
    const int NumBitsPerLimb,
    const int WitnessOffset>
  __PICO_HOSTDEV__ inline void decompress_lexicographic(
    const EllipticCurveDecompressEventFFI<NumWords, NumLimbs>* events,
    WeierstrassDecompressCols<F, NumWords, NumLimbs, NumWitnesses>& wcol,
    LexicographicChoiceCols<F, NumWords, NumLimbs, NumWitnesses>& col,
    const size_t idx,
    const size_t count,
    const uint32_t modulus[NumWords],
    const uint32_t mod_minus_two[NumWords],
    F* byte_trace)
  {
    if (idx < count) {
      const auto& event = events[idx];

      // Determine LSB of computed y.
      const uint8_t lsb = wcol.y.lsb == F::one() ? 1 : 0;

      // Reconstruct decompressed y from bytes.
      uint32_t decompressed_y[NumWords] = {0};
      for (int i = 0; i < NumLimbs; i++) {
        decompressed_y[i / 4] |= event.decompressed_y_bytes[i] << (8 * (i % 4));
      }

      uint32_t neg_y[NumWords];
      fp_op::sub_words_with_borrow<NumWords>(modulus, decompressed_y, neg_y);

      // Check if decompressed y matches computed sqrt(y).
      const bool is_y_eq_sqrt_y_result = (event.decompressed_y_bytes[0] & 1) == lsb;
      col.is_y_eq_sqrt_y_result = is_y_eq_sqrt_y_result ? F::one() : F::zero();

      // Perform lexicographic comparison based on sign bit.
      if (event.sign_bit) {
        col.when_sqrt_y_res_is_lt = !is_y_eq_sqrt_y_result ? F::one() : F::zero();
        col.when_neg_y_res_is_lt = is_y_eq_sqrt_y_result ? F::one() : F::zero();
        fp_lt::populate<F, NumLimbs, NumWitnesses>(col.comparison_lt_cols, neg_y, decompressed_y, byte_trace);
      } else {
        col.when_sqrt_y_res_is_lt = is_y_eq_sqrt_y_result ? F::one() : F::zero();
        col.when_neg_y_res_is_lt = !is_y_eq_sqrt_y_result ? F::one() : F::zero();
        fp_lt::populate<F, NumLimbs, NumWitnesses>(col.comparison_lt_cols, decompressed_y, neg_y, byte_trace);
      }
    }
  }
} // namespace pico_gpu::curve_op