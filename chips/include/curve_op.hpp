#pragma once

#include "types.hpp"
#include "fp_op.hpp"
#include "fp_lt.hpp"
#include "curves.hpp"
#include "./memory_read_write.hpp"

namespace pico_gpu::curve_op {
  using curves::AffinePoint;

  /// Populate field operations for Weierstrass curve point addition.
  /// Computes: result = p + q using the formula:
  /// slope = (q.y - p.y) / (q.x - p.x)
  /// x3 = slope^2 - p.x - q.x
  /// y3 = slope * (p.x - x3) - p.y
  template <
    class F,
    const int NumWords,
    const int NumLimbs,
    const int NumWitnesses,
    const int NumBitsPerLimb,
    const int WitnessOffset>
  __PICO_HOSTDEV__ void populate_field_ops(
    WeierstrassAddAssignCols<F, NumWords, NumLimbs, NumWitnesses>& col,
    const AffinePoint<NumWords>& p,
    const AffinePoint<NumWords>& q,
    const uint32_t modulus[NumWords],
    const uint32_t mod_minus_two[NumWords],
    F* byte_trace)
  {
    // Calculate slope = (q.y - p.y) / (q.x - p.x).
    uint32_t slope_numerator[NumWords];
    fp_op::populate_sub<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
      col.slope_numerator, q.y, p.y, slope_numerator, modulus, false, byte_trace);

    uint32_t slope_denominator[NumWords];
    fp_op::populate_sub<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
      col.slope_denominator, q.x, p.x, slope_denominator, modulus, false, byte_trace);

    uint32_t slope[NumWords];
    fp_op::populate_div<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
      col.slope, slope_numerator, slope_denominator, slope, modulus, mod_minus_two, byte_trace);

    // Calculate x3 = slope^2 - (p.x + q.x).
    uint32_t slope_squared[NumWords];
    fp_op::populate_mul<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
      col.slope_squared, slope, slope, slope_squared, modulus, false, byte_trace);

    uint32_t p_x_plus_q_x[NumWords];
    fp_op::populate_add<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
      col.p_x_plus_q_x, p.x, q.x, p_x_plus_q_x, modulus, false, byte_trace);

    uint32_t x3_ins[NumWords];
    fp_op::populate_sub<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
      col.x3_ins, slope_squared, p_x_plus_q_x, x3_ins, modulus, false, byte_trace);

    // Calculate y3 = slope * (p.x - x3) - p.y.
    uint32_t p_x_minus_x[NumWords];
    fp_op::populate_sub<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
      col.p_x_minus_x, p.x, x3_ins, p_x_minus_x, modulus, false, byte_trace);

    uint32_t slope_times_p_x_minus_x[NumWords];
    fp_op::populate_mul<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
      col.slope_times_p_x_minus_x, slope, p_x_minus_x, slope_times_p_x_minus_x, modulus, false, byte_trace);

    uint32_t y3_ins[NumWords];
    fp_op::populate_sub<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
      col.y3_ins, slope_times_p_x_minus_x, p.y, y3_ins, modulus, false, byte_trace);
  }

  /// Convert elliptic curve addition event to trace row.
  template <
    class F,
    const int NumWords,
    const int NumLimbs,
    const int NumWitnesses,
    const int NumBitsPerLimb,
    const int WitnessOffset>
  __PICO_HOSTDEV__ inline void add_event_to_row(
    const EllipticCurveAddEvent<NumWords>* events,
    WeierstrassAddAssignCols<F, NumWords, NumLimbs, NumWitnesses>& col,
    const size_t idx,
    const size_t count,
    const uint32_t modulus[NumWords],
    const uint32_t mod_minus_two[NumWords],
    F* byte_trace)
  {
    if (idx < count) {
      const auto& event = events[idx];
      AffinePoint<NumWords> p(AffinePoint<NumWords>::from_words_le(event.p));
      AffinePoint<NumWords> q(AffinePoint<NumWords>::from_words_le(event.q));

      // Populate basic columns.
      col.is_real = F::one();
      col.chunk = F::from_canonical_u32(event.chunk);
      col.clk = F::from_canonical_u32(event.clk);
      col.p_ptr = F::from_canonical_u32(event.p_ptr);
      col.q_ptr = F::from_canonical_u32(event.q_ptr);

      // Perform Weierstrass addition.
      populate_field_ops<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
        col, p, q, modulus, mod_minus_two, byte_trace);

      // Populate memory access columns.
      for (size_t i = 0; i < 2 * NumWords; i++) {
        memory_read_write::populate(col.p_access[i], event.p_memory_records[i], byte_trace);
        memory_read_write::populate(col.q_access[i], event.q_memory_records[i], byte_trace);
      }
    } else {
      // Padding row with zero points.
      populate_field_ops<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
        col, AffinePoint<NumWords>::zero(), AffinePoint<NumWords>::zero(), modulus, mod_minus_two, NULL);
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
    const int WitnessOffset>
  __PICO_HOSTDEV__ void populate_double_field_ops(
    WeierstrassDoubleAssignCols<F, NumWords, NumLimbs, NumWitnesses>& col,
    const AffinePoint<NumWords>& p,
    const uint32_t a[NumWords],
    const uint32_t modulus[NumWords],
    const uint32_t mod_minus_two[NumWords],
    F* byte_trace)
  {
    uint32_t three[NumWords] = {0};
    three[0] = 3;
    uint32_t two[NumWords] = {0};
    two[0] = 2;

    // Calculate slope = (3 * p.x^2 + a) / (2 * p.y).
    uint32_t p_x_squared[NumWords];
    fp_op::populate_mul<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
      col.p_x_squared, p.x, p.x, p_x_squared, modulus, false, byte_trace);

    uint32_t p_x_squared_times_3[NumWords];
    fp_op::populate_mul<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
      col.p_x_squared_times_3, p_x_squared, three, p_x_squared_times_3, modulus, false, byte_trace);

    uint32_t slope_numerator[NumWords];
    fp_op::populate_add<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
      col.slope_numerator, a, p_x_squared_times_3, slope_numerator, modulus, false, byte_trace);

    uint32_t slope_denominator[NumWords];
    fp_op::populate_mul<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
      col.slope_denominator, two, p.y, slope_denominator, modulus, false, byte_trace);

    uint32_t slope[NumWords];
    fp_op::populate_div<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
      col.slope, slope_numerator, slope_denominator, slope, modulus, mod_minus_two, byte_trace);

    // Calculate x3 = slope^2 - 2 * p.x.
    uint32_t slope_squared[NumWords];
    fp_op::populate_mul<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
      col.slope_squared, slope, slope, slope_squared, modulus, false, byte_trace);

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
    fp_op::populate_mul<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
      col.slope_times_p_x_minus_x, slope, p_x_minus_x, slope_times_p_x_minus_x, modulus, false, byte_trace);

    uint32_t y[NumWords];
    fp_op::populate_sub<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
      col.y3_ins, slope_times_p_x_minus_x, p.y, y, modulus, false, byte_trace);
  }

  /// Convert elliptic curve doubling event to trace row.
  template <
    class F,
    const int NumWords,
    const int NumLimbs,
    const int NumWitnesses,
    const int NumBitsPerLimb,
    const int WitnessOffset>
  __PICO_HOSTDEV__ inline void double_event_to_row(
    const EllipticCurveDoubleEvent<NumWords>* events,
    WeierstrassDoubleAssignCols<F, NumWords, NumLimbs, NumWitnesses>& col,
    const size_t idx,
    const size_t count,
    const uint32_t a[NumWords],
    const uint32_t modulus[NumWords],
    const uint32_t mod_minus_two[NumWords],
    F* byte_trace)
  {
    if (idx < count) {
      const auto& event = events[idx];
      AffinePoint<NumWords> p(AffinePoint<NumWords>::from_words_le(event.p));

      col.is_real = F::one();
      col.chunk = F::from_canonical_u32(event.chunk);
      col.clk = F::from_canonical_u32(event.clk);
      col.p_ptr = F::from_canonical_u32(event.p_ptr);

      // Perform Weierstrass doubling.
      populate_double_field_ops<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
        col, p, a, modulus, mod_minus_two, byte_trace);

      // Populate memory access columns.
      for (size_t i = 0; i < 2 * NumWords; i++) {
        memory_read_write::populate(col.p_access[i], event.p_memory_records[i], byte_trace);
      }
    } else {
      // Padding row with zero point.
      populate_double_field_ops<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
        col, AffinePoint<NumWords>::zero(), a, modulus, mod_minus_two, NULL);
    }
  }

  /// Populate field operations for point decompression.
  /// Given x-coordinate, computes y = sqrt(x^3 + b).
  template <
    class F,
    const int NumWords,
    const int NumLimbs,
    const int NumWitnesses,
    const int NumBitsPerLimb,
    const int WitnessOffset>
  __PICO_HOSTDEV__ void populate_decompress_field_ops(
    WeierstrassDecompressCols<F, NumWords, NumLimbs, NumWitnesses>& col,
    const uint32_t x[NumWords],
    const uint32_t modulus[NumWords],
    const uint32_t mod_minus_two[NumWords],
    const uint32_t sqrt_exp[NumWords],
    const uint32_t weierstrass_b[NumWords],
    F* byte_trace)
  {
    fp_lt::populate<F, NumLimbs, NumWitnesses>(col.range_x, x, modulus, byte_trace);

    // Calculate x^2 and x^3.
    uint32_t x_2[NumWords];
    fp_op::populate_mul<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
      col.x_2, x, x, x_2, modulus, false, byte_trace);

    uint32_t x_3[NumWords];
    fp_op::populate_mul<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
      col.x_3, x_2, x, x_3, modulus, false, byte_trace);

    // Calculate x^3 + b.
    uint32_t x_3_plus_b[NumWords];
    fp_op::populate_add<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
      col.x_3_plus_b, x_3, weierstrass_b, x_3_plus_b, modulus, false, byte_trace);

    // Compute square root: y = (x^3 + b)^sqrt_exp.
    uint32_t zero[NumWords] = {0};
    uint32_t sqrt[NumWords];
    fp_op::modpow_words<NumWords, NumLimbs>(x_3_plus_b, sqrt_exp, modulus, sqrt);

    // Populate y and -y columns.
    uint32_t neg_y[NumWords];
    fp_op::populate_mul<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
      col.y.multiplication, sqrt, sqrt, neg_y, modulus, false, byte_trace);

    // Rewrite result with the square root.
    fp_op::words_to_t_bytes_le<F, NumWords, NumLimbs>(sqrt, col.y.multiplication.result, byte_trace);
    fp_lt::populate<F, NumLimbs, NumWitnesses>(col.y.range, sqrt, modulus, byte_trace);
    col.y.lsb = sqrt[0] & 1 ? F::one() : F::zero();

    fp_op::populate_sub<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
      col.neg_y, zero, sqrt, neg_y, modulus, false, byte_trace);

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
    const int WitnessOffset>
  __PICO_HOSTDEV__ inline void decompress_event_to_row(
    const EllipticCurveDecompressEventFFI<NumWords, NumLimbs>* events,
    WeierstrassDecompressCols<F, NumWords, NumLimbs, NumWitnesses>& col,
    const size_t idx,
    const size_t count,
    const uint32_t dummy_x[NumWords],
    const uint32_t sqrt_exp[NumWords],
    const uint32_t b[NumWords],
    const uint32_t modulus[NumWords],
    const uint32_t mod_minus_two[NumWords],
    F* byte_trace)
  {
    if (idx < count) {
      const auto& event = events[idx];

      col.is_real = F::one();
      col.chunk = F::from_canonical_u32(event.chunk);
      col.clk = F::from_canonical_u32(event.clk);
      col.ptr = F::from_canonical_u32(event.ptr);
      col.sign_bit = event.sign_bit ? F::one() : F::zero();

      // Reconstruct x from bytes.
      uint32_t x[NumWords] = {0};
      for (int i = 0; i < NumLimbs; i++) {
        x[i / 4] |= event.x_bytes[i] << (8 * (i % 4));
      }

      // Perform Weierstrass decompression.
      populate_decompress_field_ops<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
        col, x, modulus, mod_minus_two, sqrt_exp, b, byte_trace);

      // Populate memory access columns.
      for (size_t i = 0; i < NumWords; i++) {
        memory_read_write::populate(col.x_access[i], event.x_memory_records[i], byte_trace);
        memory_read_write::populate_write(col.y_access[i], event.y_memory_records[i], byte_trace);
      }
    } else {
      // Padding row with dummy x.
      populate_decompress_field_ops<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
        col, dummy_x, modulus, mod_minus_two, sqrt_exp, b, NULL);

      // Populate dummy x access columns.
      for (size_t i = 0; i < NumWords; i++) {
        col.x_access[i].access.value._0[0] = F::from_canonical_u8((dummy_x[i] >> 0) & 0xff);
        col.x_access[i].access.value._0[1] = F::from_canonical_u8((dummy_x[i] >> 8) & 0xff);
        col.x_access[i].access.value._0[2] = F::from_canonical_u8((dummy_x[i] >> 16) & 0xff);
        col.x_access[i].access.value._0[3] = F::from_canonical_u8((dummy_x[i] >> 24) & 0xff);
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

      // Perform range checks based on which value was chosen.
      if (is_y_eq_sqrt_y_result) {
        fp_lt::populate<F, NumLimbs, NumWitnesses>(col.neg_y_range_check, neg_y, modulus, byte_trace);
      } else {
        fp_lt::populate<F, NumLimbs, NumWitnesses>(col.neg_y_range_check, decompressed_y, modulus, byte_trace);
      }

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