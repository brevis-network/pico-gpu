#pragma once

#include "types.hpp"
#include "fp_op.hpp"
#include "fp_lt.hpp"
#include "curves.hpp"
#include "./memory_read_write.hpp"

namespace pico_gpu::edwards_op {
  using curves::AffinePoint;

  /// Populate field operations for Edwards curve point addition.
  /// Edwards curve formula: x3 = (x1*y2 + y1*x2) / (1 + d*x1*y1*x2*y2)
  ///                        y3 = (y1*y2 - x1*x2) / (1 - d*x1*y1*x2*y2)
  template <
    class F,
    const int NumWords,
    const int NumLimbs,
    const int NumWitnesses,
    const int NumBitsPerLimb,
    const int WitnessOffset>
  __PICO_HOSTDEV__ void populate_ed_add_field_ops(
    EdAddAssignCols<F, NumLimbs, NumWitnesses>& col,
    const AffinePoint<NumWords>& p,
    const AffinePoint<NumWords>& q,
    const uint32_t modulus[NumWords],
    const uint32_t mod_minus_two[NumWords],
    const uint32_t edwards_d[NumWords],
    F* byte_trace)
  {
    // Calculate x3 numerator: x1*y2 + y1*x2.
    uint32_t x3_numerator[NumWords];
    const uint32_t* p_x_q_x[2] = {p.x, q.x};
    const uint32_t* q_y_p_y[2] = {q.y, p.y};
    fp_op::populate_inner_prod<F, 2, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
      col.x3_numerator, p_x_q_x, q_y_p_y, x3_numerator, modulus, byte_trace);

    // Calculate y3 numerator: y1*y2 + x1*x2.
    uint32_t y3_numerator[NumWords];
    const uint32_t* p_y_p_x[2] = {p.y, p.x};
    const uint32_t* q_y_q_x[2] = {q.y, q.x};
    fp_op::populate_inner_prod<F, 2, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
      col.y3_numerator, p_y_p_x, q_y_q_x, y3_numerator, modulus, byte_trace);

    // Calculate f = x1*y1 * x2*y2.
    uint32_t x1_mul_y1[NumWords];
    fp_op::populate_mul<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
      col.x1_mul_y1, p.x, p.y, x1_mul_y1, modulus, false, byte_trace);

    uint32_t x2_mul_y2[NumWords];
    fp_op::populate_mul<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
      col.x2_mul_y2, q.x, q.y, x2_mul_y2, modulus, false, byte_trace);

    uint32_t f[NumWords];
    fp_op::populate_mul<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
      col.f, x1_mul_y1, x2_mul_y2, f, modulus, false, byte_trace);

    // Calculate d*f for denominators.
    uint32_t d_mul_f[NumWords];
    fp_op::populate_mul<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
      col.d_mul_f, f, edwards_d, d_mul_f, modulus, false, byte_trace);

    // Calculate x3 = x3_numerator / (1 + d*f).
    uint32_t x3_ins[NumWords];
    fp_op::populate_den<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
      col.x3_ins, x3_numerator, d_mul_f, x3_ins, modulus, mod_minus_two, true, byte_trace);

    // Calculate y3 = y3_numerator / (1 - d*f).
    uint32_t y3_ins[NumWords];
    fp_op::populate_den<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
      col.y3_ins, y3_numerator, d_mul_f, y3_ins, modulus, mod_minus_two, false, byte_trace);
  }

  /// Populate field operations for Edwards curve point decompression.
  /// Given y-coordinate, computes x = sqrt((y^2 - 1) / (d*y^2 + 1)).
  template <
    class F,
    const int NumWords,
    const int NumLimbs,
    const int NumWitnesses,
    const int NumBitsPerLimb,
    const int WitnessOffset>
  __PICO_HOSTDEV__ void populate_ed_decompress_field_ops(
    EdDecompressCols<F, NumLimbs, NumWitnesses>& col,
    const uint32_t x[NumWords],
    const uint32_t y[NumWords],
    const bool sign,
    const uint32_t modulus[NumWords],
    const uint32_t mod_minus_two[NumWords],
    const uint32_t edwards_d[NumWords],
    F* byte_trace)
  {
    uint32_t one[NumWords] = {0};
    one[0] = 1;

    fp_lt::populate<F, NumLimbs, NumWitnesses>(col.y_range, y, modulus, byte_trace);

    // Calculate y^2.
    uint32_t yy[NumWords];
    fp_op::populate_mul<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
      col.yy, y, y, yy, modulus, false, byte_trace);

    // Calculate u = y^2 - 1.
    uint32_t u[NumWords];
    fp_op::populate_sub<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
      col.u, yy, one, u, modulus, false, byte_trace);

    // Calculate d*y^2.
    uint32_t dyy[NumWords];
    fp_op::populate_mul<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
      col.dyy, edwards_d, yy, dyy, modulus, false, byte_trace);

    // Calculate v = 1 + d*y^2.
    uint32_t v[NumWords];
    fp_op::populate_add<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
      col.v, one, dyy, v, modulus, false, byte_trace);

    // Calculate x^2 = u/v.
    uint32_t u_div_v[NumWords];
    fp_op::populate_div<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
      col.u_div_v, u, v, u_div_v, modulus, mod_minus_two, byte_trace);

    uint32_t zero[NumWords] = {0};

    // Populate x and -x based on sign bit.
    if (sign) {
      uint32_t x_tmp[NumWords];
      uint32_t neg_x[NumWords];
      fp_op::sub_words_with_borrow<NumWords>(modulus, x, x_tmp);

      fp_op::populate_mul<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
        col.x.multiplication, x_tmp, x_tmp, neg_x, modulus, false, byte_trace);

      fp_op::words_to_t_bytes_le<F, NumWords, NumLimbs>(x_tmp, col.x.multiplication.result, byte_trace);
      fp_lt::populate<F, NumLimbs, NumWitnesses>(col.x.range, x_tmp, modulus, byte_trace);
      col.x.lsb = x_tmp[0] & 1 ? F::one() : F::zero();

      fp_op::populate_sub<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
        col.neg_x, zero, x_tmp, neg_x, modulus, false, byte_trace);

      if (byte_trace) { byte::handle_byte_lookup_event(byte_trace, ByteOpcode::AND, x_tmp[0], 1); }
    } else {
      uint32_t neg_x[NumWords];
      fp_op::populate_mul<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
        col.x.multiplication, x, x, neg_x, modulus, false, byte_trace);

      fp_op::words_to_t_bytes_le<F, NumWords, NumLimbs>(x, col.x.multiplication.result, byte_trace);
      fp_lt::populate<F, NumLimbs, NumWitnesses>(col.x.range, x, modulus, byte_trace);
      col.x.lsb = x[0] & 1 ? F::one() : F::zero();

      fp_op::populate_sub<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
        col.neg_x, zero, x, neg_x, modulus, false, byte_trace);

      if (byte_trace) { byte::handle_byte_lookup_event(byte_trace, ByteOpcode::AND, x[0], 1); }
    }
  }

  /// Convert Edwards curve addition event to trace row.
  template <
    class F,
    const int NumWords,
    const int NumLimbs,
    const int NumWitnesses,
    const int NumBitsPerLimb,
    const int WitnessOffset>
  __PICO_HOSTDEV__ inline void ed_add_event_to_row(
    const EllipticCurveAddEvent<NumWords>* events,
    EdAddAssignCols<F, NumLimbs, NumWitnesses>& col,
    const size_t idx,
    const size_t count,
    const uint32_t modulus[NumWords],
    const uint32_t mod_minus_two[NumWords],
    const uint32_t edwards_d[NumWords],
    F* byte_trace)
  {
    if (idx < count) {
      const auto& event = events[idx];
      AffinePoint<NumWords> p(AffinePoint<NumWords>::from_words_le(event.p));
      AffinePoint<NumWords> q(AffinePoint<NumWords>::from_words_le(event.q));

      col.is_real = F::one();
      col.chunk = F::from_canonical_u32(event.chunk);
      col.clk = F::from_canonical_u32(event.clk);
      col.p_ptr = F::from_canonical_u32(event.p_ptr);
      col.q_ptr = F::from_canonical_u32(event.q_ptr);

      populate_ed_add_field_ops<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
        col, p, q, modulus, mod_minus_two, edwards_d, byte_trace);

      // Populate memory access columns.
      for (size_t i = 0; i < 2 * NumWords; i++) {
        memory_read_write::populate(col.p_access[i], event.p_memory_records[i], byte_trace);
        memory_read_write::populate(col.q_access[i], event.q_memory_records[i], byte_trace);
      }
    } else {
      // Padding row with zero points.
      populate_ed_add_field_ops<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
        col, AffinePoint<NumWords>::zero(), AffinePoint<NumWords>::zero(), modulus, mod_minus_two, edwards_d, NULL);
    }
  }

  /// Convert Edwards curve decompression event to trace row.
  template <
    class F,
    const int NumWords,
    const int NumLimbs,
    const int NumWitnesses,
    const int NumBitsPerLimb,
    const int WitnessOffset>
  __PICO_HOSTDEV__ inline void ed_decompress_event_to_row(
    const EdDecompressEvent* events,
    EdDecompressCols<F, NumLimbs, NumWitnesses>& col,
    const size_t idx,
    const size_t count,
    const uint32_t modulus[NumWords],
    const uint32_t mod_minus_two[NumWords],
    const uint32_t edwards_d[NumWords],
    F* byte_trace)
  {
    if (idx < count) {
      const auto& event = events[idx];

      col.is_real = F::one();
      col.chunk = F::from_canonical_u32(event.chunk);
      col.clk = F::from_canonical_u32(event.clk);
      col.ptr = F::from_canonical_u32(event.ptr);
      col.sign = event.sign ? F::one() : F::zero();

      // Reconstruct y and x from bytes.
      uint32_t y[COMPRESSED_POINT_BYTES / 4] = {0};
      uint32_t x[COMPRESSED_POINT_BYTES / 4] = {0};
      for (int i = 0; i < COMPRESSED_POINT_BYTES; i++) {
        y[i / 4] |= event.y_bytes[i] << (8 * (i % 4));
        x[i / 4] |= event.decompressed_x_bytes[i] << (8 * (i % 4));
      }

      populate_ed_decompress_field_ops<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
        col, x, y, event.sign, modulus, mod_minus_two, edwards_d, byte_trace);

      // Populate memory access columns.
      for (size_t i = 0; i < NumWords; i++) {
        memory_read_write::populate(col.x_access[i], event.x_memory_records[i], byte_trace);
        memory_read_write::populate(col.y_access[i], event.y_memory_records[i], byte_trace);
      }
    } else {
      // Padding row with dummy values (sqrt(-1) for Ed25519).
      uint32_t y[COMPRESSED_POINT_BYTES / 4] = {0};
      uint32_t x[COMPRESSED_POINT_BYTES / 4] = {
        0x4a0ea0b0, 0xc4ee1b27, 0xad2fe478, 0x2f431806, 0x3dfbd7a7, 0x2b4d0099, 0x4fc1df0b, 0x2b832480,
      };

      populate_ed_decompress_field_ops<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
        col, x, y, false, modulus, mod_minus_two, edwards_d, NULL);
    }
  }
} // namespace pico_gpu::edwards_op