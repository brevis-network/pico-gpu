#pragma once

#include "types.hpp"
#include "fp_op.hpp"
#include "memory_read_write.hpp"
#include <sys/types.h>

namespace pico_gpu::fp2_op {
  using namespace ::pico_gpu::memory_read_write;
  using namespace ::pico_gpu::fp_op;

  static constexpr int BitsPerLimb = 8;

  /// Populate field operations for Fp2 addition/subtraction.
  /// Fp2 elements are (a0, a1) where operations are done component-wise.
  template <
    class F,
    const int NumWords,
    const int NumLimbs,
    const int NumWitnesses,
    const int NumBitsPerLimb,
    const int WitnessOffset>
  __PICO_HOSTDEV__ void populate_fp2_add_sub_field_ops(
    Fp2AddSubCols<F, NumWords, NumLimbs, NumWitnesses>& cols,
    const uint32_t p_x[NumWords],
    const uint32_t p_y[NumWords],
    const uint32_t q_x[NumWords],
    const uint32_t q_y[NumWords],
    const FieldOperation op,
    const uint32_t modulus[NumWords],
    const uint32_t mod_minus_two[NumWords],
    F* byte_trace)
  {
    uint32_t result[NumWords];

    // Component 0: p.x op q.x
    fp_op::populate_with_modulus<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
      cols.c0, p_x, q_x, result, modulus, mod_minus_two, op, false, byte_trace);

    // Component 1: p.y op q.y
    fp_op::populate_with_modulus<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
      cols.c1, p_y, q_y, result, modulus, mod_minus_two, op, false, byte_trace);
  }

  /// Convert Fp2 addition/subtraction event to trace row.
  template <
    class F,
    const int NumWords,
    const int NumLimbs,
    const int NumWitnesses,
    const int NumBitsPerLimb,
    const int WitnessOffset>
  __PICO_HOSTDEV__ inline void add_sub_event_to_row(
    const Fp2AddSubEvent<NumWords>* events,
    Fp2AddSubCols<F, NumWords, NumLimbs, NumWitnesses>& cols,
    const size_t idx,
    const size_t count,
    const uint32_t modulus[NumWords],
    const uint32_t mod_minus_two[NumWords],
    F* byte_trace)
  {
    if (idx < count) {
      const auto& event = events[idx];

      cols.is_real = F::ONE;
      cols.is_add = F::from_bool(event.op == FieldOperation::Add);
      cols.chunk = F::from_canonical_u32(event.chunk);
      cols.clk = F::from_canonical_u32(event.clk);
      cols.x_ptr = F::from_canonical_u32(event.x_ptr);
      cols.y_ptr = F::from_canonical_u32(event.y_ptr);

      populate_fp2_add_sub_field_ops<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
        cols, event.x, event.x + NumWords, event.y, event.y + NumWords, event.op, modulus, mod_minus_two, byte_trace);

      // Populate memory access columns.
      for (size_t i = 0; i < 2 * NumWords; ++i) {
        memory_read_write::populate(cols.x_access[i], event.x_memory_records[i], byte_trace);
      }
      for (size_t i = 0; i < 2 * NumWords; ++i) {
        memory_read_write::populate(cols.y_access[i], event.y_memory_records[i], byte_trace);
      }
    } else {
      // Padding row with zero addition.
      cols.is_add = F::one();
      uint32_t zero[NumWords];
      for (int i = 0; i < NumWords; ++i) {
        zero[i] = 0;
      }
      populate_fp2_add_sub_field_ops<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
        cols, zero, zero, zero, zero, FieldOperation::Add, modulus, mod_minus_two, nullptr);
    }
  }

  /// Populate field operations for Fp2 multiplication.
  /// Fp2 multiplication: (a0 + a1*i) * (b0 + b1*i) = (a0*b0 - a1*b1) + (a0*b1 + a1*b0)*i
  template <
    class F,
    const int NumWords,
    const int NumLimbs,
    const int NumWitnesses,
    const int NumBitsPerLimb,
    const int WitnessOffset>
  __PICO_HOSTDEV__ void populate_fp2_mul_field_ops(
    Fp2MulCols<F, NumWords, NumLimbs, NumWitnesses>& cols,
    const uint32_t p_x[NumWords],
    const uint32_t p_y[NumWords],
    const uint32_t q_x[NumWords],
    const uint32_t q_y[NumWords],
    const uint32_t modulus[NumWords],
    const uint32_t mod_minus_two[NumWords],
    F* byte_trace)
  {
    // Compute intermediate products.
    uint32_t a0_mul_b0[NumWords];
    fp_op::populate_with_modulus<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
      cols.a0_mul_b0, p_x, q_x, a0_mul_b0, modulus, mod_minus_two, FieldOperation::Mul, false, byte_trace);

    uint32_t a1_mul_b1[NumWords];
    fp_op::populate_with_modulus<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
      cols.a1_mul_b1, p_y, q_y, a1_mul_b1, modulus, mod_minus_two, FieldOperation::Mul, false, byte_trace);

    uint32_t a0_mul_b1[NumWords];
    fp_op::populate_with_modulus<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
      cols.a0_mul_b1, p_x, q_y, a0_mul_b1, modulus, mod_minus_two, FieldOperation::Mul, false, byte_trace);

    uint32_t a1_mul_b0[NumWords];
    fp_op::populate_with_modulus<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
      cols.a1_mul_b0, p_y, q_x, a1_mul_b0, modulus, mod_minus_two, FieldOperation::Mul, false, byte_trace);

    // Real part: c0 = a0*b0 - a1*b1
    uint32_t result[NumWords];
    fp_op::populate_with_modulus<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
      cols.c0, a0_mul_b0, a1_mul_b1, result, modulus, mod_minus_two, FieldOperation::Sub, false, byte_trace);

    // Imaginary part: c1 = a0*b1 + a1*b0
    fp_op::populate_with_modulus<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
      cols.c1, a0_mul_b1, a1_mul_b0, result, modulus, mod_minus_two, FieldOperation::Add, false, byte_trace);
  }

  /// Convert Fp2 multiplication event to trace row.
  template <
    class F,
    const int NumWords,
    const int NumLimbs,
    const int NumWitnesses,
    const int NumBitsPerLimb,
    const int WitnessOffset>
  __PICO_HOSTDEV__ inline void mul_event_to_row(
    const Fp2MulEvent<NumWords>* events,
    Fp2MulCols<F, NumWords, NumLimbs, NumWitnesses>& cols,
    const size_t idx,
    const size_t count,
    const uint32_t modulus[NumWords],
    const uint32_t mod_minus_two[NumWords],
    F* byte_trace)
  {
    if (idx < count) {
      const auto& event = events[idx];

      cols.is_real = F::ONE;
      cols.chunk = F::from_canonical_u32(event.chunk);
      cols.clk = F::from_canonical_u32(event.clk);
      cols.x_ptr = F::from_canonical_u32(event.x_ptr);
      cols.y_ptr = F::from_canonical_u32(event.y_ptr);

      populate_fp2_mul_field_ops<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
        cols, event.x, event.x + NumWords, event.y, event.y + NumWords, modulus, mod_minus_two, byte_trace);

      // Populate memory access columns.
      for (size_t i = 0; i < 2 * NumWords; ++i) {
        memory_read_write::populate(cols.x_access[i], event.x_memory_records[i], byte_trace);
      }
      for (size_t i = 0; i < 2 * NumWords; ++i) {
        memory_read_write::populate(cols.y_access[i], event.y_memory_records[i], byte_trace);
      }
    } else {
      // Padding row with zero multiplication.
      uint32_t zero[NumWords];
      for (int i = 0; i < NumWords; ++i) {
        zero[i] = 0;
      }
      populate_fp2_mul_field_ops<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
        cols, zero, zero, zero, zero, modulus, mod_minus_two, nullptr);
    }
  }

} // namespace pico_gpu::fp2_op