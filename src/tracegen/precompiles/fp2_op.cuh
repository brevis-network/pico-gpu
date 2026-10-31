// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0

#pragma once

#include "tracegen/common/types.cuh"
#include "tracegen/precompiles/fp_op.cuh"
#include "tracegen/riscv/memory_read_write.cuh"
#include <sys/types.h>

namespace pico_gpu::fp2_op {
  using namespace ::pico_gpu::fp_op;

  static constexpr int BitsPerLimb = 8;

  /// Populate field operations for Fp2 addition/subtraction.
  /// Also populates FieldLtCols for range checking results < modulus.
  ///
  /// NumCurveDwords = dwords per curve point (= NumLimbs/4, matches Rust WordsCurvePoint).
  template <
    class F,
    const int NumCurveDwords,
    const int NumWords,
    const int NumLimbs,
    const int NumWitnesses,
    const int NumBitsPerLimb,
    const int WitnessOffset,
    bool UseScratch = false>
  __PICO_HOSTDEV__ void populate_fp2_add_sub_field_ops(
    Fp2AddSubCols<F, NumCurveDwords, NumLimbs, NumWitnesses>& cols,
    const uint32_t p_x[NumWords],
    const uint32_t p_y[NumWords],
    const uint32_t q_x[NumWords],
    const uint32_t q_y[NumWords],
    const FieldOperation op,
    const uint32_t modulus[NumWords],
    const uint32_t mod_minus_two[NumWords],
    F* byte_trace,
    fp_op::FpScratch<NumLimbs, NumWitnesses>* scratch = nullptr)
  {
    uint32_t result_c0[NumWords];
    uint32_t result_c1[NumWords];

    fp_op::populate_with_modulus<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset, UseScratch>(
      cols.c0, p_x, q_x, result_c0, modulus, mod_minus_two, op, false, byte_trace, scratch);

    fp_op::populate_with_modulus<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset, UseScratch>(
      cols.c1, p_y, q_y, result_c1, modulus, mod_minus_two, op, false, byte_trace, scratch);

    uint8_t c0_result_bytes[NumLimbs], c1_result_bytes[NumLimbs], modulus_bytes[NumLimbs];
    fp_op::words_to_bytes_le<NumWords, NumLimbs>(result_c0, c0_result_bytes);
    fp_op::words_to_bytes_le<NumWords, NumLimbs>(result_c1, c1_result_bytes);
    fp_op::words_to_bytes_le<NumWords, NumLimbs>(modulus, modulus_bytes);

    fp_op::populate_field_lt<F, NumLimbs>(cols.c0_range, c0_result_bytes, modulus_bytes, byte_trace);
    fp_op::populate_field_lt<F, NumLimbs>(cols.c1_range, c1_result_bytes, modulus_bytes, byte_trace);
  }

  /// Convert Fp2 addition/subtraction event to trace row (u64 version).
  template <
    class F,
    const int NumCurveDwords,
    const int NumFieldDwords,
    const int NumWords,
    const int NumLimbs,
    const int NumWitnesses,
    const int NumBitsPerLimb,
    const int WitnessOffset,
    bool UseScratch = false>
  __PICO_HOSTDEV__ inline void add_sub_event_to_row(
    const Fp2AddSubEvent<NumCurveDwords>* events,
    Fp2AddSubCols<F, NumCurveDwords, NumLimbs, NumWitnesses>& cols,
    const size_t idx,
    const size_t count,
    const uint32_t modulus[NumWords],
    const uint32_t mod_minus_two[NumWords],
    F* byte_trace,
    fp_op::FpScratch<NumLimbs, NumWitnesses>* scratch = nullptr)
  {
    if (idx < count) {
      const auto& event = events[idx];

      cols.is_real = F::ONE;
      cols.is_add = F::from_bool(event.op == FieldOperation::Add);
      cols.chunk = F::from_canonical_u32(event.chunk);
      cols.clk = F::from_canonical_u32(event.clk);

      fp_op::populate_syscall_addr_gadget<F>(cols.x_ptr, event.x_ptr, NumLimbs * 2, byte_trace);
      fp_op::populate_syscall_addr_gadget<F>(cols.y_ptr, event.y_ptr, NumLimbs * 2, byte_trace);

      for (int i = 0; i < NumCurveDwords; i++) {
        fp_op::populate_addr_add_gadget<F>(cols.x_addrs[i], event.x_ptr, static_cast<uint64_t>(i) * 8, byte_trace);
        fp_op::populate_addr_add_gadget<F>(cols.y_addrs[i], event.y_ptr, static_cast<uint64_t>(i) * 8, byte_trace);
      }

      uint32_t p_x_words[NumWords], p_y_words[NumWords];
      uint32_t q_x_words[NumWords], q_y_words[NumWords];
      fp_op::dwords_to_u32_words<NumFieldDwords, NumWords>(event.x, p_x_words);
      fp_op::dwords_to_u32_words<NumFieldDwords, NumWords>(event.x + NumFieldDwords, p_y_words);
      fp_op::dwords_to_u32_words<NumFieldDwords, NumWords>(event.y, q_x_words);
      fp_op::dwords_to_u32_words<NumFieldDwords, NumWords>(event.y + NumFieldDwords, q_y_words);

      populate_fp2_add_sub_field_ops<F, NumCurveDwords, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset, UseScratch>(
        cols, p_x_words, p_y_words, q_x_words, q_y_words, event.op, modulus, mod_minus_two, byte_trace, scratch);

      for (int i = 0; i < NumCurveDwords; i++) {
        fp_op::populate_write_cols_u8_from_record<F>(cols.x_access[i], event.x_memory_records[i], byte_trace);
      }
      for (int i = 0; i < NumCurveDwords; i++) {
        fp_op::populate_read_cols_u8_from_record<F>(cols.y_access[i], event.y_memory_records[i], byte_trace);
      }
    } else {
      cols.is_add = F::one();
      uint32_t zero[NumWords];
      for (int i = 0; i < NumWords; ++i) {
        zero[i] = 0;
      }
      populate_fp2_add_sub_field_ops<F, NumCurveDwords, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset, UseScratch>(
        cols, zero, zero, zero, zero, FieldOperation::Add, modulus, mod_minus_two, nullptr, scratch);
    }
  }

  /// Populate field operations for Fp2 multiplication (u64 version).
  /// Fp2 multiplication: (a0 + a1*i) * (b0 + b1*i) = (a0*b0 - a1*b1) + (a0*b1 + a1*b0)*i
  template <
    class F,
    const int NumCurveDwords,
    const int NumWords,
    const int NumLimbs,
    const int NumWitnesses,
    const int NumBitsPerLimb,
    const int WitnessOffset,
    bool UseScratch = false>
  __PICO_HOSTDEV__ void populate_fp2_mul_field_ops(
    Fp2MulCols<F, NumCurveDwords, NumLimbs, NumWitnesses>& cols,
    const uint32_t p_x[NumWords],
    const uint32_t p_y[NumWords],
    const uint32_t q_x[NumWords],
    const uint32_t q_y[NumWords],
    const uint32_t modulus[NumWords],
    const uint32_t mod_minus_two[NumWords],
    F* byte_trace,
    fp_op::FpScratch<NumLimbs, NumWitnesses>* scratch = nullptr)
  {
    uint32_t a0_mul_b0[NumWords];
    fp_op::populate_with_modulus<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset, UseScratch>(
      cols.a0_mul_b0, p_x, q_x, a0_mul_b0, modulus, mod_minus_two, FieldOperation::Mul, false, byte_trace, scratch);

    uint32_t a1_mul_b1[NumWords];
    fp_op::populate_with_modulus<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset, UseScratch>(
      cols.a1_mul_b1, p_y, q_y, a1_mul_b1, modulus, mod_minus_two, FieldOperation::Mul, false, byte_trace, scratch);

    uint32_t a0_mul_b1[NumWords];
    fp_op::populate_with_modulus<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset, UseScratch>(
      cols.a0_mul_b1, p_x, q_y, a0_mul_b1, modulus, mod_minus_two, FieldOperation::Mul, false, byte_trace, scratch);

    uint32_t a1_mul_b0[NumWords];
    fp_op::populate_with_modulus<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset, UseScratch>(
      cols.a1_mul_b0, p_y, q_x, a1_mul_b0, modulus, mod_minus_two, FieldOperation::Mul, false, byte_trace, scratch);

    uint32_t result_c0[NumWords];
    fp_op::populate_with_modulus<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset, UseScratch>(
      cols.c0, a0_mul_b0, a1_mul_b1, result_c0, modulus, mod_minus_two, FieldOperation::Sub, false, byte_trace, scratch);

    uint32_t result_c1[NumWords];
    fp_op::populate_with_modulus<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset, UseScratch>(
      cols.c1, a0_mul_b1, a1_mul_b0, result_c1, modulus, mod_minus_two, FieldOperation::Add, false, byte_trace, scratch);

    // Populate FieldLtCols: verify result < modulus for both components.
    uint8_t c0_result_bytes[NumLimbs], c1_result_bytes[NumLimbs], modulus_bytes[NumLimbs];
    fp_op::words_to_bytes_le<NumWords, NumLimbs>(result_c0, c0_result_bytes);
    fp_op::words_to_bytes_le<NumWords, NumLimbs>(result_c1, c1_result_bytes);
    fp_op::words_to_bytes_le<NumWords, NumLimbs>(modulus, modulus_bytes);

    fp_op::populate_field_lt<F, NumLimbs>(cols.c0_range, c0_result_bytes, modulus_bytes, byte_trace);
    fp_op::populate_field_lt<F, NumLimbs>(cols.c1_range, c1_result_bytes, modulus_bytes, byte_trace);
  }

  /// Convert Fp2 multiplication event to trace row (u64 version).
  template <
    class F,
    const int NumCurveDwords,
    const int NumFieldDwords,
    const int NumWords,
    const int NumLimbs,
    const int NumWitnesses,
    const int NumBitsPerLimb,
    const int WitnessOffset,
    bool UseScratch = false>
  __PICO_HOSTDEV__ inline void mul_event_to_row(
    const Fp2MulEvent<NumCurveDwords>* events,
    Fp2MulCols<F, NumCurveDwords, NumLimbs, NumWitnesses>& cols,
    const size_t idx,
    const size_t count,
    const uint32_t modulus[NumWords],
    const uint32_t mod_minus_two[NumWords],
    F* byte_trace,
    fp_op::FpScratch<NumLimbs, NumWitnesses>* scratch = nullptr)
  {
    if (idx < count) {
      const auto& event = events[idx];

      cols.is_real = F::ONE;
      cols.chunk = F::from_canonical_u32(event.chunk);
      cols.clk = F::from_canonical_u32(event.clk);

      fp_op::populate_syscall_addr_gadget<F>(cols.x_ptr, event.x_ptr, NumLimbs * 2, byte_trace);
      fp_op::populate_syscall_addr_gadget<F>(cols.y_ptr, event.y_ptr, NumLimbs * 2, byte_trace);

      for (int i = 0; i < NumCurveDwords; i++) {
        fp_op::populate_addr_add_gadget<F>(cols.x_addrs[i], event.x_ptr, static_cast<uint64_t>(i) * 8, byte_trace);
        fp_op::populate_addr_add_gadget<F>(cols.y_addrs[i], event.y_ptr, static_cast<uint64_t>(i) * 8, byte_trace);
      }

      uint32_t p_x_words[NumWords], p_y_words[NumWords];
      uint32_t q_x_words[NumWords], q_y_words[NumWords];
      fp_op::dwords_to_u32_words<NumFieldDwords, NumWords>(event.x, p_x_words);
      fp_op::dwords_to_u32_words<NumFieldDwords, NumWords>(event.x + NumFieldDwords, p_y_words);
      fp_op::dwords_to_u32_words<NumFieldDwords, NumWords>(event.y, q_x_words);
      fp_op::dwords_to_u32_words<NumFieldDwords, NumWords>(event.y + NumFieldDwords, q_y_words);

      populate_fp2_mul_field_ops<F, NumCurveDwords, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset, UseScratch>(
        cols, p_x_words, p_y_words, q_x_words, q_y_words, modulus, mod_minus_two, byte_trace, scratch);

      for (int i = 0; i < NumCurveDwords; i++) {
        fp_op::populate_write_cols_u8_from_record<F>(cols.x_access[i], event.x_memory_records[i], byte_trace);
      }
      for (int i = 0; i < NumCurveDwords; i++) {
        fp_op::populate_read_cols_u8_from_record<F>(cols.y_access[i], event.y_memory_records[i], byte_trace);
      }
    } else {
      uint32_t zero[NumWords];
      for (int i = 0; i < NumWords; ++i) {
        zero[i] = 0;
      }
      populate_fp2_mul_field_ops<F, NumCurveDwords, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset, UseScratch>(
        cols, zero, zero, zero, zero, modulus, mod_minus_two, nullptr, scratch);
    }
  }

} // namespace pico_gpu::fp2_op