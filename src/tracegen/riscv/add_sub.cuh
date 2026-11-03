// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0

#pragma once

#include <iostream>
#include "common/exception.cuh"
#include "common/rusterror.h"

#include "tracegen/common/types.cuh"
#include "tracegen/common/utils.cuh"
#include "tracegen/riscv/byte.cuh"


// ═══════════════════════════════════════════════════════════════════════
// U64 Add chip — ADD: 64-bit wrapping addition
// ═══════════════════════════════════════════════════════════════════════

namespace pico_gpu::add_u64 {
  using namespace byte;

  static constexpr size_t U64_WORD_SIZE = 4;

  template <typename T>
  struct AddGadgetU64 {
    Word<T> value;
  };

  template <typename T>
  struct AddValueColsU64 {
    AddGadgetU64<T> add_operation;
    Word<T> operand_1;
    Word<T> operand_2;
    T is_add;
  };

  template <class F>
  __device__ inline void write_word_u64(Word<F>& word, uint64_t v)
  {
    word._0[0] = F::from_canonical_u16((uint16_t)(v & 0xFFFF));
    word._0[1] = F::from_canonical_u16((uint16_t)((v >> 16) & 0xFFFF));
    word._0[2] = F::from_canonical_u16((uint16_t)((v >> 32) & 0xFFFF));
    word._0[3] = F::from_canonical_u16((uint16_t)((v >> 48) & 0xFFFF));
  }

  template <class F>
  __device__ inline void event_to_row(
    const AluEventU64& event, AddValueColsU64<F>& cols, F* byte_trace)
  {
    cols.is_add = F::one();

    uint64_t result = event.b + event.c;
    write_word_u64(cols.add_operation.value, result);
    write_word_u64(cols.operand_1, event.b);
    write_word_u64(cols.operand_2, event.c);

    add_u16_range_check<F>(byte_trace, (uint16_t)(result & 0xFFFF));
    add_u16_range_check<F>(byte_trace, (uint16_t)((result >> 16) & 0xFFFF));
    add_u16_range_check<F>(byte_trace, (uint16_t)((result >> 32) & 0xFFFF));
    add_u16_range_check<F>(byte_trace, (uint16_t)((result >> 48) & 0xFFFF));
  }
} // namespace pico_gpu::add_u64

// ═══════════════════════════════════════════════════════════════════════
// U64 Addw chip — ADDW: lower-32-bit add with sign extension
// ═══════════════════════════════════════════════════════════════════════

namespace pico_gpu::addw_u64 {
  using namespace byte;

  template <typename T>
  struct U16MSBGadgetU64 {
    T msb;
  };

  template <typename T>
  struct AddwGadgetU64 {
    T value[2];            // lower 32-bit result as 2 u16 limbs
    U16MSBGadgetU64<T> msb; // MSB of value[1]
  };

  template <typename T>
  struct AddwValueColsU64 {
    AddwGadgetU64<T> addw_operation;
    Word<T> operand_1;     // 4 u16 limbs
    Word<T> operand_2;     // 4 u16 limbs
    T is_addw;
  };

  template <class F>
  __device__ inline void write_word_u64(Word<F>& word, uint64_t v)
  {
    word._0[0] = F::from_canonical_u16((uint16_t)(v & 0xFFFF));
    word._0[1] = F::from_canonical_u16((uint16_t)((v >> 16) & 0xFFFF));
    word._0[2] = F::from_canonical_u16((uint16_t)((v >> 32) & 0xFFFF));
    word._0[3] = F::from_canonical_u16((uint16_t)((v >> 48) & 0xFFFF));
  }

  template <class F>
  __device__ inline void event_to_row(
    const AluEventU64& event, AddwValueColsU64<F>& cols, F* byte_trace)
  {
    cols.is_addw = F::one();

    uint32_t lower32 = (uint32_t)event.b + (uint32_t)event.c;
    uint16_t limb0 = (uint16_t)(lower32 & 0xFFFF);
    uint16_t limb1 = (uint16_t)((lower32 >> 16) & 0xFFFF);

    cols.addw_operation.value[0] = F::from_canonical_u16(limb0);
    cols.addw_operation.value[1] = F::from_canonical_u16(limb1);

    // u16 range checks for the two result limbs
    add_u16_range_check<F>(byte_trace, limb0);
    add_u16_range_check<F>(byte_trace, limb1);

    // MSB gadget: msb = (limb1 >> 15) & 1, range check limb1 * 2
    uint16_t msb = (limb1 >> 15) & 1;
    cols.addw_operation.msb.msb = F::from_canonical_u16(msb);
    add_u16_range_check<F>(byte_trace, (uint16_t)(limb1 * 2));

    write_word_u64(cols.operand_1, event.b);
    write_word_u64(cols.operand_2, event.c);
  }
} // namespace pico_gpu::addw_u64

// ═══════════════════════════════════════════════════════════════════════
// U64 Sub chip — SUB: 64-bit wrapping subtraction
// ═══════════════════════════════════════════════════════════════════════

namespace pico_gpu::sub_u64 {
  using namespace byte;

  template <typename T>
  struct SubGadgetU64 {
    Word<T> value;
  };

  template <typename T>
  struct SubValueColsU64 {
    SubGadgetU64<T> sub_operation;
    Word<T> operand_1;
    Word<T> operand_2;
    T is_sub;
  };

  template <class F>
  __device__ inline void write_word_u64(Word<F>& word, uint64_t v)
  {
    word._0[0] = F::from_canonical_u16((uint16_t)(v & 0xFFFF));
    word._0[1] = F::from_canonical_u16((uint16_t)((v >> 16) & 0xFFFF));
    word._0[2] = F::from_canonical_u16((uint16_t)((v >> 32) & 0xFFFF));
    word._0[3] = F::from_canonical_u16((uint16_t)((v >> 48) & 0xFFFF));
  }

  template <class F>
  __device__ inline void event_to_row(
    const AluEventU64& event, SubValueColsU64<F>& cols, F* byte_trace)
  {
    cols.is_sub = F::one();

    uint64_t result = event.b - event.c;
    write_word_u64(cols.sub_operation.value, result);
    write_word_u64(cols.operand_1, event.b);
    write_word_u64(cols.operand_2, event.c);

    add_u16_range_check<F>(byte_trace, (uint16_t)(result & 0xFFFF));
    add_u16_range_check<F>(byte_trace, (uint16_t)((result >> 16) & 0xFFFF));
    add_u16_range_check<F>(byte_trace, (uint16_t)((result >> 32) & 0xFFFF));
    add_u16_range_check<F>(byte_trace, (uint16_t)((result >> 48) & 0xFFFF));
  }
} // namespace pico_gpu::sub_u64

// ═══════════════════════════════════════════════════════════════════════
// U64 Subw chip — SUBW: lower-32-bit sub with sign extension
// ═══════════════════════════════════════════════════════════════════════

namespace pico_gpu::subw_u64 {
  using namespace byte;

  template <typename T>
  struct U16MSBGadgetU64 {
    T msb;
  };

  template <typename T>
  struct SubwGadgetU64 {
    T value[2];
    U16MSBGadgetU64<T> msb;
  };

  template <typename T>
  struct SubwValueColsU64 {
    SubwGadgetU64<T> subw_operation;
    Word<T> operand_1;
    Word<T> operand_2;
    T is_subw;
  };

  template <class F>
  __device__ inline void write_word_u64(Word<F>& word, uint64_t v)
  {
    word._0[0] = F::from_canonical_u16((uint16_t)(v & 0xFFFF));
    word._0[1] = F::from_canonical_u16((uint16_t)((v >> 16) & 0xFFFF));
    word._0[2] = F::from_canonical_u16((uint16_t)((v >> 32) & 0xFFFF));
    word._0[3] = F::from_canonical_u16((uint16_t)((v >> 48) & 0xFFFF));
  }

  template <class F>
  __device__ inline void event_to_row(
    const AluEventU64& event, SubwValueColsU64<F>& cols, F* byte_trace)
  {
    cols.is_subw = F::one();

    int32_t lower32 = (int32_t)event.b - (int32_t)event.c;
    uint64_t value = (uint64_t)(int64_t)lower32;
    uint16_t limb0 = (uint16_t)(value & 0xFFFF);
    uint16_t limb1 = (uint16_t)((value >> 16) & 0xFFFF);

    cols.subw_operation.value[0] = F::from_canonical_u16(limb0);
    cols.subw_operation.value[1] = F::from_canonical_u16(limb1);

    add_u16_range_check<F>(byte_trace, limb0);
    add_u16_range_check<F>(byte_trace, limb1);

    uint16_t msb = (limb1 >> 15) & 1;
    cols.subw_operation.msb.msb = F::from_canonical_u16(msb);
    add_u16_range_check<F>(byte_trace, (uint16_t)(limb1 * 2));

    write_word_u64(cols.operand_1, event.b);
    write_word_u64(cols.operand_2, event.c);
  }
} // namespace pico_gpu::subw_u64