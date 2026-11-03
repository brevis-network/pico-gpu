// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0

#pragma once

#include <iostream>
#include "common/rusterror.h"
#include "tracegen/common/utils.cuh"
#include "tracegen/riscv/byte.cuh"
#include "tracegen/riscv/divrem.cuh"


namespace pico_gpu::mul_u64 {
  using namespace pico_gpu::byte;
  using namespace pico_gpu::divrem;

  template <typename T>
  struct MulValueColsU64 {
    Word<T> a;
    Word<T> b;
    Word<T> c;
    MulGadgetU64<T> mul_gadget;
    T is_mul;
    T is_mulh;
    T is_mulhu;
    T is_mulhsu;
    T is_mulw;
  };

  template <class F>
  __device__ inline void event_to_row(
    const AluEventU64& event, MulValueColsU64<F>& cols, F* byte_trace)
  {
    write_word_from_u64(cols.a, event.a);
    write_word_from_u64(cols.b, event.b);
    write_word_from_u64(cols.c, event.c);

    bool is_mulh = event.opcode == Opcode::MULH;
    bool is_mulhsu = event.opcode == Opcode::MULHSU;
    bool is_mulw = event.opcode == Opcode::MULW;

    populate_mul_gadget(cols.mul_gadget, byte_trace,
                        event.b, event.c, is_mulh, is_mulhsu, is_mulw);

    cols.is_mul = F::from_bool(event.opcode == Opcode::MUL);
    cols.is_mulh = F::from_bool(is_mulh);
    cols.is_mulhu = F::from_bool(event.opcode == Opcode::MULHU);
    cols.is_mulhsu = F::from_bool(is_mulhsu);
    cols.is_mulw = F::from_bool(is_mulw);
  }
} // namespace pico_gpu::mul_u64