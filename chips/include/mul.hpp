#pragma once

#include <iostream>
#include <util/rusterror.h>
#include "utils.hpp"
#include "byte.hpp"

namespace pico_gpu::mul {
  using namespace byte;

  /// Convert multiplication event to trace row.
  /// Handles MUL, MULH, MULHU, and MULHSU operations with sign extension.
  template <class F>
  __device__ inline void event_to_row(const AluEvent& event, MulCols<F>& cols, F* field_byte_trace)
  {
    const array_t<uint8_t, 4> a = u32_to_le_bytes(event.a);
    const array_t<uint8_t, 4> b = u32_to_le_bytes(event.b);
    const array_t<uint8_t, 4> c = u32_to_le_bytes(event.c);
    uint32_t* byte_trace = reinterpret_cast<uint32_t*>(field_byte_trace);

    // Handle sign bits for signed multiplication variants.
    {
      const uint8_t b_msb = get_msb(b);
      cols.b_msb = F::from_canonical_u8(b_msb);
      const uint8_t c_msb = get_msb(c);
      cols.c_msb = F::from_canonical_u8(c_msb);

      // Sign extend b for MULH and MULHSU if negative.
      if ((event.opcode == Opcode::MULH || event.opcode == Opcode::MULHSU) && b_msb == 1) {
        cols.b_sign_extend = F::one();
      }

      // Sign extend c for MULH if negative.
      if (event.opcode == Opcode::MULH && c_msb == 1) { cols.c_sign_extend = F::one(); }

      handle_byte_lookup_event(byte_trace, ByteOpcode::MSB, b[WORD_SIZE - 1], 0);
      handle_byte_lookup_event(byte_trace, ByteOpcode::MSB, c[WORD_SIZE - 1], 0);
    }

    // Compute product byte-by-byte with sign extension.
    static_assert(2 * WORD_SIZE == LONG_WORD_SIZE);

    array_t<uint32_t, LONG_WORD_SIZE> product{};

    // Multiply each byte pair.
    for (uintptr_t i = 0; i < WORD_SIZE; ++i) {
      for (uintptr_t j = 0; j < WORD_SIZE; ++j) {
        product[i + j] += (uint32_t)b[i] * (uint32_t)c[j];
      }

      // Sign extend c if needed.
      if (cols.c_sign_extend.val != F::zero().val) {
        for (uintptr_t j = WORD_SIZE; j < LONG_WORD_SIZE - i; ++j) {
          product[i + j] += (uint32_t)b[i] * (uint32_t)0xFF;
        }
      }
    }

    // Sign extend b if needed.
    if (cols.b_sign_extend.val != F::zero().val) {
      for (uintptr_t i = WORD_SIZE; i < LONG_WORD_SIZE; ++i) {
        for (uintptr_t j = 0; j < LONG_WORD_SIZE - i; ++j) {
          product[i + j] += (uint32_t)0xFF * (uint32_t)c[j];
        }
      }
    }

    // Propagate carries through the product.
    const uint32_t base = 1 << BYTE_SIZE;
    array_t<uint32_t, LONG_WORD_SIZE> carry{};
    for (uintptr_t i = 0; i < LONG_WORD_SIZE; ++i) {
      carry[i] = product[i] / base;
      product[i] %= base;
      if (i + 1 < LONG_WORD_SIZE) product[i + 1] += carry[i];
      cols.carry[i] = F::from_canonical_u32(carry[i]);
    }

    // Populate result columns.
    for (uintptr_t i = 0; i < LONG_WORD_SIZE; ++i) {
      cols.product[i] = F::from_canonical_u32(product[i]);
    }

    word_from_le_bytes<F>(cols.a, a);
    word_from_le_bytes<F>(cols.b, b);
    word_from_le_bytes<F>(cols.c, c);
    cols.is_real = F::one();
    cols.is_mul = F::from_bool(event.opcode == Opcode::MUL);
    cols.is_mulh = F::from_bool(event.opcode == Opcode::MULH);
    cols.is_mulhu = F::from_bool(event.opcode == Opcode::MULHU);
    cols.is_mulhsu = F::from_bool(event.opcode == Opcode::MULHSU);

    // Add range checks for carries and products.
    for (int i = 0; i < LONG_WORD_SIZE; i++) {
      add_u16_range_check(byte_trace, static_cast<uint16_t>(carry[i] & 0xffff));
    }
    for (int i = 0; i < LONG_WORD_SIZE; i += 2) {
      add_u8_range_check(
        byte_trace, static_cast<uint8_t>(product[i] & 0xff), static_cast<uint8_t>(product[i + 1] & 0xff));
    }
  }

  /// Kernel to convert multiplication events to trace.
  template <class F>
  __global__ void mul_events_to_trace_kernel(
    const AluEvent* events, const size_t count, F* trace_matrix, const size_t num_cols, F* byte_trace)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    MulCols<F>* cols = reinterpret_cast<MulCols<F>*>(&trace_matrix[idx * num_cols]);
    event_to_row<F>(events[idx], *cols, byte_trace);
  }

  /// Generate trace from multiplication events.
  template <class F>
  inline RustError generate_extra_record_and_main(
    const size_t num_cols,
    const size_t event_size,
    const size_t trace_size,
    const AluEvent* events,
    F* trace,
    F* byte_trace,
    cudaStream_t stream)
  {
    try {
      if (event_size == 0) return RustError{cudaSuccess};

      // Initialize trace matrix with zeros.
      CUDA_OK(cudaMemsetAsync(trace, 0, trace_size * sizeof(F), stream));

      // Launch kernel.
      const int block_size = 256;
      const int grid_size = (event_size + block_size - 1) / block_size;

      mul_events_to_trace_kernel<<<grid_size, block_size, 0, stream>>>(events, event_size, trace, num_cols, byte_trace);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  }
} // namespace pico_gpu::mul