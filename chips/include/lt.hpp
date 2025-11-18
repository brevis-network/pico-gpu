#pragma once

#include <iostream>
#include <util/rusterror.h>
#include "utils.hpp"
#include "byte.hpp"

namespace pico_gpu::lt {
  using namespace byte;

  /// Convert less-than comparison event to trace row.
  /// Handles both SLT (signed) and SLTU (unsigned) comparisons.
  template <class F>
  __device__ inline void event_to_row(const AluEvent& event, LtCols<F>& cols, F* byte_trace)
  {
    const array_t<uint8_t, 4> a = u32_to_le_bytes(event.a);
    const array_t<uint8_t, 4> b = u32_to_le_bytes(event.b);
    const array_t<uint8_t, 4> c = u32_to_le_bytes(event.c);

    write_word_from_u32_v2<F>(cols.a, event.a);
    write_word_from_u32_v2<F>(cols.b, event.b);
    write_word_from_u32_v2<F>(cols.c, event.c);

    // For SLT, mask MSB of b and c to compare magnitudes only.
    const uint8_t masked_b = b[3] & 0x7f;
    const uint8_t masked_c = c[3] & 0x7f;
    cols.b_masked = F::from_canonical_u8(masked_b);
    cols.c_masked = F::from_canonical_u8(masked_c);

    // Record masked byte operations.
    handle_byte_lookup_event(byte_trace, ByteOpcode::AND, b[3], 0x7f);
    handle_byte_lookup_event(byte_trace, ByteOpcode::AND, c[3], 0x7f);

    // Use masked values for SLT, original values for SLTU.
    array_t<uint8_t, 4> b_comp = b;
    array_t<uint8_t, 4> c_comp = c;
    if (event.opcode == Opcode::SLT) {
      b_comp[3] = masked_b;
      c_comp[3] = masked_c;
    }

    // Find first differing byte from most significant position.
    intptr_t i = 3;
    while (true) {
      const uint8_t b_byte = b_comp[i];
      const uint8_t c_byte = c_comp[i];

      if (b_byte != c_byte) {
        cols.byte_flags[i] = F::one();
        cols.sltu = F::from_bool(b_byte < c_byte);

        const F b_byte_f = F::from_canonical_u8(b_byte);
        const F c_byte_f = F::from_canonical_u8(c_byte);
        cols.not_eq_inv = (b_byte_f - c_byte_f).reciprocal();
        cols.comparison_bytes[0] = b_byte_f;
        cols.comparison_bytes[1] = c_byte_f;
        break;
      }

      if (i == 0) {
        // All bytes equal: b_comp == c_comp.
        cols.is_comp_eq = F::one();
        break;
      }
      --i;
    }

    // Extract and compare sign bits for SLT.
    cols.msb_b = F::from_bool((b[3] >> 7) & 1);
    cols.msb_c = F::from_bool((c[3] >> 7) & 1);
    cols.is_sign_eq = F::from_bool(event.opcode != Opcode::SLT || cols.msb_b == cols.msb_c);

    cols.is_slt = F::from_bool(event.opcode == Opcode::SLT);
    cols.is_sltu = F::from_bool(event.opcode == Opcode::SLTU);

    cols.bit_b = (F(cols.msb_b) * F(cols.is_slt));
    cols.bit_c = (F(cols.msb_c) * F(cols.is_slt));

    // Record byte comparison operation.
    handle_byte_lookup_event(
      byte_trace, ByteOpcode::LTU, static_cast<uint8_t>(cols.comparison_bytes[0].as_canonical_u32() & 0xff),
      static_cast<uint8_t>(cols.comparison_bytes[1].as_canonical_u32() & 0xff));
  }

  /// Kernel to convert less-than events to trace.
  template <class F>
  __global__ void lt_events_to_trace_kernel(
    const AluEvent* events, const size_t count, F* trace_matrix, const size_t num_cols, F* byte_trace)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    LtCols<F>* cols = reinterpret_cast<LtCols<F>*>(&trace_matrix[idx * num_cols]);
    event_to_row<F>(events[idx], *cols, byte_trace);
  }

  /// Generate trace from less-than comparison events.
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

      lt_events_to_trace_kernel<<<grid_size, block_size, 0, stream>>>(events, event_size, trace, num_cols, byte_trace);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  }
} // namespace pico_gpu::lt