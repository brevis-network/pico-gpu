#pragma once

#include <iostream>
#include <util/rusterror.h>
#include "prelude.hpp"
#include "utils.hpp"
#include "byte.hpp"

namespace pico_gpu::bitwise {
  using namespace byte;

  /// Map ALU opcode to corresponding byte operation.
  __device__ inline ByteOpcode get_byte_opcode(const Opcode opcode)
  {
    switch (opcode) {
    case Opcode::AND:
      return ByteOpcode::AND;
    case Opcode::OR:
      return ByteOpcode::OR;
    case Opcode::XOR:
      return ByteOpcode::XOR;
    case Opcode::SLL:
      return ByteOpcode::SLL;
    default:
      // Invalid opcode, fallback to AND.
      return ByteOpcode::AND;
    }
  }

  /// Convert bitwise operation event to trace row.
  /// Bitwise ops (AND/OR/XOR) are processed byte by byte.
  template <class F>
  __device__ inline void event_to_row(const AluEvent& event, BitwiseCols<F>& cols, F* byte_trace)
  {
    write_word_from_u32_v2<F>(cols.a, event.a);
    write_word_from_u32_v2<F>(cols.b, event.b);
    write_word_from_u32_v2<F>(cols.c, event.c);

    // Set operation type flags.
    cols.is_xor = F::from_bool(event.opcode == Opcode::XOR);
    cols.is_or = F::from_bool(event.opcode == Opcode::OR);
    cols.is_and = F::from_bool(event.opcode == Opcode::AND);

    // Process each byte and record lookup events.
    const array_t<uint8_t, WORD_SIZE> b_bytes = u32_to_le_bytes(event.b);
    const array_t<uint8_t, WORD_SIZE> c_bytes = u32_to_le_bytes(event.c);
    for (uintptr_t i = 0; i < WORD_SIZE; ++i) {
      const ByteOpcode byte_opcode = get_byte_opcode(event.opcode);
      handle_byte_lookup_event(byte_trace, byte_opcode, b_bytes[i], c_bytes[i]);
    }
  }

  /// Kernel to convert bitwise events to trace.
  template <class F>
  __global__ void bitwise_events_to_trace_kernel(
    const AluEvent* events, const size_t count, F* trace_matrix, const size_t num_cols, F* byte_trace)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    BitwiseCols<F>* cols = reinterpret_cast<BitwiseCols<F>*>(&trace_matrix[idx * num_cols]);
    event_to_row<F>(events[idx], *cols, byte_trace);
  }

  /// Generate trace from bitwise operation events.
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

      bitwise_events_to_trace_kernel<<<grid_size, block_size, 0, stream>>>(
        events, event_size, trace, num_cols, byte_trace);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  }
} // namespace pico_gpu::bitwise