#pragma once

#include <iostream>
#include <util/exception.cuh>
#include <util/rusterror.h>

#include "types.hpp"
#include "utils.hpp"
#include "byte.hpp"

namespace pico_gpu::add_sub {
  using namespace byte;

  /// Populate addition operation with carry tracking and byte range checks.
  template <class F>
  __device__ inline uint32_t populate(AddOperation<F>& op, const uint32_t a_u32, const uint32_t b_u32, F* byte_trace)
  {
    const array_t<uint8_t, 4> a = u32_to_le_bytes(a_u32);
    const array_t<uint8_t, 4> b = u32_to_le_bytes(b_u32);

    // Calculate carries for each byte position.
    bool carry = a[0] + b[0] > 0xFF;
    op.carry[0] = F::from_bool(carry).val;
    carry = a[1] + b[1] + carry > 0xFF;
    op.carry[1] = F::from_bool(carry).val;
    carry = a[2] + b[2] + carry > 0xFF;
    op.carry[2] = F::from_bool(carry).val;

    const uint32_t expected = a_u32 + b_u32;
    write_word_from_u32_v2<F>(op.value, expected);

    // Add byte range checks for all operands and result.
    {
      const array_t<uint8_t, WORD_SIZE> a_bytes = u32_to_le_bytes(a_u32);
      const array_t<uint8_t, WORD_SIZE> b_bytes = u32_to_le_bytes(b_u32);
      const array_t<uint8_t, WORD_SIZE> expected_bytes = u32_to_le_bytes(expected);

      add_u8_range_checks(byte_trace, a_bytes);
      add_u8_range_checks(byte_trace, b_bytes);
      add_u8_range_checks(byte_trace, expected_bytes);
    }

    return expected;
  }

  /// Convert ADD/SUB event to trace row.
  /// SUB is implemented as ADD with swapped operands.
  template <class F>
  __device__ inline void event_to_row(const AluEvent& event, AddSubCols<F>& cols, F* byte_trace)
  {
    const bool is_add = event.opcode == Opcode::ADD;
    cols.is_add = F::from_bool(is_add);
    cols.is_sub = F::from_bool(!is_add);

    // For SUB, swap operands to reuse ADD logic.
    const auto operand_1 = is_add ? event.b : event.a;
    const auto operand_2 = event.c;

    populate<F>(cols.add_operation, operand_1, operand_2, byte_trace);
    write_word_from_u32_v2<F>(cols.operand_1, operand_1);
    write_word_from_u32_v2<F>(cols.operand_2, operand_2);
  }

  /// Kernel to convert ADD/SUB events to trace.
  template <class F>
  __global__ void add_sub_events_to_trace_kernel(
    const AluEvent* events, const size_t count, F* trace_matrix, const size_t num_cols, F* byte_trace)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    AddSubCols<F>* cols = reinterpret_cast<AddSubCols<F>*>(&trace_matrix[idx * num_cols]);
    event_to_row<F>(events[idx], *cols, byte_trace);
  }

  /// Generate trace from ADD/SUB events.
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

      add_sub_events_to_trace_kernel<<<grid_size, block_size, 0, stream>>>(
        events, event_size, trace, num_cols, byte_trace);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  }
} // namespace pico_gpu::add_sub