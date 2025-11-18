#pragma once

#include <cstdlib>
#include <ff/ff_config.hpp>

#include "prelude.hpp"
#include "utils.hpp"
#include "byte.hpp"

namespace pico_gpu::sll {
  using namespace byte;

  /// Convert a shift left logical event to a trace row with byte lookup.
  /// This implements the SLL (Shift Left Logical) instruction by decomposing
  /// the shift into byte-level and bit-level operations for constraint checking.
  template <class F>
  __device__ inline void event_to_row_byte(const AluEvent& event, ShiftLeftCols<F>& cols, F* field_byte_trace)
  {
    // Convert operands to little-endian byte arrays.
    const array_t<uint8_t, 4> a = u32_to_le_bytes(event.a);
    const array_t<uint8_t, 4> b = u32_to_le_bytes(event.b);
    const array_t<uint8_t, 4> c = u32_to_le_bytes(event.c);
    uint32_t* byte_trace = reinterpret_cast<uint32_t*>(field_byte_trace);

    // Populate basic operand columns.
    word_from_le_bytes<F>(cols.a, a);
    word_from_le_bytes<F>(cols.b, b);
    word_from_le_bytes<F>(cols.c, c);
    cols.is_real = F::one();

    // Decompose shift amount's least significant byte into bits.
    // This is used to verify the shift amount is in valid range.
    for (uintptr_t i = 0; i < BYTE_SIZE; ++i) {
      cols.c_least_sig_byte[i] = F::from_canonical_u32((event.c >> i) & 1);
    }

    // Calculate number of bits to shift (within a byte).
    // SLL splits the shift into: (num_bytes * 8) + num_bits_within_byte.
    const uintptr_t num_bits_to_shift = event.c % BYTE_SIZE;
    for (uintptr_t i = 0; i < BYTE_SIZE; ++i) {
      cols.shift_by_n_bits[i] = F::from_bool(num_bits_to_shift == i);
    }

    // Calculate bit shift multiplier: 2^num_bits_to_shift.
    const uint32_t bit_shift_multiplier = 1 << num_bits_to_shift;
    cols.bit_shift_multiplier = F::from_canonical_u32(bit_shift_multiplier);

    // Perform bit-level shift with carry propagation.
    // Each byte is multiplied by the shift multiplier, and carries
    // are propagated to the next byte position.
    uint32_t carry = 0;
    const uint32_t base = 1 << BYTE_SIZE;

    array_t<uint8_t, WORD_SIZE> bit_shift_result;
    array_t<uint8_t, WORD_SIZE> bit_shift_result_carry;
    for (uintptr_t i = 0; i < WORD_SIZE; ++i) {
      const uint32_t v = b[i] * bit_shift_multiplier + carry;
      carry = v / base;
      bit_shift_result[i] = (uint8_t)(v % base);
      cols.bit_shift_result[i] = F::from_canonical_u8(bit_shift_result[i]);
      bit_shift_result_carry[i] = (uint8_t)carry;
      cols.bit_shift_result_carry[i] = F::from_canonical_u8(bit_shift_result_carry[i]);
    }

    // Calculate number of full bytes to shift.
    // This handles the byte-level portion of the shift operation.
    const uintptr_t num_bytes_to_shift = (uintptr_t)(event.c & 0b11111) / BYTE_SIZE;
    for (uintptr_t i = 0; i < WORD_SIZE; ++i) {
      cols.shift_by_n_bytes[i] = F::from_bool(num_bytes_to_shift == i);
    }

    // Add byte range checks for all intermediate results.
    // This ensures all values are valid bytes (0-255).
    for (int i = 0; i < WORD_SIZE; i += 2) {
      add_u8_range_check(byte_trace, bit_shift_result[i], bit_shift_result[i + 1]);
      add_u8_range_check(byte_trace, bit_shift_result_carry[i], bit_shift_result_carry[i + 1]);
    }
  }

  /// Kernel to convert shift left events to trace rows.
  template <class F>
  __global__ void sll_events_to_trace_kernel(
    const size_t event_size,
    const size_t trace_size,
    const AluEvent* events,
    F* trace_matrix,
    const size_t num_cols,
    F* byte_trace)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx * num_cols < trace_size) {
      ShiftLeftCols<F>* cols = reinterpret_cast<ShiftLeftCols<F>*>(&trace_matrix[idx * num_cols]);

      if (idx < event_size) {
        // Process real event.
        event_to_row_byte<F>(events[idx], *cols, byte_trace);
      } else {
        // Pad with dummy values for unused rows.
        cols->shift_by_n_bits[0] = F::one();
        cols->shift_by_n_bytes[0] = F::one();
        cols->bit_shift_multiplier = F::one();
      }
    }
  }

  /// Generate trace from shift left logical events.
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
      const int num_rows = trace_size / num_cols;
      const int grid_size = (num_rows + block_size - 1) / block_size;

      sll_events_to_trace_kernel<<<grid_size, block_size, 0, stream>>>(
        event_size, trace_size, events, trace, num_cols, byte_trace);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  }
} // namespace pico_gpu::sll