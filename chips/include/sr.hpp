#pragma once

#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <tuple>
#include <ff/ff_config.hpp>

#include "prelude.hpp"
#include "utils.hpp"
#include "byte.hpp"

namespace pico_gpu::sr {
  using namespace byte;

  /// Convert a shift right event to a trace row with byte lookup.
  /// Handles both SRL (logical) and SRA (arithmetic) shift operations.
  /// The implementation decomposes the shift into byte-level and bit-level
  /// operations for constraint checking in the zero-knowledge proof system.
  template <class F>
  __device__ inline void event_to_row(const AluEvent& event, ShiftRightCols<F>& cols, F* field_byte_trace)
  {
    uint32_t* byte_trace = reinterpret_cast<uint32_t*>(field_byte_trace);

    // Initialize columns with basic operands and flags.
    {
      write_word_from_u32_v2<F>(cols.a, event.a);
      write_word_from_u32_v2<F>(cols.b, event.b);
      write_word_from_u32_v2<F>(cols.c, event.c);

      // Extract most significant bit for arithmetic shift sign extension.
      cols.b_msb = F::from_canonical_u32((event.b >> 31) & 1);

      // Set operation type flags.
      cols.is_srl = F::from_bool(event.opcode == Opcode::SRL);
      cols.is_sra = F::from_bool(event.opcode == Opcode::SRA);
      cols.is_real = F::one();

      // Decompose shift amount's least significant byte into bits.
      for (uintptr_t i = 0; i < BYTE_SIZE; ++i) {
        cols.c_least_sig_byte[i] = F::from_canonical_u32((event.c >> i) & 1);
      }

      // Record MSB lookup for most significant byte of operand b.
      handle_byte_lookup_event(byte_trace, ByteOpcode::MSB, (event.b >> 24) & 0xff, 0);
    }

    // Calculate shift amounts according to RISC-V spec (take lowest 5 bits).
    const uintptr_t num_bytes_to_shift = (event.c % 32) / BYTE_SIZE;
    const uintptr_t num_bits_to_shift = (event.c % 32) % BYTE_SIZE;

    // Byte-level shifting with sign extension for SRA.
    array_t<uint8_t, LONG_WORD_SIZE> byte_shift_result{};
    {
      // Set indicator flags for number of bytes to shift.
      for (uintptr_t i = 0; i < WORD_SIZE; ++i) {
        cols.shift_by_n_bytes[i] = F::from_bool(num_bytes_to_shift == i);
      }

      // For SRA, sign-extend to 64 bits; for SRL, zero-extend.
      array_t<uint8_t, 8> sign_extended_b =
        event.opcode == Opcode::SRA ? u64_to_le_bytes((int64_t)(int32_t)event.b) : u64_to_le_bytes((uint64_t)event.b);

      // Perform byte-level shift by copying bytes from shifted positions.
      for (uintptr_t i = 0; i < LONG_WORD_SIZE - num_bytes_to_shift; ++i) {
        byte_shift_result[i] = sign_extended_b[i + num_bytes_to_shift];
        cols.byte_shift_result[i] = F::from_canonical_u8(sign_extended_b[i + num_bytes_to_shift]);
      }
    }

    // Bit-level shifting with carry propagation.
    {
      // Set indicator flags for number of bits to shift.
      for (uintptr_t i = 0; i < BYTE_SIZE; ++i) {
        cols.shift_by_n_bits[i] = F::from_bool(num_bits_to_shift == i);
      }

      // Calculate carry multiplier for combining shifted bytes.
      const uint32_t carry_multiplier = 1 << (8 - num_bits_to_shift);
      uint32_t last_carry = 0;

      array_t<uint8_t, LONG_WORD_SIZE> bit_shift_result;
      array_t<uint8_t, LONG_WORD_SIZE> shr_carry_output_carry;
      array_t<uint8_t, LONG_WORD_SIZE> shr_carry_output_shifted_byte;

      // Process bytes from most significant to least significant.
      // Each byte is shifted, and the carry is propagated to the next byte.
      for (intptr_t i = LONG_WORD_SIZE - 1; i >= 0; --i) {
        auto [shift, carry] = shr_carry(byte_shift_result[i], num_bits_to_shift);

        // Record shr_carry operation for byte lookup.
        handle_byte_lookup_event(
          byte_trace, ByteOpcode::ShrCarry, byte_shift_result[i], static_cast<uint8_t>(num_bits_to_shift & 0xff));

        shr_carry_output_carry[i] = carry;
        cols.shr_carry_output_carry[i] = F::from_canonical_u8(carry);

        shr_carry_output_shifted_byte[i] = shift;
        cols.shr_carry_output_shifted_byte[i] = F::from_canonical_u8(shift);

        // Combine shifted byte with carry from previous byte.
        const uint8_t res = (uint8_t)(((uint32_t)shift + last_carry * carry_multiplier) & 0xFF);
        bit_shift_result[i] = res;
        cols.bit_shift_result[i] = F::from_canonical_u8(res);
        last_carry = (uint32_t)carry;
      }

      // Add byte range checks for all intermediate results.
      for (int i = 0; i < LONG_WORD_SIZE; i += 2) {
        add_u8_range_check(byte_trace, byte_shift_result[i], byte_shift_result[i + 1]);
        add_u8_range_check(byte_trace, bit_shift_result[i], bit_shift_result[i + 1]);
        add_u8_range_check(byte_trace, shr_carry_output_carry[i], shr_carry_output_carry[i + 1]);
        add_u8_range_check(byte_trace, shr_carry_output_shifted_byte[i], shr_carry_output_shifted_byte[i + 1]);
      }
    }
  }

  /// Kernel to convert shift right events to trace rows.
  template <class F>
  __global__ void sr_events_to_trace_kernel(
    const size_t event_size,
    const size_t trace_size,
    const AluEvent* events,
    F* trace_matrix,
    const size_t num_cols,
    F* byte_trace)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx * num_cols < trace_size) {
      ShiftRightCols<F>* cols = reinterpret_cast<ShiftRightCols<F>*>(&trace_matrix[idx * num_cols]);

      if (idx < event_size) {
        // Process real event.
        event_to_row<F>(events[idx], *cols, byte_trace);
      } else {
        // Pad with dummy values for unused rows.
        cols->shift_by_n_bits[0] = F::one();
        cols->shift_by_n_bytes[0] = F::one();
      }
    }
  }

  /// Generate trace from shift right events.
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

      sr_events_to_trace_kernel<<<grid_size, block_size, 0, stream>>>(
        event_size, trace_size, events, trace, num_cols, byte_trace);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  }
} // namespace pico_gpu::sr