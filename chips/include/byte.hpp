#pragma once

#include <cassert>
#include <iostream>
#include <util/rusterror.h>
#include "types.hpp"
#include "utils.hpp"

namespace pico_gpu::byte {
  // Kernel implementation
  template <class F>
  __global__ void byte_preprocess_inputs_kernel(F* trace_matrix, const size_t num_cols)
  {
    // Calculate thread index
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = idx & 0xffff;
    uint8_t b = (row >> 8) & 0xff;
    uint8_t c = (row >> 0) & 0xff;
    if (row >= (1 << 16)) return;

    BytePreprocessedCols<F>* cols = reinterpret_cast<BytePreprocessedCols<F>*>(&trace_matrix[row * num_cols]);
    cols->b = b;
    cols->c = c;
  }

  template <class F>
  __global__ void byte_preprocess_results_kernel(F* trace_matrix, const size_t num_cols)
  {
    // Calculate thread index
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = idx & 0xffff;
    uint8_t b = (row >> 8) & 0xff;
    uint8_t c = (row >> 0) & 0xff;
    uint8_t opcode = (idx >> 16) & 0xff;
    if (row >= (1 << 16)) return;

    BytePreprocessedCols<F>* cols = reinterpret_cast<BytePreprocessedCols<F>*>(&trace_matrix[row * num_cols]);
    switch (opcode) {
    case (uint8_t)ByteOpcode::AND:
      cols->and_result = F::from_canonical_u8(b & c);
      break;
    case (uint8_t)ByteOpcode::OR:
      cols->or_result = F::from_canonical_u8(b | c);
      break;
    case (uint8_t)ByteOpcode::XOR:
      cols->xor_result = F::from_canonical_u8(b ^ c);
      break;
    case (uint8_t)ByteOpcode::SLL:
      cols->sll = F::from_canonical_u8(b << (c & 0x7));
      break;
    case (uint8_t)ByteOpcode::ShrCarry: {
      uint8_t c_mod = c & 0x7;
      uint8_t res;
      uint8_t carry;
      if (c_mod != 0) {
        res = b >> c_mod;
        // we should keep this operation as 2 lines of code
        // b << (0x8 - c_mod) gets promoted to an int which is which keeps the bits we "shifted out"
        // because left shift cannot overflow due to UB, right shifting back results in a no-op so
        // carry = (b << c) >> c transforms into carry = carry, or a no-op
        carry = b << (0x8 - c_mod);
        carry = carry >> (0x8 - c_mod);
      } else {
        res = b;
        carry = 0;
      }
      cols->shr = F::from_canonical_u8(res);
      cols->shr_carry = F::from_canonical_u8(carry);
      break;
    }
    case (uint8_t)ByteOpcode::LTU:
      cols->ltu = b < c ? F::one() : F::zero();
      break;
    case (uint8_t)ByteOpcode::MSB:
      cols->msb = b & 0x80 ? F::one() : F::zero();
      break;
    case (uint8_t)ByteOpcode::U8Range:
      break;
    case (uint8_t)ByteOpcode::U16Range:
      cols->value_u16 = F::from_canonical_u32(row);
      break;
    default:
      assert(false);
      break;
    }
  }

  template <class F>
  __global__ void byte_events_to_trace_kernel(
    const ByteLookupEvent* events, const uint32_t* multiplicities, size_t count, F* trace_matrix, size_t num_cols)
  {
    // Calculate thread index
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    const ByteLookupEvent& e = events[idx];
    uint16_t row_index = (uint16_t(e.b) << 8) + uint16_t(e.c);
    trace_matrix[row_index * num_cols + uint8_t(e.opcode)] = F::from_canonical_u32(multiplicities[idx]);
  }

  template <class F>
  inline RustError generate_preprocessed(F* trace, const size_t num_cols, cudaStream_t stream)
  {
    try {
      // launch kernel
      const int block_size = 256;
      int grid_size = (0x10000 + block_size - 1) / block_size;
      byte_preprocess_inputs_kernel<<<grid_size, block_size, 0, stream>>>(trace, num_cols);
      CUDA_OK(cudaGetLastError());
      // 9 * 0x10000 for 9 byte events
      grid_size = (0x90000 + block_size - 1) / block_size;
      byte_preprocess_results_kernel<<<grid_size, block_size, 0, stream>>>(trace, num_cols);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  } // generate_preprocessed

  static constexpr size_t NUM_ROWS = 1 << 16; // 65536
  static constexpr size_t NUM_BYTE_MULT_COLS = 9;
  static constexpr size_t TRACE_SIZE = NUM_ROWS * NUM_BYTE_MULT_COLS;

  /// Initialize byte trace GPU memory with zeros.
  template <class F>
  inline RustError initialize_byte_trace(F* d_byte_trace, const size_t byte_trace_size, cudaStream_t stream)
  {
    try {
      if (byte_trace_size == 0) return RustError{cudaSuccess};

      CUDA_OK(cudaMemsetAsync(d_byte_trace, 0, byte_trace_size * sizeof(F), stream));
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  }

  /// Kernel to finalize byte trace by converting raw counts to field elements.
  template <class F>
  __global__ void finalize_kernel(F* trace_matrix, const size_t n)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
      uint32_t* raw_ptr = reinterpret_cast<uint32_t*>(&trace_matrix[idx]);
      const uint32_t v = *raw_ptr;
      trace_matrix[idx] = F::from_canonical_u32(v);
    }
  }

  /// Finalize byte trace after all events are processed.
  template <class F>
  inline RustError finalize_byte_trace(F* d_byte_trace, const size_t byte_trace_size, cudaStream_t stream)
  {
    if (byte_trace_size == 0) return RustError{cudaSuccess};

    const int block_size = 256;
    const int grid_size = (byte_trace_size + block_size - 1) / block_size;

    finalize_kernel<F><<<grid_size, block_size, 0, stream>>>(d_byte_trace, byte_trace_size);

    return RustError{cudaSuccess};
  }

  /// Handle byte lookup event by incrementing the corresponding trace entry.
  template <class F>
  __device__ inline void
  handle_byte_lookup_event(F* trace_matrix, const ByteOpcode opcode, const uint8_t b, const uint8_t c)
  {
    const uint16_t row_index = (uint16_t(b) << 8) | uint16_t(c);
    const uint32_t trace_index = row_index * NUM_BYTE_MULT_COLS + uint8_t(opcode);

    uint32_t* raw_ptr = reinterpret_cast<uint32_t*>(&trace_matrix[trace_index]);
    atomicAdd(raw_ptr, 1);
  }

  /// Add u8 range check for a pair of bytes.
  template <class F>
  __device__ inline void add_u8_range_check(F* trace_matrix, const uint8_t b, const uint8_t c)
  {
    handle_byte_lookup_event<F>(trace_matrix, ByteOpcode::U8Range, b, c);
  }

  /// Add u16 range check for a 16-bit value.
  template <class F>
  __device__ inline void add_u16_range_check(F* trace_matrix, const uint16_t a)
  {
    const uint16_t b = a >> 8;
    const uint16_t c = a & 0xFF;
    handle_byte_lookup_event<F>(trace_matrix, ByteOpcode::U16Range, static_cast<uint8_t>(b), static_cast<uint8_t>(c));
  }

  /// Add u8 range checks for a byte array.
  template <class F>
  __device__ inline void add_u8_range_checks(F* trace_matrix, const uint8_t* bytes, const size_t bytes_size)
  {
    for (size_t i = 0; i < bytes_size; i += 2) {
      const uint8_t b = bytes[i];
      const uint8_t c = (i + 1 < bytes_size) ? bytes[i + 1] : 0;
      add_u8_range_check<F>(trace_matrix, b, c);
    }
  }

  /// Add u8 range checks for a fixed-size word.
  template <class F>
  __device__ inline void add_u8_range_checks(F* trace_matrix, const array_t<uint8_t, WORD_SIZE>& bytes)
  {
    for (uintptr_t i = 0; i < WORD_SIZE; i += 2) {
      add_u8_range_check<F>(trace_matrix, bytes[i], bytes[i + 1]);
    }
  }

  /// Add u8 range checks for field element array.
  template <class F>
  __device__ inline void
  add_u8_range_checks_field(F* trace_matrix, const F* field_values, const size_t field_values_size)
  {
    for (size_t i = 0; i < field_values_size; i++) {
      const uint8_t byte_value = static_cast<uint8_t>(field_values[i].as_canonical_u32());
      add_u8_range_check<F>(trace_matrix, byte_value, 0);
    }
  }

  /// Add u16 range checks for an array of 16-bit values.
  template <class F>
  __device__ inline void add_u16_range_checks(F* trace_matrix, const uint16_t* values, const size_t values_size)
  {
    for (size_t i = 0; i < values_size; i++) {
      add_u16_range_check<F>(trace_matrix, values[i]);
    }
  }
} // namespace pico_gpu::byte
