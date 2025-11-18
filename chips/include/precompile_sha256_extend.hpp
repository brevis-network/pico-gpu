#pragma once

#include "memory_read_write.hpp"
#include "utils.hpp"
#include "types.hpp"
#include "util/exception.cuh"
#include "util/rusterror.h"
#include <iostream>
#include <ostream>
#include <cstring>
#include "../../ff/ff_config.hpp"
#include "byte.hpp"
#include "sha256_operations.hpp"

namespace pico_gpu::precompile_sha_extend {
  using namespace sha256_operations;
  using namespace memory_read_write;
  using namespace byte;

  template <typename T>
  struct Add4Operation;
  template <typename T>
  struct FixedShiftRightOperation;
  template <typename T>
  struct IsZeroGadget;

  /// SHA256 sigma0 function for message schedule.
  __device__ inline uint32_t sigma0(uint32_t x) { return ((x >> 7) | (x << 25)) ^ ((x >> 18) | (x << 14)) ^ (x >> 3); }

  /// SHA256 sigma1 function for message schedule.
  __device__ inline uint32_t sigma1(uint32_t x)
  {
    return ((x >> 17) | (x << 15)) ^ ((x >> 19) | (x << 13)) ^ (x >> 10);
  }

  /// Fixed right shift operation with carry tracking.
  template <typename T>
  struct FixedShiftRightOperation {
    Word<T> value;
    Word<T> shift;
    Word<T> carry;

    __device__ FixedShiftRightOperation() {}

    __device__ static constexpr size_t nb_bytes_to_shift(size_t rotation) { return rotation / 8; }

    __device__ static constexpr size_t nb_bits_to_shift(size_t rotation) { return rotation % 8; }

    __device__ static constexpr uint32_t carry_multiplier(size_t rotation)
    {
      const size_t nb_bits_to_shift_val = rotation % 8;
      return 1 << (8 - nb_bits_to_shift_val);
    }

    __device__ static void shr_carry(uint8_t input, uint8_t shift_amount, uint8_t* shift_out, uint8_t* carry_out)
    {
      *shift_out = input >> shift_amount;
      *carry_out = input & ((1 << shift_amount) - 1);
    }

    __device__ uint32_t populate(uint32_t input, const size_t rotation, field_t* byte_trace)
    {
      const uint32_t expected = input >> rotation;

      // Extract bytes in little endian order.
      uint8_t input_bytes_raw[4] = {
        (uint8_t)(input & 0xFF), (uint8_t)((input >> 8) & 0xFF), (uint8_t)((input >> 16) & 0xFF),
        (uint8_t)((input >> 24) & 0xFF)};

      // Compute rotation constants.
      const size_t nb_bytes_to_shift = FixedShiftRightOperation::nb_bytes_to_shift(rotation);
      const size_t nb_bits_to_shift = FixedShiftRightOperation::nb_bits_to_shift(rotation);
      const uint32_t carry_mult = FixedShiftRightOperation::carry_multiplier(rotation);

      // Perform byte shift.
      uint8_t input_bytes_shifted[4] = {0, 0, 0, 0};
      for (int i = 0; i < 4; i++) {
        if (i + nb_bytes_to_shift < 4) input_bytes_shifted[i] = input_bytes_raw[(i + nb_bytes_to_shift) % 4];
      }

      // Process in reverse order.
      T first_shift = T::zero();
      T last_carry = T::zero();

      for (int i = 3; i >= 0; i--) {
        const uint8_t b = input_bytes_shifted[i];
        const uint8_t c = (uint8_t)nb_bits_to_shift;

        uint8_t shift_val, carry_val;
        shr_carry(b, c, &shift_val, &carry_val);

        handle_byte_lookup_event(byte_trace, ByteOpcode::ShrCarry, b, c);

        shift._0[i] = T::from_canonical_u8(shift_val);
        carry._0[i] = T::from_canonical_u8(carry_val);

        if (i == 3) {
          first_shift = shift._0[i];
        } else {
          value._0[i] = shift._0[i] + last_carry * T::from_canonical_u32(carry_mult);
        }

        last_carry = carry._0[i];
      }

      // For shift operation, don't move carry to first byte.
      value._0[3] = first_shift;

      return expected;
    }
  };

  /// Four-way addition operation with carry tracking.
  template <typename T>
  struct Add4Operation {
    Word<T> value;
    Word<T> is_carry_0;
    Word<T> is_carry_1;
    Word<T> is_carry_2;
    Word<T> is_carry_3;
    Word<T> carry;

    __device__ Add4Operation() {}

    __device__ uint32_t populate(uint32_t a_u32, uint32_t b_u32, uint32_t c_u32, uint32_t d_u32, field_t* byte_trace)
    {
      const uint32_t expected = a_u32 + b_u32 + c_u32 + d_u32;
      write_word_from_u32_v2(value, expected);

      // Extract bytes.
      uint8_t a[4], b[4], c[4], d[4];
      for (int i = 0; i < 4; i++) {
        a[i] = (a_u32 >> (i * 8)) & 0xFF;
        b[i] = (b_u32 >> (i * 8)) & 0xFF;
        c[i] = (c_u32 >> (i * 8)) & 0xFF;
        d[i] = (d_u32 >> (i * 8)) & 0xFF;
      }

      const uint32_t base = 256;
      uint8_t carry_vals[4] = {0, 0, 0, 0};

      // Calculate carries and set indication fields.
      for (int i = 0; i < WORD_SIZE; i++) {
        uint32_t res = (uint32_t)a[i] + (uint32_t)b[i] + (uint32_t)c[i] + (uint32_t)d[i];
        if (i > 0) res += (uint32_t)carry_vals[i - 1];
        carry_vals[i] = (uint8_t)(res / base);

        is_carry_0._0[i] = (carry_vals[i] == 0) ? T::one() : T::zero();
        is_carry_1._0[i] = (carry_vals[i] == 1) ? T::one() : T::zero();
        is_carry_2._0[i] = (carry_vals[i] == 2) ? T::one() : T::zero();
        is_carry_3._0[i] = (carry_vals[i] == 3) ? T::one() : T::zero();

        carry._0[i] = T::from_canonical_u8(carry_vals[i]);
      }

      // Add byte range check events.
      for (int i = 0; i < 4; i += 2) {
        add_u8_range_check(byte_trace, a[i], a[i + 1]);
        add_u8_range_check(byte_trace, b[i], b[i + 1]);
        add_u8_range_check(byte_trace, c[i], c[i + 1]);
        add_u8_range_check(byte_trace, d[i], d[i + 1]);
      }
      array_t<uint8_t, WORD_SIZE> expected_bytes = u32_to_le_bytes(expected);
      add_u8_range_checks(byte_trace, expected_bytes);

      return expected;
    }
  };

  /// Gadget to check if a value is zero.
  template <typename T>
  struct IsZeroGadget {
    T inverse;
    T result;

    __device__ IsZeroGadget()
    {
      inverse = T::zero();
      result = T::zero();
    }

    __device__ bool populate(T input)
    {
      if (input == T::zero()) {
        inverse = T::zero();
        result = T::one();
        return true;
      } else {
        inverse = input.reciprocal();
        result = T::zero();
        return false;
      }
    }
  };

  /// SHA256 extend columns structure.
  template <typename T>
  struct ShaExtendCols {
    // Basic inputs.
    T chunk;
    T clk;
    T w_ptr;

    // Control flags.
    T i;
    T cycle_16;
    IsZeroGadget<T> cycle_16_start;
    IsZeroGadget<T> cycle_16_end;
    T cycle_48[3];
    T cycle_48_start;
    T cycle_48_end;

    // Memory reads for w[i-15].
    MemoryReadCols<T> w_i_minus_15;
    FixedRotateRightOperation<T> w_i_minus_15_rr_7;
    FixedRotateRightOperation<T> w_i_minus_15_rr_18;
    FixedShiftRightOperation<T> w_i_minus_15_rs_3;
    XorOperation<T> s0_intermediate;
    XorOperation<T> s0;

    // Memory reads for w[i-2].
    MemoryReadCols<T> w_i_minus_2;
    FixedRotateRightOperation<T> w_i_minus_2_rr_17;
    FixedRotateRightOperation<T> w_i_minus_2_rr_19;
    FixedShiftRightOperation<T> w_i_minus_2_rs_10;
    XorOperation<T> s1_intermediate;
    XorOperation<T> s1;

    // Memory reads for other inputs.
    MemoryReadCols<T> w_i_minus_16;
    MemoryReadCols<T> w_i_minus_7;

    // Final computation.
    Add4Operation<T> s2;

    // Result.
    MemoryWriteCols<T> w_i;

    // Selector.
    T is_real;

    __device__ ShaExtendCols() {}

    /// Populate control flags for the given row.
    __device__ void populate_flags(const size_t row_idx)
    {
      const size_t j = 16 + (row_idx % 48);
      i = T::from_canonical_u32(j);

      cycle_16 = T::from_canonical_u32((row_idx + 1) % 16);

      // Check if start of 16-row cycle.
      T cycle_16_minus_g = cycle_16 - T::one();
      cycle_16_start.populate(cycle_16_minus_g);

      // Check if end of 16-row cycle.
      cycle_16_end.populate(cycle_16);

      // Set cycle_48 flags based on j value.
      cycle_48[0] = ((j >= 16 && j < 32) ? T::one() : T::zero());
      cycle_48[1] = ((j >= 32 && j < 48) ? T::one() : T::zero());
      cycle_48[2] = ((j >= 48 && j < 64) ? T::one() : T::zero());

      // Start and end of 48-row cycles.
      cycle_48_start = cycle_48[0] * cycle_16_start.result * is_real;
      cycle_48_end = cycle_48[2] * cycle_16_end.result * is_real;
    }
  };

  /// Convert SHA extend event to trace rows.
  template <typename T>
  __device__ inline void
  event_to_rows(const ShaExtendFfiEvent& event, ShaExtendCols<T>* rows, size_t& row_count, field_t* byte_trace)
  {
    size_t current_row = 0;

    // Process 48 extend operations (w[16] to w[63]).
    for (size_t j = 0; j < 48; j++) {
      ShaExtendCols<T>& cols = rows[current_row];

      cols.is_real = T::one();
      cols.populate_flags(j);
      cols.chunk = T::from_canonical_u32(event.chunk);
      cols.clk = T::from_canonical_u32(event.clk);
      cols.w_ptr = T::from_canonical_u32(event.w_ptr);

      populate(cols.w_i_minus_15, event.w_i_minus_15_reads[j], byte_trace);
      populate(cols.w_i_minus_2, event.w_i_minus_2_reads[j], byte_trace);
      populate(cols.w_i_minus_16, event.w_i_minus_16_reads[j], byte_trace);
      populate(cols.w_i_minus_7, event.w_i_minus_7_reads[j], byte_trace);

      // Compute s0 := (w[i-15] rightrotate 7) xor (w[i-15] rightrotate 18)
      // xor (w[i-15] rightshift 3).
      const uint32_t w_i_minus_15 = event.w_i_minus_15_reads[j].value;
      const uint32_t w_i_minus_15_rr_7 = cols.w_i_minus_15_rr_7.populate(w_i_minus_15, 7, byte_trace);
      const uint32_t w_i_minus_15_rr_18 = cols.w_i_minus_15_rr_18.populate(w_i_minus_15, 18, byte_trace);
      const uint32_t w_i_minus_15_rs_3 = cols.w_i_minus_15_rs_3.populate(w_i_minus_15, 3, byte_trace);
      const uint32_t s0_intermediate = cols.s0_intermediate.populate(w_i_minus_15_rr_7, w_i_minus_15_rr_18, byte_trace);
      const uint32_t s0 = cols.s0.populate(s0_intermediate, w_i_minus_15_rs_3, byte_trace);

      // Compute s1 := (w[i-2] rightrotate 17) xor (w[i-2] rightrotate 19)
      // xor (w[i-2] rightshift 10).
      const uint32_t w_i_minus_2 = event.w_i_minus_2_reads[j].value;
      const uint32_t w_i_minus_2_rr_17 = cols.w_i_minus_2_rr_17.populate(w_i_minus_2, 17, byte_trace);
      const uint32_t w_i_minus_2_rr_19 = cols.w_i_minus_2_rr_19.populate(w_i_minus_2, 19, byte_trace);
      const uint32_t w_i_minus_2_rs_10 = cols.w_i_minus_2_rs_10.populate(w_i_minus_2, 10, byte_trace);
      const uint32_t s1_intermediate = cols.s1_intermediate.populate(w_i_minus_2_rr_17, w_i_minus_2_rr_19, byte_trace);
      const uint32_t s1 = cols.s1.populate(s1_intermediate, w_i_minus_2_rs_10, byte_trace);

      // Compute s2 := w[i-16] + s0 + w[i-7] + s1.
      const uint32_t w_i_minus_7 = event.w_i_minus_7_reads[j].value;
      const uint32_t w_i_minus_16 = event.w_i_minus_16_reads[j].value;
      const uint32_t s2 = cols.s2.populate(w_i_minus_16, s0, w_i_minus_7, s1, byte_trace);

      populate(cols.w_i, event.w_i_writes[j], byte_trace);

      current_row++;
    }

    row_count = current_row;
  }

  /// Kernel to convert events to trace rows.
  template <typename T>
  __global__ void events_to_rows_kernel(
    const ShaExtendFfiEvent* __restrict__ events,
    T* __restrict__ trace_matrix,
    field_t* byte_trace,
    const size_t events_count,
    const size_t num_cols)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= events_count) return;

    ShaExtendCols<T>* cols = reinterpret_cast<ShaExtendCols<T>*>(&trace_matrix[idx * num_cols * 48]);
    size_t row_count;
    event_to_rows(events[idx], cols, row_count, byte_trace);
  }

  /// Kernel to pad trace with dummy rows.
  template <typename T>
  __global__ void pad_dummy_rows_kernel(
    T* __restrict__ trace_matrix, const size_t pad_count, const size_t num_cols, const size_t events_count)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pad_count) return;

    const size_t row_idx = events_count * 48 + idx;
    ShaExtendCols<T>* cols = reinterpret_cast<ShaExtendCols<T>*>(&trace_matrix[row_idx * num_cols]);

    memset(cols, 0, sizeof(ShaExtendCols<T>));
    cols->populate_flags(idx);
  }

  /// Generate trace from SHA extend events.
  template <class F>
  inline RustError generate_extra_record_and_main(
    const ShaExtendFfiEvent* events,
    const size_t events_count,
    F* trace,
    const size_t total_rows,
    const size_t num_cols,
    F* byte_trace,
    cudaStream_t stream)
  {
    try {
      if (events_count == 0) return RustError{cudaSuccess};

      // Initialize trace matrix with zeros.
      CUDA_OK(cudaMemsetAsync(trace, 0, (total_rows * num_cols) * sizeof(F), stream));

      const int block_size = 256;
      const int grid_size = (events_count + block_size - 1) / block_size;

      // Pad dummy rows if needed.
      if (total_rows > events_count * 48) {
        const size_t pad_count = total_rows - (events_count * 48);
        const int pad_size = (pad_count + block_size - 1) / block_size;

        pad_dummy_rows_kernel<<<pad_size, block_size, 0, stream>>>(trace, pad_count, num_cols, events_count);
        CUDA_OK(cudaGetLastError());
      }

      // Launch main kernel.
      events_to_rows_kernel<<<grid_size, block_size, 0, stream>>>(events, trace, byte_trace, events_count, num_cols);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  }
} // namespace pico_gpu::precompile_sha_extend