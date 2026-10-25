// Copyright (c) 2025-2026 Brevis. All Rights Reserved.
// Licensed under the Business Source License 1.1 (BUSL-1.1).
// See LICENSE.md for details.

#pragma once

#include "tracegen/riscv/memory_read_write.cuh"
#include "tracegen/common/utils.cuh"
#include "tracegen/common/types.cuh"
#include "common/exception.cuh"
#include "common/rusterror.h"
#include <iostream>
#include <ostream>
#include <cstring>
#include "field/ff_config.hpp"
#include "tracegen/riscv/syscall_addr.cuh"
#include "tracegen/riscv/byte.cuh"
#include "tracegen/precompiles/sha256_operations.cuh"
#include "tracegen/precompiles/precompile_sha256_compress.cuh"
#include "tracegen/common/is_zero.cuh"

namespace pico_gpu::precompile_sha_extend {
  using namespace sha256_operations;
  using namespace memory_read_write_u64;
  using namespace byte;
  using namespace pico_gpu::precompile_sha_compress;

  template <typename T>
  struct FixedShiftRightU32Gadget {
    T value[2];
    T higher_limb[2];

    __device__ uint32_t populate(uint32_t input, uint32_t shift, field_t* byte_trace) {
        uint32_t expected = input >> shift;
        value[0] = T::from_canonical_u16(static_cast<uint16_t>(expected & 0xFFFF));
        value[1] = T::from_canonical_u16(static_cast<uint16_t>((expected >> 16) & 0xFFFF));

        uint16_t input_limbs[2] = {static_cast<uint16_t>(input & 0xFFFF), static_cast<uint16_t>((input >> 16) & 0xFFFF)};
        uint32_t nb_limbs_to_shift = shift / 16;
        uint32_t nb_bits_to_shift = shift % 16;

        uint16_t word[2] = {0, 0};
        for (int i = 0; i < 2; i++) {
            if (i + nb_limbs_to_shift < 2) {
                word[i] = input_limbs[i + nb_limbs_to_shift];
            }
        }

        for (int i = 1; i >= 0; i--) {
            uint16_t limb = word[i];
            uint16_t lower_limb = limb & ((1 << nb_bits_to_shift) - 1);
            uint16_t higher_limb_val = limb >> nb_bits_to_shift;
            higher_limb[i] = T::from_canonical_u16(higher_limb_val);
            pico_gpu::byte::handle_bit_range_check_event(byte_trace, (uint8_t)(lower_limb >> 8), (uint8_t)(lower_limb & 0xFF), (uint8_t)nb_bits_to_shift);
            pico_gpu::byte::handle_bit_range_check_event(byte_trace, (uint8_t)(higher_limb_val >> 8), (uint8_t)(higher_limb_val & 0xFF), (uint8_t)(16 - nb_bits_to_shift));
        }
        return expected;
    }
  };

  template <typename T>
  struct Add4U32Gadget {
    T value[2];
    __device__ uint32_t populate(uint32_t a, uint32_t b, uint32_t c, uint32_t d, field_t* byte_trace) {
        uint32_t expected = a + b + c + d;
        uint16_t limbs[2] = {static_cast<uint16_t>(expected & 0xFFFF), static_cast<uint16_t>((expected >> 16) & 0xFFFF)};
        value[0] = T::from_canonical_u16(limbs[0]);
        value[1] = T::from_canonical_u16(limbs[1]);
        pico_gpu::byte::add_u16_range_check(byte_trace, limbs[0]);
        pico_gpu::byte::add_u16_range_check(byte_trace, limbs[1]);

        uint32_t base = 1u << 16;
        uint32_t carry = 0;
        uint16_t a_l[2] = {static_cast<uint16_t>(a & 0xFFFF), static_cast<uint16_t>((a >> 16) & 0xFFFF)};
        uint16_t b_l[2] = {static_cast<uint16_t>(b & 0xFFFF), static_cast<uint16_t>((b >> 16) & 0xFFFF)};
        uint16_t c_l[2] = {static_cast<uint16_t>(c & 0xFFFF), static_cast<uint16_t>((c >> 16) & 0xFFFF)};
        uint16_t d_l[2] = {static_cast<uint16_t>(d & 0xFFFF), static_cast<uint16_t>((d >> 16) & 0xFFFF)};
        uint8_t carry_limbs[2];
        for (int i = 0; i < 2; i++) {
            carry = ((uint32_t)a_l[i] + (uint32_t)b_l[i] + (uint32_t)c_l[i] + (uint32_t)d_l[i] + carry - (uint32_t)limbs[i]) / base;
            carry_limbs[i] = (uint8_t)carry;
        }
        pico_gpu::byte::add_u8_range_check(byte_trace, carry_limbs[0], carry_limbs[1]);

        return expected;
    }
  };

  template <typename T>
  struct ShaExtendCols {
    T chunk;
    T clk;
    T w_ptr[3];
    pico_gpu::precompile_sha_compress::AddrAddGadget<T> w_i_minus_15_ptr;
    pico_gpu::precompile_sha_compress::AddrAddGadget<T> w_i_minus_2_ptr;
    pico_gpu::precompile_sha_compress::AddrAddGadget<T> w_i_minus_16_ptr;
    pico_gpu::precompile_sha_compress::AddrAddGadget<T> w_i_minus_7_ptr;
    pico_gpu::precompile_sha_compress::AddrAddGadget<T> w_i_ptr;

    T i;

    MemoryReadCols<T> w_i_minus_15;
    FixedRotateRightU32Gadget<T> w_i_minus_15_rr_7;
    FixedRotateRightU32Gadget<T> w_i_minus_15_rr_18;
    FixedShiftRightU32Gadget<T> w_i_minus_15_rs_3;
    XorU32Gadget<T> s0_intermediate;
    XorU32Gadget<T> s0;

    MemoryReadCols<T> w_i_minus_2;
    FixedRotateRightU32Gadget<T> w_i_minus_2_rr_17;
    FixedRotateRightU32Gadget<T> w_i_minus_2_rr_19;
    FixedShiftRightU32Gadget<T> w_i_minus_2_rs_10;
    XorU32Gadget<T> s1_intermediate;
    XorU32Gadget<T> s1;

    MemoryReadCols<T> w_i_minus_16;
    MemoryReadCols<T> w_i_minus_7;

    Add4U32Gadget<T> s2;

    MemoryWriteCols<T> w_i;

    T is_real;
  };

  template <typename T>
  __device__ inline void split_addr(T* w_ptr, uint64_t addr) {
        w_ptr[0] = T::from_canonical_u16(static_cast<uint16_t>(addr & 0xFFFF));
        w_ptr[1] = T::from_canonical_u16(static_cast<uint16_t>((addr >> 16) & 0xFFFF));
        w_ptr[2] = T::from_canonical_u16(static_cast<uint16_t>((addr >> 32) & 0xFFFF));
  }

  template <typename T>
  __device__ inline void
  event_to_rows(const ShaExtendFfiEvent& event, ShaExtendCols<T>* rows, size_t& row_count, field_t* byte_trace)
  {
    size_t current_row = 0;

    // Process 48 extend operations (w[16] to w[63]).
    for (size_t j = 0; j < 48; j++) {
      ShaExtendCols<T>& cols = rows[current_row];

      cols.is_real = T::one();
      uint64_t i_val = 16 + j;
      cols.i = T::from_canonical_u32(static_cast<uint32_t>(i_val));
      cols.chunk = T::from_canonical_u32(event.chunk);
      cols.clk = T::from_canonical_u32(static_cast<uint32_t>(event.clk + j));

      split_addr(cols.w_ptr, event.w_ptr);

      cols.w_i_minus_15_ptr.populate(byte_trace, event.w_ptr, (i_val - 15) * 8);
      cols.w_i_minus_2_ptr.populate(byte_trace, event.w_ptr, (i_val - 2) * 8);
      cols.w_i_minus_16_ptr.populate(byte_trace, event.w_ptr, (i_val - 16) * 8);
      cols.w_i_minus_7_ptr.populate(byte_trace, event.w_ptr, (i_val - 7) * 8);
      cols.w_i_ptr.populate(byte_trace, event.w_ptr, i_val * 8);

      populate_access_u64(cols.w_i_minus_15.access, event.w_i_minus_15_reads[j].chunk, event.w_i_minus_15_reads[j].timestamp, event.w_i_minus_15_reads[j].value, event.w_i_minus_15_reads[j].prev_chunk, event.w_i_minus_15_reads[j].prev_timestamp, event.w_i_minus_15_reads[j].value, byte_trace);

      populate_access_u64(cols.w_i_minus_2.access, event.w_i_minus_2_reads[j].chunk, event.w_i_minus_2_reads[j].timestamp, event.w_i_minus_2_reads[j].value, event.w_i_minus_2_reads[j].prev_chunk, event.w_i_minus_2_reads[j].prev_timestamp, event.w_i_minus_2_reads[j].value, byte_trace);

      populate_access_u64(cols.w_i_minus_16.access, event.w_i_minus_16_reads[j].chunk, event.w_i_minus_16_reads[j].timestamp, event.w_i_minus_16_reads[j].value, event.w_i_minus_16_reads[j].prev_chunk, event.w_i_minus_16_reads[j].prev_timestamp, event.w_i_minus_16_reads[j].value, byte_trace);

      populate_access_u64(cols.w_i_minus_7.access, event.w_i_minus_7_reads[j].chunk, event.w_i_minus_7_reads[j].timestamp, event.w_i_minus_7_reads[j].value, event.w_i_minus_7_reads[j].prev_chunk, event.w_i_minus_7_reads[j].prev_timestamp, event.w_i_minus_7_reads[j].value, byte_trace);

      // Compute s0 := (w[i-15] rightrotate 7) xor (w[i-15] rightrotate 18) xor (w[i-15] rightshift 3).
      const uint32_t w_i_minus_15 = static_cast<uint32_t>(event.w_i_minus_15_reads[j].value);
      const uint32_t w_i_minus_15_rr_7 = cols.w_i_minus_15_rr_7.populate(w_i_minus_15, 7, byte_trace);
      const uint32_t w_i_minus_15_rr_18 = cols.w_i_minus_15_rr_18.populate(w_i_minus_15, 18, byte_trace);
      const uint32_t w_i_minus_15_rs_3 = cols.w_i_minus_15_rs_3.populate(w_i_minus_15, 3, byte_trace);
      const uint32_t s0_intermediate = cols.s0_intermediate.populate(w_i_minus_15_rr_7, w_i_minus_15_rr_18, byte_trace);
      const uint32_t s0 = cols.s0.populate(s0_intermediate, w_i_minus_15_rs_3, byte_trace);

      // Compute s1 := (w[i-2] rightrotate 17) xor (w[i-2] rightrotate 19) xor (w[i-2] rightshift 10).
      const uint32_t w_i_minus_2 = static_cast<uint32_t>(event.w_i_minus_2_reads[j].value);
      const uint32_t w_i_minus_2_rr_17 = cols.w_i_minus_2_rr_17.populate(w_i_minus_2, 17, byte_trace);
      const uint32_t w_i_minus_2_rr_19 = cols.w_i_minus_2_rr_19.populate(w_i_minus_2, 19, byte_trace);
      const uint32_t w_i_minus_2_rs_10 = cols.w_i_minus_2_rs_10.populate(w_i_minus_2, 10, byte_trace);
      const uint32_t s1_intermediate = cols.s1_intermediate.populate(w_i_minus_2_rr_17, w_i_minus_2_rr_19, byte_trace);
      const uint32_t s1 = cols.s1.populate(s1_intermediate, w_i_minus_2_rs_10, byte_trace);

      // Compute s2 := w[i-16] + s0 + w[i-7] + s1.
      const uint32_t w_i_minus_7 = static_cast<uint32_t>(event.w_i_minus_7_reads[j].value);
      const uint32_t w_i_minus_16 = static_cast<uint32_t>(event.w_i_minus_16_reads[j].value);
      const uint32_t s2 = cols.s2.populate(w_i_minus_16, s0, w_i_minus_7, s1, byte_trace);

      write_word_from_u64(cols.w_i.prev_value, event.w_i_writes[j].prev_value);
      populate_access_u64(cols.w_i.access, event.w_i_writes[j].chunk, event.w_i_writes[j].timestamp, event.w_i_writes[j].value, event.w_i_writes[j].prev_chunk, event.w_i_writes[j].prev_timestamp, event.w_i_writes[j].prev_value, byte_trace);
      pico_gpu::byte::handle_byte_lookup_event(byte_trace, ByteOpcode::LTU, (uint8_t)j, (uint8_t)48);

      current_row++;
    }

    row_count = current_row;
  }

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

  template <typename T>
  __global__ void pad_dummy_rows_kernel(
    T* __restrict__ trace_matrix, const size_t pad_count, const size_t num_cols, const size_t events_count)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pad_count) return;

    const size_t row_idx = events_count * 48 + idx;
    ShaExtendCols<T>* cols = reinterpret_cast<ShaExtendCols<T>*>(&trace_matrix[row_idx * num_cols]);

    memset(cols, 0, sizeof(ShaExtendCols<T>));
  }

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

      CUDA_OK(cudaMemsetAsync(trace, 0, (total_rows * num_cols) * sizeof(F), stream));

      const int block_size = 256;
      const int grid_size = (events_count + block_size - 1) / block_size;

      if (total_rows > events_count * 48) {
        const size_t pad_count = total_rows - (events_count * 48);
        const int pad_size = (pad_count + block_size - 1) / block_size;
        pad_dummy_rows_kernel<<<pad_size, block_size, 0, stream>>>(trace, pad_count, num_cols, events_count);
        CUDA_OK(cudaGetLastError());
      }

      if (byte_trace != nullptr) {
        events_to_rows_kernel<<<grid_size, block_size, 0, stream>>>(events, trace, byte_trace, events_count, num_cols);
        CUDA_OK(cudaGetLastError());
      }
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  }

  // ==========================================
  template <typename T>
  struct ShaExtendControlCols {
    // Leads the ShaExtend state bus so the state machine is pinned to one chunk. clk restarts
    // at zero in every chunk, so without this the bus cannot tell two chunks apart.
    T chunk;
    T clk;
    SyscallAddrGadget<T> w_ptr;
    pico_gpu::precompile_sha_compress::AddrAddGadget<T> w_16th_addr;
    pico_gpu::precompile_sha_compress::AddrAddGadget<T> w_17th_addr;
    pico_gpu::precompile_sha_compress::AddrAddGadget<T> w_64th_addr;

    T is_real;
  };

  template <typename T>
  __device__ inline void
  event_to_row_control(const ShaExtendFfiEvent& event, ShaExtendControlCols<T>* cols, field_t* byte_trace)
  {
      cols->chunk = T::from_canonical_u32(event.chunk);
      cols->clk = T::from_canonical_u32(event.clk);
      cols->w_ptr.populate(byte_trace, event.w_ptr, 512);
      cols->w_16th_addr.populate(byte_trace, event.w_ptr, 15 * 8);
      cols->w_17th_addr.populate(byte_trace, event.w_ptr, 16 * 8);
      cols->w_64th_addr.populate(byte_trace, event.w_ptr, 63 * 8);
      cols->is_real = T::one();
  }

  template <typename T>
  __global__ void control_events_to_rows_kernel(
    const ShaExtendFfiEvent* __restrict__ events,
    T* __restrict__ trace_matrix,
    field_t* byte_trace,
    const size_t events_count,
    const size_t num_cols)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= events_count) return;

    ShaExtendControlCols<T>* cols = reinterpret_cast<ShaExtendControlCols<T>*>(&trace_matrix[idx * num_cols]);
    event_to_row_control(events[idx], cols, byte_trace);
  }

  template <typename T>
  __global__ void control_pad_dummy_rows_kernel(
    T* __restrict__ trace_matrix, const size_t pad_count, const size_t num_cols, const size_t events_count)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pad_count) return;

    const size_t row_idx = events_count + idx;
    ShaExtendControlCols<T>* cols = reinterpret_cast<ShaExtendControlCols<T>*>(&trace_matrix[row_idx * num_cols]);

    memset(cols, 0, sizeof(ShaExtendControlCols<T>));
  }

  template <class F>
  inline RustError generate_control_extra_record_and_main(
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

      CUDA_OK(cudaMemsetAsync(trace, 0, (total_rows * num_cols) * sizeof(F), stream));

      const int block_size = 256;
      const int grid_size = (events_count + block_size - 1) / block_size;

      if (total_rows > events_count) {
        const size_t pad_count = total_rows - events_count;
        const int pad_size = (pad_count + block_size - 1) / block_size;
        control_pad_dummy_rows_kernel<<<pad_size, block_size, 0, stream>>>(trace, pad_count, num_cols, events_count);
        CUDA_OK(cudaGetLastError());
      }

      if (byte_trace != nullptr) {
        control_events_to_rows_kernel<<<grid_size, block_size, 0, stream>>>(events, trace, byte_trace, events_count, num_cols);
        CUDA_OK(cudaGetLastError());
      }
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  }

} // namespace pico_gpu::precompile_sha_extend
