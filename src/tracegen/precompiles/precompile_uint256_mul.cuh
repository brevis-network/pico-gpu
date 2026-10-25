// Copyright (c) 2025-2026 Brevis. All Rights Reserved.
// Licensed under the Business Source License 1.1 (BUSL-1.1).
// See LICENSE.md for details.

#pragma once

#include "common/rusterror.h"
#include "tracegen/common/types.cuh"
#include "tracegen/riscv/memory_read_write.cuh"
#include "tracegen/common/utils.cuh"
#include "tracegen/common/is_zero.cuh"
#include "tracegen/precompiles/fp_op.cuh"
#include "tracegen/precompiles/fp_lt.cuh"
#include "tracegen/riscv/syscall_addr.cuh"

namespace pico_gpu::precompile_uint256_mul {
  // In the u64 branch, each "word" is a u64 dword (4 u16 limbs).
  // NumWords = 4 dwords = 32 bytes = 256 bits
  // NumLimbs = 32 (byte-level limbs for field ops, unchanged)
  // Note: the FFI event still uses 8 u32 words per value (pairs of u32 = 1 dword).
  const size_t NumLimbs = 32;
  const size_t NumWords = 4;       // u64 dwords (was 8 u32 words in 32-bit branch)
  const size_t FfiWords = 8;       // u32 words in FFI event arrays
  const size_t NumWitnesses = 63;
  const size_t WitnessOffset = 1 << 14;

  template <typename T>
  struct AddrAddGadget {
    T value[3];
    __device__ uint64_t populate(field_t* byte_trace, uint64_t base, uint64_t offset) {
        uint64_t expected = base + offset;
        uint16_t limbs[3] = {
            static_cast<uint16_t>(expected & 0xFFFF),
            static_cast<uint16_t>((expected >> 16) & 0xFFFF),
            static_cast<uint16_t>((expected >> 32) & 0xFFFF)
        };
        value[0] = T::from_canonical_u16(limbs[0]);
        value[1] = T::from_canonical_u16(limbs[1]);
        value[2] = T::from_canonical_u16(limbs[2]);
        pico_gpu::byte::add_u16_range_check(byte_trace, limbs[0]);
        pico_gpu::byte::add_u16_range_check(byte_trace, limbs[1]);
        pico_gpu::byte::add_u16_range_check(byte_trace, limbs[2]);
        return expected;
    }
  };

  // U16ToU8Gadget: stores low bytes of 4 u16 limbs.
  template <class T>
  struct U16ToU8Gadget {
    T low_bytes[4]; // WORD_SIZE = 4
  };

  // MemoryWriteColsU8 = MemoryWriteCols + U16ToU8Gadget for prev_value.
  template <class T>
  struct MemoryWriteColsU8 {
    MemoryWriteCols<T> inner;
    U16ToU8Gadget<T> prev_value_u8;
  };

  // MemoryReadColsU8 = MemoryReadCols + U16ToU8Gadget for prev_value.
  template <class T>
  struct MemoryReadColsU8 {
    MemoryReadCols<T> inner;
    U16ToU8Gadget<T> prev_value_u8;
  };

  // Populate U16ToU8Gadget with safe range checks.
  template <class F>
  __device__ void populate_u16_to_u8_safe(U16ToU8Gadget<F>& gadget, F* byte_trace, uint64_t value) {
    uint16_t limbs[4] = {
      static_cast<uint16_t>(value & 0xFFFF),
      static_cast<uint16_t>((value >> 16) & 0xFFFF),
      static_cast<uint16_t>((value >> 32) & 0xFFFF),
      static_cast<uint16_t>((value >> 48) & 0xFFFF),
    };
    for (int i = 0; i < 4; ++i) {
      uint8_t low_byte = limbs[i] & 0xFF;
      uint8_t high_byte = (limbs[i] >> 8) & 0xFF;
      gadget.low_bytes[i] = F::from_canonical_u8(low_byte);
      pico_gpu::byte::add_u8_range_check(byte_trace, low_byte, high_byte);
    }
  }

  // GPU Uint256MulCols must match the Rust Uint256MulCols layout exactly.
  // In u64 branch: NumWords = 4 (dwords), each memory column uses *U8 variants.
  template <class T>
  struct Uint256MulCols {
    T chunk;
    T clk;
    SyscallAddrGadget<T> x_ptr;
    SyscallAddrGadget<T> y_ptr;
    AddrAddGadget<T> x_addrs[NumWords];              // 4
    AddrAddGadget<T> y_and_modulus_addrs[NumWords * 2]; // 8
    MemoryWriteColsU8<T> x_memory[NumWords];          // 4
    MemoryReadColsU8<T> y_memory[NumWords];            // 4
    MemoryReadColsU8<T> modulus_memory[NumWords];      // 4
    IsZeroOperation<T> modulus_is_zero;
    T modulus_is_not_zero;
    FieldOpCols<T, 32, 63> output;
    FieldLtCols<T, 32, 63> output_range_check;
    T is_real;
  };

  // Helper: populate a u64 MemoryWriteColsU8 from two paired u32 FfiMemoryWriteRecords.
  // In the FFI event, each u64 dword is split into two u32 records (low, high).
  template <class F>
  __device__ void populate_write_u64(
    MemoryWriteColsU8<F>& col,
    const MemoryWriteRecord& rec_lo,
    const MemoryWriteRecord& rec_hi,
    F* byte_trace)
  {
    // Current value = rec_lo.value | (rec_hi.value << 32)
    uint64_t cur_value = (uint64_t)rec_lo.value | ((uint64_t)rec_hi.value << 32);
    uint64_t prev_value = (uint64_t)rec_lo.prev_value | ((uint64_t)rec_hi.prev_value << 32);

    // Write prev_value as Word (4 u16 limbs)
    col.inner.prev_value._0[0] = F::from_canonical_u16(static_cast<uint16_t>(prev_value & 0xFFFF));
    col.inner.prev_value._0[1] = F::from_canonical_u16(static_cast<uint16_t>((prev_value >> 16) & 0xFFFF));
    col.inner.prev_value._0[2] = F::from_canonical_u16(static_cast<uint16_t>((prev_value >> 32) & 0xFFFF));
    col.inner.prev_value._0[3] = F::from_canonical_u16(static_cast<uint16_t>((prev_value >> 48) & 0xFFFF));

    // Populate access cols
    col.inner.access.value._0[0] = F::from_canonical_u16(static_cast<uint16_t>(cur_value & 0xFFFF));
    col.inner.access.value._0[1] = F::from_canonical_u16(static_cast<uint16_t>((cur_value >> 16) & 0xFFFF));
    col.inner.access.value._0[2] = F::from_canonical_u16(static_cast<uint16_t>((cur_value >> 32) & 0xFFFF));
    col.inner.access.value._0[3] = F::from_canonical_u16(static_cast<uint16_t>((cur_value >> 48) & 0xFFFF));

    // Sentinel-or-real prev clk; see populate_access_u64 in memory_read_write.cuh.
    const uint32_t prev_clk =
      (rec_lo.prev_chunk == rec_lo.chunk) ? rec_lo.prev_timestamp : 0u;
    col.inner.access.prev_clk = F::from_canonical_u32(prev_clk);

    const uint32_t diff_minus_one = rec_lo.timestamp - prev_clk - 1;
    const uint16_t diff_16bit = static_cast<uint16_t>(diff_minus_one & 0xFFFF);
    const uint16_t diff_hi9 = static_cast<uint16_t>((diff_minus_one >> 16) & 0x1FF);
    col.inner.access.diff_16bit_limb = F::from_canonical_u16(diff_16bit);
    // Single 9-bit high limb (bits 16..25), BitRange(9)-checked.
    col.inner.access.diff_hi9 = F::from_canonical_u16(diff_hi9);

    pico_gpu::byte::add_u16_range_check(byte_trace, diff_16bit);
    pico_gpu::byte::add_bit_range_check(byte_trace, diff_hi9, 9);

    // Populate U16ToU8Gadget for prev_value.
    populate_u16_to_u8_safe(col.prev_value_u8, byte_trace, prev_value);
  }

  // Helper: populate a u64 MemoryReadColsU8 from two paired u32 FfiMemoryReadRecords.
  template <class F>
  __device__ void populate_read_u64(
    MemoryReadColsU8<F>& col,
    const MemoryReadRecord& rec_lo,
    const MemoryReadRecord& rec_hi,
    F* byte_trace,
    uint64_t u8_source_value)
  {
    uint64_t cur_value = (uint64_t)rec_lo.value | ((uint64_t)rec_hi.value << 32);

    col.inner.access.value._0[0] = F::from_canonical_u16(static_cast<uint16_t>(cur_value & 0xFFFF));
    col.inner.access.value._0[1] = F::from_canonical_u16(static_cast<uint16_t>((cur_value >> 16) & 0xFFFF));
    col.inner.access.value._0[2] = F::from_canonical_u16(static_cast<uint16_t>((cur_value >> 32) & 0xFFFF));
    col.inner.access.value._0[3] = F::from_canonical_u16(static_cast<uint16_t>((cur_value >> 48) & 0xFFFF));

    // Sentinel-or-real prev clk; see populate_access_u64 in memory_read_write.cuh.
    const uint32_t prev_clk =
      (rec_lo.prev_chunk == rec_lo.chunk) ? rec_lo.prev_timestamp : 0u;
    col.inner.access.prev_clk = F::from_canonical_u32(prev_clk);

    const uint32_t diff_minus_one = rec_lo.timestamp - prev_clk - 1;
    const uint16_t diff_16bit = static_cast<uint16_t>(diff_minus_one & 0xFFFF);
    const uint16_t diff_hi9 = static_cast<uint16_t>((diff_minus_one >> 16) & 0x1FF);
    col.inner.access.diff_16bit_limb = F::from_canonical_u16(diff_16bit);
    // Single 9-bit high limb (bits 16..25), BitRange(9)-checked.
    col.inner.access.diff_hi9 = F::from_canonical_u16(diff_hi9);

    pico_gpu::byte::add_u16_range_check(byte_trace, diff_16bit);
    pico_gpu::byte::add_bit_range_check(byte_trace, diff_hi9, 9);

    // Populate U16ToU8Gadget for prev_value (uses u8_source_value).
    populate_u16_to_u8_safe(col.prev_value_u8, byte_trace, u8_source_value);
  }

  /// Convert a uint256 multiply event to a trace row.
  template <class F>
  __PICO_HOSTDEV__ inline void event_to_row(
    const Uint256MulEvent* events, Uint256MulCols<F>& cols, F* byte_trace, const size_t idx, const size_t count)
  {
    if (idx < count) {
      const auto& event = events[idx];

      // Assign basic values to the columns.
      cols.is_real = F::one();
      cols.chunk = F::from_canonical_u32(static_cast<uint32_t>(event.chunk));
      cols.clk = F::from_canonical_u32(static_cast<uint32_t>(event.clk));

      cols.x_ptr.populate(byte_trace, event.x_ptr, 32);
      cols.y_ptr.populate(byte_trace, event.y_ptr, 64);

      // Populate memory columns.
      // Each dword (i) maps to two u32 FFI records (2*i, 2*i+1).
      for (int i = 0; i < (int)NumWords; ++i) {
        // x: write, u8 source is prev_value (handled internally)
        populate_write_u64(cols.x_memory[i],
          event.x_memory_records[2*i], event.x_memory_records[2*i+1], byte_trace);

        // y: read, u8 source is value (for reads, prev_value == value)
        uint64_t y_val = (uint64_t)event.y_memory_records[2*i].value
                       | ((uint64_t)event.y_memory_records[2*i+1].value << 32);
        populate_read_u64(cols.y_memory[i],
          event.y_memory_records[2*i], event.y_memory_records[2*i+1], byte_trace, y_val);

        // modulus: read, u8 source is value
        uint64_t mod_val = (uint64_t)event.modulus_memory_records[2*i].value
                         | ((uint64_t)event.modulus_memory_records[2*i+1].value << 32);
        populate_read_u64(cols.modulus_memory[i],
          event.modulus_memory_records[2*i], event.modulus_memory_records[2*i+1], byte_trace, mod_val);

        cols.x_addrs[i].populate(byte_trace, event.x_ptr, i * 8);
        cols.y_and_modulus_addrs[i].populate(byte_trace, event.y_ptr, i * 8);
        cols.y_and_modulus_addrs[NumWords + i].populate(byte_trace, event.y_ptr, (NumWords + i) * 8);
      }

      // Check if modulus is zero.
      // The FFI event still uses uint32_t modulus[8].
      const uint32_t* modulus = event.modulus;
      uint32_t modulus_byte_sum = 0;
      uint8_t modulus_bytes[NumLimbs];
      fp_op::words_to_bytes_le<FfiWords, NumLimbs>(modulus, modulus_bytes);
      for (int i = 0; i < (int)NumLimbs; ++i) {
        modulus_byte_sum += modulus_bytes[i];
      }
      is_zero::populate(cols.modulus_is_zero, modulus_byte_sum);

      // Populate the output column.
      const bool modulus_is_zero = fp_op::are_words_zero<FfiWords>(modulus);

      uint32_t two[FfiWords];
      for (int i = 1; i < (int)FfiWords; ++i)
        two[i] = 0;
      two[0] = 2;

      uint32_t modulus_minus_two[FfiWords];
      fp_op::sub_words_with_borrow<FfiWords>(modulus, two, modulus_minus_two);

      uint32_t result[FfiWords];
      fp_op::populate_with_modulus<F, FfiWords, NumLimbs, NumWitnesses, 8, WitnessOffset>(
        cols.output, event.x, event.y, result, modulus, modulus_minus_two, FieldOperation::Mul, modulus_is_zero,
        byte_trace);

      cols.modulus_is_not_zero = F::one() - cols.modulus_is_zero.result;
      if (cols.modulus_is_not_zero == F::one()) {
        fp_lt::populate<F, NumLimbs, NumWitnesses>(cols.output_range_check, result, modulus, byte_trace);
      }
    } else {
      // Populate dummy row.
      uint32_t zero[FfiWords];
      uint32_t result[FfiWords];
      for (int i = 0; i < (int)FfiWords; ++i)
        zero[i] = 0;

      uint32_t neg_two[FfiWords];
      neg_two[0] = 0xFFFFFFFF - 0x1;
      for (int i = 1; i < (int)FfiWords; ++i)
        neg_two[i] = 0xFFFFFFFF;

      fp_op::populate_with_modulus<F, FfiWords, NumLimbs, NumWitnesses, 8, WitnessOffset>(
        cols.output, zero, zero, result, zero, neg_two, FieldOperation::Mul, true, nullptr);
    }
  }

  /// Kernel to convert events to trace rows.
  template <class F>
  __global__ void events_to_trace_kernel(
    const Uint256MulEvent* events,
    const size_t count,
    F* trace_matrix,
    F* byte_trace,
    const size_t num_cols,
    const size_t num_rows)
  {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;

    Uint256MulCols<F>* cols = reinterpret_cast<Uint256MulCols<F>*>(&trace_matrix[idx * num_cols]);
    event_to_row<F>(events, *cols, byte_trace, idx, count);
  }

  /// Generate trace from uint256 multiply events.
  template <class F>
  inline RustError generate_extra_record_and_main(
    const size_t num_cols,
    const size_t event_size,
    const size_t trace_size,
    const Uint256MulEvent* events,
    F* trace,
    F* byte_trace,
    cudaStream_t stream)
  {
    try {
      // Initialize trace matrix with zeros.
      CUDA_OK(cudaMemsetAsync(trace, 0, trace_size * sizeof(F), stream));
      if (event_size == 0) return RustError{cudaSuccess};

      // Launch kernel.
      const uint32_t block_size = 256;
      const uint32_t num_rows = trace_size / num_cols;
      const uint32_t grid_size = (num_rows + block_size - 1) / block_size;

      events_to_trace_kernel<F>
        <<<grid_size, block_size, 0, stream>>>(events, event_size, trace, byte_trace, num_cols, num_rows);
      CUDA_OK(cudaGetLastError());

      CUDA_OK(cudaStreamSynchronize(stream));
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  }
} // namespace pico_gpu::precompile_uint256_mul