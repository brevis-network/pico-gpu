// Copyright (c) 2025-2026 Brevis. All Rights Reserved.
// Licensed under the Business Source License 1.1 (BUSL-1.1).
// See LICENSE.md for details.

#pragma once

#include <iostream>
#include "common/rusterror.h"
#include "tracegen/common/prelude.cuh"
#include "tracegen/common/utils.cuh"
#include "tracegen/riscv/byte.cuh"


// ─── u64 bitwise (BitwiseValueCols matching Rust layout) ────────────────────

#include "tracegen/riscv/divrem.cuh"

namespace pico_gpu::bitwise_u64 {
  using namespace pico_gpu::byte;
  using namespace pico_gpu::divrem;

  static constexpr size_t WORD_BYTE_SIZE = 8;

  /// BitwiseValueColsU64 matching Rust BitwiseValueCols layout (#[repr(C)]).
  /// Layout: b, c, BitWiseU16Gadget { U16ToU8Gadget(b), U16ToU8Gadget(c), BitWiseGadget(result) }, is_xor, is_or, is_and
  template <typename T>
  struct BitwiseValueColsU64 {
    Word<T> b;
    Word<T> c;
    // U16ToU8Gadget for b: low_bytes[4]
    T b_low_bytes[4];
    // U16ToU8Gadget for c: low_bytes[4]
    T c_low_bytes[4];
    // BitWiseGadget: result[8]
    T result[WORD_BYTE_SIZE];
    T is_xor;
    T is_or;
    T is_and;
  };

  template <class F>
  __device__ inline void event_to_row(
    const AluEventU64& event, BitwiseValueColsU64<F>& cols, F* byte_trace)
  {
    uint64_t b = event.b;
    uint64_t c = event.c;
    uint64_t a = event.a;

    write_word_from_u64(cols.b, b);
    write_word_from_u64(cols.c, c);

    cols.is_xor = F::from_bool(event.opcode == Opcode::XOR);
    cols.is_or = F::from_bool(event.opcode == Opcode::OR);
    cols.is_and = F::from_bool(event.opcode == Opcode::AND);

    // Populate U16ToU8Gadget: extract low byte from each u16 limb
    // b as 4 u16 limbs
    uint16_t b_limbs[4] = {
      (uint16_t)(b & 0xFFFF), (uint16_t)((b >> 16) & 0xFFFF),
      (uint16_t)((b >> 32) & 0xFFFF), (uint16_t)((b >> 48) & 0xFFFF)
    };
    uint16_t c_limbs[4] = {
      (uint16_t)(c & 0xFFFF), (uint16_t)((c >> 16) & 0xFFFF),
      (uint16_t)((c >> 32) & 0xFFFF), (uint16_t)((c >> 48) & 0xFFFF)
    };
    for (int i = 0; i < 4; ++i) {
      cols.b_low_bytes[i] = F::from_canonical_u8((uint8_t)(b_limbs[i] & 0xFF));
      cols.c_low_bytes[i] = F::from_canonical_u8((uint8_t)(c_limbs[i] & 0xFF));
    }

    const array_t<uint8_t, WORD_BYTE_SIZE> a_bytes = u64_to_le_bytes(a);
    const array_t<uint8_t, WORD_BYTE_SIZE> b_bytes = u64_to_le_bytes(b);
    const array_t<uint8_t, WORD_BYTE_SIZE> c_bytes = u64_to_le_bytes(c);

    ByteOpcode byte_opcode = ByteOpcode::AND;
    if (event.opcode == Opcode::XOR) byte_opcode = ByteOpcode::XOR;
    else if (event.opcode == Opcode::OR) byte_opcode = ByteOpcode::OR;
    else if (event.opcode == Opcode::AND) byte_opcode = ByteOpcode::AND;

    for (uintptr_t i = 0; i < WORD_BYTE_SIZE; ++i) {
      cols.result[i] = F::from_canonical_u8(a_bytes[i]);
      handle_byte_lookup_event(byte_trace, byte_opcode, b_bytes[i], c_bytes[i]);
    }
  }

  template <class F, size_t NumCols>
  __global__ void bitwise_events_to_trace_kernel(
    const AluEventU64* events,
    const size_t count,
    F* trace_matrix,
    const size_t num_cols,
    F* byte_trace)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if constexpr (NumCols > 0 && NumCols == sizeof(BitwiseValueColsU64<F>) / sizeof(F) &&
                  sizeof(F) == sizeof(uint32_t)) {
      // WSTAGE fast path (compile-time-width specialization only): the 32 rows of a warp
      // form ONE contiguous row-major span in the trace. Each thread builds its row in
      // dynamic shared memory (stride-27 staging writes are bank-conflict-free:
      // gcd(27,32)=1), then the warp flushes the whole span linearly = fully coalesced
      // 128B stores, removing the sector amplification of per-thread strided 108B row
      // stores (3.0x standalone). Byte-identical: the staged row is zero-initialized and
      // flushed in full, matching the pre-zeroed buffer; event_to_row and the byte-lookup
      // side effects run unmodified.
      extern __shared__ uint32_t wstage_smem[];
      const int warp = threadIdx.x >> 5;
      const int lane = threadIdx.x & 31;
      uint32_t* wrow = wstage_smem + (size_t)warp * 32 * NumCols; // this warp's 32xNumCols tile
      uint32_t* myrow = wrow + (size_t)lane * NumCols;            // this thread's staged row
#pragma unroll
      for (int i = 0; i < (int)NumCols; i++) myrow[i] = 0u;
      __syncwarp();
      if (idx < count) {
        BitwiseValueColsU64<F>* cols = reinterpret_cast<BitwiseValueColsU64<F>*>(myrow);
        event_to_row<F>(events[idx], *cols, byte_trace);
      }
      __syncwarp();
      const size_t warp_base = (size_t)blockIdx.x * blockDim.x + (size_t)warp * 32;
      if (warp_base < count) {
        const size_t rem = count - warp_base;
        const int words = (rem >= 32 ? 32 : (int)rem) * (int)NumCols;
        uint32_t* out = reinterpret_cast<uint32_t*>(&trace_matrix[warp_base * NumCols]);
        for (int k = lane; k < words; k += 32) out[k] = wrow[k];
      }
    } else {
      if (idx >= count) return;

      BitwiseValueColsU64<F>* cols =
        reinterpret_cast<BitwiseValueColsU64<F>*>(&trace_matrix[idx * num_cols]);
      event_to_row<F>(events[idx], *cols, byte_trace);
    }
  }

  template <class F>
  inline RustError generate_extra_record_and_main(
    const size_t num_cols,
    const size_t event_size,
    const size_t trace_size,
    const AluEventU64* events,
    F* trace,
    F* byte_trace,
    cudaStream_t stream)
  {
    try {
      CUDA_OK(cudaMemsetAsync(trace, 0, trace_size * sizeof(F), stream));
      if (event_size == 0) return RustError{cudaSuccess};

      const int block_size = 256;
      const int grid_size = (event_size + block_size - 1) / block_size;

      // Dispatch the WSTAGE fast path on the real compile-time struct width (27 on this
      // branch). Deriving it from sizeof(BitwiseValueColsU64<F>) keeps the dispatch in
      // sync with the kernel's internal guard (NumCols == sizeof(...)/sizeof(F)); other
      // widths fall through to the generic (NumCols=0) path.
      constexpr size_t kBitwiseU64Cols = sizeof(BitwiseValueColsU64<F>) / sizeof(F);
      if (num_cols == kBitwiseU64Cols) {
        // Block 128 = 4 warps x 32 rows x 27 cols x 4B = 13.5KB dynamic smem — under the
        // 48KB default, so no cudaFuncSetAttribute opt-in is needed.
        const int wstage_block = 128;
        const size_t wstage_shmem =
          (size_t)(wstage_block / 32) * 32 * kBitwiseU64Cols * sizeof(F);
        const int wstage_grid = (event_size + wstage_block - 1) / wstage_block;
        bitwise_events_to_trace_kernel<F, kBitwiseU64Cols>
          <<<wstage_grid, wstage_block, wstage_shmem, stream>>>(
            events, event_size, trace, num_cols, byte_trace);
      } else {
        bitwise_events_to_trace_kernel<F, 0><<<grid_size, block_size, 0, stream>>>(
          events, event_size, trace, num_cols, byte_trace);
      }
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  }
} // namespace pico_gpu::bitwise_u64