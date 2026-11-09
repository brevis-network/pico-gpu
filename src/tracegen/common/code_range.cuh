// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0

#pragma once

#include "common/rusterror.h"
#include "tracegen/common/prelude.cuh"
#include "tracegen/common/types.cuh"
#include "tracegen/riscv/byte.cuh"

// A write's check that it misses the program's code, the device half of Rust
// `chips/gadgets/code_range.rs`. `populate` mirrors `CodeRangeCols::populate` value for value, so a
// GPU trace equals the CPU one: either the write ends at or below the code's start, or it starts at
// or above its end, by a distance whose two u16 limbs are range-checked.
namespace pico_gpu::code_range {
  using namespace pico_gpu::byte;

  // The columns of a write of `width` bytes at `addr`, the code being [code_start, code_end), and
  // the u16 range checks of the distance's limbs. A write that meets the code gets limbs no
  // assignment of which satisfies the AIR, as on the host.
  template <class F>
  __device__ inline void populate(
    CodeRangeCols<F>& cols,
    const uint64_t addr,
    const uint64_t width,
    const uint64_t code_start,
    const uint64_t code_end,
    F* byte_trace)
  {
    const uint16_t start0 = static_cast<uint16_t>(code_start);
    const uint16_t start1 = static_cast<uint16_t>(code_start >> 16);
    const uint16_t end0 = static_cast<uint16_t>(code_end);
    const uint16_t end1 = static_cast<uint16_t>(code_end >> 16);
    cols.code_start[0] = F::from_canonical_u16(start0);
    cols.code_start[1] = F::from_canonical_u16(start1);
    cols.code_end[0] = F::from_canonical_u16(end0);
    cols.code_end[1] = F::from_canonical_u16(end1);
    const bool below = addr + width <= code_start;
    // Wraps for a write into the code, as the host's `wrapping_sub` does.
    const uint64_t distance = below ? code_start - (addr + width) : addr - code_end;
    const uint16_t low = static_cast<uint16_t>(distance);
    const uint16_t high = static_cast<uint16_t>(distance >> 16);
    const int64_t low_operand = below
      ? static_cast<int64_t>(start0) - static_cast<int64_t>(addr & 0xffff) - static_cast<int64_t>(width)
      : static_cast<int64_t>(addr & 0xffff) - static_cast<int64_t>(end0);
    // What makes the low limb of the distance the low operand plus whole 2^16s: an arithmetic
    // shift, as the host's `>>` on an i64, clamped to [0, 2].
    int64_t borrow = (static_cast<int64_t>(low) - low_operand) >> 16;
    borrow = borrow < 0 ? 0 : (borrow > 2 ? 2 : borrow);
    cols.below = F::from_bool(below);
    cols.borrow = F::from_canonical_u32(static_cast<uint32_t>(borrow));
    cols.distance[0] = F::from_canonical_u16(low);
    cols.distance[1] = F::from_canonical_u16(high);
    add_u16_range_check<F>(byte_trace, low);
    add_u16_range_check<F>(byte_trace, high);
  }

  // The width of a store's write, 0 for any other instruction.
  __device__ inline uint64_t store_width(const Opcode opcode)
  {
    switch (opcode) {
      case Opcode::SB: return 1;
      case Opcode::SH: return 2;
      case Opcode::SW: return 4;
      case Opcode::SD: return 8;
      default: return 0;
    }
  }

  // One thread per real row: the `CodeRangeCols` at column `offset` of row `idx`, for the write of
  // `width` bytes from `writes[idx]`. A precompile's kernel leaves these columns, the last of its
  // row, zero; this fills them, and its padding rows stay zero.
  template <class F>
  __global__ void fill_kernel(
    F* __restrict__ trace,
    const size_t num_cols,
    const size_t offset,
    const uint64_t* __restrict__ writes,
    const size_t rows,
    const uint64_t width,
    const uint64_t code_start,
    const uint64_t code_end,
    F* __restrict__ byte_trace)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows) return;
    CodeRangeCols<F>& cols = *reinterpret_cast<CodeRangeCols<F>*>(&trace[idx * num_cols + offset]);
    populate<F>(cols, writes[idx], width, code_start, code_end, byte_trace);
  }

  template <class F>
  inline RustError fill(
    F* trace,
    const size_t num_cols,
    const size_t offset,
    const uint64_t* writes,
    const size_t rows,
    const uint64_t width,
    const uint64_t code_start,
    const uint64_t code_end,
    F* byte_trace,
    cudaStream_t stream)
  {
    try {
      if (rows == 0) return RustError{cudaSuccess};
      if (offset + sizeof(CodeRangeCols<F>) / sizeof(F) > num_cols) return RustError{cudaErrorInvalidValue};
      const int block = 256;
      const int grid = static_cast<int>((rows + block - 1) / block);
      fill_kernel<F><<<grid, block, 0, stream>>>(
        trace, num_cols, offset, writes, rows, width, code_start, code_end, byte_trace);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }
    return RustError{cudaSuccess};
  }
} // namespace pico_gpu::code_range
