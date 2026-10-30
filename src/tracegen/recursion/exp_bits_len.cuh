// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// ExpBitsLenChip GPU trace generation (KoalaBear only).
// Each request is a fixed 32-row block. main: one thread per (padded) row recomputes the squaring
// ladder base^(2^step), the reverse-product `result`, and the KoalaBear canonicality flags in closed
// form; padding rows fold into the same launch (result = result_multiplier = 1). preproc: one thread
// per real row, two-buffer (per-instruction metadata + flat per-row bit-output (addr,mult) pairs).

#pragma once

#include "common/rusterror.h"
#include "tracegen/common/types.cuh"
#include "tracegen/common/utils.cuh"

namespace pico_gpu::exp_bits_len {

  // Event / Cols / Meta structs and EXP_BITS_LEN_ROWS / EXP_BITS_LEN_LOW_BITS live in types.cuh.

  /// One thread per row over the whole padded trace. `bit_src` in the event is the canonical u32.
  template <class F>
  __global__ void
  exp_bits_len_events_to_trace_kernel(const ExpBitsLenFlatEvent<F>* events,
                                      size_t num_events,
                                      F* trace_matrix,
                                      size_t num_cols,
                                      size_t num_rows)
  {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;
    ExpBitsLenCols<F>* cols =
        reinterpret_cast<ExpBitsLenCols<F>*>(&trace_matrix[idx * num_cols]);
    uint32_t event_idx = idx / EXP_BITS_LEN_ROWS;
    uint32_t step = idx % EXP_BITS_LEN_ROWS;

    if (event_idx >= num_events) {
      // Padding row: result = result_multiplier = 1, everything else 0 (from memset).
      cols->result = F::one();
      cols->result_multiplier = F::one();
      return;
    }

    const ExpBitsLenFlatEvent<F>& ev = events[event_idx];
    uint32_t decomp_src = ev.bit_src;
    uint32_t n = ev.num_bits;

    // bases[step] = base^(2^step): square `step` times.
    F base_step = ev.base;
    for (uint32_t k = 0; k < step; ++k) base_step = base_step * base_step;

    // result[step] = product of bases[p] for p in [step, ROWS-1) with p < n and bit p set.
    F result = F::one();
    {
      F cur = base_step; // bases[step]
      for (uint32_t p = step; p + 1 < EXP_BITS_LEN_ROWS; ++p) {
        if (p < n && ((decomp_src >> p) & 1u) == 1u) result = result * cur;
        cur = cur * cur;
      }
    }

    uint32_t shifted = decomp_src >> step;
    uint32_t num_bits = (step < n) ? (n - step) : 0u;
    uint32_t low_bits_left =
        (step < EXP_BITS_LEN_LOW_BITS) ? (uint32_t)(EXP_BITS_LEN_LOW_BITS - step) : 0u;
    uint32_t bit = shifted & 1u;

    cols->bit_idx = F::from_canonical_u32(step);
    cols->base = base_step;
    cols->bit_src = F::from_canonical_u32(shifted);
    cols->num_bits = F::from_canonical_u32(num_bits);
    cols->apply_bit = (num_bits != 0u) ? F::one() : F::zero();
    cols->low_bits_left = F::from_canonical_u32(low_bits_left);
    cols->in_low_region = (low_bits_left != 0u) ? F::one() : F::zero();
    cols->result = result;
    cols->result_multiplier = (num_bits != 0u && bit == 1u) ? base_step : F::one();
    cols->bit_src_mod_2 = (bit == 1u) ? F::one() : F::zero();

    // Running flags recomputed per row as closed forms:
    // low_bits_are_zero[step] = (decomp_src & ((1<<min(step,24))-1)) == 0.
    uint32_t low_mask_bits = (step < EXP_BITS_LEN_LOW_BITS) ? step : (uint32_t)EXP_BITS_LEN_LOW_BITS;
    uint32_t low_mask = (1u << low_mask_bits) - 1u;
    cols->low_bits_are_zero = ((decomp_src & low_mask) == 0u) ? F::one() : F::zero();

    // high_bits_all_one[step] = (step>=24) && ((decomp_src>>24) & ((1<<(step-24))-1)) == that mask.
    if (step >= EXP_BITS_LEN_LOW_BITS) {
      uint32_t hb = step - (uint32_t)EXP_BITS_LEN_LOW_BITS;
      uint32_t hmask = (1u << hb) - 1u;
      uint32_t high = (decomp_src >> EXP_BITS_LEN_LOW_BITS) & hmask;
      cols->high_bits_all_one = (high == hmask) ? F::one() : F::zero();
    } else {
      cols->high_bits_all_one = F::zero();
    }
  }

  /// One thread per real row (num_instrs * ROWS). Padding rows stay zero (memset).
  template <class F>
  __global__ void
  exp_bits_len_instrs_to_trace_kernel(const ExpBitsLenPreprocMeta<F>* metas,
                                      size_t num_instrs,
                                      const RecursionMemoryAccessCols<F>* bit_outs_flat,
                                      F* trace_matrix,
                                      size_t num_cols,
                                      size_t num_real_rows)
  {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_real_rows) return;
    uint32_t instr = idx / EXP_BITS_LEN_ROWS;
    uint32_t i = idx % EXP_BITS_LEN_ROWS;
    const ExpBitsLenPreprocMeta<F>& meta = metas[instr];
    ExpBitsLenPreprocessedCols<F>* cols =
        reinterpret_cast<ExpBitsLenPreprocessedCols<F>*>(&trace_matrix[idx * num_cols]);
    bool is_first = (i == 0);
    bool is_last = (i == EXP_BITS_LEN_ROWS - 1);
    cols->is_first = is_first ? F::one() : F::zero();
    cols->is_last = is_last ? F::one() : F::zero();
    cols->is_real = F::one();
    // Input read on the first row only (negative mult = read).
    cols->bit_src_in.addr = meta.bit_src_in_addr;
    cols->bit_src_in.mult = is_first ? (F::zero() - F::one()) : F::zero();
    // Result write on the first row only.
    cols->result_out.addr = meta.result_out_addr;
    cols->result_out.mult = is_first ? meta.result_mult : F::zero();
    // Per-row bit output (zero addr/mult past the requested export width).
    if (i < meta.bit_outs_count) {
      cols->bit_out = bit_outs_flat[meta.bit_outs_offset + i];
    } else {
      cols->bit_out.addr = F::zero();
      cols->bit_out.mult = F::zero();
    }
    // Per-request constants injected on the first row.
    cols->base_const = is_first ? meta.base : F::zero();
    cols->num_bits_const = is_first ? F::from_canonical_u32(meta.num_bits) : F::zero();
  }

  /// `num_rows` is the padded row count; the kernel fills both real and padding rows.
  template <class F>
  inline RustError
  generate_main(const ExpBitsLenFlatEvent<F>* events,
                size_t num_events,
                F* trace,
                size_t num_cols,
                size_t num_rows,
                cudaStream_t stream)
  {
    try {
      if (num_rows == 0) return RustError{cudaSuccess};
      const int block_size = 256;
      int grid_size = (num_rows + block_size - 1) / block_size;
      exp_bits_len_events_to_trace_kernel<<<grid_size, block_size, 0, stream>>>(
          events, num_events, trace, num_cols, num_rows);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) { return RustError{e.code()}; }
    return RustError{cudaSuccess};
  }

  /// `bit_outs_flat` holds every instruction's per-row bit-output (addr,mult) pairs concatenated;
  /// `num_real_rows` = num_instrs * ROWS (padding rows are left zeroed by the caller's memset).
  template <class F>
  inline RustError
  generate_preprocessed(const ExpBitsLenPreprocMeta<F>* metas,
                        size_t num_instrs,
                        const RecursionMemoryAccessCols<F>* bit_outs_flat,
                        F* trace,
                        size_t num_cols,
                        size_t num_real_rows,
                        cudaStream_t stream)
  {
    try {
      if (num_real_rows == 0) return RustError{cudaSuccess};
      const int block_size = 256;
      int grid_size = (num_real_rows + block_size - 1) / block_size;
      exp_bits_len_instrs_to_trace_kernel<<<grid_size, block_size, 0, stream>>>(
          metas, num_instrs, bit_outs_flat, trace, num_cols, num_real_rows);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) { return RustError{e.code()}; }
    return RustError{cudaSuccess};
  }

} // namespace pico_gpu::exp_bits_len
