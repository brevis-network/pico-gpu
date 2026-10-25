// Copyright (c) 2025-2026 Brevis. All Rights Reserved.
// Licensed under the Business Source License 1.1 (BUSL-1.1).
// See LICENSE.md for details.
//
// ExtDotProductChip GPU trace generation.
// Extension-field dot product with a running accumulator over PACKED base-field
// values. The main event is fixed-size and copies 1:1 to a trace row. The
// preprocessed rows (each instruction spans a data-dependent number of rows) are
// expanded fully on the device from a two-buffer layout: per-instruction metadata
// plus flat mu/val address side buffers concatenated across instructions.

#pragma once

#include "common/rusterror.h"
#include "tracegen/common/types.cuh"
#include "tracegen/common/utils.cuh"

namespace pico_gpu::ext_dot_product {

  // Event / Cols structs live in types.cuh (namespace `pico_gpu`).

  // One thread per event: copy the fixed-size event into its trace row.
  template <class F>
  __global__ void
  ext_dot_product_events_to_trace_kernel(const ExtDotProductEvent<F>* events,
                                         size_t count,
                                         F* trace_matrix,
                                         size_t num_cols)
  {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    ExtDotProductCols<F>* cols =
        reinterpret_cast<ExtDotProductCols<F>*>(&trace_matrix[idx * num_cols]);
    for (int m = 0; m < EXT_DOT_PRODUCT_DATAPAR; ++m)
      for (int k = 0; k < EXTENSION_DEGREE; ++k)
        cols->mus[m]._0[k] = events[idx].mus[m]._0[k];
    for (int g = 0; g < EXT_DOT_PRODUCT_GROUPS; ++g)
      for (int k = 0; k < EXTENSION_DEGREE; ++k)
        cols->vals[g]._0[k] = events[idx].vals[g]._0[k];
    for (int k = 0; k < EXTENSION_DEGREE; ++k)
      cols->acc._0[k] = events[idx].acc._0[k];
  }

  // One thread per instruction: expand its n_rows preprocessed rows on the device, reading the
  // per-row mu/val element addresses from the flat side buffers.
  template <class F>
  __global__ void
  ext_dot_product_instrs_to_trace_kernel(const ExtDotProductPreprocMeta<F>* metas,
                                         size_t count,
                                         const F* mu_addrs_flat,
                                         const F* val_addrs_flat,
                                         F* trace_matrix,
                                         size_t num_cols)
  {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    const ExtDotProductPreprocMeta<F>& meta = metas[idx];
    for (uint32_t i = 0; i < meta.n_rows; ++i) {
      ExtDotProductPreprocessedCols<F>* cols = reinterpret_cast<ExtDotProductPreprocessedCols<F>*>(
          &trace_matrix[(meta.row_offset + i) * num_cols]);
      cols->is_real = F::one();
      cols->is_end = (i == meta.n_rows - 1) ? F::one() : F::zero();
      for (uint32_t j = 0; j < EXT_DOT_PRODUCT_DATAPAR; ++j)
        cols->mu_addrs[j] = mu_addrs_flat[meta.mu_offset + EXT_DOT_PRODUCT_DATAPAR * i + j];
      for (uint32_t g = 0; g < EXT_DOT_PRODUCT_GROUPS; ++g)
        cols->val_addrs[g] = val_addrs_flat[meta.val_offset + EXT_DOT_PRODUCT_GROUPS * i + g];
      cols->acc_addr = meta.acc_addr;
      cols->mult = (i == meta.n_rows - 1) ? meta.mult : F::zero();
    }
  }

  template <class F>
  inline RustError
  generate_main(const ExtDotProductEvent<F>* events,
                size_t num_events,
                F* trace,
                size_t num_cols,
                cudaStream_t stream)
  {
    try {
      if (num_events == 0) return RustError{cudaSuccess};
      const int block_size = 256;
      int grid_size = (num_events + block_size - 1) / block_size;
      ext_dot_product_events_to_trace_kernel<<<grid_size, block_size, 0, stream>>>(
          events, num_events, trace, num_cols);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) { return RustError{e.code()}; }
    return RustError{cudaSuccess};
  }

  // `mu_addrs_flat`/`val_addrs_flat` hold every instruction's mu/val addresses concatenated; each
  // meta carries its offsets, row count, and the device-computed `row_offset`.
  template <class F>
  inline RustError
  generate_preprocessed(const ExtDotProductPreprocMeta<F>* metas,
                        size_t num_instrs,
                        const F* mu_addrs_flat,
                        const F* val_addrs_flat,
                        F* trace,
                        size_t num_cols,
                        cudaStream_t stream)
  {
    try {
      if (num_instrs == 0) return RustError{cudaSuccess};
      const int block_size = 256;
      int grid_size = (num_instrs + block_size - 1) / block_size;
      ext_dot_product_instrs_to_trace_kernel<<<grid_size, block_size, 0, stream>>>(
          metas, num_instrs, mu_addrs_flat, val_addrs_flat, trace, num_cols);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) { return RustError{e.code()}; }
    return RustError{cudaSuccess};
  }

} // namespace pico_gpu::ext_dot_product
