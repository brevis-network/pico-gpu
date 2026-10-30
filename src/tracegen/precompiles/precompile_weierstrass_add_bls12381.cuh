// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0

#pragma once

#include "tracegen/common/types.cuh"
#include "tracegen/common/field_params.cuh"
#include "tracegen/precompiles/curve_op.cuh"

namespace pico_gpu::weierstrass_add::bls12381 {
  using namespace params_bls12381;
  using namespace curve_op;
  using EllipticCurveAddEvent_t = WeierstrassAddEvent<NumWords>;

  using Scratch = fp_op::FpScratch<NumLimbs, NumWitnesses>;

  /// Kernel to process Weierstrass curve addition events.
  template <class F>
  __global__ void weierstrass_add_kernel(
    const EllipticCurveAddEvent_t* events,
    const size_t count,
    F* trace_matrix,
    const size_t num_cols,
    const size_t num_rows,
    F* byte_trace,
    Scratch* workspace)
  {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;

    WeierstrassAddAssignCols<F, NumWords, NumLimbs, NumWitnesses>* col =
      reinterpret_cast<WeierstrassAddAssignCols<F, NumWords, NumLimbs, NumWitnesses>*>(&trace_matrix[idx * num_cols]);

    add_event_to_row<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset, true>(
      events, *col, idx, count, MODULUS, MOD_MINUS_TWO,
      mont_params::BLS12381_RR, mont_params::BLS12381_M0, byte_trace, &workspace[idx]);
  }

  /// Generate trace from Weierstrass curve addition events.
  template <class F>
  inline RustError generate_extra_record_and_main(
    const size_t num_cols,
    const size_t event_size,
    const size_t trace_size,
    const EllipticCurveAddEvent_t* events,
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

      Scratch* d_scratch = nullptr;
      CUDA_OK(cudaMalloc(&d_scratch, num_rows * sizeof(Scratch)));

      weierstrass_add_kernel<F>
        <<<grid_size, block_size, 0, stream>>>(events, event_size, trace, num_cols, num_rows, byte_trace, d_scratch);
      CUDA_OK(cudaGetLastError());

      CUDA_OK(cudaStreamSynchronize(stream));
      CUDA_OK(cudaFree(d_scratch));
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  }
} // namespace pico_gpu::weierstrass_add::bls12381