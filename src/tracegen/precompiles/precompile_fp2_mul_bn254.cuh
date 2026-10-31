// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0

#pragma once

#include "common/exception.cuh"
#include "common/rusterror.h"
#include "tracegen/common/types.cuh"
#include "tracegen/common/field_params.cuh"
#include "tracegen/precompiles/fp2_op.cuh"
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <ostream>

namespace pico_gpu::precompile_fp2_mul_bn254 {
  using namespace ::pico_gpu::fp2_op;
  using namespace ::pico_gpu::params_bn254;

  // NumFieldDwords = dwords per field element (= NumLimbs/8)
  static constexpr int NumFieldDwords = NumLimbs / 8;
  // NumCurveDwords = dwords per curve point (= 2 field elements)
  static constexpr int NumCurveDwords = NumFieldDwords * 2;

  using Fp2MulEvent_t = Fp2MulEvent<NumCurveDwords>;

  template <class F>
  __global__ void fp2_mul_kernel(
    const Fp2MulEvent_t* events,
    const size_t count,
    F* trace_matrix,
    F* byte_trace,
    const size_t num_cols,
    const size_t num_rows)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;

    Fp2MulCols<F, NumCurveDwords, NumLimbs, NumWitnesses>* col =
      reinterpret_cast<Fp2MulCols<F, NumCurveDwords, NumLimbs, NumWitnesses>*>(&trace_matrix[idx * num_cols]);

    mul_event_to_row<F, NumCurveDwords, NumFieldDwords, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
      events, *col, idx, count, MODULUS, MOD_MINUS_TWO, byte_trace);
  }

  template <class F>
  inline RustError generate_extra_record_and_main(
    const size_t num_cols,
    const size_t event_size,
    const size_t trace_size,
    const Fp2MulEvent_t* events,
    F* trace,
    F* byte_trace,
    cudaStream_t stream)
  {
    try {
      CUDA_OK(cudaMemsetAsync(trace, 0, trace_size * sizeof(F), stream));
      if (event_size == 0) return RustError{cudaSuccess};

      const uint32_t block_size = 256;
      const uint32_t num_rows = trace_size / num_cols;
      const uint32_t grid_size = (num_rows + block_size - 1) / block_size;

      fp2_mul_kernel<F>
        <<<grid_size, block_size, 0, stream>>>(events, event_size, trace, byte_trace, num_cols, num_rows);
      CUDA_OK(cudaGetLastError());

      CUDA_OK(cudaStreamSynchronize(stream));
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  } // generate_extra_record_and_main
} // namespace pico_gpu::precompile_fp2_mul_bn254