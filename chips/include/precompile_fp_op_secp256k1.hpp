#pragma once

#include "ff/ff_config.hpp"
#include "util/rusterror.h"

#include "types.hpp"
#include "field_params.hpp"
#include "fp_op.hpp"
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <ostream>

namespace pico_gpu::precompile_fp_op_secp256k1 {
  using namespace ::pico_gpu::fp_op;
  using namespace ::pico_gpu::params_secp256k1;

  using FpEvent_t = FpEvent<NumWords>;
  using FieldOpCols_t = FieldOpCols<field_t, NumLimbs, NumWitnesses>;
  using FpOpCols_t = FpOpCols<field_t, NumWords, NumLimbs, NumWitnesses>;

  __global__ void events_to_rows_kernel(
    const FpEvent_t* __restrict__ events,
    field_t* __restrict__ trace_matrix,
    field_t* byte_trace,
    const size_t events_count,
    const size_t num_cols)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= events_count) return;

    FpOpCols_t* cols = reinterpret_cast<FpOpCols_t*>(&trace_matrix[idx * num_cols]);
    event_to_row<field_t, NumWords, NumLimbs, NumWitnesses, BitsPerLimb, WitnessOffset>(
      events[idx], *cols, MODULUS, MOD_MINUS_TWO, byte_trace);
  }

  __global__ void pad_dummy_rows_kernel(
    field_t* __restrict__ trace_matrix,
    const size_t pad_count,
    const size_t num_cols,
    const field_t* __restrict__ dummy_row)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pad_count) return;

    field_t* row = &trace_matrix[idx * num_cols];
    for (size_t i = 0; i < num_cols; i++) {
      row[i] = dummy_row[i];
    }
  }

  inline RustError generate_extra_record_and_main(
    const size_t num_cols,
    const size_t event_size,
    const size_t trace_size,
    const FpEvent_t* events,
    field_t* trace,
    const field_t* dummy_row,
    field_t* byte_trace,
    cudaStream_t stream)
  {
    try {
      if (event_size == 0) return RustError{cudaSuccess};

      // Initialize trace matrix with zeros.
      CUDA_OK(cudaMemsetAsync(trace, 0, trace_size * sizeof(field_t), stream));

      // Pad dummy rows if needed.
      const size_t num_total_rows = trace_size / num_cols;
      if (num_total_rows > event_size) {
        const int block_size = 256;
        const size_t pad_count = num_total_rows - event_size;
        const int grid_size = int((pad_count + block_size - 1) / block_size);
        const size_t idx_dummy_start = event_size * num_cols;

        pad_dummy_rows_kernel<<<grid_size, block_size, 0, stream>>>(
          &trace[idx_dummy_start], pad_count, num_cols, dummy_row);
        CUDA_OK(cudaGetLastError());
      }

      // Launch main kernel.
      const int block_size = 256;
      const int grid_size = (event_size + block_size - 1) / block_size;

      events_to_rows_kernel<<<grid_size, block_size, 0, stream>>>(events, trace, byte_trace, event_size, num_cols);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  }
} // namespace pico_gpu::precompile_fp_op_secp256k1