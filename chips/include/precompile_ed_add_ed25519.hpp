#pragma once

#include "types.hpp"
#include "field_params.hpp"
#include "./edwards_op.hpp"

namespace pico_gpu::ed_add::ed25519 {
  using namespace params_ed25519;
  using namespace edwards_op;
  using EllipticCurveAddEvent_t = EllipticCurveAddEvent<NumWords>;

  template <class F>
  __global__ void edwards_add_kernel(
    const EllipticCurveAddEvent_t* events,
    const size_t count,
    F* trace_matrix,
    const size_t num_cols,
    const size_t num_rows,
    F* byte_trace)
  {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;

    // Get pointer to current row in trace matrix.
    EdAddAssignCols<F, NumLimbs, NumWitnesses>* col =
      reinterpret_cast<EdAddAssignCols<F, NumLimbs, NumWitnesses>*>(&trace_matrix[idx * num_cols]);

    // Process event and populate row.
    ed_add_event_to_row<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
      events, *col, idx, count, MODULUS, MOD_MINUS_TWO, EDWARDS_D, byte_trace);
  }

  template <class F>
  inline RustError generate_extra_record_and_main(
    const EllipticCurveAddEvent_t* events,
    const size_t num_events,
    F* trace,
    const size_t num_rows,
    const size_t num_cols,
    F* byte_trace,
    cudaStream_t stream)
  {
    try {
      if (num_events == 0) return RustError{cudaSuccess};

      // Initialize trace matrix with zeros.
      CUDA_OK(cudaMemsetAsync(trace, 0, num_rows * num_cols * sizeof(F), stream));

      // Launch kernel.
      const uint32_t block_size = 256;
      const uint32_t grid_size = (num_rows + block_size - 1) / block_size;

      edwards_add_kernel<F>
        <<<grid_size, block_size, 0, stream>>>(events, num_events, trace, num_cols, num_rows, byte_trace);

      CUDA_OK(cudaGetLastError());
      CUDA_OK(cudaStreamSynchronize(stream));
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  } // generate_extra_record_and_main
} // namespace pico_gpu::ed_add::ed25519