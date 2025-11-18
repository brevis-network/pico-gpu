#pragma once

#include "types.hpp"
#include "field_params.hpp"
#include "./curve_op.hpp"

namespace pico_gpu::weierstrass_add::secp256k1 {
  using namespace params_secp256k1;
  using namespace curve_op;
  using EllipticCurveAddEvent_t = EllipticCurveAddEvent<NumWords>;

  /// Kernel to process Weierstrass curve addition events.
  template <class F>
  __global__ void weierstrass_add_kernel(
    const EllipticCurveAddEvent_t* events,
    const size_t count,
    F* trace_matrix,
    const size_t num_cols,
    const size_t num_rows,
    F* byte_trace)
  {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;

    WeierstrassAddAssignCols<F, NumWords, NumLimbs, NumWitnesses>* col =
      reinterpret_cast<WeierstrassAddAssignCols<F, NumWords, NumLimbs, NumWitnesses>*>(&trace_matrix[idx * num_cols]);

    add_event_to_row<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
      events, *col, idx, count, MODULUS, MOD_MINUS_TWO, byte_trace);
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
      if (event_size == 0) return RustError{cudaSuccess};

      // Initialize trace matrix with zeros.
      CUDA_OK(cudaMemsetAsync(trace, 0, trace_size * sizeof(F), stream));

      // Launch kernel.
      const uint32_t block_size = 256;
      const uint32_t num_rows = trace_size / num_cols;
      const uint32_t grid_size = (num_rows + block_size - 1) / block_size;

      weierstrass_add_kernel<F>
        <<<grid_size, block_size, 0, stream>>>(events, event_size, trace, num_cols, num_rows, byte_trace);
      CUDA_OK(cudaGetLastError());

      CUDA_OK(cudaStreamSynchronize(stream));
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  }
} // namespace pico_gpu::weierstrass_add::secp256k1