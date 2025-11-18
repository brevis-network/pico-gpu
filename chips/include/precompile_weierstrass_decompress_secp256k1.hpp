#pragma once

#include "types.hpp"
#include "field_params.hpp"
#include "./curve_op.hpp"

namespace pico_gpu::weierstrass_decompress::secp256k1 {
  using namespace params_secp256k1;
  using namespace curve_op;
  using EllipticCurveDecompressEvent_t = EllipticCurveDecompressEventFFI<NumWords, NumLimbs>;

  /// Kernel to process Weierstrass curve decompression events.
  template <class F>
  __global__ void weierstrass_decompress_kernel(
    const EllipticCurveDecompressEvent_t* events,
    const size_t count,
    F* trace_matrix,
    const size_t num_cols,
    const size_t num_rows,
    F* byte_trace)
  {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;

    WeierstrassDecompressCols<F, NumWords, NumLimbs, NumWitnesses>* col =
      reinterpret_cast<WeierstrassDecompressCols<F, NumWords, NumLimbs, NumWitnesses>*>(&trace_matrix[idx * num_cols]);

    decompress_event_to_row<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
      events, *col, idx, count, GEN_X, SQRT_EXP, WEIERSTRASS_B, MODULUS, MOD_MINUS_TWO, byte_trace);
  }

  /// Kernel to process lexicographic choice for decompression.
  template <class F>
  __global__ void weierstrass_lexicographic_kernel(
    const EllipticCurveDecompressEvent_t* events,
    const size_t count,
    F* trace_matrix,
    const size_t num_cols,
    const size_t num_decompress_cols,
    const size_t num_rows,
    F* byte_trace)
  {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;

    WeierstrassDecompressCols<F, NumWords, NumLimbs, NumWitnesses>* wcol =
      reinterpret_cast<WeierstrassDecompressCols<F, NumWords, NumLimbs, NumWitnesses>*>(&trace_matrix[idx * num_cols]);

    LexicographicChoiceCols<F, NumWords, NumLimbs, NumWitnesses>* col =
      reinterpret_cast<LexicographicChoiceCols<F, NumWords, NumLimbs, NumWitnesses>*>(
        &trace_matrix[idx * num_cols + num_decompress_cols]);

    decompress_lexicographic<F, NumWords, NumLimbs, NumWitnesses, NumBitsPerLimb, WitnessOffset>(
      events, *wcol, *col, idx, count, MODULUS, MOD_MINUS_TWO, byte_trace);
  }

  /// Generate trace from Weierstrass curve decompression events.
  template <class F>
  inline RustError generate_extra_record_and_main(
    const EllipticCurveDecompressEvent_t* events,
    const size_t num_events,
    F* trace,
    const size_t num_rows,
    const size_t num_cols,
    const size_t num_lexi_cols,
    const bool lexi,
    F* byte_trace,
    cudaStream_t stream)
  {
    try {
      if (num_events == 0) return RustError{cudaSuccess};

      const size_t real_cols = lexi ? num_cols + num_lexi_cols : num_cols;

      // Initialize trace matrix with zeros.
      CUDA_OK(cudaMemsetAsync(trace, 0, (num_rows * real_cols) * sizeof(F), stream));

      // Launch kernels.
      const uint32_t block_size = 256;
      const uint32_t grid_size = (num_rows + block_size - 1) / block_size;

      weierstrass_decompress_kernel<F>
        <<<grid_size, block_size, 0, stream>>>(events, num_events, trace, real_cols, num_rows, byte_trace);
      CUDA_OK(cudaGetLastError());

      if (lexi) {
        weierstrass_lexicographic_kernel<F>
          <<<grid_size, block_size, 0, stream>>>(events, num_events, trace, real_cols, num_cols, num_rows, byte_trace);
        CUDA_OK(cudaGetLastError());
      }

      CUDA_OK(cudaStreamSynchronize(stream));
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  }
} // namespace pico_gpu::weierstrass_decompress::secp256k1