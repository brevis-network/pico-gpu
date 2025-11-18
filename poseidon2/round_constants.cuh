#pragma once

#include <cuda_runtime.h>

// Direct include of the host round-constant definitions
#if defined(FEATURE_BABY_BEAR)
  #include "poseidon2/constants/babybear_poseidon2.h"
using namespace poseidon2_constants_babybear::t16;
#elif defined(FEATURE_KOALA_BEAR)
  #include "poseidon2/constants/koalabear_poseidon2.h"
using namespace poseidon2_constants_koalabear::t16;
#else
  #error "Must define FEATURE_BABY_BEAR or FEATURE_KOALA_BEAR"
#endif

namespace poseidon2_round_constants {

  static constexpr int EXTERNAL_ROUNDS = 8;
  static constexpr int WIDTH = 16;

  template <typename S>
  struct RoundConstants {
    int width;
    int alpha;
    int internal_rounds;
    int external_rounds;
    S* d_round_constants = nullptr;
  };

  template <typename S>
  cudaError_t init_round_constants(RoundConstants<S>& rc, cudaStream_t stream)
  {
    rc.width = WIDTH;
    rc.alpha = alpha;
    rc.internal_rounds = internal_rounds;
    rc.external_rounds = EXTERNAL_ROUNDS;

    size_t len = static_cast<size_t>(rc.width) * (rc.internal_rounds + rc.external_rounds);
    const S* h_constants = reinterpret_cast<const S*>(round_constants);

    cudaError_t err = cudaMallocAsync(&rc.d_round_constants, len * sizeof(S), stream);
    if (err != cudaSuccess) return err;

    err = cudaMemcpyAsync(rc.d_round_constants, h_constants, len * sizeof(S), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) return err;

    return cudaStreamSynchronize(stream);
  }

  template <typename S>
  cudaError_t free_round_constants(RoundConstants<S>& rc, cudaStream_t stream)
  {
    if (!rc.d_round_constants) return cudaSuccess;
    cudaError_t err = cudaFreeAsync(rc.d_round_constants, stream);
    if (err != cudaSuccess) return err;
    rc.d_round_constants = nullptr;
    return cudaSuccess;
  }

} // namespace poseidon2_round_constants
