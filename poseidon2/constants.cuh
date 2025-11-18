#ifndef POSEIDON2_CONSTANTS_CUH
#define POSEIDON2_CONSTANTS_CUH

#if defined(FEATURE_BABY_BEAR)
  #include "poseidon2/constants/babybear_poseidon2.h"
using namespace poseidon2_constants_babybear;
#elif defined(FEATURE_KOALA_BEAR)
  #include "poseidon2/constants/koalabear_poseidon2.h"
using namespace poseidon2_constants_koalabear;
#else
  #error "no FEATURE"
#endif

namespace poseidon2 {
  const int EXTERNAL_ROUNDS_DEFAULT = 8;

  enum MdsType { DEFAULT_MDS, PLONKY };

  template <typename S>
  struct Poseidon2Constants {
    int width;
    int alpha;
    int internal_rounds;
    int external_rounds;
    S* round_constants = nullptr;
    MdsType mds_type;
  };

  template <typename S>
  cudaError_t create_poseidon2_constants(
    int width,
    int alpha,
    int internal_rounds,
    int external_rounds,
    const S* round_constants,
    MdsType mds_type,
    cudaStream_t stream,
    Poseidon2Constants<S>* poseidon_constants)
  {
    if (!(alpha == 3 || alpha == 5 || alpha == 7 || alpha == 11)) { throw "invalid alpha value"; }
    if (external_rounds % 2) { throw "invalid external rounds"; }

    int round_constants_len = width * (external_rounds + internal_rounds);

    // malloc memory for copying round constants
    S* d_round_constants;
    cudaError_t err = cudaMallocAsync(&d_round_constants, sizeof(S) * round_constants_len, stream);
    if (err != cudaSuccess) { return err; }

    // copy round constants
    err = cudaMemcpyAsync(
      d_round_constants, round_constants, sizeof(S) * round_constants_len, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) { return err; }

    // make sure all the constants have been copied
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) { return err; }

    *poseidon_constants = {width, alpha, internal_rounds, external_rounds, d_round_constants, mds_type};

    return cudaGetLastError();
  }

  template <typename S>
  cudaError_t
  init_poseidon2_constants(int width, MdsType mds_type, cudaStream_t stream, Poseidon2Constants<S>* poseidon2_constants)
  {
#define P2_CONSTANTS_DEF(width)                                                                                        \
case width:                                                                                                            \
  internal_rounds = t##width::internal_rounds;                                                                         \
  round_constants = t##width::round_constants;                                                                         \
  alpha = t##width::alpha;                                                                                             \
  break;

    int alpha;
    int external_rounds = EXTERNAL_ROUNDS_DEFAULT;
    int internal_rounds;
    uint32_t* round_constants;
    switch (width) {
      P2_CONSTANTS_DEF(16)
    default:
      throw "init_poseidon2_constants: width must be 16";
    }

    S* h_round_constants = reinterpret_cast<S*>(round_constants);

    return create_poseidon2_constants(
      width, alpha, internal_rounds, external_rounds, h_round_constants, mds_type, stream, poseidon2_constants);
  }

  template <typename S>
  cudaError_t release_poseidon2_constants(Poseidon2Constants<S>* constants, cudaStream_t stream)
  {
    cudaError_t err = cudaFreeAsync(constants->round_constants, stream);
    if (err != cudaSuccess) { return err; }

    constants->alpha = 0;
    constants->width = 0;
    constants->external_rounds = 0;
    constants->internal_rounds = 0;
    constants->round_constants = nullptr;
  }

} // namespace poseidon2

#endif