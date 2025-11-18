#ifndef POSEIDON2_KERNELS_CUH
#define POSEIDON2_KERNELS_CUH

#include "poseidon2/constants.cuh"

namespace poseidon2 {

  static __host__ __device__ __forceinline__ unsigned int d_next_pow_of_two(unsigned int v)
  {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
  }

  // state print function for debugging
  template <typename S, int T>
  __host__ __device__ void print_state(const char* id, S state[T])
  {
    printf("%s:\n", id);
    for (int i = 0; i < 16; ++i) {
      printf("s[%d] = %d\n", i, (uint32_t)state[i]);
    }
    printf("\n");
  }

  template <typename S>
  __host__ __device__ __forceinline__ S sbox_el(S element, const int alpha)
  {
    // TRICKY: function `sqr` changes the original element value
    S x1 = element;
    S x2 = element.sqr();
    S x3, x4;

    switch (alpha) {
    case 3:
      return x2 * x1;
    case 7:
      x3 = x2 * x1;
      x4 = x2.sqr();
      return x4 * x3;
    default:
      printf("unsupported sbox alpha: %d\n", alpha);
      assert(false);
    }

    return x1;
  }

  template <typename S, int T>
  __host__ __device__ __forceinline__ void sbox(S state[T], const int alpha)
  {
    for (int i = 0; i < T; i++) {
      state[i] = sbox_el(state[i], alpha);
    }
  }

  template <typename S, int T>
  __host__ __device__ __forceinline__ void add_rc(S state[T], size_t rc_offset, const S* rc)
  {
    for (int i = 0; i < T; i++) {
      state[i] = state[i] + rc[rc_offset + i];
    }
  }

  template <typename S>
  __host__ __device__ inline void mds_light_4x4(S s[4])
  {
    S t0 = s[0] + s[1];
    S t1 = s[2] + s[3];
    S t2 = s[1] + s[1] + t1;
    S t3 = s[3] + s[3] + t0;
    S t4 = t1 + t1 + t1 + t1 + t3;
    S t5 = t0 + t0 + t0 + t0 + t2;
    s[0] = t3 + t5;
    s[1] = t5;
    s[2] = t2 + t4;
    s[3] = t4;
  }

  // Multiply a 4-element vector x by:
  // [ 2 3 1 1 ]
  // [ 1 2 3 1 ]
  // [ 1 1 2 3 ]
  // [ 3 1 1 2 ]
  // https://github.com/Plonky3/Plonky3/blob/main/poseidon2/src/matrix.rs#L36
  template <typename S>
  __host__ __device__ inline void mds_light_plonky_4x4(S s[4])
  {
    S t01 = s[0] + s[1];
    S t23 = s[2] + s[3];
    S t0123 = t01 + t23;
    S t01123 = t0123 + s[1];
    S t01233 = t0123 + s[3];
    s[3] = t01233 + s[0] + s[0];
    s[1] = t01123 + s[2] + s[2];
    s[0] = t01123 + t01;
    s[2] = t01233 + t23;
  }

  template <typename S, int T>
  __host__ __device__ inline void mds_light(S state[T], MdsType mds)
  {
    S sum;
    switch (T) {
    case 2:
      // Matrix circ(2, 1)
      // [2, 1]
      // [1, 2]
      sum = state[0] + state[1];
      state[0] = state[0] + sum;
      state[1] = state[1] + sum;
      break;
    case 3:
      // Matrix circ(2, 1, 1)
      // [2, 1, 1]
      // [1, 2, 1]
      // [1, 1, 2]
      sum = state[0] + state[1] + state[2];
      state[0] = state[0] + sum;
      state[1] = state[1] + sum;
      state[2] = state[2] + sum;
      break;
    case 4:
    case 8:
    case 12:
    case 16:
    case 20:
    case 24:
      for (int i = 0; i < T; i += 4) {
        switch (mds) {
        case MdsType::DEFAULT_MDS:
          mds_light_4x4(&state[i]);
          break;
        case MdsType::PLONKY:
          mds_light_plonky_4x4(&state[i]);
        }
      }

      S sums[4] = {state[0], state[1], state[2], state[3]};
      for (int i = 4; i < T; i += 4) {
        sums[0] = sums[0] + state[i];
        sums[1] = sums[1] + state[i + 1];
        sums[2] = sums[2] + state[i + 2];
        sums[3] = sums[3] + state[i + 3];
      }
      for (int i = 0; i < T; i++) {
        state[i] = state[i] + sums[i % 4];
      }
      break;
    }
  }

#if defined(FEATURE_BABY_BEAR)
  template <typename S, int T>
  __host__ __device__ inline void internal_layer_mat_mul(S state[T], S sum)
  {
    // The diagonal matrix is defined by the vector:
    // V = [-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/2^8, 1/4, 1/8, 1/2^27, -1/2^8,
    // -1/16, -1/2^27]

    state[1] += sum;
    state[2] = state[2].dbl() + sum;
    state[3] = state[3].halve() + sum;
    state[4] = sum + state[4].dbl() + state[4];
    state[5] = sum + state[5].dbl().dbl();
    state[6] = sum - state[6].halve();
    state[7] = sum - (state[7].dbl() + state[7]);
    state[8] = sum - state[8].dbl().dbl();
    state[9] = state[9].mul_2exp_neg_n(8);
    state[9] += sum;
    state[10] = state[10].mul_2exp_neg_n(2);
    state[10] += sum;
    state[11] = state[11].mul_2exp_neg_n(3);
    state[11] += sum;
    state[12] = state[12].mul_2exp_neg_n(27);
    state[12] += sum;
    state[13] = state[13].mul_2exp_neg_n(8);
    state[13] = sum - state[13];
    state[14] = state[14].mul_2exp_neg_n(4);
    state[14] = sum - state[14];
    state[15] = state[15].mul_2exp_neg_n(27);
    state[15] = sum - state[15];
  }
#elif defined(FEATURE_KOALA_BEAR)
  template <typename S, int T>
  __host__ __device__ inline void internal_layer_mat_mul(S state[T], S sum)
  {
    // The diagonal matrix is defined by the vector:
    // V = [-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/2^8, 1/8, 1/2^24, -1/2^8, -1/8,
    // -1/16, -1/2^24]

    state[1] += sum;
    state[2] = state[2].dbl() + sum;
    state[3] = state[3].halve() + sum;
    state[4] = sum + state[4].dbl() + state[4];
    state[5] = sum + state[5].dbl().dbl();
    state[6] = sum - state[6].halve();
    state[7] = sum - (state[7].dbl() + state[7]);
    state[8] = sum - state[8].dbl().dbl();
    state[9] = state[9].mul_2exp_neg_n(8);
    state[9] += sum;
    state[10] = state[10].mul_2exp_neg_n(3);
    state[10] += sum;
    state[11] = state[11].mul_2exp_neg_n(24);
    state[11] += sum;
    state[12] = state[12].mul_2exp_neg_n(8);
    state[12] = sum - state[12];
    state[13] = state[13].mul_2exp_neg_n(3);
    state[13] = sum - state[13];
    state[14] = state[14].mul_2exp_neg_n(4);
    state[14] = sum - state[14];
    state[15] = state[15].mul_2exp_neg_n(24);
    state[15] = sum - state[15];
  }
#else
  #error "no FEATURE"
#endif

  template <typename S, int T>
  __host__ __device__ inline void internal_round(S state[T], size_t rc_offset, const Poseidon2Constants<S>& constants)
  {
    S element = state[0];
    element = element + constants.round_constants[rc_offset];
    element = sbox_el<S>(element, constants.alpha);

    switch (T) {
    case 4:
    case 8:
    case 12:
    case 16:
    case 20:
    case 24:
      S part_sum = 0;
      for (int i = 1; i < T; i++) {
        part_sum = part_sum + state[i];
      }

      S full_sum = part_sum + element;
      state[0] = part_sum - element;

      internal_layer_mat_mul<S, T>(state, full_sum);
    }
  }

  template <typename S, int T>
  __host__ __device__ inline void permute_state(S state[T], const Poseidon2Constants<S>& constants)
  {
    unsigned int rn;

    mds_light<S, T>(state, constants.mds_type);

    // external initial rounds
    size_t rc_offset = 0;
    for (rn = 0; rn < constants.external_rounds / 2; rn++) {
      add_rc<S, T>(state, rc_offset, constants.round_constants);
      sbox<S, T>(state, constants.alpha);
      mds_light<S, T>(state, constants.mds_type);
      rc_offset += T;
    }

    // internal rounds
    for (; rn < constants.external_rounds / 2 + constants.internal_rounds; rn++) {
      internal_round<S, T>(state, rc_offset, constants);
      rc_offset += T;
    }

    // External terminal rounds
    for (rn = constants.external_rounds / 2; rn < constants.external_rounds; rn++) {
      add_rc<S, T>(state, rc_offset, constants.round_constants);
      sbox<S, T>(state, constants.alpha);
      mds_light<S, T>(state, constants.mds_type);
      rc_offset += T;
    }
  }

  template <typename S, int T>
  __global__ void permutation_kernel(
    const S* states, S* states_out, unsigned int number_of_states, const Poseidon2Constants<S> constants)
  {
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= number_of_states) { return; }

    S state[T];
#pragma unroll
    for (int i = 0; i < T; i++) {
      state[i] = states[idx * T + i];
    }

    permute_state<S, T>(state, constants);

#pragma unroll
    for (int i = 0; i < T; i++) {
      states_out[idx * T + i] = state[i];
    }
  }

  template <typename S, int T>
  __global__ void hash_many_kernel(
    const S* input,
    S* output,
    uint64_t number_of_states,
    unsigned int input_len,
    unsigned int output_len,
    const Poseidon2Constants<S> constants)
  {
    uint64_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= number_of_states) { return; }

    S state[T] = {0};
#pragma unroll
    for (int i = 0; i < input_len; i++) {
      state[i] = input[idx * input_len + i];
    }

    permute_state<S, T>(state, constants);

#pragma unroll
    for (int i = 0; i < output_len; i++) {
      output[idx * output_len + i] = state[i];
    }
  }

  // T: width, R: rate
  template <typename S, int T, int R>
  __global__ void padding_free_sponge_hash_many_kernel(
    const S* input,
    S* output,
    uint64_t number_of_states,
    unsigned int input_len,
    unsigned int output_len,
    const Poseidon2Constants<S> constants)
  {
    uint64_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= number_of_states) { return; }

    int pos = 0;
    S state[T] = {0};
    while (true) {
      bool done = false;
      for (int i = 0; i < R; ++i) {
        if (pos < input_len) {
          state[i] = input[pos++];
        } else {
          done = true;

          if (i != 0) { permute_state<S, T>(state, constants); }

          break;
        }
      }

      if (done) { break; }

      permute_state<S, T>(state, constants);
    }

    print_state<S, T>("final", state);

#pragma unroll
    for (int i = 0; i < output_len; ++i) {
      output[idx * output_len + i] = state[i];
    }
  }

} // namespace poseidon2

#endif
