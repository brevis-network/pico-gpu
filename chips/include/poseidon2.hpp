#pragma once

#include "types.hpp"
#include <poseidon2/round_constants.cuh>
#include <util/rusterror.h>
#include "ff/ff_config.hpp"

using namespace poseidon2_round_constants;

namespace pico_gpu::poseidon2 {
  using FullRoundT = FullRound<field_t>;
  using PartialRoundT = PartialRound<field_t>;
  using Poseidon2ValueColsT = Poseidon2ValueCols<field_t, half_external_rounds, internal_rounds>;

  __host__ __device__ __forceinline__ void mds_light_4x4(field_t s[4])
  {
    field_t t01 = s[0] + s[1];
    field_t t23 = s[2] + s[3];
    field_t t0123 = t01 + t23;
    field_t t01123 = t0123 + s[1];
    field_t t01233 = t0123 + s[3];
    s[3] = t01233 + s[0] + s[0];
    s[1] = t01123 + s[2] + s[2];
    s[0] = t01123 + t01;
    s[2] = t01233 + t23;
  }

  __host__ __device__ inline void mds_light_permutation(field_t s[WIDTH])
  {
    for (int i = 0; i < WIDTH; i += 4) {
      mds_light_4x4(&s[i]);
    }

    field_t sums[4] = {s[0], s[1], s[2], s[3]};

    for (int i = 4; i < WIDTH; i += 4) {
      sums[0] = sums[0] + s[i];
      sums[1] = sums[1] + s[i + 1];
      sums[2] = sums[2] + s[i + 2];
      sums[3] = sums[3] + s[i + 3];
    }

    for (int i = 0; i < WIDTH; i++) {
      s[i] = s[i] + sums[i & 3];
    }
  }

  __host__ __device__ inline void
  add_round_constants(field_t state[WIDTH], size_t rc_offset, const field_t* __restrict__ round_constants)
  {
    for (int i = 0; i < WIDTH; i++) {
      state[i] = state[i] + round_constants[rc_offset + i];
    }
  }

#if defined(FEATURE_BABY_BEAR)
  __host__ __device__ __forceinline__ field_t sbox_el(field_t x, field_t& sbox)
  {
    const field_t x1 = x;
    field_t x2 = x.sqr();
    const field_t x3 = x2 * x1;

    // Compute x^4 and x^7
    const field_t x4 = x2.sqr();
    const field_t x7 = x4 * x3;

    sbox = x3;
    return x7;
  }
#elif defined(FEATURE_KOALA_BEAR)
  __host__ __device__ __forceinline__ field_t sbox_el(field_t x)
  {
    const field_t x1 = x;
    const field_t x2 = x.sqr();
    return x2 * x1;
  }
#else
  #error "no FEATURE"
#endif

  __host__ __device__ inline void sbox(field_t state[WIDTH], FullRoundT& full_round)
  {
    for (int i = 0; i < WIDTH; i++) {
#if defined(FEATURE_BABY_BEAR)
      state[i] = sbox_el(state[i], full_round.sbox[i]);
#elif defined(FEATURE_KOALA_BEAR)
      state[i] = sbox_el(state[i]);
#else
  #error "no FEATURE"
#endif
    }
  }

#if defined(FEATURE_BABY_BEAR)
  __host__ __device__ inline void internal_layer_mat_mul(field_t state[WIDTH], field_t sum)
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
  __host__ __device__ inline void internal_layer_mat_mul(field_t state[WIDTH], field_t sum)
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

  __host__ __device__ inline void internal_round(
    field_t state[WIDTH], size_t rc_offset, PartialRoundT& partial_round, const field_t* __restrict__ round_constants)
  {
    field_t element = state[0] + round_constants[rc_offset];
#if defined(FEATURE_BABY_BEAR)
    element = sbox_el(element, partial_round.sbox);
#elif defined(FEATURE_KOALA_BEAR)
    element = sbox_el(element);
#else
  #error "no FEATURE"
#endif

    partial_round.post_sbox = element;

    field_t part_sum = 0;
    for (int i = 1; i < WIDTH; i++) {
      part_sum = part_sum + state[i];
    }

    field_t full_sum = part_sum + element;
    state[0] = part_sum - element;

    internal_layer_mat_mul(state, full_sum);
  }

  __host__ __device__ inline void
  permute_and_populate(Poseidon2ValueColsT& cols, const field_t* __restrict__ round_constants)
  {
    field_t* state = cols.input;
    mds_light_permutation(state);

    size_t rc_offset = 0;
    unsigned int cnt;

    for (cnt = 0; cnt < half_external_rounds; cnt++) {
      add_round_constants(state, rc_offset, round_constants);
      sbox(state, cols.beginning_full_rounds[cnt]);
      mds_light_permutation(state);
      rc_offset += WIDTH;

      for (int i = 0; i < WIDTH; i++) {
        cols.beginning_full_rounds[cnt].post[i] = state[i];
      }
    }

    for (cnt = 0; cnt < internal_rounds; cnt++) {
      internal_round(state, rc_offset, cols.partial_rounds[cnt], round_constants);
      rc_offset += WIDTH;
    }

    for (cnt = 0; cnt < half_external_rounds; cnt++) {
      add_round_constants(state, rc_offset, round_constants);
      sbox(state, cols.ending_full_rounds[cnt]);
      mds_light_permutation(state);
      rc_offset += WIDTH;

      for (int i = 0; i < WIDTH; i++) {
        cols.ending_full_rounds[cnt].post[i] = state[i];
      }
    }
  }

} // namespace pico_gpu::poseidon2
