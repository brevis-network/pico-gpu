// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0

/**
 * KoalaBear GPU-accelerated Poseidon2 duplex sponge grinding kernel.
 *
 * KoalaBear field (kb31_t) and KoalaBear Poseidon2 (alpha=3, 20 partial rounds).
 */

#include "ff/koala_bear.hpp"
#include "common/launcher.cuh"
#include "hash/poseidon2/poseidon2.cuh"
#include <cstdint>

using Fp = kb31_t;

// CELLS/CELLS_RATE/CELLS_OUT are preprocessor macros from hash/poseidon2/poseidon2.cuh — use directly

// Device sponge state — must match Rust KbDeviceSpongeState layout exactly.
struct KbDeviceSpongeState {
    Fp state[CELLS];       // 16 KoalaBear field elements (CELLS=16 is a macro)
    uint32_t absorb_idx;
    uint32_t sample_idx;
};

static_assert(sizeof(KbDeviceSpongeState) ==
              CELLS * sizeof(Fp) + 2 * sizeof(uint32_t),
              "KbDeviceSpongeState size mismatch with Rust");

// Sponge absorb (observe)
__device__ void kb_sponge_observe(KbDeviceSpongeState& sponge, Fp value) {
    sponge.state[sponge.absorb_idx] = value;
    sponge.absorb_idx += 1;
    if (sponge.absorb_idx == CELLS_RATE) {
        poseidon2_kb::poseidon2_mix_kb(sponge.state);
        sponge.absorb_idx = 0;
        sponge.sample_idx = CELLS_RATE;
    }
}

// Sponge squeeze (sample)
__device__ Fp kb_sponge_sample(KbDeviceSpongeState& sponge) {
    if (sponge.absorb_idx != 0 || sponge.sample_idx == 0) {
        poseidon2_kb::poseidon2_mix_kb(sponge.state);
        sponge.absorb_idx = 0;
        sponge.sample_idx = CELLS_RATE;
    }
    sponge.sample_idx -= 1;
    return sponge.state[sponge.sample_idx];
}

// Get the canonical uint32 from a field element.
// Canonical bits: the Rust verifier checks `as_canonical_u64() & mask`, so the grind must test
// canonical bits. Reinterpreting the raw (Montgomery-form) word here is what made GPU witnesses
// fail CPU verification and forced the slow CPU grind path.
static __device__ __forceinline__ uint32_t kb_fp_to_u32(Fp x) {
    // mont32_t::operator uint32_t() performs the Montgomery reduction (mul_by_1).
    return static_cast<uint32_t>(x);
}

__device__ uint32_t kb_sponge_sample_bits(KbDeviceSpongeState& sponge, uint32_t bits) {
    Fp rand_f = kb_sponge_sample(sponge);
    uint32_t rand_u32 = kb_fp_to_u32(rand_f);
    return rand_u32 & ((1u << bits) - 1);
}

__device__ bool kb_sponge_check_witness(KbDeviceSpongeState& sponge,
                                         uint32_t bits, Fp witness) {
    kb_sponge_observe(sponge, witness);
    return kb_sponge_sample_bits(sponge, bits) == 0;
}

// KoalaBear grinding kernel
__global__ void kb_grind_kernel(
    const KbDeviceSpongeState* init_state,
    uint32_t bits,
    uint32_t min_witness,
    uint32_t max_witness,
    uint32_t* result
) {
    uint32_t w = min_witness + blockIdx.x * blockDim.x + threadIdx.x;
    if (w > max_witness || *result != UINT32_MAX) {
        return;
    }

    KbDeviceSpongeState local_state = *init_state;
    // Use Fp((int32_t)w) so the int-constructor converts w from canonical to
    // Montgomery form: val = (w * R) % MOD. This matches Rust's from_u32(w) =
    // new(w) = to_monty(w). KoalaBear ORDER < 2^31 so the cast is safe.
    Fp witness = Fp((int32_t)w);
    if (kb_sponge_check_witness(local_state, bits, witness)) {
        atomicCAS(result, UINT32_MAX, w);
    }
}

extern "C" int _kb_sponge_grind(
    const KbDeviceSpongeState* init_state,
    uint32_t bits,
    uint32_t min_witness,
    uint32_t max_witness,
    uint32_t* result   // device pointer, must be initialised to UINT32_MAX before call
) {
    // kb31_t::MOD = 0x7F000001 = KoalaBear prime
    if (bits >= 32 || (uint64_t{1} << bits) >= (uint64_t)kb31_t::MOD) {
        return cudaErrorInvalidValue;
    }
    auto const [grid, block] = kernel_launch_params(1 << bits);
    kb_grind_kernel<<<grid, block>>>(init_state, bits, min_witness, max_witness, result);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    // Wait on this thread's stream, never the whole device: a device-wide sync cannot coexist
    // with a CUDA-graph capture open on any stream of that card, so with two prover lanes it
    // returns cudaErrorStreamCaptureUnsupported here AND kills the other lane's capture.
    err = cudaStreamSynchronize(cudaStreamPerThread);
    if (err != cudaSuccess) return err;

    return CHECK_KERNEL();
}

// ─── Test-support: single-state KoalaBear Poseidon2 permutation ──────────────
// Exposes poseidon2_kb::poseidon2_mix_kb for element-wise CPU<->GPU parity tests.

__global__ void kb_poseidon2_permute_kernel(Fp* state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        Fp local[CELLS];
        for (uint i = 0; i < CELLS; i++) local[i] = state[i];
        poseidon2_kb::poseidon2_mix_kb(local);
        for (uint i = 0; i < CELLS; i++) state[i] = local[i];
    }
}

extern "C" int _kb_poseidon2_permute(Fp* state) {
    kb_poseidon2_permute_kernel<<<1, 1>>>(state);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    err = cudaStreamSynchronize(cudaStreamPerThread);
    if (err != cudaSuccess) return err;
    return CHECK_KERNEL();
}
