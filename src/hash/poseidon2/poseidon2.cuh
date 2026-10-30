// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0

/*
 * KoalaBear Poseidon2 permutation for GPU.
 *
 * Parameters (matching Plonky3 0.5 `default_koalabear_poseidon2_16`, which is what
 * mle-core/src/config/koala_bear_poseidon2.rs re-exports):
 *   - Field: KoalaBear (p = 2^31 - 2^24 + 1)
 *   - Width:  16
 *   - S-box:  x^3  (alpha = 3)
 *   - External rounds: 8  (4 initial + 4 terminal)
 *   - Internal rounds: 20
 *   - Round constants: Plonky3's KOALABEAR_POSEIDON2_RC_16_{EXTERNAL_INITIAL, EXTERNAL_FINAL,
 *     INTERNAL} (Grain LFSR, field_type=1, alpha=3, n=31, t=16, R_F=8, R_P=20). Every round adds
 *     a non-zero constant; there are no zero rounds.
 *
 * Internal matrix diagonal V (matching KoalaBearInternalLayerParameters in Plonky3):
 *   [-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/2^8, 1/8, 1/2^24, -1/2^8, -1/8, -1/16, -1/2^24]
 */

#pragma once

#include "ff/koala_bear.hpp"
#include "hash/poseidon2/params.cuh"

namespace poseidon2_kb {

using Fp = kb31_t;

// ── Round constants (standard/canonical form; kb31_t constructor converts to Montgomery) ──

// 4 initial external rounds × 16 elements
// 4 initial external rounds × 16 elements = KOALABEAR_POSEIDON2_RC_16_EXTERNAL_INITIAL
static __device__ __constant__ Fp INITIAL_ROUND_CONSTANTS[64] = {
    // round 0
    Fp{0x7ee56a48}, Fp{0x11367045}, Fp{0x12e41941}, Fp{0x7ebbc12b},
    Fp{0x1970b7d5}, Fp{0x662b60e8}, Fp{0x3e4990c6}, Fp{0x679f91f5},
    Fp{0x350813bb}, Fp{0x00874ad4}, Fp{0x28a0081a}, Fp{0x18fa5872},
    Fp{0x5f25b071}, Fp{0x5e5d5998}, Fp{0x5e6fd3e7}, Fp{0x5b2e2660},
    // round 1
    Fp{0x6f1837bf}, Fp{0x3fe6182b}, Fp{0x1edd7ac5}, Fp{0x57470d00},
    Fp{0x43d486d5}, Fp{0x1982c70f}, Fp{0x0ea53af9}, Fp{0x61d6165b},
    Fp{0x51639c00}, Fp{0x2dec352c}, Fp{0x2950e531}, Fp{0x2d2cb947},
    Fp{0x08256cef}, Fp{0x1a0109f6}, Fp{0x1f51faf3}, Fp{0x5cef1c62},
    // round 2
    Fp{0x3d65e50e}, Fp{0x33d91626}, Fp{0x133d5a1e}, Fp{0x0ff49b0d},
    Fp{0x38900cd1}, Fp{0x2c22cc3f}, Fp{0x28852bb2}, Fp{0x06c65a02},
    Fp{0x7b2cf7bc}, Fp{0x68016e1a}, Fp{0x15e16bc0}, Fp{0x5248149a},
    Fp{0x6dd212a0}, Fp{0x18d6830a}, Fp{0x5001be82}, Fp{0x64dac34e},
    // round 3
    Fp{0x5902b287}, Fp{0x426583a0}, Fp{0x0c921632}, Fp{0x3fe028a5},
    Fp{0x245f8e49}, Fp{0x43bb297e}, Fp{0x7873dbd9}, Fp{0x3cc987df},
    Fp{0x286bb4ce}, Fp{0x640a8dcd}, Fp{0x512a8e36}, Fp{0x03a4cf55},
    Fp{0x481837a2}, Fp{0x03d6da84}, Fp{0x73726ac7}, Fp{0x760e7fdf},
};

// 4 terminal external rounds × 16 elements = KOALABEAR_POSEIDON2_RC_16_EXTERNAL_FINAL
static __device__ __constant__ Fp TERMINAL_ROUND_CONSTANTS[64] = {
    // round 0
    Fp{0x43e7dc24}, Fp{0x259a5d61}, Fp{0x27e85a3b}, Fp{0x1b9133fa},
    Fp{0x343e5628}, Fp{0x485cd4c2}, Fp{0x16e269f5}, Fp{0x165b60c6},
    Fp{0x25f683d9}, Fp{0x124f81f9}, Fp{0x174331f9}, Fp{0x77344dc5},
    Fp{0x5a821dba}, Fp{0x5fc4177f}, Fp{0x54153bf5}, Fp{0x5e3f1194},
    // round 1
    Fp{0x3bdbf191}, Fp{0x088c84a3}, Fp{0x68256c9b}, Fp{0x3c90bbc6},
    Fp{0x6846166a}, Fp{0x03f4238d}, Fp{0x463335fb}, Fp{0x5e3d3551},
    Fp{0x6e59ae6f}, Fp{0x32d06cc0}, Fp{0x596293f3}, Fp{0x6c87edb2},
    Fp{0x08fc60b5}, Fp{0x34bcca80}, Fp{0x24f007f3}, Fp{0x62731c6f},
    // round 2
    Fp{0x1e1db6c6}, Fp{0x0ca409bb}, Fp{0x585c1e78}, Fp{0x56e94edc},
    Fp{0x16d22734}, Fp{0x18e11467}, Fp{0x7b2c3730}, Fp{0x770075e4},
    Fp{0x35d1b18c}, Fp{0x22be3db5}, Fp{0x4fb1fbb7}, Fp{0x477cb3ed},
    Fp{0x7d5311c6}, Fp{0x5b62ae7d}, Fp{0x559c5fa8}, Fp{0x77f15048},
    // round 3
    Fp{0x3211570b}, Fp{0x490fef6a}, Fp{0x77ec311f}, Fp{0x2247171b},
    Fp{0x4e0ac711}, Fp{0x2edf69c9}, Fp{0x3b5a8850}, Fp{0x65809421},
    Fp{0x5619b4aa}, Fp{0x362019a7}, Fp{0x6bf9d4ed}, Fp{0x5b413dff},
    Fp{0x617e181e}, Fp{0x5e7ab57b}, Fp{0x33ad7833}, Fp{0x3466c7ca},
};

// 20 internal round constants (only index 0 of the state is used) = KOALABEAR_POSEIDON2_RC_16_INTERNAL
static __device__ __constant__ Fp INTERNAL_ROUND_CONSTANTS[20] = {
    Fp{0x54dfeb5d}, Fp{0x7d40afd6}, Fp{0x722cb316}, Fp{0x106a4573},
    Fp{0x45a7ccdb}, Fp{0x44061375}, Fp{0x154077a5}, Fp{0x45744faa},
    Fp{0x4eb5e5ee}, Fp{0x3794e83f}, Fp{0x47c7093c}, Fp{0x5694903c},
    Fp{0x69cb6299}, Fp{0x373df84c}, Fp{0x46a0df58}, Fp{0x46b8758a},
    Fp{0x3241ebcb}, Fp{0x0b09d233}, Fp{0x1af42357}, Fp{0x1e66cec2},
};

// Permutation shape from params.cuh; rate and output width are this sponge's own choices.
#define CELLS            (::poseidon2::state_width)
#define CELLS_RATE 8
#define CELLS_OUT  8
#define ROUNDS_FULL      (::poseidon2::EXTERNAL_ROUNDS)
#define ROUNDS_HALF_FULL (::poseidon2::half_external_rounds)
#define ROUNDS_PARTIAL   (::poseidon2::internal_rounds)

// The S-box and the internal matrix below are written out for this exact shape.
static_assert(::poseidon2::alpha == 3, "sbox_d3 implements the x^3 S-box");
static_assert(::poseidon2::state_width == 16, "internal_layer_mat_mul_kb is written for width 16");

// S-box: x^3 for KoalaBear (alpha = 3)
// Fused x^3 (ported from FRI merkle_tree.cu sbox_cube): skip the intermediate
// Montgomery final_sub between the two muls. For a 31-bit prime with R=2^32,
// REDC(x*x) < 2*MOD; then REDC(x2*x) is valid since x2*x < 2*MOD^2 < R*MOD
// (2*MOD < R). Final REDC + final_sub yields the canonical result < MOD, so the
// value is bit-identical to (x*x)*x. Wires MLE kb31_t Montgomery constants:
// MOD = 0x7F000001, M0 = -p^{-1} mod 2^32 = 0x7EFFFFFF (== MOD-2 for KoalaBear).
static __device__ __forceinline__ Fp sbox_d3(Fp x) {
    const uint32_t MOD = Fp::MOD;   // 0x7F000001
    const uint32_t M   = Fp::M0;    // 0x7EFFFFFF
    uint32_t xv = *x;               // raw Montgomery limb
    uint32_t t0, t1, red, x2, x3;

    // x^2 = REDC(x * x), skip final_sub -> result < 2*MOD
    asm volatile("mul.lo.u32 %0, %2, %3; mul.hi.u32 %1, %2, %3;"
        : "=r"(t0), "=r"(t1) : "r"(xv), "r"(xv));
    asm volatile("mul.lo.u32 %0, %1, %2;"
        : "=r"(red) : "r"(t0), "r"(M));
    asm volatile("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.u32 %1, %2, %3, %4;"
        : "+r"(t0), "=r"(x2) : "r"(red), "r"(MOD), "r"(t1));

    // x^3 = REDC(x2 * x), WITH final_sub -> canonical result < MOD
    asm volatile("mul.lo.u32 %0, %2, %3; mul.hi.u32 %1, %2, %3;"
        : "=r"(t0), "=r"(t1) : "r"(x2), "r"(xv));
    asm volatile("mul.lo.u32 %0, %1, %2;"
        : "=r"(red) : "r"(t0), "r"(M));
    asm volatile("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.u32 %1, %2, %3, %4;"
        : "+r"(t0), "=r"(x3) : "r"(red), "r"(MOD), "r"(t1));
    asm volatile("{");
    asm volatile(".reg.pred %p;");
    asm volatile("setp.ge.u32 %p, %0, %1;" :: "r"(x3), "r"(MOD));
    asm volatile("@%p sub.u32 %0, %0, %1;" : "+r"(x3) : "r"(MOD));
    asm volatile("}");

    return Fp(x3);
}

static __device__ void do_full_sboxes_kb(Fp *cells) {
    #pragma unroll
    for (uint i = 0; i < CELLS; i++) {
        cells[i] = sbox_d3(cells[i]);
    }
}

static __device__ void do_partial_sbox_kb(Fp *cells) {
    cells[0] = sbox_d3(cells[0]);
}

// Field helpers: kb31_t uses mont32_t but lacks doubled()/halve() methods
// doubled(x) = x + x;  halve(x) = x * inv2 where inv2 = (P+1)/2 = 1065353217
static __device__ __forceinline__ Fp kb_dbl(Fp x)  { return x + x; }
// KoalaBear inv2 = (2130706433+1)/2 = 1065353217  →  Fp{1065353217} converts to Montgomery
// IMPORTANT: must use int literal (not uint32_t) so kb31_t(int) constructor is called,
// which performs the Montgomery conversion val = (1065353217 << 32) % P.
// FIX_APPLIED_V2: kb_hlv now uses int literal for correct Montgomery form
static __device__ __forceinline__ Fp kb_hlv(Fp x)  { return x * Fp{1065353217}; }

// Multiply a 4-element vector by circulant MDS [2,3,1,1]
static __device__ void multiply_by_4x4_circulant_kb(Fp *x) {
    Fp t01 = x[0] + x[1];
    Fp t23 = x[2] + x[3];
    Fp t0123 = t01 + t23;
    Fp t01123 = t0123 + x[1];
    Fp t01233 = t0123 + x[3];
    x[3] = t01233 + kb_dbl(x[0]);
    x[1] = t01123 + kb_dbl(x[2]);
    x[0] = t01123 + t01;
    x[2] = t01233 + t23;
}

static __device__ void multiply_by_m_ext_kb(Fp *old_cells) {
    Fp tmp_sums[4] = {Fp{0}, Fp{0}, Fp{0}, Fp{0}};
    #pragma unroll
    for (uint i = 0; i < CELLS / 4; i++) {
        multiply_by_4x4_circulant_kb(old_cells + i * 4);
        #pragma unroll
        for (uint j = 0; j < 4; j++) {
            tmp_sums[j] = tmp_sums[j] + old_cells[i * 4 + j];
        }
    }
    #pragma unroll
    for (uint i = 0; i < CELLS; i++) {
        old_cells[i] = old_cells[i] + tmp_sums[i % 4];
    }
}

static __device__ void add_round_constants_full_kb(const Fp *rc, Fp *cells, uint round) {
    #pragma unroll
    for (uint i = 0; i < CELLS; i++) {
        cells[i] = cells[i] + rc[round * CELLS + i];
    }
}

// Helper: x / 2^n via repeated halving (uses kb_hlv for kb31_t compatibility)
static __device__ __forceinline__ Fp halve_n(Fp x, int n) {
    for (int i = 0; i < n; i++) x = kb_hlv(x);
    return x;
}

// Internal matrix diagonal:
// V = [-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/2^8, 1/8, 1/2^24, -1/2^8, -1/8, -1/16, -1/2^24]
static __device__ __forceinline__ void internal_layer_mat_mul_kb(Fp *cells) {
    // Compute sum = cells[1] + cells[2] + ... + cells[15]
    Fp part_sum = cells[1];
    for (uint i = 2; i < CELLS; i++) {
        part_sum = part_sum + cells[i];
    }
    Fp full_sum = part_sum + cells[0];

    // cells[0] gets coefficient -2: part_sum - cells[0]
    cells[0] = part_sum - cells[0];

    // cells[1]: +1  → += full_sum
    cells[1] = cells[1] + full_sum;
    // cells[2]: +2  → x+x + full_sum
    cells[2] = kb_dbl(cells[2]) + full_sum;
    // cells[3]: +1/2 → halve + full_sum
    cells[3] = kb_hlv(cells[3]) + full_sum;
    // cells[4]: +3  → dbl+orig + full_sum
    cells[4] = full_sum + kb_dbl(cells[4]) + cells[4];
    // cells[5]: +4  → dbl(dbl) + full_sum
    cells[5] = full_sum + kb_dbl(kb_dbl(cells[5]));
    // cells[6]: -1/2 → full_sum - halve
    cells[6] = full_sum - kb_hlv(cells[6]);
    // cells[7]: -3  → full_sum - (dbl+orig)
    cells[7] = full_sum - (kb_dbl(cells[7]) + cells[7]);
    // cells[8]: -4  → full_sum - dbl(dbl)
    cells[8] = full_sum - kb_dbl(kb_dbl(cells[8]));
    // cells[9]:  +1/2^8 → halve 8 times + full_sum
    cells[9] = halve_n(cells[9], 8) + full_sum;
    // cells[10]: +1/8 = 1/2^3 → halve 3 times + full_sum
    cells[10] = halve_n(cells[10], 3) + full_sum;
    // cells[11]: +1/2^24 → halve 24 times + full_sum
    cells[11] = halve_n(cells[11], 24) + full_sum;
    // cells[12]: -1/2^8 → full_sum - halve_8
    cells[12] = full_sum - halve_n(cells[12], 8);
    // cells[13]: -1/8 = -1/2^3 → full_sum - halve_3
    cells[13] = full_sum - halve_n(cells[13], 3);
    // cells[14]: -1/16 = -1/2^4 → full_sum - halve_4
    cells[14] = full_sum - halve_n(cells[14], 4);
    // cells[15]: -1/2^24 → full_sum - halve_24
    cells[15] = full_sum - halve_n(cells[15], 24);
}

static __device__ void full_round_half_kb(const Fp *rc, Fp *cells, uint round) {
    add_round_constants_full_kb(rc, cells, round);
    do_full_sboxes_kb(cells);
    multiply_by_m_ext_kb(cells);
}

static __device__ void partial_round_kb(Fp *cells, uint round) {
    cells[0] = cells[0] + INTERNAL_ROUND_CONSTANTS[round];
    do_partial_sbox_kb(cells);
    internal_layer_mat_mul_kb(cells);
}

// Reduce a uint64 sum of up to 15 raw Montgomery residues (< MOD < 2^31 each) to a canonical
// residue. Uses 2^31 ≡ (2^31 − MOD) (mod MOD); for KoalaBear 2^31 − MOD = 2^24 − 1.
// hi ≤ 15 ⇒ result < 2^31 + 15·(2^24−1) < 2·MOD ⇒ one conditional subtraction suffices.
static __device__ __forceinline__ Fp kb_field_sum_u64(uint64_t v) {
    uint32_t lo = (uint32_t)(v & 0x7FFFFFFFu);
    uint32_t hi = (uint32_t)(v >> 31);
    uint32_t r  = lo + hi * ((1u << 31) - Fp::MOD);
    if (r >= Fp::MOD) r -= Fp::MOD;
    return Fp{r}; // raw Montgomery-residue constructor (uint32_t)
}

// Full KoalaBear Poseidon2 permutation (width=16).
// Optimizations (techniques from merkle_tree.cu's p2_permute; the schedule is
// the Plonky3 0.5 KoalaBear width-16 one, bit-exact with `default_koalabear_poseidon2_16`):
//   1. #pragma unroll on all round/state loops.
//   2. Partial rounds keep a running part_sum: full_sum = part_sum + state[0] is 1 add per
//      round instead of a serial 15-add chain; the next round's part_sum is accumulated as a
//      uint64 of raw residues during the M_int update (independent adds, no field reductions)
//      and reduced once via kb_field_sum_u64.
//   3. Diagonal 1/2^k multiplications use mont32_t's fused `>> k` (3 instructions for k > 2)
//      instead of halve_n's k sequential Montgomery multiplications (was 8/3/24/8/3/4/24 muls
//      per round for cells 9..15 — ~1400 muls per permutation removed).
static __device__ void poseidon2_mix_kb(Fp *cells) {
    // Initial linear layer
    multiply_by_m_ext_kb(cells);

    // 4 initial external (full) rounds
    #pragma unroll
    for (uint i = 0; i < ROUNDS_HALF_FULL; i++) {
        full_round_half_kb(INITIAL_ROUND_CONSTANTS, cells, i);
    }

    // 20 internal (partial) rounds with running part_sum
    Fp part_sum = cells[1];
    #pragma unroll
    for (uint i = 2; i < CELLS; i++) {
        part_sum = part_sum + cells[i];
    }

    #pragma unroll 4
    for (uint r = 0; r < ROUNDS_PARTIAL; r++) {
        cells[0] = cells[0] + INTERNAL_ROUND_CONSTANTS[r];
        cells[0] = sbox_d3(cells[0]);

        Fp full_sum = part_sum + cells[0];
        // V[0] = −2: part_sum − state[0]  (≡ full_sum − 2·state[0])
        cells[0] = part_sum - cells[0];

        // Apply M_int diag V = [−2,1,2,1/2,3,4,−1/2,−3,−4,1/2^8,1/8,1/2^24,−1/2^8,−1/8,−1/16,−1/2^24]
        // while accumulating the next part_sum (Σ cells[1..15]) as raw uint64.
        uint64_t nps = 0;
        cells[1]  = cells[1] + full_sum;                       nps += *cells[1];
        cells[2]  = kb_dbl(cells[2]) + full_sum;               nps += *cells[2];
        cells[3]  = (cells[3] >> 1) + full_sum;                nps += *cells[3];
        cells[4]  = full_sum + kb_dbl(cells[4]) + cells[4];    nps += *cells[4];
        cells[5]  = full_sum + kb_dbl(kb_dbl(cells[5]));       nps += *cells[5];
        cells[6]  = full_sum - (cells[6] >> 1);                nps += *cells[6];
        cells[7]  = full_sum - (kb_dbl(cells[7]) + cells[7]);  nps += *cells[7];
        cells[8]  = full_sum - kb_dbl(kb_dbl(cells[8]));       nps += *cells[8];
        cells[9]  = (cells[9] >> 8) + full_sum;                nps += *cells[9];
        cells[10] = (cells[10] >> 3) + full_sum;               nps += *cells[10];
        cells[11] = (cells[11] >> 24) + full_sum;              nps += *cells[11];
        cells[12] = full_sum - (cells[12] >> 8);               nps += *cells[12];
        cells[13] = full_sum - (cells[13] >> 3);               nps += *cells[13];
        cells[14] = full_sum - (cells[14] >> 4);               nps += *cells[14];
        cells[15] = full_sum - (cells[15] >> 24);              nps += *cells[15];

        part_sum = kb_field_sum_u64(nps);
    }

    // 4 terminal external (full) rounds
    #pragma unroll
    for (uint r = 0; r < ROUNDS_HALF_FULL; r++) {
        full_round_half_kb(TERMINAL_ROUND_CONSTANTS, cells, r);
    }
}

} // namespace poseidon2_kb
