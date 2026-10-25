// Copyright (c) 2025-2026 Brevis. All Rights Reserved.
// Licensed under the Business Source License 1.1 (BUSL-1.1).
// See LICENSE.md for details.

#include "field/fp.h"
#include "field/fpext.h"
#include "common/launcher.cuh"
#include "prover/reduce.cuh"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector_types.h>

namespace {
constexpr int S_DEG = 2;
}

// Mirrors `BatchingTracePacket` in cuda-mle/src/whir.rs -- field order and types must match.
// `lifted_height_shift` / `stride_shift` are log2 values derived and power-of-two-checked on the
// host, which is why this kernel takes no `skip_domain`: the host owns that formula.
struct BatchingTracePacket {
    const Fp *__restrict__ ptr;
    uint32_t height;
    uint32_t width;
    uint32_t stacked_row_start;
    uint32_t mu_idx;
    uint32_t lifted_height_shift;
    uint32_t stride_shift;
};

/// Produces the mu-weighted combination of the stacked columns directly from the per-trace
/// (unstacked) buffers, without ever materialising the stacked matrix.
__global__ void whir_algebraic_batch_traces_kernel(
    Fp *__restrict__ dst,                                  // 4 base columns of total_rows each
    const BatchingTracePacket *__restrict__ trace_packets, // one descriptor per source trace
    const FpExt *mu_pows,                                  // one power per stacked column
    size_t total_rows,
    size_t packet_count
) {

    size_t out_row = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_row >= total_rows) {
        return;
    }

    // A wide trace set could benefit from a shared-memory reduction here; not done yet.
    // All of total_rows, lifted_height and stride are powers of two, so the index math below uses
    // shifts and masks: a runtime 64-bit divide would compile to the slow __cuda_sm20_div_u64
    // routine. The host supplies the per-trace log2 values already validated, so nothing is
    // recomputed per (row, trace) pair here.
    const uint32_t total_rows_log = __ffsll((long long)total_rows) - 1;
    FpExt acc(Fp::zero());
    for (int p = 0; p < packet_count; p++) {
        auto desc = trace_packets[p];
        auto src = desc.ptr;
        uint32_t src_height = desc.height;
        uint32_t src_width = desc.width;
        uint32_t start_row = desc.stacked_row_start;
        uint32_t mu_base = desc.mu_idx;
        const uint32_t lifted_height_shift = desc.lifted_height_shift;
        const uint32_t lifted_height = 1u << lifted_height_shift;
        const uint32_t lifted_height_mask = lifted_height - 1;
        const uint32_t stride_shift = desc.stride_shift; // stride == 1 -> 0
        const uint32_t stride_mask = (1u << stride_shift) - 1; // stride == 1 -> 0 (x & 0 == 0)
        // The trace occupies stacked positions [start_row, start_row + lifted_height * src_width),
        // wrapping modulo total_rows. Visit every lap with
        //   start_row <= lap * total_rows + out_row < window_end.
        auto window_end = start_row + lifted_height * src_width;
        if (out_row >= window_end)
            continue;
        uint32_t first_lap = start_row <= out_row ? 0 : 1;
        for (uint32_t pos = first_lap * total_rows + out_row; pos < window_end;
             pos += total_rows) {
            auto lap = (pos - out_row) >> total_rows_log;
            uint32_t local_pos = pos - start_row; // < lifted_height * src_width
            uint32_t src_col = local_pos >> lifted_height_shift;
            uint32_t lifted_row = local_pos & lifted_height_mask;
            // only every stride-th lifted row carries a real trace value; the rest are zero
            auto val = (lifted_row & stride_mask) == 0
                                 ? src[src_col * src_height + (lifted_row >> stride_shift)]
                                 : Fp::zero();
            auto coeff = mu_pows[mu_base + lap];
            acc += coeff * val;
        }
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        dst[i * total_rows + out_row] = acc.elems[i];
    }
}

// One WHIR sumcheck round. `coeffs` holds f as multilinear coefficients and `moments` holds w as
// superset sums (m[T] = sum over x containing T of w(x)). Each block writes its partial sums of
//   s(X) = sum_y f(X, y) * w(X, y)   for X = 1 and X = 2,
// computed pair by pair straight from the coefficient / moment representation.
__global__ void whir_sumcheck_coeff_moments_round_kernel(
    const FpExt *coeffs,
    const FpExt *moments,
    FpExt *partials,
    const uint32_t n
) {
    extern __shared__ char smem[];
    FpExt *scratch = (FpExt *)smem;

    int num_pairs = n >> 1;

    FpExt thread_acc[S_DEG];

#pragma unroll
    for (int k = 0; k < S_DEG; k++) {
        thread_acc[k] = FpExt(0);
    }

    for (int pair = blockIdx.x * blockDim.x + threadIdx.x; pair < num_pairs;
         pair += gridDim.x * blockDim.x) {
        int lo = pair << 1;
        int hi = lo + 1;

        FpExt f_lo = coeffs[lo];
        FpExt f_hi = coeffs[hi];
        FpExt w_lo = moments[lo];
        FpExt w_hi = moments[hi];

        // At X = 1 the coefficient form gives f = f_lo + f_hi and the moment form leaves only w_hi.
        FpExt f_at_1 = f_lo + f_hi;
        FpExt prod_at_1 = f_at_1 * w_hi;

        // At X = 2: f = f_lo + 2 f_hi, and the moment becomes 3 w_hi - w_lo.
        FpExt f_at_2 = f_lo + f_hi + f_hi;
        FpExt w_at_2 = w_hi * Fp(3) - w_lo;
        FpExt prod_at_2 = f_at_2 * w_at_2;

        thread_acc[0] += prod_at_1;
        thread_acc[1] += prod_at_2;
    }

#pragma unroll
    for (int d = 0; d < S_DEG; d++) {
        FpExt block_total = reduce::block_reduce_sum(thread_acc[d], scratch);
        if (threadIdx.x == 0) {
            partials[blockIdx.x * S_DEG + d] = block_total;
        }
        __syncthreads();
    }
}

// Binds the leading variable to `r` in both the coefficient vector of f and the superset-sum
// (moment) vector of w, halving each.
__global__ void whir_fold_coeffs_and_moments_kernel(
    const FpExt *coeffs,
    const FpExt *moments,
    FpExt *coeffs_out,
    FpExt *moments_out,
    FpExt r,
    uint32_t num_pairs
) {
    uint32_t pair = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair >= num_pairs) {
        return;
    }

    uint32_t lo = pair << 1;
    uint32_t hi = lo + 1;

    FpExt f_lo = coeffs[lo];
    FpExt f_hi = coeffs[hi];
    coeffs_out[pair] = f_lo + r * f_hi;

    FpExt w_lo = moments[lo];
    FpExt w_hi = moments[hi];
    FpExt lo_weight = FpExt(Fp(1)) - r;
    FpExt hi_weight = r + r - FpExt(Fp(1));
    moments_out[pair] = lo_weight * w_lo + hi_weight * w_hi;
}

// x^e given sq_pows[b] = x^(2^b) for the low `nbits` bits of e.
__device__ __forceinline__ FpExt pow_from_square_chain_ext(
    const FpExt *sq_pows,
    uint32_t nbits,
    uint32_t e
) {
    FpExt prod = FpExt(Fp(1));
    for (uint32_t b = 0; b < nbits; b++) {
        if (e & (1u << b)) {
            prod *= sq_pows[b];
        }
    }
    return prod;
}

__device__ __forceinline__ Fp pow_from_square_chain_base(
    const Fp *sq_pows,
    uint32_t nbits,
    uint32_t e
) {
    Fp prod = Fp(1);
    for (uint32_t b = 0; b < nbits; b++) {
        if (e & (1u << b)) {
            prod *= sq_pows[b];
        }
    }
    return prod;
}

__global__ void w_moments_accumulate_kernel(
    FpExt *moments,
    const FpExt *z0_sq, // z0 raised to 2^b, for b < nbits
    const Fp *z_sq,     // nq rows of nbits squarings, one row per query point
    FpExt g,
    uint32_t nq,
    uint32_t nbits,
    uint32_t n
) {
    size_t pt = blockIdx.x * blockDim.x + threadIdx.x;
    if (pt >= n) {
        return;
    }

    uint32_t e = static_cast<uint32_t>(pt);

    FpExt sum = g * pow_from_square_chain_ext(z0_sq, nbits, e);
    FpExt g_pow = g;
    for (uint32_t q = 0; q < nq; q++) {
        g_pow *= g;
        const Fp *q_sq = z_sq + q * nbits;
        Fp zq_pow = pow_from_square_chain_base(q_sq, nbits, e);
        sum += g_pow * zq_pow;
    }
    moments[pt] += sum;
}


// ── w_moments accumulate: NIB+FOLD fast path (5.31x standalone) ─────────────
// The original kernel makes every thread (a) walk the identical serial
// gamma_pow *= gamma chain (num_queries FpExt muls) and (b) rebuild each
// z_i^row as a log_height-level predicated bit-product per query. Instead:
//   * s_gpow[q] = gamma^(q+2) built once per block via square-and-multiply,
//     with the block-common hi-prefix (exponent bits 8..log_height, shared by
//     all rows in a block) folded in;
//   * the low-8-bit product comes from two per-query 16-entry smem nibble
//     tables (bits 0-3 / 4-7) built once per block;
//   * inner loop = 2 smem loads + 1 Fp mul + 1 ext-by-base mul-add.
// Byte-equal: Fp/FpExt mul ends in an unconditional final_sub, so every
// product is the canonical [0,p) representative and exact-field
// re-association is bit-safe. Requires blockDim.x == 256 (launcher-enforced).
__global__ void w_moments_accumulate_nib_kernel(
    FpExt *w_moments,
    const FpExt *z0_pows2, // [log_height], where z0_pows2[i] = z0^(2^i)
    const Fp *z_pows2,     // [num_queries][log_height], row-major
    FpExt gamma,
    uint32_t num_queries,
    uint32_t log_height,
    uint32_t height
) {
    constexpr uint32_t LOWB = 8; // log2(blockDim.x); launcher guarantees blockDim.x == 256
    extern __shared__ unsigned char w_moments_smem[];
    FpExt *s_gpow = reinterpret_cast<FpExt *>(w_moments_smem); // [num_queries] gamma^(q+2)·hi_q
    const uint32_t tid = threadIdx.x;

    // Per-block setup: gamma^(q+2) via square-and-multiply (canonical => byte-equal).
    for (uint32_t q = tid; q < num_queries; q += blockDim.x) {
        uint32_t e = q + 2;
        FpExt b = gamma;
        FpExt r = FpExt(Fp(1));
        while (true) {
            if (e & 1u) r *= b;
            e >>= 1;
            if (!e) break;
            b *= b;
        }
        s_gpow[q] = r;
    }

    Fp *s_hi = reinterpret_cast<Fp *>(s_gpow + num_queries); // [num_queries] hi-prefix
    Fp *s_n0 = s_hi + num_queries;                           // [num_queries][16] bits 0-3
    Fp *s_n1 = s_n0 + (size_t)num_queries * 16;              // [num_queries][16] bits 4-7
    const uint32_t lowb = (log_height < LOWB) ? log_height : LOWB;
    // Exponent bits [lowb, log_height) are block-common: row >> lowb == blockIdx.x for
    // every thread in the block (blockDim.x == 1 << lowb).
    const uint32_t hi_bits = (uint32_t)(((size_t)blockIdx.x * blockDim.x) >> lowb);

    for (uint32_t q = tid; q < num_queries; q += blockDim.x) {
        const Fp *qp = z_pows2 + (size_t)q * log_height;
        Fp h = Fp(1);
        for (uint32_t bit = lowb; bit < log_height; bit++) {
            if (hi_bits & (1u << (bit - lowb))) h *= qp[bit];
        }
        s_hi[q] = h;
        // Nibble tables; entry k = ascending product of qp[bit] over set bits of k
        // (t[k] = t[k ^ (1<<hb)] * qp[hb] with hb = highest set bit keeps ascending order).
        Fp t[16];
        t[0] = Fp(1);
        s_n0[q * 16 + 0] = t[0];
        for (uint32_t k = 1; k < 16; k++) {
            uint32_t hb = 31u - __clz(k);
            Fp v = t[k ^ (1u << hb)];
            if (hb < lowb) v *= qp[hb];
            t[k] = v;
            s_n0[q * 16 + k] = v;
        }
        t[0] = Fp(1);
        s_n1[q * 16 + 0] = t[0];
        for (uint32_t k = 1; k < 16; k++) {
            uint32_t hb = 31u - __clz(k);
            uint32_t bit = 4u + hb;
            Fp v = t[k ^ (1u << hb)];
            if (bit < lowb) v *= qp[bit];
            t[k] = v;
            s_n1[q * 16 + k] = v;
        }
        // Same thread wrote s_gpow[q] in the first strided loop (same q assignment) — no race.
        s_gpow[q] = s_gpow[q] * h;
    }

    __syncthreads();

    size_t row = (size_t)blockIdx.x * blockDim.x + tid;
    if (row >= height) return;
    uint32_t exponent = static_cast<uint32_t>(row);

    // z0 term: 1 of num_queries+1 terms — keep the original per-thread ext pow (negligible).
    FpExt acc = gamma * pow_from_square_chain_ext(z0_pows2, log_height, exponent);

    const uint32_t i0 = tid & 15u;
    const uint32_t i1 = (tid >> 4) & 15u;
    for (uint32_t i = 0; i < num_queries; i++) {
        Fp lo = s_n0[i * 16 + i0] * s_n1[i * 16 + i1];
        acc += s_gpow[i] * lo; // s_gpow already carries the hi-prefix factor
    }

    w_moments[row] += acc;
}

// ============================================================================
// LAUNCHERS
// ============================================================================

extern "C" int _kb_whir_algebraic_batch_traces(
    Fp *dst,
    const BatchingTracePacket *trace_packets,
    const FpExt *mu_pows,
    size_t total_rows,
    size_t packet_count
) {
    auto [grid, block] = kernel_launch_params(total_rows);
    whir_algebraic_batch_traces_kernel<<<grid, block>>>(
        dst, trace_packets, mu_pows, total_rows, packet_count
    );
    return CHECK_KERNEL();
}

inline std::pair<dim3, dim3> round_launch_geometry(uint32_t n) {
    return kernel_launch_params(n >> 1, 256);
}

extern "C" uint32_t _kb_whir_sumcheck_coeff_moments_required_temp_buffer_size(uint32_t n) {
    auto [grid, block] = round_launch_geometry(n);
    return grid.x * S_DEG;
}

extern "C" int _kb_whir_sumcheck_coeff_moments_round(
    const FpExt *coeffs,
    const FpExt *moments,
    FpExt *result,   // S_DEG evaluations, s(1) and s(2)
    FpExt *partials, // scratch, S_DEG entries per block of the first pass
    const uint32_t n
) {
    auto [grid, block] = round_launch_geometry(n);
    unsigned int warps_per_block = (block.x + WARP_SIZE - 1) / WARP_SIZE;
    size_t smem_bytes = std::max(1u, warps_per_block) * sizeof(FpExt);

    whir_sumcheck_coeff_moments_round_kernel<<<grid, block, smem_bytes>>>(
        coeffs, moments, partials, n
    );

    int err = CHECK_KERNEL();
    if (err != 0)
        return err;

    auto partial_count = grid.x;
    auto [reduce_grid, reduce_block] = kernel_launch_params(partial_count);
    unsigned int reduce_warps = (reduce_block.x + WARP_SIZE - 1) / WARP_SIZE;
    size_t reduce_shmem = std::max(1u, reduce_warps) * sizeof(FpExt);
    reduce::static_final_reduce_block_sums<S_DEG>
        <<<S_DEG, reduce_block, reduce_shmem>>>(partials, result, partial_count);

    return CHECK_KERNEL();
}

extern "C" int _kb_whir_fold_coeffs_and_moments(
    const FpExt *coeffs,
    const FpExt *moments,
    FpExt *coeffs_out,
    FpExt *moments_out,
    FpExt r,
    uint32_t n
) {
    auto [grid, block] = kernel_launch_params(n >> 1);
    whir_fold_coeffs_and_moments_kernel<<<grid, block>>>(
        coeffs, moments, coeffs_out, moments_out, r, n >> 1
    );
    return CHECK_KERNEL();
}

extern "C" int _kb_w_moments_accumulate(
    FpExt *moments,
    const FpExt *z0_sq,
    const Fp *z_sq,
    FpExt g,
    uint32_t nq,
    uint32_t nbits,
    uint32_t n
) {
    // NIB+FOLD fast path (5.31x standalone): needs blockDim.x == 256 == 2^8 exactly
    // (n a power of two >= 256 so every block is full and the hi-prefix is
    // block-common) and the smem tables within the 48KB no-opt-in limit.
    size_t nib_smem = (size_t)nq * (sizeof(FpExt) + 33 * sizeof(Fp));
    if (nbits >= 8 && n == (1u << nbits) &&
        nib_smem <= 48 * 1024) {
        uint32_t nib_grid = n >> 8;
        w_moments_accumulate_nib_kernel<<<nib_grid, 256, nib_smem>>>(
            moments, z0_sq, z_sq, g, nq, nbits, n
        );
        return CHECK_KERNEL();
    }
    auto [grid, block] = kernel_launch_params(n, 256);
    w_moments_accumulate_kernel<<<grid, block>>>(
        moments, z0_sq, z_sq, g, nq, nbits, n
    );
    return CHECK_KERNEL();
}
