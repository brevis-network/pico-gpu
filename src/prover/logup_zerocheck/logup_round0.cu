// Copyright (c) 2025-2026 Brevis. All Rights Reserved.
// Licensed under the Business Source License 1.1 (BUSL-1.1).
// See LICENSE.md for details.

#include "prover/dag/codec.cuh"
#include "prover/dag/dag_entry.cuh"
#include "prover/logup_zerocheck/eval_config.cuh"
#include "field/fp.h"
#include "field/fpext.h"
#include "prover/frac_ext.cuh"
#include "common/launcher.cuh"
#include "prover/reduce.cuh"
#include "common/utils.cuh"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector_types.h>

// Rule, the rule decoders and the OP_* opcodes live in symbolic_dag.
using namespace symbolic_dag;

namespace logup_round0 {

// Tuned independently of the zerocheck round-0 kernels.
constexpr uint32_t BUFFER_THRESHOLD = 16;
// Work-size cut-over between the two kernel shapes. Below it (cube_size * skip_len small) every coset
// gets its own grid row so the GPU stays busy; at or above it one thread walks all cosets together,
// which shares a single inverse NTT per operand across them.
constexpr uint32_t COSET_PARALLEL_THRESHOLD = 32768; // 2^15

// Walks the interaction DAG once and, for every coset at the same time, produces the weighted numerator
// and denominator totals of this thread's point. Marked __inline__ (not forced) so the compiler can
// weigh inlining against register pressure.
//
// FIRST_COSET_IS_IDENTITY (compile time): the lockstep kernel sets it because coset 0 is unshifted.
// bypass_ntt (run time): the per-coset kernel sets it while it is working on the unshifted coset.
template <uint32_t NUM_COSETS, bool NEEDS_SHMEM, bool FIRST_COSET_IS_IDENTITY = false>
__device__ __inline__ void eval_interaction_dag(
    const NttEvalContext<NUM_COSETS> &ctx,
    const Fp *sel_first, // [NUM_COSETS] first-row selector, one per coset, for this x_idx
    const Fp *sel_last,  // [NUM_COSETS] last-row selector, one per coset, for this x_idx
    uint32_t x_idx,
    const FpExt *__restrict__ numer_wts,
    const FpExt *__restrict__ denom_wts,
    const Rule *__restrict__ dag_rules,
    size_t num_rules,
    FpExt *__restrict__ numer_out, // written: [NUM_COSETS]
    FpExt *__restrict__ denom_out, // written: [NUM_COSETS]
    bool bypass_ntt = false        // true while handling the unshifted coset
) {
#pragma unroll
    for (uint32_t ci = 0; ci < NUM_COSETS; ci++) {
        numer_out[ci] = FpExt(Fp::zero());
        denom_out[ci] = FpExt(Fp::zero());
    }

    for (size_t rule_idx = 0; rule_idx < num_rules; ++rule_idx) {
        Rule cur_rule = dag_rules[rule_idx];
        // Only the header (opcode, flags, first operand) is decoded here; the rest on demand.
        RuleHeader hdr = decode_rule_header(cur_rule);

        Fp a_vals[NUM_COSETS];
        ntt_eval_dag_entry<NUM_COSETS, NEEDS_SHMEM, FIRST_COSET_IS_IDENTITY>(
            a_vals, hdr.x, ctx, sel_first, sel_last, x_idx, bypass_ntt
        );

        Fp out_vals[NUM_COSETS];

        switch (hdr.op) {
        case OP_ADD: {
            // Second operand exists only for two-input ops.
            SourceInfo b_src = decode_y(cur_rule);
            Fp b_vals[NUM_COSETS];
            ntt_eval_dag_entry<NUM_COSETS, NEEDS_SHMEM, FIRST_COSET_IS_IDENTITY>(
                b_vals, b_src, ctx, sel_first, sel_last, x_idx, bypass_ntt
            );
#pragma unroll
            for (uint32_t ci = 0; ci < NUM_COSETS; ci++) {
                out_vals[ci] = a_vals[ci] + b_vals[ci];
            }
            break;
        }
        case OP_SUB: {
            SourceInfo b_src = decode_y(cur_rule);
            Fp b_vals[NUM_COSETS];
            ntt_eval_dag_entry<NUM_COSETS, NEEDS_SHMEM, FIRST_COSET_IS_IDENTITY>(
                b_vals, b_src, ctx, sel_first, sel_last, x_idx, bypass_ntt
            );
#pragma unroll
            for (uint32_t ci = 0; ci < NUM_COSETS; ci++) {
                out_vals[ci] = a_vals[ci] - b_vals[ci];
            }
            break;
        }
        case OP_MUL: {
            SourceInfo b_src = decode_y(cur_rule);
            Fp b_vals[NUM_COSETS];
            ntt_eval_dag_entry<NUM_COSETS, NEEDS_SHMEM, FIRST_COSET_IS_IDENTITY>(
                b_vals, b_src, ctx, sel_first, sel_last, x_idx, bypass_ntt
            );
#pragma unroll
            for (uint32_t ci = 0; ci < NUM_COSETS; ci++) {
                out_vals[ci] = a_vals[ci] * b_vals[ci];
            }
            break;
        }
        case OP_NEG:
#pragma unroll
            for (uint32_t ci = 0; ci < NUM_COSETS; ci++) {
                out_vals[ci] = -a_vals[ci];
            }
            break;
        case OP_VAR:
#pragma unroll
            for (uint32_t ci = 0; ci < NUM_COSETS; ci++) {
                out_vals[ci] = a_vals[ci];
            }
            break;
        case OP_INV:
#pragma unroll
            for (uint32_t ci = 0; ci < NUM_COSETS; ci++) {
                out_vals[ci] = inv(a_vals[ci]);
            }
            break;
        }

        // Keep the node's values if later nodes read them back; the slot is decoded only in that case.
        if (hdr.buffer_result && ctx.buffer_size > 0) {
            uint32_t scratch_slot = decode_z_index(cur_rule);
#ifdef CUDA_DEBUG
            assert(scratch_slot < ctx.buffer_size);
#endif
            // Per-thread scratch is slot-major: [scratch_len][NUM_COSETS].
#pragma unroll
            for (uint32_t ci = 0; ci < NUM_COSETS; ci++) {
                ctx.inter_buffer[scratch_slot * ctx.buffer_stride + ci] = out_vals[ci];
            }
        }

        // Constraint nodes feed the weighted numerator/denominator totals.
        if (hdr.is_constraint) {
            FpExt nw = numer_wts[rule_idx];
            FpExt dw = denom_wts[rule_idx];
#pragma unroll
            for (uint32_t ci = 0; ci < NUM_COSETS; ci++) {
                numer_out[ci] += nw * out_vals[ci];
                denom_out[ci] += dw * out_vals[ci];
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------
// Device kernels
// ---------------------------------------------------------------------------------------------------

// Round-0 interaction kernel, lockstep shape: a thread evaluates every coset for its (x, domain) point.
template <uint32_t NUM_COSETS, bool GLOBAL, bool NEEDS_SHMEM>
__global__ void logup_r0_lockstep_kernel(
    FracExt *__restrict__ block_partials, // [NUM_COSETS][gridDim.x][skip_len]
    const Fp *__restrict__ sel_hypercube, // [3][cube_size]
    const Fp *__restrict__ prep_trace,
    const Fp *const *__restrict__ main_traces,
    const FpExt *__restrict__ eq_hypercube, // [cube_size]
    const Fp *__restrict__ pub_vals,
    const FpExt *__restrict__ numer_wts,
    const FpExt *__restrict__ denom_wts,
    FpExt denom_bias,
    const Rule *__restrict__ dag_rules,
    size_t num_rules,
    uint32_t scratch_len,
    Fp *__restrict__ scratch_global,
    uint32_t skip_len,
    uint32_t cube_size,
    uint32_t rows_total,
    Fp coset_gen
) {
    extern __shared__ char dyn_smem[];
    // Dynamic shared memory: blockDim.x FracExt slots for the per-coset block reduction, followed (only
    // with NEEDS_SHMEM) by blockDim.x Fp of NTT scratch, i.e. one skip_len window per x group.
    FracExt *block_scratch = reinterpret_cast<FracExt *>(dyn_smem);
    Fp *ntt_scratch_base =
        NEEDS_SHMEM ? reinterpret_cast<Fp *>(dyn_smem + blockDim.x * sizeof(FracExt)) : nullptr;

    uint32_t const log_skip = __ffs(skip_len) - 1;
    bool const trivial_skip = (log_skip == 0);

    // Threads sharing an x value share one skip_len-sized NTT window.
    uint32_t const group_in_block = threadIdx.x >> log_skip;
    Fp *ntt_scratch = NEEDS_SHMEM ? (ntt_scratch_base + group_in_block * skip_len) : nullptr;

    // Low log_skip bits of the thread id pick the domain position, the rest pick x.
    uint32_t const gtid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t const domain_pos = gtid & (skip_len - 1);
    uint32_t const first_x = gtid >> log_skip;

    uint32_t const domain_pos_rev = trivial_skip ? 0 : rev_len(domain_pos, log_skip);

    // Lifted-domain factors that turn the hypercube selectors into first/last-row selectors.
    uint32_t const log_h = __ffs(rows_total) - 1;
    uint32_t const log_seg = min(log_skip, log_h);
    uint32_t const seg_len = 1u << log_seg;
    uint32_t const log_step = log_skip - log_seg;

    Fp first_factor[NUM_COSETS];
    Fp last_factor[NUM_COSETS];
    Fp const row_gen = trivial_skip ? Fp::one() : TWO_ADIC_GENERATORS[log_skip - log_step];
    Fp const omega_pos =
        trivial_skip ? Fp::one() : device_ntt::get_twiddle(log_skip, domain_pos);

    // Coset ci is shifted by coset_gen^ci (ci = 0 is the domain itself).
    Fp cur_shift = Fp::one();
#pragma unroll
    for (uint32_t ci = 0; ci < NUM_COSETS; ci++) {
        Fp point = cur_shift * omega_pos;
        Fp omega_lifted = exp_power_of_2(point, log_step);
        first_factor[ci] = avg_gp(omega_lifted, seg_len);
        last_factor[ci] = avg_gp(omega_lifted * row_gen, seg_len);
        cur_shift *= coset_gen; // advance to the next coset's shift
    }

    // Scratch for buffered DAG nodes: registers when small, global memory otherwise.
    Fp reg_scratch[GLOBAL ? 1 : BUFFER_THRESHOLD * NUM_COSETS];
    Fp *scratch_ptr;
    uint32_t scratch_stride;
    if constexpr (GLOBAL) {
        // Global scratch is slot-major across the whole grid: [scratch_len][threads][NUM_COSETS].
        scratch_ptr = scratch_global + gtid * NUM_COSETS;
        scratch_stride = gridDim.x * blockDim.x * NUM_COSETS;
    } else {
        scratch_ptr = reg_scratch;
        scratch_stride = NUM_COSETS;
    }

    FracExt acc_sums[NUM_COSETS];
#pragma unroll
    for (uint32_t ci = 0; ci < NUM_COSETS; ci++) {
        acc_sums[ci] = {FpExt(Fp::zero()), FpExt(Fp::zero())};
    }

    // The launch config makes blockDim.x divisible by skip_len.
    uint32_t const x_step = (gridDim.x * blockDim.x) >> log_skip;

    // Everything in the context except the selectors is fixed across the x loop.
    NttEvalContext<NUM_COSETS> ctx{
        prep_trace,
        main_traces,
        pub_vals,
        scratch_ptr,
        ntt_scratch,
        {}, // filled in right after
        skip_len,
        rows_total,
        scratch_stride,
        scratch_len,
        domain_pos,
    };
    // Per-coset twist for this domain position; the unshifted coset needs none.
    ctx.omega_shifts[0] = Fp::one();
    cur_shift = coset_gen;
#pragma unroll
    for (uint32_t ci = 1; ci < NUM_COSETS; ci++) {
        ctx.omega_shifts[ci] = pow(cur_shift, domain_pos_rev);
        cur_shift *= coset_gen;
    }

    // Grid-stride loop over hypercube points.
    for (uint32_t x_idx = first_x; x_idx < cube_size; x_idx += x_step) {
        Fp sel_first[NUM_COSETS];
        Fp sel_last[NUM_COSETS];
#pragma unroll
        for (uint32_t ci = 0; ci < NUM_COSETS; ci++) {
            sel_first[ci] = first_factor[ci] * sel_hypercube[x_idx];
            sel_last[ci] = last_factor[ci] * sel_hypercube[2 * cube_size + x_idx];
        }

        FpExt numer_vals[NUM_COSETS];
        FpExt denom_vals[NUM_COSETS];
        eval_interaction_dag<NUM_COSETS, NEEDS_SHMEM, /*FIRST_COSET_IS_IDENTITY=*/true>(
            ctx, sel_first, sel_last, x_idx,
            numer_wts, denom_wts, dag_rules, num_rules, numer_vals, denom_vals
        );

        FpExt eq_w = eq_hypercube[x_idx];
#pragma unroll
        for (uint32_t ci = 0; ci < NUM_COSETS; ci++) {
            acc_sums[ci].p += eq_w * numer_vals[ci];
            acc_sums[ci].q += eq_w * (denom_vals[ci] + denom_bias);
        }
    }

    // Reduce coset by coset so a single blockDim.x-sized shared array suffices.
#pragma unroll
    for (uint32_t ci = 0; ci < NUM_COSETS; ci++) {
        block_scratch[threadIdx.x] = acc_sums[ci];
        __syncthreads();

        if (threadIdx.x < skip_len) {
            FracExt col_total = block_scratch[threadIdx.x];
            for (uint32_t grp = 1; grp < (blockDim.x >> log_skip); ++grp) {
                auto src_idx = (grp << log_skip) + threadIdx.x;
                col_total.p += block_scratch[src_idx].p;
                col_total.q += block_scratch[src_idx].q;
            }
            // Partials are [block][NUM_COSETS * skip_len], the row layout final_reduce_block_sums consumes.
            block_partials[blockIdx.x * NUM_COSETS * skip_len + ci * skip_len + domain_pos] =
                col_total;
        }
        __syncthreads();
    }
}

// Round-0 interaction kernel, per-coset shape: blockIdx.y selects the coset and each thread evaluates
// only that one (the DAG walker is instantiated with a single coset). Chosen for small workloads, where
// the extra grid dimension buys occupancy.
template <bool GLOBAL, bool NEEDS_SHMEM>
__global__ void logup_r0_per_coset_kernel(
    FracExt *__restrict__ block_partials, // [n_blocks][coset_count * skip_len]
    const Fp *__restrict__ sel_hypercube, // [3][cube_size]
    const Fp *__restrict__ prep_trace,
    const Fp *const *__restrict__ main_traces,
    const FpExt *__restrict__ eq_hypercube, // [cube_size]
    const Fp *__restrict__ pub_vals,
    const FpExt *__restrict__ numer_wts,
    const FpExt *__restrict__ denom_wts,
    FpExt denom_bias,
    const Rule *__restrict__ dag_rules,
    size_t num_rules,
    uint32_t scratch_len,
    Fp *__restrict__ scratch_global,
    uint32_t skip_len,
    uint32_t cube_size,
    uint32_t rows_total,
    Fp coset_gen
) {
    extern __shared__ char dyn_smem[];
    FracExt *block_scratch = reinterpret_cast<FracExt *>(dyn_smem);
    Fp *ntt_scratch_base =
        NEEDS_SHMEM ? reinterpret_cast<Fp *>(dyn_smem + blockDim.x * sizeof(FracExt)) : nullptr;

    uint32_t const log_skip = __ffs(skip_len) - 1;
    uint32_t const coset_count = gridDim.y;

    // Threads sharing an x value share one skip_len-sized NTT window.
    uint32_t const group_in_block = threadIdx.x >> log_skip;
    Fp *ntt_scratch = NEEDS_SHMEM ? (ntt_scratch_base + group_in_block * skip_len) : nullptr;

    // Low log_skip bits of the thread id pick the domain position, the rest pick x.
    uint32_t const gtid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t const domain_pos = gtid & (skip_len - 1);
    uint32_t const first_x = gtid >> log_skip;

    // Unlike the lockstep kernel, the coset comes from the grid.
    uint32_t const coset_id = blockIdx.y;
    bool const on_identity = (coset_id == 0);

    bool const trivial_skip = (log_skip == 0);
    uint32_t const domain_pos_rev = trivial_skip ? 0 : rev_len(domain_pos, log_skip);

    uint32_t const log_h = __ffs(rows_total) - 1;
    uint32_t const log_seg = min(log_skip, log_h);
    uint32_t const seg_len = 1u << log_seg;
    uint32_t const log_step = log_skip - log_seg;

    Fp const row_gen = trivial_skip ? Fp::one() : TWO_ADIC_GENERATORS[log_skip - log_step];
    Fp const omega_pos =
        trivial_skip ? Fp::one() : device_ntt::get_twiddle(log_skip, domain_pos);

    // Same factors as the lockstep kernel, specialised to the shift coset_gen^coset_id.
    Fp const cur_shift = on_identity ? Fp::one() : pow(coset_gen, coset_id);
    Fp const point = on_identity ? omega_pos : (cur_shift * omega_pos);
    Fp const omega_lifted = exp_power_of_2(point, log_step);
    Fp const first_factor = avg_gp(omega_lifted, seg_len);
    Fp const last_factor = avg_gp(omega_lifted * row_gen, seg_len);
    Fp const pos_shift = on_identity ? Fp::one() : pow(cur_shift, domain_pos_rev);

    // Node scratch for one coset per thread.
    Fp reg_scratch[GLOBAL ? 1 : BUFFER_THRESHOLD];
    Fp *scratch_ptr;
    uint32_t scratch_stride;
    if constexpr (GLOBAL) {
        // Slot-major over every thread of every coset row of the grid.
        uint32_t flat_tid = coset_id * gridDim.x * blockDim.x + gtid;
        scratch_ptr = scratch_global + flat_tid;
        scratch_stride = gridDim.x * gridDim.y * blockDim.x;
    } else {
        scratch_ptr = reg_scratch;
        scratch_stride = 1;
    }

    FracExt acc_sum = {FpExt(Fp::zero()), FpExt(Fp::zero())};

    uint32_t const x_step = (gridDim.x * blockDim.x) >> log_skip;

    // One-coset context; nothing in it changes inside the x loop.
    NttEvalContext<1> ctx{
        prep_trace,
        main_traces,
        pub_vals,
        scratch_ptr,
        ntt_scratch,
        {pos_shift},
        skip_len,
        rows_total,
        scratch_stride,
        scratch_len,
        domain_pos,
    };

    // Grid-stride loop over hypercube points.
    for (uint32_t x_idx = first_x; x_idx < cube_size; x_idx += x_step) {
        Fp sel_first = first_factor * sel_hypercube[x_idx];
        Fp sel_last = last_factor * sel_hypercube[2 * cube_size + x_idx];

        FpExt numer_vals[1];
        FpExt denom_vals[1];
        // Instantiated for one coset; whether that coset is the unshifted one is only known at run time,
        // so it is passed as bypass_ntt rather than through the template flag.
        eval_interaction_dag</*NUM_COSETS=*/1, NEEDS_SHMEM, /*FIRST_COSET_IS_IDENTITY=*/false>(
            ctx,
            &sel_first,
            &sel_last,
            x_idx,
            numer_wts,
            denom_wts,
            dag_rules,
            num_rules,
            numer_vals,
            denom_vals,
            on_identity
        );

        FpExt eq_w = eq_hypercube[x_idx];
        acc_sum.p += eq_w * numer_vals[0];
        acc_sum.q += eq_w * (denom_vals[0] + denom_bias);
    }

    // Block reduction for this block's coset.
    block_scratch[threadIdx.x] = acc_sum;
    __syncthreads();

    if (threadIdx.x < skip_len) {
        FracExt col_total = block_scratch[threadIdx.x];
        for (uint32_t grp = 1; grp < (blockDim.x >> log_skip); ++grp) {
            auto src_idx = (grp << log_skip) + threadIdx.x;
            col_total.p += block_scratch[src_idx].p;
            col_total.q += block_scratch[src_idx].q;
        }
        // Same [block][coset_count * skip_len] partials layout as the lockstep kernel, so the final
        // reduction is shared between both shapes.
        block_partials[blockIdx.x * coset_count * skip_len + coset_id * skip_len + domain_pos] =
            col_total;
    }
}

// ---------------------------------------------------------------------------------------------------
// Host entry points
// ---------------------------------------------------------------------------------------------------

constexpr uint32_t MAX_THREADS = 128;

// True when the workload is small enough for the per-coset kernel shape.
inline bool prefer_per_coset(uint32_t cube_size, uint32_t skip_len) {
    return (cube_size * skip_len) < COSET_PARALLEL_THRESHOLD;
}

// Size query, launches nothing: number of FracExt entries the caller must allocate for block_partials.
extern "C" size_t _kb_logup_r0_temp_sums_buffer_size(
    uint32_t scratch_len,
    uint32_t skip_len,
    uint32_t cube_size,
    uint32_t coset_count,
    size_t temp_byte_cap
) {
    // The size works out the same for either kernel shape; the flag only selects its launch geometry.
    return DISPATCH_BOOL(
        round0_config_impl::temp_sums_buffer_size,
        prefer_per_coset(cube_size, skip_len),
        scratch_len,
        skip_len,
        cube_size,
        coset_count,
        temp_byte_cap,
        BUFFER_THRESHOLD,
        MAX_THREADS
    );
}

extern "C" size_t _kb_logup_r0_intermediates_buffer_size(
    uint32_t scratch_len,
    uint32_t skip_len,
    uint32_t cube_size,
    uint32_t coset_count,
    size_t temp_byte_cap
) {
    // The size works out the same for either kernel shape; the flag only selects its launch geometry.
    return DISPATCH_BOOL(
        round0_config_impl::intermediates_buffer_size,
        prefer_per_coset(cube_size, skip_len),
        scratch_len,
        skip_len,
        cube_size,
        coset_count,
        temp_byte_cap,
        BUFFER_THRESHOLD,
        MAX_THREADS
    );
}

template <uint32_t NUM_COSETS, bool GLOBAL, bool NEEDS_SHMEM>
int run_logup_lockstep(
    FracExt *block_partials,
    FracExt *out_sums,
    const Fp *sel_hypercube,
    const Fp *prep_trace,
    const Fp *const *main_traces,
    const FpExt *eq_hypercube,
    const Fp *pub_vals,
    const FpExt *numer_wts,
    const FpExt *denom_wts,
    FpExt denom_bias,
    const Rule *dag_rules,
    size_t num_rules,
    uint32_t scratch_len,
    Fp *scratch_global,
    uint32_t skip_len,
    uint32_t cube_size,
    uint32_t rows_total,
    Fp coset_gen,
    size_t temp_byte_cap
) {
    auto [grid_dim, block_dim] = coset_round0_config::eval_constraints_launch_params(
        scratch_len, skip_len, cube_size, NUM_COSETS, temp_byte_cap, BUFFER_THRESHOLD, MAX_THREADS
    );

    size_t sum_bytes = sizeof(FracExt) * block_dim.x;
    // NTT scratch: a skip_len window for each x group of the block.
    size_t ntt_bytes = NEEDS_SHMEM ? sizeof(Fp) * block_dim.x : 0;
    size_t smem_bytes = sum_bytes + ntt_bytes;

    logup_r0_lockstep_kernel<NUM_COSETS, GLOBAL, NEEDS_SHMEM>
        <<<grid_dim, block_dim, smem_bytes>>>(
            block_partials,
            sel_hypercube,
            prep_trace,
            main_traces,
            eq_hypercube,
            pub_vals,
            numer_wts,
            denom_wts,
            denom_bias,
            dag_rules,
            num_rules,
            scratch_len,
            scratch_global,
            skip_len,
            cube_size,
            rows_total,
            coset_gen
        );

    int status = CHECK_KERNEL();
    if (status != 0) {
        return status;
    }

    auto n_blocks = grid_dim.x;
    auto [red_grid, red_block] = kernel_launch_params(n_blocks);
    unsigned int red_warps = div_ceil(red_block.x, WARP_SIZE);
    size_t red_smem = std::max(1u, red_warps) * sizeof(FpExt);
    auto out_len = NUM_COSETS * skip_len;
    reduce::final_reduce_block_sums<<<2 * out_len, red_block, red_smem>>>(
        reinterpret_cast<FpExt *>(block_partials), reinterpret_cast<FpExt *>(out_sums), n_blocks
    );
    return CHECK_KERNEL();
}

// Maps a runtime coset count in 1..4 plus the two flags onto run_logup_lockstep instantiations.
DEFINE_DISPATCH_N_B1_B2(dispatch_logup_lockstep, run_logup_lockstep, 4)

// Launches the per-coset kernel (one grid row per coset) and then the cross-block reduction.
template <bool GLOBAL, bool NEEDS_SHMEM>
int run_logup_per_coset(
    FracExt *block_partials,
    FracExt *out_sums,
    const Fp *sel_hypercube,
    const Fp *prep_trace,
    const Fp *const *main_traces,
    const FpExt *eq_hypercube,
    const Fp *pub_vals,
    const FpExt *numer_wts,
    const FpExt *denom_wts,
    FpExt denom_bias,
    const Rule *dag_rules,
    size_t num_rules,
    uint32_t scratch_len,
    Fp *scratch_global,
    uint32_t skip_len,
    uint32_t cube_size,
    uint32_t rows_total,
    uint32_t coset_count,
    Fp coset_gen,
    size_t temp_byte_cap
) {
    auto [grid_dim, block_dim] = coset_parallel_round0_config::eval_constraints_launch_params(
        scratch_len, skip_len, cube_size, coset_count, temp_byte_cap, BUFFER_THRESHOLD, MAX_THREADS
    );

    size_t sum_bytes = sizeof(FracExt) * block_dim.x;
    size_t ntt_bytes = NEEDS_SHMEM ? sizeof(Fp) * block_dim.x : 0;
    size_t smem_bytes = sum_bytes + ntt_bytes;

    logup_r0_per_coset_kernel<GLOBAL, NEEDS_SHMEM>
        <<<grid_dim, block_dim, smem_bytes>>>(
            block_partials,
            sel_hypercube,
            prep_trace,
            main_traces,
            eq_hypercube,
            pub_vals,
            numer_wts,
            denom_wts,
            denom_bias,
            dag_rules,
            num_rules,
            scratch_len,
            scratch_global,
            skip_len,
            cube_size,
            rows_total,
            coset_gen
        );
    int status = CHECK_KERNEL();
    if (status != 0) {
        return status;
    }

    auto n_blocks = grid_dim.x;
    auto [red_grid, red_block] = kernel_launch_params(n_blocks);
    unsigned int red_warps = div_ceil(red_block.x, WARP_SIZE);
    size_t red_smem = std::max(1u, red_warps) * sizeof(FpExt);
    auto out_len = coset_count * skip_len;
    reduce::final_reduce_block_sums<<<2 * out_len, red_block, red_smem>>>(
        reinterpret_cast<FpExt *>(block_partials), reinterpret_cast<FpExt *>(out_sums), n_blocks
    );

    return CHECK_KERNEL();
}

extern "C" int _kb_logup_bary_eval_interactions_round0(
    FracExt *block_partials, // per-block partials, [blocks][coset_count * skip_len]
    FracExt *out_sums,       // [coset_count * skip_len]
    const Fp *sel_hypercube, // [3][cube_size]
    const Fp *prep_trace,
    const Fp *const *main_traces,
    const FpExt *eq_hypercube, // [cube_size]
    const Fp *pub_vals,
    const FpExt *numer_wts,
    const FpExt *denom_wts,
    FpExt denom_bias,
    const Rule *dag_rules,
    size_t num_rules,
    uint32_t scratch_len,
    Fp *scratch_global,
    uint32_t skip_len,
    uint32_t cube_size,
    uint32_t rows_total,
    uint32_t coset_count,
    Fp coset_gen,
    size_t temp_byte_cap
) {
    bool use_global = scratch_len > BUFFER_THRESHOLD;
    bool per_coset = prefer_per_coset(cube_size, skip_len);
    if (!per_coset && coset_count > 4) {
        return cudaErrorInvalidValue;
    }
    if (skip_len == 1) {
        if (per_coset) {
            return use_global ? run_logup_per_coset<true, false>(
                                   block_partials,
                                   out_sums,
                                   sel_hypercube,
                                   prep_trace,
                                   main_traces,
                                   eq_hypercube,
                                   pub_vals,
                                   numer_wts,
                                   denom_wts,
                                   denom_bias,
                                   dag_rules,
                                   num_rules,
                                   scratch_len,
                                   scratch_global,
                                   skip_len,
                                   cube_size,
                                   rows_total,
                                   coset_count,
                                   coset_gen,
                                   temp_byte_cap
                               )
                             : run_logup_per_coset<false, false>(
                                   block_partials,
                                   out_sums,
                                   sel_hypercube,
                                   prep_trace,
                                   main_traces,
                                   eq_hypercube,
                                   pub_vals,
                                   numer_wts,
                                   denom_wts,
                                   denom_bias,
                                   dag_rules,
                                   num_rules,
                                   scratch_len,
                                   scratch_global,
                                   skip_len,
                                   cube_size,
                                   rows_total,
                                   coset_count,
                                   coset_gen,
                                   temp_byte_cap
                               );
        }
        return dispatch_logup_lockstep(
            coset_count,
            use_global,
            false,
            block_partials,
            out_sums,
            sel_hypercube,
            prep_trace,
            main_traces,
            eq_hypercube,
            pub_vals,
            numer_wts,
            denom_wts,
            denom_bias,
            dag_rules,
            num_rules,
            scratch_len,
            scratch_global,
            skip_len,
            cube_size,
            rows_total,
            coset_gen,
            temp_byte_cap
        );
    }

    bool want_smem = skip_len > WARP_SIZE;

    // Small problems go to the per-coset kernel, larger ones to the lockstep kernel.
    if (per_coset) {
        if (use_global) {
            if (want_smem) {
                return run_logup_per_coset<true, true>(
                    block_partials,
                    out_sums,
                    sel_hypercube,
                    prep_trace,
                    main_traces,
                    eq_hypercube,
                    pub_vals,
                    numer_wts,
                    denom_wts,
                    denom_bias,
                    dag_rules,
                    num_rules,
                    scratch_len,
                    scratch_global,
                    skip_len,
                    cube_size,
                    rows_total,
                    coset_count,
                    coset_gen,
                    temp_byte_cap
                );
            } else {
                return run_logup_per_coset<true, false>(
                    block_partials,
                    out_sums,
                    sel_hypercube,
                    prep_trace,
                    main_traces,
                    eq_hypercube,
                    pub_vals,
                    numer_wts,
                    denom_wts,
                    denom_bias,
                    dag_rules,
                    num_rules,
                    scratch_len,
                    scratch_global,
                    skip_len,
                    cube_size,
                    rows_total,
                    coset_count,
                    coset_gen,
                    temp_byte_cap
                );
            }
        } else {
            if (want_smem) {
                return run_logup_per_coset<false, true>(
                    block_partials,
                    out_sums,
                    sel_hypercube,
                    prep_trace,
                    main_traces,
                    eq_hypercube,
                    pub_vals,
                    numer_wts,
                    denom_wts,
                    denom_bias,
                    dag_rules,
                    num_rules,
                    scratch_len,
                    scratch_global,
                    skip_len,
                    cube_size,
                    rows_total,
                    coset_count,
                    coset_gen,
                    temp_byte_cap
                );
            } else {
                return run_logup_per_coset<false, false>(
                    block_partials,
                    out_sums,
                    sel_hypercube,
                    prep_trace,
                    main_traces,
                    eq_hypercube,
                    pub_vals,
                    numer_wts,
                    denom_wts,
                    denom_bias,
                    dag_rules,
                    num_rules,
                    scratch_len,
                    scratch_global,
                    skip_len,
                    cube_size,
                    rows_total,
                    coset_count,
                    coset_gen,
                    temp_byte_cap
                );
            }
        }
    } else {
        return dispatch_logup_lockstep(
            coset_count,
            use_global,
            want_smem,
            block_partials,
            out_sums,
            sel_hypercube,
            prep_trace,
            main_traces,
            eq_hypercube,
            pub_vals,
            numer_wts,
            denom_wts,
            denom_bias,
            dag_rules,
            num_rules,
            scratch_len,
            scratch_global,
            skip_len,
            cube_size,
            rows_total,
            coset_gen,
            temp_byte_cap
        );
    }
}

} // namespace logup_round0
