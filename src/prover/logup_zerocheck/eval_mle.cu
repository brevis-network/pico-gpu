// Copyright (c) 2025-2026 Brevis. All Rights Reserved.
// Licensed under the Business Source License 1.1 (BUSL-1.1).
// See LICENSE.md for details.

#include "prover/dag/codec.cuh"
#include "prover/logup_zerocheck/eval_config.cuh"
#include "field/fp.h"
#include "field/fpext.h"
#include "prover/frac_ext.cuh"
#include "common/launcher.cuh"
#include "prover/matrix.cuh"
#include "prover/reduce.cuh"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <stdio.h>
#include <vector_types.h>

namespace logup_zerocheck_mle {

__device__ __forceinline__ FpExt load_dag_operand(
    const SourceInfo &entry,
    uint32_t row_idx,
    const FpExt *__restrict__ sel_cols,
    const MainMatrixPtrs<FpExt> prep_mat,
    const MainMatrixPtrs<FpExt> *__restrict__ main_mats,
    const Fp *__restrict__ pub_vals,
    const FpExt *__restrict__ scratch_buf,
    uint32_t scratch_stride,
    const uint32_t n_rows,
    const FpExt *__restrict__ challenge_vals = nullptr
) {
    switch (entry.type) {
    case ENTRY_PREPROCESSED: {
#ifdef CUDA_DEBUG
        assert(prep_mat.data);
#endif
        const auto mat_span = n_rows * prep_mat.air_width;
        const FpExt *__restrict__ mat_base = prep_mat.data + mat_span * entry.offset;
        const FpExt *__restrict__ col = mat_base + n_rows * entry.index;
        return col[row_idx];
    }
    case ENTRY_MAIN: {
        auto part_mat = main_mats[entry.part];
        const auto mat_span = n_rows * part_mat.air_width;
        const FpExt *__restrict__ mat_base = part_mat.data + mat_span * entry.offset;
        const FpExt *__restrict__ col = mat_base + n_rows * entry.index;
        return col[row_idx];
    }
    case SRC_INTERMEDIATE:
        return scratch_buf[entry.index * scratch_stride];
    case SRC_IS_FIRST: {
        const FpExt *__restrict__ col = sel_cols;
        return FpExt(col[row_idx]);
    }
    case SRC_IS_LAST: {
        const FpExt *__restrict__ col = sel_cols + 2 * n_rows;
        return FpExt(col[row_idx]);
    }
    case SRC_IS_TRANSITION: {
        const FpExt *__restrict__ col = sel_cols + n_rows;
        return FpExt(col[row_idx]);
    }
    case ENTRY_CHALLENGE:
#ifdef CUDA_DEBUG
        assert(challenge_vals);
#endif
        return challenge_vals[entry.index];
    case ENTRY_PUBLIC:
        return FpExt(pub_vals[entry.index]);
    case SRC_CONSTANT:
        return FpExt(Fp(entry.index));
    default:
        assert(false);
    }
    return FpExt(Fp::zero());
}

// ---------------------------------------------------------------------------------------------------
// Device kernels
// ---------------------------------------------------------------------------------------------------

constexpr uint32_t ZEROCHECK_BUFFER_THRESHOLD = 16;
constexpr uint32_t LOGUP_BUFFER_THRESHOLD = 10;

template <bool GLOBAL>
__global__ void zerocheck_mle_eval_kernel(
    FpExt *__restrict__ block_partials,
    const FpExt *__restrict__ eq_weights,
    const FpExt *__restrict__ sel_cols,
    const MainMatrixPtrs<FpExt> prep_mat,
    const MainMatrixPtrs<FpExt> *__restrict__ main_mats,
    const FpExt *__restrict__ lambda_powers,
    const Fp *__restrict__ pub_vals,
    const Rule *__restrict__ dag_rules,
    size_t num_rules,
    const size_t *__restrict__ constraint_nodes,
    size_t num_used,
    size_t num_lambdas,
    uint32_t scratch_len,
    FpExt *__restrict__ scratch_global,
    uint32_t y_count
) {
    extern __shared__ char dyn_smem[];
    FpExt *warp_scratch = (FpExt *)dyn_smem;

    uint32_t x_count = gridDim.y;
    uint32_t x_idx = blockIdx.y;
    uint32_t y_idx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t y_span = blockDim.x * gridDim.x;
    bool const in_range = (y_idx < y_count);

    FpExt weighted(Fp::zero());

    if (in_range) {
        FpExt reg_scratch[ZEROCHECK_BUFFER_THRESHOLD];
        FpExt *scratch_buf;
        uint32_t scratch_stride;
        if constexpr (GLOBAL) {
            uint32_t slot_base = x_idx * y_span + y_idx;
            uint32_t slot_step = y_span * x_count;
            scratch_buf = scratch_global + slot_base;
            scratch_stride = slot_step;
        } else {
            scratch_buf = reg_scratch;
            scratch_stride = 1;
        }

        uint32_t n_rows = x_count * y_count;
        uint32_t row_idx = x_idx * y_count + y_idx;
        uint32_t pow_idx = 0;

#define ZEROCHECK_EVAL_ARGS                                                                        \
    row_idx, sel_cols, prep_mat, main_mats, pub_vals, scratch_buf, scratch_stride, n_rows

        for (size_t rule_idx = 0; rule_idx < num_rules; ++rule_idx) {
            Rule cur_rule = dag_rules[rule_idx];
            DecodedRule dec = decode_rule(cur_rule);

            FpExt a_val = load_dag_operand(dec.x, ZEROCHECK_EVAL_ARGS);
            FpExt node_out;
            switch (dec.op) {
            case OP_ADD: {
                FpExt b_val = load_dag_operand(dec.y, ZEROCHECK_EVAL_ARGS);
                node_out = a_val + b_val;
                break;
            }
            case OP_SUB: {
                FpExt b_val = load_dag_operand(dec.y, ZEROCHECK_EVAL_ARGS);
                node_out = a_val - b_val;
                break;
            }
            case OP_MUL: {
                FpExt b_val = load_dag_operand(dec.y, ZEROCHECK_EVAL_ARGS);
                node_out = a_val * b_val;
                break;
            }
            case OP_NEG:
                node_out = -a_val;
                break;
            case OP_VAR:
                node_out = a_val;
                break;
            default:
                assert(false);
            }

            if (dec.buffer_result && scratch_len > 0) {
                if constexpr (GLOBAL) {
                    scratch_buf[dec.z_index * scratch_stride] = node_out;
                } else {
                    reg_scratch[dec.z_index] = node_out;
                }
            }

            if (dec.is_constraint) {
                while (pow_idx < num_lambdas && pow_idx < num_used &&
                       constraint_nodes[pow_idx] == rule_idx) {
                    FpExt coeff = lambda_powers[pow_idx];
                    pow_idx++;
                    weighted += coeff * node_out;
                }
            }
        }
        weighted *= eq_weights[y_idx];
    }

    FpExt block_total = reduce::block_reduce_sum(weighted, warp_scratch);
    if (threadIdx.x == 0) {
        block_partials[blockIdx.x * x_count + x_idx] = block_total;
    }
}

template <bool GLOBAL>
__global__ void logup_mle_eval_kernel(
    FracExt *__restrict__ block_partials,
    const FpExt *__restrict__ eq_weights,
    const FpExt *__restrict__ sel_cols,
    const MainMatrixPtrs<FpExt> prep_mat,
    const MainMatrixPtrs<FpExt> *__restrict__ main_mats,
    const FpExt *__restrict__ challenge_vals,
    const FpExt *__restrict__ eq_pair_weights,
    const Fp *__restrict__ pub_vals,
    const Rule *__restrict__ dag_rules,
    const size_t *__restrict__ constraint_nodes,
    const uint32_t *__restrict__ pair_ids,
    size_t num_used,
    uint32_t scratch_len,
    FpExt *__restrict__ scratch_global,
    uint32_t y_count
) {
    extern __shared__ char dyn_smem[];
    FpExt *warp_scratch = reinterpret_cast<FpExt *>(dyn_smem);

    uint32_t x_count = gridDim.y;
    uint32_t x_idx = blockIdx.y;
    uint32_t y_idx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t y_span = blockDim.x * gridDim.x;
    bool const in_range = (y_idx < y_count);

    FpExt numer_acc = FpExt(Fp::zero());
    FpExt denom_acc = FpExt(Fp::zero());

    if (in_range) {
        FpExt reg_scratch[LOGUP_BUFFER_THRESHOLD];
        FpExt *scratch_buf;
        uint32_t scratch_stride;
        if constexpr (GLOBAL) {
            uint32_t slot_base = x_idx * y_span + y_idx;
            uint32_t slot_step = y_span * x_count;
            scratch_buf = scratch_global + slot_base;
            scratch_stride = slot_step;
        } else {
            scratch_buf = reg_scratch;
            scratch_stride = 1;
        }

        uint32_t n_rows = x_count * y_count;
        uint32_t row_idx = x_idx * y_count + y_idx;
        size_t n_done = 0;

#define LOGUP_EVAL_ARGS                                                                            \
    row_idx, sel_cols, prep_mat, main_mats, pub_vals, scratch_buf, scratch_stride, n_rows,       \
        challenge_vals

        // constraint_nodes lists the numerator/denominator nodes in interaction order.
        for (size_t use_pos = 0; use_pos < num_used; ++use_pos) {
            size_t target = constraint_nodes[use_pos];
            FpExt node_out(Fp::zero());

            if (target < n_done) {
                // Already computed earlier in this thread: read it back (or re-read a plain variable).
                Rule cur_rule = dag_rules[target];
                DecodedRule dec = decode_rule(cur_rule);
                if (dec.op == OP_VAR) {
                    node_out = load_dag_operand(dec.x, LOGUP_EVAL_ARGS);
                } else if (scratch_len > 0 && dec.buffer_result) {
                    if constexpr (GLOBAL) {
                        node_out = scratch_buf[dec.z_index * scratch_stride];
                    } else {
                        node_out = reg_scratch[dec.z_index];
                    }
                }
            } else {
                // Not reached yet: advance the DAG walk up to and including the target node.
                for (; n_done <= target; ++n_done) {
                    Rule cur_rule = dag_rules[n_done];
                    DecodedRule dec = decode_rule(cur_rule);

                    FpExt a_val = load_dag_operand(dec.x, LOGUP_EVAL_ARGS);
                    FpExt node_val;
                    switch (dec.op) {
                    case OP_ADD: {
                        FpExt b_val = load_dag_operand(dec.y, LOGUP_EVAL_ARGS);
                        node_val = a_val + b_val;
                        break;
                    }
                    case OP_SUB: {
                        FpExt b_val = load_dag_operand(dec.y, LOGUP_EVAL_ARGS);
                        node_val = a_val - b_val;
                        break;
                    }
                    case OP_MUL: {
                        FpExt b_val = load_dag_operand(dec.y, LOGUP_EVAL_ARGS);
                        node_val = a_val * b_val;
                        break;
                    }
                    case OP_NEG:
                        node_val = -a_val;
                        break;
                    case OP_VAR:
                        node_val = a_val;
                        break;
                    default:
                        assert(false);
                    }

                    if (dec.buffer_result && scratch_len > 0) {
                        if constexpr (GLOBAL) {
                            scratch_buf[dec.z_index * scratch_stride] = node_val;
                        } else {
                            reg_scratch[dec.z_index] = node_val;
                        }
                    }

                    if (n_done == target) {
                        node_out = node_val;
                    }
                }
            }

            // pair_id packs (interaction << 1) | is_denominator; the high part indexes the eq weights.
            uint32_t pair_id = pair_ids[use_pos];
            node_out *= eq_pair_weights[pair_id >> 1];

            if (pair_id & 1) {
                denom_acc += node_out;
            } else {
                numer_acc += node_out;
            }
        }

        FpExt eq_w = eq_weights[y_idx];
        numer_acc *= eq_w;
        denom_acc *= eq_w;
    }

    FpExt numer_total = reduce::block_reduce_sum(numer_acc, warp_scratch);
    __syncthreads(); // warp_scratch is reused by the second reduction
    FpExt denom_total = reduce::block_reduce_sum(denom_acc, warp_scratch);
    if (threadIdx.x == 0) {
        block_partials[blockIdx.x * x_count + x_idx] = {numer_total, denom_total};
    }
}

// ---------------------------------------------------------------------------------------------------
// Host entry points
// ---------------------------------------------------------------------------------------------------
constexpr uint32_t MAX_THREADS = 128;

// Size query, launches nothing: FpExt entries needed for the zerocheck per-block partials.
extern "C" size_t _kb_zerocheck_mle_temp_sums_buffer_size(uint32_t x_count, uint32_t y_count) {
    return mle_rounds_config::temp_sums_buffer_size(x_count, y_count, MAX_THREADS);
}

// Global node-scratch size, counted in FpExt.
extern "C" size_t _kb_zerocheck_mle_intermediates_buffer_size(
    uint32_t scratch_len,
    uint32_t x_count,
    uint32_t y_count
) {
    return mle_rounds_config::intermediates_buffer_size(
        scratch_len, x_count, y_count, ZEROCHECK_BUFFER_THRESHOLD, MAX_THREADS
    );
}

extern "C" int _kb_zerocheck_eval_mle(
    FpExt *block_partials,
    FpExt *final_out,
    const FpExt *eq_weights,
    const FpExt *sel_cols,
    const MainMatrixPtrs<FpExt> prep_mat,
    const MainMatrixPtrs<FpExt> *main_mats,
    const FpExt *lambda_powers,
    const Fp *pub_vals,
    const Rule *dag_rules,
    size_t num_rules,
    const size_t *constraint_nodes,
    size_t num_used,
    size_t num_lambdas,
    uint32_t scratch_len,
    FpExt *scratch_global,
    uint32_t y_count,
    uint32_t x_count
) {
    if (!valid_grid_y_dim(x_count)) {
        return cudaErrorInvalidValue;
    }
    auto [grid_dim, block_dim] =
        mle_rounds_config::eval_constraints_launch_params(x_count, y_count, MAX_THREADS);
    size_t smem_bytes = div_ceil(block_dim.x, WARP_SIZE) * sizeof(FpExt);

#define ZEROCHECK_KERNEL_ARGS                                                                      \
    block_partials, eq_weights, sel_cols, prep_mat, main_mats, lambda_powers, pub_vals, dag_rules,      \
        num_rules, constraint_nodes, num_used, num_lambdas, scratch_len, scratch_global, y_count

    if (scratch_len > ZEROCHECK_BUFFER_THRESHOLD) {
        zerocheck_mle_eval_kernel<true><<<grid_dim, block_dim, smem_bytes>>>(ZEROCHECK_KERNEL_ARGS);
    } else {
        zerocheck_mle_eval_kernel<false><<<grid_dim, block_dim, smem_bytes>>>(ZEROCHECK_KERNEL_ARGS);
    }
    int status = CHECK_KERNEL();
    if (status != 0)
        return status;

    // Second pass: fold the per-block partials into final_out.
    auto n_blocks = grid_dim.x;
    auto [red_grid, red_block] = kernel_launch_params(n_blocks);
    unsigned int red_warps = (red_block.x + WARP_SIZE - 1) / WARP_SIZE;
    size_t red_smem = std::max(1u, red_warps) * sizeof(FpExt);
    reduce::final_reduce_block_sums<<<x_count, red_block, red_smem>>>(
        block_partials, final_out, n_blocks
    );
    return CHECK_KERNEL();
}

// Size query, launches nothing: FracExt entries needed for the logup per-block partials.
extern "C" size_t _kb_logup_mle_temp_sums_buffer_size(uint32_t x_count, uint32_t y_count) {
    return mle_rounds_config::temp_sums_buffer_size(x_count, y_count, MAX_THREADS);
}

// Global node-scratch size, counted in FpExt.
extern "C" size_t _kb_logup_mle_intermediates_buffer_size(
    uint32_t scratch_len,
    uint32_t x_count,
    uint32_t y_count
) {
    return mle_rounds_config::intermediates_buffer_size(
        scratch_len, x_count, y_count, LOGUP_BUFFER_THRESHOLD, MAX_THREADS
    );
}

extern "C" int _kb_logup_eval_mle(
    FracExt *block_partials,
    FracExt *final_out,
    const FpExt *eq_weights,
    const FpExt *sel_cols,
    const MainMatrixPtrs<FpExt> prep_mat,
    const MainMatrixPtrs<FpExt> *main_mats,
    const FpExt *challenge_vals,
    const FpExt *eq_pair_weights,
    const Fp *pub_vals,
    const Rule *dag_rules,
    const size_t *constraint_nodes,
    const uint32_t *pair_ids,
    size_t num_used,
    uint32_t scratch_len,
    FpExt *scratch_global,
    uint32_t y_count,
    uint32_t x_count
) {
    if (!valid_grid_y_dim(x_count)) {
        return cudaErrorInvalidValue;
    }
    auto [grid_dim, block_dim] =
        mle_rounds_config::eval_constraints_launch_params(x_count, y_count, MAX_THREADS);
    size_t smem_bytes = div_ceil(block_dim.x, WARP_SIZE) * sizeof(FpExt);

#define LOGUP_KERNEL_ARGS                                                                          \
    block_partials, eq_weights, sel_cols, prep_mat, main_mats, challenge_vals, eq_pair_weights, pub_vals,      \
        dag_rules, constraint_nodes, pair_ids, num_used, scratch_len, scratch_global, y_count

    if (scratch_len > LOGUP_BUFFER_THRESHOLD) {
        logup_mle_eval_kernel<true><<<grid_dim, block_dim, smem_bytes>>>(LOGUP_KERNEL_ARGS);
    } else {
        logup_mle_eval_kernel<false><<<grid_dim, block_dim, smem_bytes>>>(LOGUP_KERNEL_ARGS);
    }
    int status = CHECK_KERNEL();
    if (status != 0)
        return status;

    // Second pass: fold the per-block partials into final_out.
    auto n_blocks = grid_dim.x;
    auto [red_grid, red_block] = kernel_launch_params(n_blocks);
    unsigned int red_warps = (red_block.x + WARP_SIZE - 1) / WARP_SIZE;
    size_t red_smem = std::max(1u, red_warps) * sizeof(FpExt);
    // Each FracExt is two FpExt, so the reduction runs over 2 * x_count columns.
    reduce::final_reduce_block_sums<<<2 * x_count, red_block, red_smem>>>(
        reinterpret_cast<FpExt *>(block_partials), reinterpret_cast<FpExt *>(final_out), n_blocks
    );
    return CHECK_KERNEL();
}

} // namespace logup_zerocheck_mle
