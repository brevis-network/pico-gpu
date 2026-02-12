#pragma once

#include <cstdio>
#include <cstring>
#include <cstddef>  // for offsetof
#include <cstdint>  // for uintptr_t
#include <cuda_runtime.h>
// Directly use KoalaBear types (no BabyBear support)
#include <ff/kb31_t.hpp>
#include <ff/kb31_septic_extension_t.hpp>
// Include matrix type before air/folder.cuh which uses Matrix
#include <matrix/type.cuh>
// Include kb31_extension_t - KoalaBear extension field
#include <ff/kb31_extension_t.hpp>
// Include air/folder.cuh after Matrix is defined
#include <air/folder.cuh>
#include "quotient_types.cuh"

// Type aliases for field types - using KoalaBear only
using field_t = kb31_t;
using extension_t = kb31_extension_t;  // KoalaBear extension field
using septic_digest_t = kb31_septic_digest_t;

#define DEBUG_FLAG 0  // Set this to 0 or 1

#if DEBUG_FLAG == 1
    #define DEBUG(...) printf(__VA_ARGS__)
#else
    #define DEBUG(...)  // Do nothing
#endif

namespace brevis_quotient_kernels {
template<
    typename Val,
    typename Challenge,
    typename GlobalSum,
    size_t MEMORY_SIZE,
    size_t EF_SIZE>
__global__ void compute_quotient_values(
// __global__ __launch_bounds__(512) void compute_quotient_values( // large register spilling
    EvalOp* ops,
    size_t ops_len,
    Val* const_base,
    Challenge* const_ext,
    Challenge local_cum_sum,
    GlobalSum global_cum_sum,
    TwoAdicMultiplicativeCoset<Val> trace_domain,
    TwoAdicMultiplicativeCoset<Val> quotient_domain,
    Matrix<Val> prep_on_domain,
    Matrix<Val> main_on_domain,
    Matrix<Val> perm_on_domain,
    Challenge* perm_challenges,
    Challenge* alpha_powers,
    Val* pub_values,
    Val trace_gen,
    Val quotient_gen,
    Matrix<Val> out_values
) {
    size_t eval_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    size_t domain_size = quotient_domain.size();
    size_t prep_width = prep_on_domain.width;
    size_t main_width = main_on_domain.width;
    size_t perm_width = perm_on_domain.width;
    size_t log_blowup = quotient_domain.log_n - trace_domain.log_n;
    size_t next_offset = 1 << log_blowup;

    if (eval_idx >= domain_size) {
        return;
    }

    Val gen_power =
        quotient_gen ^ ((blockIdx.x * blockDim.x) + threadIdx.x);
    Val point = gen_power * quotient_domain.shift;

    LagrangeSelectorsAtPoint<Val> selectors =
        trace_domain.compute_selectors(trace_gen, point);

    Val first_row = selectors.first_row[eval_idx];
    Val last_row = selectors.last_row[eval_idx];
    Val transition = selectors.transition[eval_idx];
    Val vanishing_inv = selectors.vanishing_inv[eval_idx];

    AirEvaluator<Val, Challenge, GlobalSum, 2> evaluator =
        AirEvaluator<Val, Challenge, GlobalSum, 2>();
    evaluator.preprocessed = prep_on_domain;
    evaluator.main_trace = main_on_domain;
    evaluator.pub_values = pub_values;
    evaluator.perm_trace = perm_on_domain;
    evaluator.perm_challenges = perm_challenges;
    evaluator.local_cum_sum = local_cum_sum;
    evaluator.global_cum_sum = global_cum_sum;
    evaluator.isFirstRow = first_row;
    evaluator.isLastRow = last_row;
    evaluator.isTransition = transition;
    // Check for null alpha_powers (should not happen, but safety check)
    if (alpha_powers == nullptr) {
        return;
    }
    evaluator.alpha_powers = alpha_powers;
    evaluator.constraint_idx = 0;
    evaluator.accum = Challenge::zero();
    evaluator.eval_idx = eval_idx;
    evaluator.eval_size = domain_size;
    evaluator.next_offset = next_offset;

    Val base_regs[MEMORY_SIZE];
    for (size_t i = 0; i < MEMORY_SIZE; i++) {
        base_regs[i] = Val {0};
    }
    Challenge ext_regs[EF_SIZE];
    for (size_t i = 0; i < EF_SIZE; i++) {
        ext_regs[i] = Challenge::zero();
    }

    for (size_t i = 0; i < ops_len; i++) {
        EvalOp instr = ops[i];
        switch (instr.op) {
            case 0:
                DEBUG("EMPTY\n");
                break;

            case 1:
                DEBUG("FAssignC: %d <- %d\n", instr.dst, instr.src1);
                base_regs[instr.dst] = const_base[instr.src1];
                break;
            case 2:
                DEBUG(
                    "FAssignV: %d <- (%d, %d)\n",
                    instr.dst,
                    instr.src1_kind,
                    instr.src1
                );
                base_regs[instr.dst] = evaluator.read_base(instr.src1_kind, instr.src1);
                break;
            case 3:
                DEBUG("FAssignE: %d <- %d\n", instr.dst, instr.src1);
                base_regs[instr.dst] = base_regs[instr.src1];
                break;

            case 4:
                DEBUG(
                    "FAddVC: %d <- var_f(%d, %d) + constant_f[%d]\n",
                    instr.dst,
                    instr.src1_kind,
                    instr.src1,
                    instr.src2
                );
                base_regs[instr.dst] = evaluator.read_base(instr.src1_kind, instr.src1)
                    + const_base[instr.src2];
                break;
            case 5:
                DEBUG(
                    "FAddVV: %d <- (%d, %d) + (%d, %d)\n",
                    instr.dst,
                    instr.src1_kind,
                    instr.src1,
                    instr.src2_kind,
                    instr.src2
                );
                base_regs[instr.dst] = evaluator.read_base(instr.src1_kind, instr.src1)
                    + evaluator.read_base(instr.src2_kind, instr.src2);
                break;
            case 6:
                DEBUG(
                    "FAddVE: %d <- (%d, %d) + %d\n",
                    instr.dst,
                    instr.src1_kind,
                    instr.src1,
                    instr.src2
                );
                base_regs[instr.dst] =
                    evaluator.read_base(instr.src1_kind, instr.src1) + base_regs[instr.src2];
                break;

            case 7:
                DEBUG(
                    "FAddEC: %d <- f[%d] + constant_f[%d]\n",
                    instr.dst,
                    instr.src1,
                    instr.src2
                );
                base_regs[instr.dst] = base_regs[instr.src1] + const_base[instr.src2];
                break;
            case 8:
                DEBUG(
                    "FAddEV: %d <- %d + (%d, %d)\n",
                    instr.dst,
                    instr.src1,
                    instr.src2_kind,
                    instr.src2
                );
                base_regs[instr.dst] =
                    base_regs[instr.src1] + evaluator.read_base(instr.src2_kind, instr.src2);
                break;
            case 9:
                DEBUG("FAddEE: %d <- %d + %d\n", instr.dst, instr.src1, instr.src2);
                base_regs[instr.dst] = base_regs[instr.src1] + base_regs[instr.src2];
                break;
            case 10:
                DEBUG("FAddAssignE: %d <- %d\n", instr.dst, instr.src1);
                base_regs[instr.dst] += base_regs[instr.src1];
                break;

            case 11:
                DEBUG(
                    "FSubVC: %d <- var_f(%d, %d) - constant_f[%d]\n",
                    instr.dst,
                    instr.src1_kind,
                    instr.src1,
                    instr.src2
                );
                base_regs[instr.dst] = evaluator.read_base(instr.src1_kind, instr.src1)
                    - const_base[instr.src2];
                break;
            case 12:
                DEBUG(
                    "FSubVV: %d <- (%d, %d) - (%d, %d)\n",
                    instr.dst,
                    instr.src1_kind,
                    instr.src1,
                    instr.src2_kind,
                    instr.src2
                );
                base_regs[instr.dst] = evaluator.read_base(instr.src1_kind, instr.src1)
                    - evaluator.read_base(instr.src2_kind, instr.src2);
                break;
            case 13:
                DEBUG(
                    "FSubVE: %d <- (%d, %d) - %d\n",
                    instr.dst,
                    instr.src1_kind,
                    instr.src1,
                    instr.src2
                );
                base_regs[instr.dst] =
                    evaluator.read_base(instr.src1_kind, instr.src1) - base_regs[instr.src2];
                break;

            case 14:
                DEBUG("FSubEC: %d <- %d - %d\n", instr.dst, instr.src1, instr.src2);
                base_regs[instr.dst] = base_regs[instr.src1] - const_base[instr.src2];
                break;
            case 15:
                DEBUG(
                    "FSubEV: %d <- %d - (%d, %d)\n",
                    instr.dst,
                    instr.src1,
                    instr.src2_kind,
                    instr.src2
                );
                base_regs[instr.dst] =
                    base_regs[instr.src1] - evaluator.read_base(instr.src2_kind, instr.src2);
                break;
            case 16:
                DEBUG("FSubEE: %d <- %d - %d\n", instr.dst, instr.src1, instr.src2);
                base_regs[instr.dst] = base_regs[instr.src1] - base_regs[instr.src2];
                break;
            case 17:
                DEBUG("FSubAssignE: %d <- %d\n", instr.dst, instr.src1);
                base_regs[instr.dst] -= base_regs[instr.src1];
                break;

            case 18:
                DEBUG(
                    "FMulVC: %d <- var_f(%d, %d) * constant_f[%d]\n",
                    instr.dst,
                    instr.src1_kind,
                    instr.src1,
                    instr.src2
                );
                base_regs[instr.dst] = evaluator.read_base(instr.src1_kind, instr.src1)
                    * const_base[instr.src2];
                break;
            case 19:
                DEBUG(
                    "FMulVV: %d <- (%d, %d) * (%d, %d)\n",
                    instr.dst,
                    instr.src1_kind,
                    instr.src1,
                    instr.src2_kind,
                    instr.src2
                );
                base_regs[instr.dst] = evaluator.read_base(instr.src1_kind, instr.src1)
                    * evaluator.read_base(instr.src2_kind, instr.src2);
                break;
            case 20:
                DEBUG(
                    "FMulVE: %d <- (%d, %d) * %d\n",
                    instr.dst,
                    instr.src1_kind,
                    instr.src1,
                    instr.src2
                );
                base_regs[instr.dst] =
                    evaluator.read_base(instr.src1_kind, instr.src1) * base_regs[instr.src2];
                break;

            case 21:
                DEBUG(
                    "FMulEC: %d <- f[%d] * constant_f[%d]\n",
                    instr.dst,
                    instr.src1,
                    instr.src2
                );
                base_regs[instr.dst] = base_regs[instr.src1] * const_base[instr.src2];
                break;
            case 22:
                DEBUG(
                    "FMulEV: %d <- %d * (%d, %d)\n",
                    instr.dst,
                    instr.src1,
                    instr.src2_kind,
                    instr.src2
                );
                base_regs[instr.dst] =
                    base_regs[instr.src1] * evaluator.read_base(instr.src2_kind, instr.src2);
                break;
            case 23:
                DEBUG("FMulEE: %d <- %d * %d\n", instr.dst, instr.src1, instr.src2);
                DEBUG(
                    "FMulEE Input: %d, %d\n",
                    base_regs[instr.src1],
                    base_regs[instr.src2]
                );
                base_regs[instr.dst] = base_regs[instr.src1] * base_regs[instr.src2];
                DEBUG("FMulEE Output: %d\n", base_regs[instr.dst]);
                break;
            case 24:
                DEBUG("FMulAssignE: %d <- %d\n", instr.dst, instr.src1);
                base_regs[instr.dst] *= base_regs[instr.src1];
                break;

            case 25:
                DEBUG("FNegE: %d <- -%d\n", instr.dst, instr.src1);
                base_regs[instr.dst] = -base_regs[instr.src1];
                break;

            case 26:
                DEBUG("EAssignC: %d <- %d\n", instr.dst, instr.src1);
                ext_regs[instr.dst] = const_ext[instr.src1];
                break;
            case 27:
                DEBUG(
                    "EAssignV: %d <- (%d, %d)\n",
                    instr.dst,
                    instr.src1_kind,
                    instr.src1
                );
                ext_regs[instr.dst] = evaluator.read_ext(instr.src1_kind, instr.src1);
                break;
            case 28:
                DEBUG("EAssignE: %d <- %d\n", instr.dst, instr.src1);
                ext_regs[instr.dst] = ext_regs[instr.src1];
                break;

            case 29:
                DEBUG(
                    "EAddVC: %d <- var_ef(%d, %d) + constant_ef[%d]\n",
                    instr.dst,
                    instr.src1_kind,
                    instr.src1,
                    instr.src2
                );
                ext_regs[instr.dst] = evaluator.read_ext(instr.src1_kind, instr.src1)
                    + const_ext[instr.src2];
                break;
            case 30:
                DEBUG(
                    "EAddVV: %d <- (%d, %d) + (%d, %d)\n",
                    instr.dst,
                    instr.src1_kind,
                    instr.src1,
                    instr.src2_kind,
                    instr.src2
                );
                ext_regs[instr.dst] = evaluator.read_ext(instr.src1_kind, instr.src1)
                    + evaluator.read_ext(instr.src2_kind, instr.src2);
                break;
            case 31:
                DEBUG(
                    "EAddVE: %d <- (%d, %d) + %d\n",
                    instr.dst,
                    instr.src1_kind,
                    instr.src1,
                    instr.src2
                );
                ext_regs[instr.dst] =
                    evaluator.read_ext(instr.src1_kind, instr.src1) + ext_regs[instr.src2];
                break;

            case 32:
                DEBUG(
                    "EAddEC: %d <- ef[%d] + constant_ef[%d]\n",
                    instr.dst,
                    instr.src1,
                    instr.src2
                );
                ext_regs[instr.dst] = ext_regs[instr.src1] + const_ext[instr.src2];
                break;
            case 33:
                DEBUG(
                    "EAddEV: %d <- %d + (%d, %d)\n",
                    instr.dst,
                    instr.src1,
                    instr.src2_kind,
                    instr.src2
                );
                ext_regs[instr.dst] =
                    ext_regs[instr.src1] + evaluator.read_ext(instr.src2_kind, instr.src2);
                break;
            case 34:
                DEBUG("EAddEE: %d <- %d + %d\n", instr.dst, instr.src1, instr.src2);
                ext_regs[instr.dst] = ext_regs[instr.src1] + ext_regs[instr.src2];
                break;
            case 35:
                DEBUG("EAddAssignE: %d <- %d\n", instr.dst, instr.src1);
                ext_regs[instr.dst] += ext_regs[instr.src1];
                break;

            case 36:
                DEBUG(
                    "ESubVC: %d <- var_ef(%d, %d) - constant_ef[%d]\n",
                    instr.dst,
                    instr.src1_kind,
                    instr.src1,
                    instr.src2
                );
                ext_regs[instr.dst] = evaluator.read_ext(instr.src1_kind, instr.src1)
                    - const_ext[instr.src2];
                break;
            case 37:
                DEBUG(
                    "ESubVV: %d <- (%d, %d) - (%d, %d)\n",
                    instr.dst,
                    instr.src1_kind,
                    instr.src1,
                    instr.src2_kind,
                    instr.src2
                );
                ext_regs[instr.dst] = evaluator.read_ext(instr.src1_kind, instr.src1)
                    - evaluator.read_ext(instr.src2_kind, instr.src2);
                break;
            case 38:
                DEBUG(
                    "ESubVE: %d <- (%d, %d) - %d\n",
                    instr.dst,
                    instr.src1_kind,
                    instr.src1,
                    instr.src2
                );
                ext_regs[instr.dst] =
                    evaluator.read_ext(instr.src1_kind, instr.src1) - ext_regs[instr.src2];
                break;

            case 39:
                DEBUG(
                    "ESubEC: %d <- ef[%d] - constant_ef[%d]\n",
                    instr.dst,
                    instr.src1,
                    instr.src2
                );
                ext_regs[instr.dst] = ext_regs[instr.src1] - const_ext[instr.src2];
                break;
            case 40:
                DEBUG(
                    "ESubEV: %d <- %d - (%d, %d)\n",
                    instr.dst,
                    instr.src1,
                    instr.src2_kind,
                    instr.src2
                );
                ext_regs[instr.dst] =
                    ext_regs[instr.src1] - evaluator.read_ext(instr.src2_kind, instr.src2);
                break;
            case 41:
                DEBUG("ESubEE: %d <- %d - %d\n", instr.dst, instr.src1, instr.src2);
                ext_regs[instr.dst] = ext_regs[instr.src1] - ext_regs[instr.src2];
                break;
            case 42:
                DEBUG("ESubAssignE: %d <- %d\n", instr.dst, instr.src1);
                ext_regs[instr.dst] -= ext_regs[instr.src1];
                break;

            case 43:
                DEBUG(
                    "EMulVC: %d <- var_ef(%d, %d) * constant_ef[%d]\n",
                    instr.dst,
                    instr.src1_kind,
                    instr.src1,
                    instr.src2
                );
                ext_regs[instr.dst] = evaluator.read_ext(instr.src1_kind, instr.src1)
                    * const_ext[instr.src2];
                break;
            case 44:
                DEBUG(
                    "EMulVV: %d <- (%d, %d) * (%d, %d)\n",
                    instr.dst,
                    instr.src1_kind,
                    instr.src1,
                    instr.src2_kind,
                    instr.src2
                );
                ext_regs[instr.dst] = evaluator.read_ext(instr.src1_kind, instr.src1)
                    * evaluator.read_ext(instr.src2_kind, instr.src2);
                break;
            case 45:
                DEBUG(
                    "EMulVE: %d <- (%d, %d) * %d\n",
                    instr.dst,
                    instr.src1_kind,
                    instr.src1,
                    instr.src2
                );
                ext_regs[instr.dst] =
                    evaluator.read_ext(instr.src1_kind, instr.src1) * ext_regs[instr.src2];
                break;

            case 46:
                DEBUG(
                    "EMulEC: %d <- ef[%d] * constant_ef[%d]\n",
                    instr.dst,
                    instr.src1,
                    instr.src2
                );
                ext_regs[instr.dst] = ext_regs[instr.src1] * const_ext[instr.src2];
                break;
            case 47:
                DEBUG(
                    "EMulEV: %d <- %d * (%d, %d)\n",
                    instr.dst,
                    instr.src1,
                    instr.src2_kind,
                    instr.src2
                );
                ext_regs[instr.dst] =
                    ext_regs[instr.src1] * evaluator.read_ext(instr.src2_kind, instr.src2);
                break;
            case 48:
                DEBUG("EMulEE: %d <- %d * %d\n", instr.dst, instr.src1, instr.src2);
                ext_regs[instr.dst] = ext_regs[instr.src1] * ext_regs[instr.src2];
                break;
            case 49:
                DEBUG("EMulAssignE: %d <- %d\n", instr.dst, instr.src1);
                ext_regs[instr.dst] *= ext_regs[instr.src1];
                break;

            case 50:
                DEBUG("ENegE: %d <- -%d\n", instr.dst, instr.src1);
                ext_regs[instr.dst] = Challenge::zero() - ext_regs[instr.src1];
                break;

            case 51: {
                DEBUG("EFFromE: %d <- %d\n", instr.dst, instr.src1);
                Challenge result;
                result.value[0] = base_regs[instr.src1];
                result.value[1] = Val {0};
                result.value[2] = Val {0};
                result.value[3] = Val {0};
                ext_regs[instr.dst] = result;
                break;
            }
            case 52:
                DEBUG("EFAddEE: %d <- %d + %d\n", instr.dst, instr.src1, instr.src2);
                ext_regs[instr.dst] = ext_regs[instr.src1] + base_regs[instr.src2];
                break;
            case 53:
                DEBUG("EFAddAssignE: %d <- %d\n", instr.dst, instr.src1);
                ext_regs[instr.dst] += base_regs[instr.src1];
                break;
            case 54:
                DEBUG("EFSubEE: %d <- %d - %d\n", instr.dst, instr.src1, instr.src2);
                ext_regs[instr.dst] = ext_regs[instr.src1] - base_regs[instr.src2];
                break;
            case 55:
                DEBUG("EFSubAssignE: %d <- %d\n", instr.dst, instr.src1);
                ext_regs[instr.dst] -= base_regs[instr.src1];
                break;
            case 56:
                DEBUG("EFMulEE: %d <- %d * %d\n", instr.dst, instr.src1, instr.src2);
                ext_regs[instr.dst] = ext_regs[instr.src1] * base_regs[instr.src2];
                break;
            case 57:
                DEBUG("EFMulAssignE: %d <- %d\n", instr.dst, instr.src1);
                ext_regs[instr.dst] *= base_regs[instr.src1];
                break;
            case 58:
                DEBUG(
                    "EFAsBaseSlice: %d <- (%d, %d)\n",
                    instr.dst,
                    instr.src1_kind,
                    instr.src1
                );
                // UNSUPPORTED
                break;

            case 59:
                DEBUG("FAssertZero: %d\n", instr.dst);
                evaluator.accum += (evaluator.alpha_powers[evaluator.constraint_idx] * base_regs[instr.dst]);
                evaluator.constraint_idx++;
                break;
            case 60:
                DEBUG("EAssertZero: %d\n", instr.dst);
                evaluator.accum += (evaluator.alpha_powers[evaluator.constraint_idx] * ext_regs[instr.dst]);
                evaluator.constraint_idx++;
                break;
        }
    }

    kb31_extension_t quotient_result = evaluator.accum * vanishing_inv;

    #pragma unroll
    for (size_t k = 0; k < kb31_extension_t::D; k++) {
        out_values.values[k * out_values.height + eval_idx] =
            quotient_result.value[k];
    }
}
}  // namespace brevis_quotient_kernels

namespace brevis_quotient {

extern "C" void compute_quotient_values(
    EvalOp* ops,
    size_t ops_len,
    field_t* const_base,
    extension_t* const_ext,
    size_t base_reg_count,
    size_t ext_reg_count,
    const extension_t* local_cum_sum,
    const septic_digest_t* global_cum_sum,
    const TwoAdicMultiplicativeCoset<field_t>* trace_domain,
    const TwoAdicMultiplicativeCoset<field_t>* quotient_domain,
    Matrix<field_t> prep_on_domain,
    Matrix<field_t> main_on_domain,
    Matrix<field_t> perm_on_domain,
    extension_t* perm_challenges,
    extension_t* alpha_powers,
    field_t* pub_values,
    field_t trace_gen,
    field_t quotient_gen,
    Matrix<field_t> out_values,
    cudaStream_t stream
) {
    // Validate critical pointers
    if (trace_domain == nullptr) {
        DEBUG("[CUDA] ERROR: trace_domain is null\n");
        return;
    }
    if (quotient_domain == nullptr) {
        DEBUG("[CUDA] ERROR: quotient_domain is null\n");
        return;
    }
    if (alpha_powers == nullptr) {
        DEBUG("[CUDA] ERROR: alpha_powers is null\n");
        return;
    }
    if (local_cum_sum == nullptr) {
        DEBUG("[CUDA] ERROR: local_cum_sum is null\n");
        return;
    }
    if (global_cum_sum == nullptr) {
        DEBUG("[CUDA] ERROR: global_cum_sum is null\n");
        return;
    }
    if (out_values.values == nullptr || out_values.width == 0 || out_values.height == 0) {
        DEBUG("[CUDA] ERROR: out_values is invalid (values=%p, width=%zu, height=%zu)\n", 
               out_values.values, out_values.width, out_values.height);
        return;
    }
    
    DEBUG("[CUDA] All pointer validations passed\n");
    
    // Dereference pointers to get actual structures
    TwoAdicMultiplicativeCoset<field_t> trace_domain_val;
    trace_domain_val.log_n = trace_domain->log_n;
    trace_domain_val.shift = trace_domain->shift;
    
    TwoAdicMultiplicativeCoset<field_t> quotient_domain_val;
    quotient_domain_val.log_n = quotient_domain->log_n;
    quotient_domain_val.shift = quotient_domain->shift;
    
    // Debug: Print struct addresses and size info before dereferencing
    DEBUG("[CUDA] local_cum_sum ptr=%p, expected size=%zu\n", (void*)local_cum_sum, sizeof(extension_t));
    DEBUG("[CUDA] global_cum_sum ptr=%p, expected size=%zu\n", (void*)global_cum_sum, sizeof(septic_digest_t));
    DEBUG("[CUDA] C++ struct sizes: extension_t=%zu, septic_digest_t=%zu, field_t=%zu\n",
           sizeof(extension_t), sizeof(septic_digest_t), sizeof(field_t));
    DEBUG("[CUDA] About to dereference for kernel launch...\n");
    
    size_t threads_per_block = 512;
    size_t domain_size = quotient_domain_val.size();
    size_t block_count = (domain_size - 1) / threads_per_block + 1;
    DEBUG("[CUDA] Kernel config: domain_size=%zu, block_count=%zu, threads_per_block=%zu\n", 
           domain_size, block_count, threads_per_block);
    
#define LAUNCH_KERNELS(MSIZE) \
    { \
        auto launch = [&](auto k_ptr, int ef_s) { \
            cudaFuncSetAttribute((const void*)k_ptr, cudaFuncAttributePreferredSharedMemoryCarveout, 0); \
            cudaFuncSetCacheConfig((const void*)k_ptr, cudaFuncCachePreferL1); \
            k_ptr<<<block_count, threads_per_block, 0, stream>>>( \
                ops, ops_len, const_base, const_ext, \
                *local_cum_sum, *global_cum_sum, trace_domain_val, quotient_domain_val, \
                prep_on_domain, main_on_domain, perm_on_domain, \
                perm_challenges, alpha_powers, pub_values, trace_gen, quotient_gen, out_values); \
        }; \
        if (ext_reg_count <= 10) { \
            DEBUG("[CUDA] Launching kernel with MEMORY_SIZE=%d, EF_SIZE=10 (L1 Optimized)\n", MSIZE); \
            launch(brevis_quotient_kernels::compute_quotient_values<field_t, extension_t, septic_digest_t, MSIZE, 10>, 10); \
        } else if (ext_reg_count <= 20) { \
            DEBUG("[CUDA] Launching kernel with MEMORY_SIZE=%d, EF_SIZE=20 (L1 Optimized)\n", MSIZE); \
            launch(brevis_quotient_kernels::compute_quotient_values<field_t, extension_t, septic_digest_t, MSIZE, 20>, 20); \
        } else if (ext_reg_count <= 30) { \
            DEBUG("[CUDA] Launching kernel with MEMORY_SIZE=%d, EF_SIZE=30 (L1 Optimized)\n", MSIZE); \
            launch(brevis_quotient_kernels::compute_quotient_values<field_t, extension_t, septic_digest_t, MSIZE, 30>, 30); \
        } else if (ext_reg_count <= 40) { \
            DEBUG("[CUDA] Launching kernel with MEMORY_SIZE=%d, EF_SIZE=40 (L1 Optimized)\n", MSIZE); \
            launch(brevis_quotient_kernels::compute_quotient_values<field_t, extension_t, septic_digest_t, MSIZE, 40>, 40); \
        } else if (ext_reg_count <= 50) { \
            DEBUG("[CUDA] Launching kernel with MEMORY_SIZE=%d, EF_SIZE=50 (L1 Optimized)\n", MSIZE); \
            launch(brevis_quotient_kernels::compute_quotient_values<field_t, extension_t, septic_digest_t, MSIZE, 50>, 50); \
        } else if (ext_reg_count <= 60) { \
            DEBUG("[CUDA] Launching kernel with MEMORY_SIZE=%d, EF_SIZE=60 (L1 Optimized)\n", MSIZE); \
            launch(brevis_quotient_kernels::compute_quotient_values<field_t, extension_t, septic_digest_t, MSIZE, 60>, 60); \
        } else if (ext_reg_count <= 70) { \
            DEBUG("[CUDA] Launching kernel with MEMORY_SIZE=%d, EF_SIZE=70 (L1 Optimized)\n", MSIZE); \
            launch(brevis_quotient_kernels::compute_quotient_values<field_t, extension_t, septic_digest_t, MSIZE, 70>, 70); \
        } else if (ext_reg_count <= 290) { \
            DEBUG("[CUDA] Launching kernel with MEMORY_SIZE=%d, EF_SIZE=290 (L1 Optimized)\n", MSIZE); \
            launch(brevis_quotient_kernels::compute_quotient_values<field_t, extension_t, septic_digest_t, MSIZE, 290>, 290); \
        } else { \
            DEBUG("[CUDA] ERROR: ext_reg_count=%zu exceeds maximum (290)\n", ext_reg_count); \
            assert(false && "ext_reg_count exceeds maximum (290)"); \
        } \
    } \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        DEBUG("[CUDA] ERROR: Kernel launch failed (MEMORY_SIZE=%d): %s\n", MSIZE, cudaGetErrorString(err)); \
    }

    if (base_reg_count <= 32) {
        LAUNCH_KERNELS(32);
    } else if (base_reg_count <= 64) {
        LAUNCH_KERNELS(64);
    } else if (base_reg_count <= 128) {
        LAUNCH_KERNELS(128);
    } else if (base_reg_count <= 256) {
        LAUNCH_KERNELS(256);
    } else if (base_reg_count <= 512) {
        LAUNCH_KERNELS(512);
    } else if (base_reg_count <= 1024) {
        LAUNCH_KERNELS(1024);
    } else if (base_reg_count <= 2048) {
        LAUNCH_KERNELS(2048);
    } else if (base_reg_count <= 4096) {
        LAUNCH_KERNELS(4096);
    } else {
        DEBUG("[CUDA] ERROR: base_reg_count=%zu exceeds maximum (4096)\n", base_reg_count);
        assert(false && "base_reg_count exceeds maximum (4096)");
    }
#undef LAUNCH_KERNELS
}
}  // namespace brevis_quotient
