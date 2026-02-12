#pragma once

#include <ff/kb31_t.hpp>
#include <ff/kb31_septic_extension_t.hpp>
#include <matrix/type.cuh>
#include <ff/kb31_extension_t.hpp>
#include <air/folder.cuh>
#include <quotient/quotient_types.cuh>

using field_t = kb31_t;
using extension_t = kb31_extension_t;
using septic_digest_t = kb31_septic_digest_t;

extern "C" __global__ void jit_computeValues(
    field_t* const_base,
    extension_t* const_ext,
    extension_t local_cum_sum,
    septic_digest_t global_cum_sum,
    TwoAdicMultiplicativeCoset<field_t> trace_domain,
    TwoAdicMultiplicativeCoset<field_t> quotient_domain,
    Matrix<field_t> prep_on_domain,
    Matrix<field_t> main_on_domain,
    Matrix<field_t> perm_on_domain,
    extension_t* perm_challenges,
    extension_t* alpha_powers,
    field_t* pub_values,
    // Precomputed selectors
    field_t* isFirstRowArr,
    field_t* isLastRowArr,
    field_t* isTransitionArr,
    field_t* invZeroifierArr,
    Matrix<field_t> out_values
) {
    size_t eval_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    size_t domain_size = quotient_domain.size();
    if (eval_idx >= domain_size) return;

    size_t log_blowup = quotient_domain.log_n - trace_domain.log_n;
    size_t next_offset = 1 << log_blowup;

    AirEvaluator<field_t, extension_t, septic_digest_t, 2> evaluator;
    evaluator.preprocessed = prep_on_domain;
    evaluator.main_trace = main_on_domain;
    evaluator.pub_values = pub_values;
    evaluator.perm_trace = perm_on_domain;
    evaluator.perm_challenges = perm_challenges;
    evaluator.local_cum_sum = local_cum_sum;
    evaluator.global_cum_sum = global_cum_sum;
    
    // Read precomputed selectors
    evaluator.isFirstRow = isFirstRowArr[eval_idx];
    evaluator.isLastRow = isLastRowArr[eval_idx];
    evaluator.isTransition = isTransitionArr[eval_idx];
    
    evaluator.alpha_powers = alpha_powers;
    evaluator.constraint_idx = 0;
    evaluator.accum = extension_t::zero();
    evaluator.eval_idx = eval_idx;
    evaluator.eval_size = domain_size;
    evaluator.next_offset = next_offset;

    // --- JIT_LOGIC_START ---
{{JIT_BODY}}
    // --- JIT_LOGIC_END ---

    extension_t v_inv = extension_t(invZeroifierArr[eval_idx]);
    extension_t quotient_result = evaluator.accum * v_inv;

    #pragma unroll
    for (size_t k = 0; k < extension_t::D; k++) {
        out_values.values[k * out_values.height + eval_idx] = quotient_result.value[k];
    }
}
