#pragma once

template <typename Val, typename Challenge, typename GlobalSum, size_t N>
struct AirEvaluator {
   public:
    Matrix<Val> preprocessed;
    Matrix<Val> main_trace;
    Val* pub_values;
    Matrix<Val> perm_trace;
    Challenge* perm_challenges;
    Challenge local_cum_sum;
    GlobalSum global_cum_sum;
    Val isFirstRow;
    Val isLastRow;
    Val isTransition;
    Challenge* alpha_powers;
    size_t constraint_idx;
    Challenge accum;
    size_t next_offset;
    size_t eval_idx;
    size_t eval_size;

   public:
    __device__ AirEvaluator() {}

    __inline__ __device__ Val read_base(unsigned char variant, unsigned int idx) {
        switch (variant) {
            case 0:
                return Val{0};
            case 1:
                return Val{idx};
            case 2:
                return preprocessed.values[idx * preprocessed.height + (eval_idx % eval_size)];
            case 3:
                return preprocessed.values[idx * preprocessed.height + ((eval_idx + next_offset) % eval_size)];
            case 4:
                return main_trace.values[idx * main_trace.height + (eval_idx % eval_size)];
            case 5:
                return main_trace.values[idx * main_trace.height + ((eval_idx + next_offset) % eval_size)];
            case 6:
                return isFirstRow;
            case 7:
                return isLastRow;
            case 8:
                return isTransition;
            case 9:
                return pub_values[idx];
            case 10:
                if (idx < 7) {
                    return global_cum_sum.point.x.value[idx];
                }
                else {
                    return global_cum_sum.point.y.value[idx - 7];
                }
        }
        return Val{0};
    }

    __inline__ __device__ Challenge read_ext(unsigned char variant, unsigned int idx) {
        switch (variant) {
            case 0:
                return Challenge::zero();
            case 1:
                {
                Challenge result;
                for (size_t k = 0 ; k < Challenge::D; k++)
                    result.value[k] = perm_trace.values[(idx * Challenge::D + k) * perm_trace.height + 
                        (eval_idx % eval_size)];
                return result;
                }
            case 2:
                {
                Challenge result;
                for (size_t k = 0 ; k < Challenge::D; k++)
                    result.value[k] = perm_trace.values[(idx * Challenge::D + k) * perm_trace.height + 
                    ((eval_idx + next_offset) % eval_size)];
                return result;
                }
            case 3:
                return perm_challenges[idx];
            case 4:
                return local_cum_sum;
        }
        return Challenge::zero();
    }
};
