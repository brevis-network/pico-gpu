/*
 * KB NTT wrapper for supra_ntt_kb — ensures KB parameters.cuh is loaded
 * before the generic NTT kernels.
 *
 * This file is found first when compiling supra_ntt_kb (because
 * cuda/supra_kb/include is prepended to the -I list).  It:
 *   1. Includes the KB parameters.cuh (from this same directory), which
 *      defines fr_t = kb31_t, MAX_LG_DOMAIN_SIZE = 24, and sets the
 *      include guard __SPPARK_NTT_PARAMETERS_CUH__.
 *   2. Uses #include_next to include the real supra ntt.cuh from the next
 *      directory in the include path (cuda/supra/include/ntt/ntt.cuh).
 *      Because the include guard is already set, supra's ntt.cuh will skip
 *      re-including parameters.cuh, preserving our KB definitions.
 */

// Step 1: Include KB field parameters.
#include "parameters.cuh"

// Step 2: Delegate to the real supra ntt.cuh (the next occurrence in the
// include path, i.e. cuda/supra/include/ntt/ntt.cuh).
#include_next "ntt/ntt.cuh"
