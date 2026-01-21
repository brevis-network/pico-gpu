#include <cuda.h>
#include <cuda_runtime.h>
#include "jit_compiler.cuh"
#include <iostream>

// Replicate includes from quotient_eval.cuh to get types without the compute_quotient_values implementation
#include <ff/kb31_t.hpp>
#include <ff/kb31_septic_extension_t.hpp>
#include <matrix/type.cuh>
#include <ff/kb31_extension_t.hpp>
#include <air/folder.cuh>
#include "quotient_types.cuh"

// Type aliases matching quotient_eval.cuh
using field_t = kb31_t;
using extension_t = kb31_extension_t;
using septic_digest_t = kb31_septic_digest_t;

extern "C" void computeValuesJIT(
    const char* jit_source,
    const char* cache_key,
    field_t* const_base,
    extension_t* const_ext,
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
    // Precomputed selectors
    field_t* isFirstRowArr,
    field_t* isLastRowArr,
    field_t* isTransitionArr,
    field_t* invZeroifierArr,
    Matrix<field_t> out_values,
    cudaStream_t stream
) {
    auto& compiler = quotient_jit::JITCompiler::getInstance();
    CUfunction func = compiler.getOrCompile(jit_source, cache_key);

    // Using 256 threads per block for better occupancy
    size_t threads_per_block = 256;
    size_t domain_size = quotient_domain->log_n == 0 ? 0 : (1 << quotient_domain->log_n);
    size_t block_count = (domain_size + threads_per_block - 1) / threads_per_block;

    if (domain_size == 0) return;

    // Package arguments for Driver API
    void* args[] = {
        &const_base,
        &const_ext,
        (void*)local_cum_sum,
        (void*)global_cum_sum,
        (void*)trace_domain,
        (void*)quotient_domain,
        &prep_on_domain,
        &main_on_domain,
        &perm_on_domain,
        &perm_challenges,
        &alpha_powers,
        &pub_values,
        &isFirstRowArr,
        &isLastRowArr,
        &isTransitionArr,
        &invZeroifierArr,
        &out_values
    };

    CUresult res = cuLaunchKernel(
        func,
        block_count, 1, 1,
        threads_per_block, 1, 1,
        0, (CUstream)stream,
        args, nullptr
    );

    if (res != CUDA_SUCCESS) {
        const char* err_name;
        cuGetErrorName(res, &err_name);
        std::cerr << "[JIT] cuLaunchKernel failed: " << err_name << " (Code: " << res << ")" << std::endl;
        
        // If 256 fails due to resources, fallback to 64 as a safety measure
        if (res == CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES) {
            threads_per_block = 64;
            block_count = (domain_size + threads_per_block - 1) / threads_per_block;
            res = cuLaunchKernel(func, block_count, 1, 1, threads_per_block, 1, 1, 0, (CUstream)stream, args, nullptr);
            if (res == CUDA_SUCCESS) return;
        }
        exit(1);
    }
}

// Added function to only compile and cache, without running
extern "C" void compileCacheJIT(
    const char* jit_source,
    const char* cache_key
) {
    auto& compiler = quotient_jit::JITCompiler::getInstance();
    // This will trigger compilation and saving to disk if not exists
    compiler.getOrCompile(jit_source, cache_key);
}
