#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include <util/rusterror.h>

__host__ __inline__ void checkCudaError(cudaError_t error, const char* message)
{
  if (error != cudaSuccess) {
    fprintf(stderr, "Error: %s: %s\n", message, cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
}

// Basic kernel that generates xs and computes selectors in one pass
template <class F>
__global__ void basic_selectors_kernel(
  const F* coset_constants, // [coset_1, coset_2, subgroup_last, generator, shift]
  size_t xs_len,
  size_t evals_len,
  const F* evals,
  F* is_first_row,
  F* is_last_row,
  F* is_transition,
  F* inv_zeroifier)
{
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < xs_len) {
    // Generate x value on-the-fly: shift * generator^idx
    F gen_power = coset_constants[3] ^ idx; // generator^idx
    F x = coset_constants[4] * gen_power;   // shift * generator^idx

    // Get corresponding eval value (cycling through evals)
    size_t cycle_idx = idx % evals_len;
    F eval_val = evals[cycle_idx];
    F inv_eval = eval_val.reciprocal();

    // Compute selectors
    F denom1 = x - coset_constants[0]; // x - coset_1
    F denom2 = x - coset_constants[1]; // x - coset_2
    F denom3 = x - coset_constants[2]; // x - subgroup_last

    is_first_row[idx] = denom1.reciprocal() * eval_val;
    is_last_row[idx] = denom2.reciprocal() * eval_val;
    is_transition[idx] = denom3; // Note: not multiplied by eval_val
    inv_zeroifier[idx] = inv_eval;
  }
}

// Shared memory + vectorized kernel (for small evals_len <= 1024)
template <class F>
__global__ void shared_mem_vectorized_kernel(
  const F* coset_constants,
  size_t xs_len,
  size_t evals_len,
  const F* evals,
  F* is_first_row,
  F* is_last_row,
  F* is_transition,
  F* inv_zeroifier)
{
  // Shared memory for evals
  extern __shared__ char shared_mem[];
  F* shared_evals = reinterpret_cast<F*>(shared_mem);

  size_t base_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;

  // Load evals into shared memory cooperatively
  for (size_t i = threadIdx.x; i < evals_len; i += blockDim.x) {
    shared_evals[i] = evals[i];
  }
  __syncthreads();

// Process 2 elements per thread
#pragma unroll
  for (int offset = 0; offset < 2; offset++) {
    uint32_t idx = base_idx + offset;
    if (idx < xs_len) {
      // Generate x value: shift * generator^idx
      F gen_power = coset_constants[3] ^ idx; // generator^idx
      F x = coset_constants[4] * gen_power;   // shift * generator^idx

      // Get corresponding eval value (cycling through evals)
      size_t cycle_idx = idx % evals_len;
      F eval_val = shared_evals[cycle_idx];
      F inv_eval = eval_val.reciprocal();

      F denom1 = x - coset_constants[0];
      F denom2 = x - coset_constants[1];
      F denom3 = x - coset_constants[2];

      is_first_row[idx] = denom1.reciprocal() * eval_val;
      is_last_row[idx] = denom2.reciprocal() * eval_val;
      is_transition[idx] = denom3;
      inv_zeroifier[idx] = inv_eval;
    }
  }
}

// Vectorized kernel (for large evals_len > 1024)
template <class F>
__global__ void vectorized_selectors_kernel(
  const F* coset_constants,
  size_t xs_len,
  size_t evals_len,
  const F* evals,
  F* is_first_row,
  F* is_last_row,
  F* is_transition,
  F* inv_zeroifier)
{
  size_t base_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;

// Process 2 elements per thread
#pragma unroll
  for (int offset = 0; offset < 2; offset++) {
    uint32_t idx = base_idx + offset;
    if (idx < xs_len) {
      // Generate x value: shift * generator^idx
      F gen_power = coset_constants[3] ^ idx; // generator^idx
      F x = coset_constants[4] * gen_power;   // shift * generator^idx

      // Get corresponding eval value (cycling through evals)
      size_t cycle_idx = idx % evals_len;
      F eval_val = evals[cycle_idx];
      F inv_eval = eval_val.reciprocal();

      F denom1 = x - coset_constants[0];
      F denom2 = x - coset_constants[1];
      F denom3 = x - coset_constants[2];

      is_first_row[idx] = denom1.reciprocal() * eval_val;
      is_last_row[idx] = denom2.reciprocal() * eval_val;
      is_transition[idx] = denom3;
      inv_zeroifier[idx] = inv_eval;
    }
  }
}

template <class F>
inline RustError selectors_on_coset_gpu(
  const F* coset_constants, // [coset_1, coset_2, subgroup_last, generator, shift]
  size_t xs_len,
  size_t evals_len,
  const F* evals,
  F* is_first_row,
  F* is_last_row,
  F* is_transition,
  F* inv_zeroifier,
  cudaStream_t stream)
{
  try {
    const size_t block_size = 256;

    if (evals_len <= 1024) { // Only uses max 4KB shared memory - very safe
      // Use shared memory + vectorized version for small evals
      size_t num_threads = (xs_len + 1) / 2; // Each thread processes 2 elements
      size_t grid_size = (num_threads + block_size - 1) / block_size;
      size_t shared_mem_size = evals_len * sizeof(F);

      shared_mem_vectorized_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(
        coset_constants, xs_len, evals_len, evals, is_first_row, is_last_row, is_transition, inv_zeroifier);
    } else {
      // Use vectorized version only for large evals
      size_t num_threads = (xs_len + 1) / 2; // Each thread processes 2 elements
      size_t grid_size = (num_threads + block_size - 1) / block_size;

      vectorized_selectors_kernel<<<grid_size, block_size, 0, stream>>>(
        coset_constants, xs_len, evals_len, evals, is_first_row, is_last_row, is_transition, inv_zeroifier);
    }

    checkCudaError(cudaGetLastError(), "Failed to launch selector kernel");

  } catch (const cuda_error& e) {
    return RustError{e.code()};
  }

  return RustError{cudaSuccess};
}