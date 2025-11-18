#pragma once

#include "poseidon2.hpp"

using namespace ::pico_gpu::poseidon2;

namespace pico_gpu::recursion_poseidon2 {
  using RecursionPoseidon2EventT = RecursionPoseidon2Event<field_t>;
  __global__ void poseidon2_instrs_to_trace_kernel_phase1(
    const Poseidon2SkinnyInstr<field_t>* instrs, size_t count, field_t n1, field_t* trace_matrix, size_t num_cols)
  {
    // Calculate thread index
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    const Poseidon2SkinnyInstr<field_t>& inst = instrs[idx];
    uint32_t row_index = idx / POSEIDON2_DATAPAR;
    uint32_t datapar_index = idx % POSEIDON2_DATAPAR;
    Poseidon2PreprocessedCols<field_t>* cols =
      reinterpret_cast<Poseidon2PreprocessedCols<field_t>*>(&trace_matrix[row_index * num_cols]);

    // use the super parallelism to write the negative 1
    cols->values[datapar_index].is_real_neg = n1;
  }

  __global__ void poseidon2_instrs_to_trace_kernel_phase2(
    const Poseidon2SkinnyInstr<field_t>* instrs, size_t count, field_t* trace_matrix, size_t num_cols)
  {
    // Calculate thread index
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    // WIDTH = 16 => mask = 0x0f
    size_t p2_idx = idx & 0x0f;
    idx = idx >> 4;
    if (idx >= count) return;

    const Poseidon2SkinnyInstr<field_t>& inst = instrs[idx];
    uint32_t row_index = idx / POSEIDON2_DATAPAR;
    uint32_t datapar_index = idx % POSEIDON2_DATAPAR;
    Poseidon2PreprocessedCols<field_t>* cols =
      reinterpret_cast<Poseidon2PreprocessedCols<field_t>*>(&trace_matrix[row_index * num_cols]);

    // use the super parallelism to write the array cols
    cols->values[datapar_index].input[p2_idx] = inst.addrs.input[p2_idx];
    cols->values[datapar_index].output[p2_idx].addr = inst.addrs.output[p2_idx];
    cols->values[datapar_index].output[p2_idx].mult = inst.mults[p2_idx];
  }

  __host__ __device__ inline void event_to_row(
    const RecursionPoseidon2EventT& event, Poseidon2ValueColsT& cols, const field_t* __restrict__ round_constants)
  {
    cols.is_real = field_t::from_bool(true);
    for (int i = 0; i < WIDTH; i++) {
      cols.input[i] = event.input[i];
    }
    permute_and_populate(cols, round_constants);
    // Since permute_and_populate changes the state, we need to copy it back to
    // cols.input
    for (int i = 0; i < WIDTH; i++) {
      cols.input[i] = event.input[i];
    }
  }

  __global__ void events_to_rows_kernel(
    const RecursionPoseidon2EventT* __restrict__ events,
    field_t* __restrict__ trace_matrix,
    size_t events_count,
    size_t num_cols,
    const field_t* __restrict__ round_constants)
  {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= events_count) return;

    Poseidon2ValueColsT* cols = reinterpret_cast<Poseidon2ValueColsT*>(&trace_matrix[idx * num_cols]);
    event_to_row(events[idx], *cols, round_constants);
  }

  __global__ void pad_dummy_rows_kernel(
    field_t* __restrict__ trace_matrix, size_t pad_count, size_t num_cols, const field_t* __restrict__ dummy_row)
  {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pad_count) return;

    field_t* row = &trace_matrix[idx * num_cols];
    for (size_t i = 0; i < num_cols; i++) {
      row[i] = dummy_row[i];
    }
  }

  inline RustError generate_preprocessed(
    const Poseidon2SkinnyInstr<field_t>* instrs,
    size_t num_instrs,
    field_t* trace,
    size_t num_cols,
    cudaStream_t stream)
  {
    try {
      if (num_instrs == 0) { return RustError{cudaSuccess}; }

      // launch kernel
      const int block_size = 256;
      int grid_size = (num_instrs + block_size - 1) / block_size;
      field_t n1 = field_t::one();
      n1 = -n1;
      poseidon2_instrs_to_trace_kernel_phase1<<<grid_size, block_size, 0, stream>>>(
        instrs, num_instrs, n1, trace, num_cols);
      CUDA_OK(cudaGetLastError());
      grid_size = (16 * num_instrs + block_size - 1) / block_size;
      poseidon2_instrs_to_trace_kernel_phase2<<<grid_size, block_size, 0, stream>>>(
        instrs, num_instrs, trace, num_cols);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  } // generate_preprocessed

  inline RustError generate_main(
    size_t num_cols,
    size_t event_size,
    size_t trace_size,
    const RecursionPoseidon2EventT* events,
    field_t* trace,
    field_t* dummy_row,
    cudaStream_t stream)
  {
    try {
      if (event_size == 0) { return RustError{cudaSuccess}; }

      // Initialize trace matrix with zeros
      CUDA_OK(cudaMemsetAsync(trace, 0, trace_size * sizeof(field_t), stream));

      // Launch kernel

      size_t num_total_rows = trace_size / num_cols;
      if (num_total_rows > event_size) {
        const int block_size = 256;
        size_t pad_count = num_total_rows - event_size;
        int grid_size = int((pad_count + block_size - 1) / block_size);
        size_t idx_dummy_start = event_size * num_cols;
        pad_dummy_rows_kernel<<<grid_size, block_size, 0, stream>>>(
          &trace[idx_dummy_start], pad_count, num_cols, dummy_row);
        CUDA_OK(cudaGetLastError());
      }

      RoundConstants<field_t> rc;
      init_round_constants(rc, stream);
      const int block_size = 256;
      const int grid_size = (event_size + block_size - 1) / block_size;
      events_to_rows_kernel<<<grid_size, block_size, 0, stream>>>(
        events, trace, event_size, num_cols, rc.d_round_constants);
      CUDA_OK(cudaGetLastError());
      free_round_constants(rc, stream);
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  }
} // namespace pico_gpu::recursion_poseidon2
