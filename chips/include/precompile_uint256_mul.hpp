#pragma once

#include <util/rusterror.h>
#include "types.hpp"
#include "memory_read_write.hpp"
#include "utils.hpp"
#include "is_zero.hpp"
#include "fp_op.hpp"
#include "fp_lt.hpp"

namespace pico_gpu::precompile_uint256_mul {
  const size_t NumLimbs = 32;
  const size_t NumWords = NumLimbs / 4;
  const size_t NumWitnesses = 63;
  const size_t WitnessOffset = 1 << 14;

  /// Convert a uint256 multiply event to a trace row.
  template <class F>
  __PICO_HOSTDEV__ inline void event_to_row(
    const Uint256MulEvent* events, Uint256MulCols<F>& cols, F* byte_trace, const size_t idx, const size_t count)
  {
    if (idx < count) {
      const auto& event = events[idx];

      // Assign basic values to the columns.
      cols.is_real = F::one();
      cols.chunk = F::from_canonical_u32(event.chunk);
      cols.clk = F::from_canonical_u32(event.clk);
      cols.x_ptr = F::from_canonical_u32(event.x_ptr);
      cols.y_ptr = F::from_canonical_u32(event.y_ptr);

      // Populate memory columns.
      for (int i = 0; i < NumWords; ++i) {
        memory_read_write::populate(cols.x_memory[i], event.x_memory_records[i], byte_trace);
        memory_read_write::populate(cols.y_memory[i], event.y_memory_records[i], byte_trace);
        memory_read_write::populate(cols.modulus_memory[i], event.modulus_memory_records[i], byte_trace);
      }

      // Check if modulus is zero.
      const uint32_t* modulus = event.modulus;
      uint32_t modulus_byte_sum = 0;
      uint8_t modulus_bytes[NumLimbs];
      fp_op::words_to_bytes_le<NumWords, NumLimbs>(modulus, modulus_bytes);
      for (int i = 0; i < NumLimbs; ++i) {
        modulus_byte_sum += modulus_bytes[i];
      }
      is_zero::populate(cols.modulus_is_zero, modulus_byte_sum);

      // Populate the output column.
      const bool modulus_is_zero = fp_op::are_words_zero<NumWords>(modulus);

      uint32_t two[NumWords];
      for (int i = 1; i < NumWords; ++i)
        two[i] = 0;
      two[0] = 2;

      uint32_t modulus_minus_two[NumWords];
      fp_op::sub_words_with_borrow<NumWords>(modulus, two, modulus_minus_two);

      uint32_t result[NumWords];
      fp_op::populate_with_modulus<F, NumWords, NumLimbs, NumWitnesses, 8, WitnessOffset>(
        cols.output, event.x, event.y, result, modulus, modulus_minus_two, FieldOperation::Mul, modulus_is_zero,
        byte_trace);

      cols.modulus_is_not_zero = F::one() - cols.modulus_is_zero.result;
      if (cols.modulus_is_not_zero == F::one()) {
        fp_lt::populate<F, NumLimbs, NumWitnesses>(cols.output_range_check, result, modulus, byte_trace);
      }
    } else {
      // Populate dummy row.
      uint32_t zero[NumWords];
      uint32_t result[NumWords];
      for (int i = 0; i < NumWords; ++i)
        zero[i] = 0;

      uint32_t neg_two[NumWords];
      neg_two[0] = 0xFFFFFFFF - 0x1;
      for (int i = 1; i < NumWords; ++i)
        neg_two[i] = 0xFFFFFFFF;

      fp_op::populate_with_modulus<F, NumWords, NumLimbs, NumWitnesses, 8, WitnessOffset>(
        cols.output, zero, zero, result, zero, neg_two, FieldOperation::Mul, true, nullptr);
    }
  }

  /// Kernel to convert events to trace rows.
  template <class F>
  __global__ void events_to_trace_kernel(
    const Uint256MulEvent* events,
    const size_t count,
    F* trace_matrix,
    F* byte_trace,
    const size_t num_cols,
    const size_t num_rows)
  {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;

    Uint256MulCols<F>* cols = reinterpret_cast<Uint256MulCols<F>*>(&trace_matrix[idx * num_cols]);
    event_to_row<F>(events, *cols, byte_trace, idx, count);
  }

  /// Generate trace from uint256 multiply events.
  template <class F>
  inline RustError generate_extra_record_and_main(
    const size_t num_cols,
    const size_t event_size,
    const size_t trace_size,
    const Uint256MulEvent* events,
    F* trace,
    F* byte_trace,
    cudaStream_t stream)
  {
    try {
      if (event_size == 0) return RustError{cudaSuccess};

      // Initialize trace matrix with zeros.
      CUDA_OK(cudaMemsetAsync(trace, 0, trace_size * sizeof(F), stream));

      // Launch kernel.
      const uint32_t block_size = 256;
      const uint32_t num_rows = trace_size / num_cols;
      const uint32_t grid_size = (num_rows + block_size - 1) / block_size;

      events_to_trace_kernel<F>
        <<<grid_size, block_size, 0, stream>>>(events, event_size, trace, byte_trace, num_cols, num_rows);
      CUDA_OK(cudaGetLastError());

      CUDA_OK(cudaStreamSynchronize(stream));
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  }
} // namespace pico_gpu::precompile_uint256_mul