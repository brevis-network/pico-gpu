#pragma once

#include <ff/ff_config.hpp>
#include <iostream>
#include <poseidon2/constants.cuh>
#include <poseidon2/round_constants.cuh>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <util/rusterror.h>
#include <ff/kb31_septic_extension_t.hpp>
#include <cub/cub.cuh>

#include "types.hpp"
#include "global_interaction.hpp"
#include "global_accumulation.hpp"
#include "utils.hpp"
#include "byte.hpp"

#if defined(FEATURE_BABY_BEAR)
  #include "poseidon2/constants/babybear_poseidon2.h"
using namespace poseidon2_constants_babybear::t16;
#elif defined(FEATURE_KOALA_BEAR)
  #include "poseidon2/constants/koalabear_poseidon2.h"
using namespace poseidon2_constants_koalabear::t16;
#else
  #error "Must define FEATURE_BABY_BEAR or FEATURE_KOALA_BEAR"
#endif

using namespace poseidon2;

namespace pico_gpu::riscv_global {
  using namespace pico_gpu::byte;

  struct GlobalExtraEventIndices {
    size_t byte;
    size_t poseidon2;
  };

  /// Initialize a padding row with dummy values.
  template <class F>
  __device__ inline void assign_padding_row(GlobalCols<F>& cols)
  {
    for (int i = 0; i < 7; i++) {
      cols.message[i] = F::zero();
    }
    cols.kind = F::zero();
    cols.is_real = F::zero();
    cols.is_receive = F::zero();
    cols.is_send = F::zero();

    global_interaction::populate_dummy(&cols.interaction);
  }

  /// Convert a global interaction event to a trace row with extra record.
  template <class F>
  __device__ inline septic_curve_t event_to_row_with_extra_record(
    const GlobalInteractionEvent& event,
    GlobalCols<F>& cols,
    const Poseidon2Constants<F>& poseidon2_constants,
    F* byte_trace,
    Poseidon2Event& poseidon2_events)
  {
    // Set message and kind.
    for (int i = 0; i < 7; i++) {
      cols.message[i] = F::from_canonical_u32(event.message[i]);
    }
    cols.kind = F::from_canonical_u8(event.kind);
    cols.is_real = F::one();

    if (event.is_receive) {
      cols.is_receive = F::one();
      cols.is_send = F::zero();
    } else {
      cols.is_send = F::one();
      cols.is_receive = F::zero();
    }

    // Populate global interaction operation.
    auto poseidon2_result = global_interaction::populate(
      &cols.interaction, event.message, event.is_receive, true, event.kind, poseidon2_constants);

    // Handle byte lookup event for u16 range check.
    {
      const uint16_t message_u16 = static_cast<uint16_t>(event.message[0]);
      add_u16_range_check<F>(byte_trace, message_u16);
    }

    // Handle poseidon2 event.
    if (poseidon2_result.tag == FfiOption<Poseidon2Event>::Tag::Some) poseidon2_events = poseidon2_result.some._0;

    // Create septic curve point for accumulation.
    septic_curve_t pt;
    for (int i = 0; i < 7; i++) {
      pt.x.value[i] = cols.interaction.x_coordinate._0[i];
      pt.y.value[i] = cols.interaction.y_coordinate._0[i];
    }
    return pt;
  }

  /// Kernel to convert events to trace with extra record.
  template <class F>
  __global__ void events_to_trace_with_extra_record_kernel(
    const size_t event_size,
    const size_t trace_size,
    const GlobalInteractionEvent* events,
    F* trace_matrix,
    const size_t num_cols,
    F* __restrict__ round_constants,
    septic_curve_t* d_data,
    F* byte_trace,
    Poseidon2Event* poseidon2_events)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= event_size) return;

    if (idx == 0) d_data[0] = septic_curve_t::start_point();

    Poseidon2Constants<F> poseidon2_constants = {
      WIDTH, alpha, internal_rounds, poseidon2_round_constants::EXTERNAL_ROUNDS, round_constants, MdsType::PLONKY};

    GlobalCols<F>* cols = reinterpret_cast<GlobalCols<F>*>(&trace_matrix[idx * num_cols]);

    d_data[idx + 1] =
      event_to_row_with_extra_record<F>(events[idx], *cols, poseidon2_constants, byte_trace, poseidon2_events[idx]);
  }

  /// Kernel to perform global accumulation.
  template <class F>
  __global__ void global_accumulate_kernel(
    const septic_curve_t& final_digest,
    const septic_curve_t* partial_sums,
    const size_t padded_event_count,
    const size_t event_count,
    F* trace_matrix,
    const size_t num_cols)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= padded_event_count) return;

    GlobalCols<F>* cols = reinterpret_cast<GlobalCols<F>*>(&trace_matrix[idx * num_cols]);
    auto final_sum_checker = septic_curve_t::sum_checker_x(final_digest, septic_curve_t::dummy_point(), final_digest);

    if (idx < event_count) {
      global_accumulation::populate_real(cols->accumulation, partial_sums + idx, final_digest, final_sum_checker);
    } else {
      global_interaction::populate_dummy(&(cols->interaction));
      global_accumulation::populate_dummy(cols->accumulation, final_digest, final_sum_checker);
    }
  }

  /// Elliptic curve addition with safety checks.
  __device__ __forceinline__ septic_curve_t elliptic_curve_add(const septic_curve_t& a, const septic_curve_t& b)
  {
    // Basic checks.
    if (a.is_infinity()) return b;
    if (b.is_infinity()) return a;

    kb31_septic_extension_t x_diff = b.x - a.x;

    // Check if x coordinates are the same.
    if (x_diff == kb31_septic_extension_t::zero()) {
      if (a.y == b.y) {
        // Point doubling case - check if it would cause division by zero.
        if (a.y == kb31_septic_extension_t::zero()) return septic_curve_t::infinity();

        kb31_septic_extension_t y_doubled = a.y + a.y;
        if (y_doubled == kb31_septic_extension_t::zero()) return septic_curve_t::infinity();

        // Safe point doubling.
        kb31_septic_extension_t x_squared = a.x * a.x;
        kb31_septic_extension_t slope_numerator = x_squared + x_squared + x_squared + kb31_t::two();
        kb31_septic_extension_t slope = slope_numerator / y_doubled;

        kb31_septic_extension_t result_x = slope * slope - a.x - a.x;
        kb31_septic_extension_t result_y = slope * (a.x - result_x) - a.y;

        return septic_curve_t(result_x, result_y);
      } else {
        // Same x coordinate but different y coordinate -> point at infinity.
        return septic_curve_t::infinity();
      }
    } else {
      // General addition case.
      kb31_septic_extension_t y_diff = b.y - a.y;
      kb31_septic_extension_t slope = y_diff / x_diff;

      kb31_septic_extension_t result_x = slope * slope - a.x - b.x;
      kb31_septic_extension_t result_y = slope * (a.x - result_x) - a.y;

      return septic_curve_t(result_x, result_y);
    }
  }

  /// CUB custom operator for elliptic curve addition.
  struct EllipticCurveAddOp {
    __device__ __forceinline__ septic_curve_t operator()(const septic_curve_t& a, const septic_curve_t& b) const
    {
      return elliptic_curve_add(a, b);
    }
  };

  /// Block-level CUB scan kernel.
  template <class F>
  __global__ void
  cub_scan_kernel(const septic_curve_t* input, septic_curve_t* output, septic_curve_t* block_sums, const size_t n)
  {
    typedef cub::BlockScan<septic_curve_t, 256> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;

    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    septic_curve_t thread_data = septic_curve_t::infinity();
    if (global_idx < n) thread_data = input[global_idx];

    septic_curve_t thread_result;
    EllipticCurveAddOp safe_add_op;

    // Use custom safe addition operator.
    BlockScan(temp_storage).InclusiveScan(thread_data, thread_result, safe_add_op);

    if (global_idx < n) output[global_idx] = thread_result;

    if (threadIdx.x == blockDim.x - 1 && block_sums != nullptr) block_sums[blockIdx.x] = thread_result;
  }

  /// Add block prefix kernel.
  template <class F>
  __global__ void add_block_prefix_kernel(septic_curve_t* data, const septic_curve_t* block_prefixes, const size_t n)
  {
    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_idx < n && blockIdx.x > 0)
      data[global_idx] = elliptic_curve_add(block_prefixes[blockIdx.x - 1], data[global_idx]);
  }

  /// Perform CUB-based elliptic curve scan.
  template <class F>
  inline RustError
  cub_elliptic_curve_scan(const septic_curve_t* d_input, septic_curve_t* d_output, const size_t n, cudaStream_t stream)
  {
    if (n == 0) return RustError{cudaSuccess};

    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if (NUM_BLOCKS == 1) {
      // Single block processing with safety checks.
      cub_scan_kernel<F><<<1, BLOCK_SIZE, 0, stream>>>(d_input, d_output, nullptr, n);

    } else {
      // Multi-block processing with safety checks.
      dev_ptr_t<septic_curve_t> d_block_sums{NUM_BLOCKS, stream};

      // Phase 1: Each block performs local scan.
      cub_scan_kernel<F><<<NUM_BLOCKS, BLOCK_SIZE, 0, stream>>>(d_input, d_output, &d_block_sums[0], n);

      cudaError_t err1 = cudaGetLastError();
      if (err1 != cudaSuccess) {
        printf("Phase 1 error: %s\n", cudaGetErrorString(err1));
        return RustError{err1};
      }

      // Phase 2: Scan the block sums.
      if (NUM_BLOCKS <= BLOCK_SIZE) {
        cub_scan_kernel<F><<<1, NUM_BLOCKS, 0, stream>>>(&d_block_sums[0], &d_block_sums[0], nullptr, NUM_BLOCKS);
      } else {
        // Recursive processing (rarely reached).
        RustError recursive_result = cub_elliptic_curve_scan<F>(&d_block_sums[0], &d_block_sums[0], NUM_BLOCKS, stream);
        if (recursive_result.code != cudaSuccess) return recursive_result;
      }

      cudaError_t err2 = cudaGetLastError();
      if (err2 != cudaSuccess) {
        printf("Phase 2 error: %s\n", cudaGetErrorString(err2));
        return RustError{err2};
      }

      // Phase 3: Add block prefixes (also uses safe addition).
      add_block_prefix_kernel<F><<<NUM_BLOCKS, BLOCK_SIZE, 0, stream>>>(d_output, &d_block_sums[0], n);

      cudaError_t err3 = cudaGetLastError();
      if (err3 != cudaSuccess) {
        printf("Phase 3 error: %s\n", cudaGetErrorString(err3));
        return RustError{err3};
      }
    }

    return RustError{cudaSuccess};
  }

  /// Generate trace from global interaction events.
  template <class F>
  inline RustError generate_extra_record_and_main(
    const size_t num_cols,
    const size_t event_size,
    const size_t trace_size,
    const GlobalInteractionEvent* events,
    F* trace,
    cudaStream_t stream,
    F* byte_trace,
    Poseidon2Event* poseidon2_events,
    const size_t poseidon2_events_start_index)
  {
    try {
      if (event_size == 0) return RustError{cudaSuccess};

      CUDA_OK(cudaMemsetAsync(trace, 0, trace_size * sizeof(F), stream));

      poseidon2_round_constants::RoundConstants<F> rc;
      poseidon2_round_constants::init_round_constants(rc, stream);

      const int block_size = 256;
      const int num_rows = trace_size / num_cols;
      const int grid_size = (num_rows + block_size - 1) / block_size;

      dev_ptr_t<septic_curve_t> d_curve_points{event_size + 1, stream};

      events_to_trace_with_extra_record_kernel<<<grid_size, block_size, 0, stream>>>(
        event_size, trace_size, events, trace, num_cols, rc.d_round_constants, &d_curve_points[0], byte_trace,
        &poseidon2_events[poseidon2_events_start_index]);

      CUDA_OK(cudaStreamSynchronize(stream));

      // Use CUB instead of Thrust for inclusive scan.
      RustError scan_result =
        cub_elliptic_curve_scan<F>(&d_curve_points[0], &d_curve_points[0], event_size + 1, stream);
      if (scan_result.code != cudaSuccess) return scan_result;

      // Launch accumulation kernel.
      const size_t padded_event_size = trace_size / num_cols;
      const int accumulate_grid_size = (padded_event_size + block_size - 1) / block_size;

      global_accumulate_kernel<<<accumulate_grid_size, block_size, 0, stream>>>(
        d_curve_points[event_size], &d_curve_points[0], padded_event_size, event_size, trace, num_cols);
      CUDA_OK(cudaGetLastError());

      poseidon2_round_constants::free_round_constants(rc, stream);

    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  }
} // namespace pico_gpu::riscv_global