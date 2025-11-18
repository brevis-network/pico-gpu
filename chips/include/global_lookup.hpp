
#pragma once

#include <ff/ff_config.hpp>
#include <iostream>
#include <poseidon2/constants.cuh>
#include <poseidon2/round_constants.cuh>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <util/rusterror.h>
#include "types.hpp"
#include "utils.hpp"
#include "global_interaction.hpp"
#include "global_accumulation.hpp"
#include "global_interaction.hpp"
#include "utils.hpp"

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

namespace pico_gpu::global_lookup {
  __PICO_HOSTDEV__ inline septic_curve_t event_to_row(
    const GlobalInteractionEvent* event,
    GlobalCols<field_t>* col,
    const Poseidon2Constants<field_t>& poseidon2_constants)
  {
#pragma unroll
    for (int i = 0; i < 7; i++) {
      col->message[i] = field_t::from_canonical_u32(event->message[i]);
    }
    col->kind = event->kind;
    global_interaction::populate(
      &col->interaction, event->message, event->is_receive, true, event->kind, poseidon2_constants);
    col->is_real = field_t::one();
    col->is_receive = event->is_receive ? field_t::one() : field_t::zero();
    col->is_send = event->is_receive ? field_t::zero() : field_t::one();

    septic_curve_t pt;
    for (int i = 0; i < 7; i++) {
      pt.x.value[i] = col->interaction.x_coordinate._0[i];
      pt.y.value[i] = col->interaction.y_coordinate._0[i];
    }
    return pt;
  }

  // Kernel implementation

  __global__ void global_events_to_trace_kernel(
    const GlobalInteractionEvent* events,
    size_t count,
    field_t* trace_matrix,
    size_t num_col,
    field_t* __restrict__ round_constants,
    septic_curve_t* d_data)
  {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    if (idx == 0) { d_data[0] = septic_curve_t::start_point(); }
    Poseidon2Constants<field_t> poseidon2_constants = {
      WIDTH, alpha, internal_rounds, poseidon2_round_constants::EXTERNAL_ROUNDS, round_constants, MdsType::PLONKY};
    GlobalCols<field_t>* col = reinterpret_cast<GlobalCols<field_t>*>(&trace_matrix[idx * num_col]);
    d_data[idx + 1] = event_to_row(events + idx, col, poseidon2_constants);
  }

  __global__ void global_accumulate_kernel(
    const septic_curve_t& final_digest,
    const septic_curve_t* partial_sums,
    size_t padded_event_count,
    size_t event_count,
    field_t* trace_matrix,
    size_t num_col)
  {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= padded_event_count) return;
    GlobalCols<field_t>* col = reinterpret_cast<GlobalCols<field_t>*>(&trace_matrix[idx * num_col]);
    auto final_sum_checker = septic_curve_t::sum_checker_x(final_digest, septic_curve_t::dummy_point(), final_digest);
    if (idx < event_count) {
      global_accumulation::populate_real(col->accumulation, partial_sums + idx, final_digest, final_sum_checker);
    } else {
      global_interaction::populate_dummy(&(col->interaction));
      global_accumulation::populate_dummy(col->accumulation, final_digest, final_sum_checker);
    }
  }

  template <class F>
  inline RustError generate_main(
    size_t num_cols,
    size_t event_size,
    size_t trace_size,
    const GlobalInteractionEvent* events,
    F* trace,
    cudaStream_t stream)
  {
    try {
      if (event_size == 0) { return RustError{cudaSuccess}; }

      // initialize trace matrix with zeros
      CUDA_OK(cudaMemsetAsync(trace, 0, trace_size * sizeof(F), stream));

      // generate real rows by events
      poseidon2_round_constants::RoundConstants<field_t> rc;
      poseidon2_round_constants::init_round_constants(rc, stream);
      size_t block_size = 256;
      size_t grid_size = (event_size + block_size - 1) / block_size;
      dev_ptr_t<septic_curve_t> d_curve_points{event_size + 1, stream};
      global_events_to_trace_kernel<<<grid_size, block_size, 0, stream>>>(
        events, event_size, trace, num_cols, rc.d_round_constants, &d_curve_points[0]);
      CUDA_OK(cudaGetLastError());

      CUDA_OK(cudaStreamSynchronize(stream));

      thrust::device_ptr<septic_curve_t> d_curve_points_begin =
        thrust::device_pointer_cast((septic_curve_t*)d_curve_points);
      thrust::inclusive_scan(d_curve_points_begin, d_curve_points_begin + event_size + 1, d_curve_points_begin);

      // generate padding rows
      size_t padded_event_size = trace_size / num_cols;
      grid_size = (padded_event_size + block_size - 1) / block_size;
      global_accumulate_kernel<<<grid_size, block_size, 0, stream>>>(
        d_curve_points[event_size], &d_curve_points[0], padded_event_size, event_size, trace, num_cols);
      CUDA_OK(cudaGetLastError());

      poseidon2_round_constants::free_round_constants(rc, stream);
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  } // generate_main
} // namespace pico_gpu::global_lookup
