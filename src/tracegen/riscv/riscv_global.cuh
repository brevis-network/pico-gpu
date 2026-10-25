// Copyright (c) 2025-2026 Brevis. All Rights Reserved.
// Licensed under the Business Source License 1.1 (BUSL-1.1).
// See LICENSE.md for details.

#pragma once

#include "field/ff_config.hpp"
#include <iostream>
#include "hash/poseidon2/constants.cuh"
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include "common/rusterror.h"
#include "field/kb31_septic_extension_t.hpp"
#include <cub/cub.cuh>

#include "tracegen/common/types.cuh"
#include "tracegen/common/global_interaction.cuh"
#include "tracegen/common/global_accumulation.cuh"
#include "tracegen/common/utils.cuh"
#include "tracegen/riscv/byte.cuh"
#include "tracegen/riscv/riscv_global_septic_local.cuh"

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
    for (int i = 0; i < 8; i++) {
      cols.message[i] = F::zero();
    }
    cols.message_0_16bit_limb = F::zero();
    cols.message_0_8bit_limb = F::zero();
    cols.message_0_high_bit = F::zero();
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
    for (int i = 0; i < 8; i++) {
      cols.message[i] = F::from_canonical_u32(event.message[i]);
    }
    cols.message_0_16bit_limb = F::from_canonical_u32(event.message[0] & 0xFFFF);
    cols.message_0_8bit_limb = F::from_canonical_u32((event.message[0] >> 16) & 0xFF);
    // 25-bit widening: boolean high bit (no byte lookup), mirrors Rust tracegen.
    cols.message_0_high_bit = F::from_canonical_u32((event.message[0] >> (CHUNK_CLK_BITS - 1)) & 1);
    cols.kind = F::from_canonical_u8(event.kind);
    cols.is_real = F::one();

    if (event.is_receive) {
      cols.is_receive = F::one();
      cols.is_send = F::zero();
    } else {
      cols.is_send = F::one();
      cols.is_receive = F::zero();
    }

    // Populate global interaction (uses first 7 message elements for septic curve).
    // HOIST+INLINE local copy (riscv_global_septic_local.cuh), scoped to this kernel's TU
    // only -- see that header's top comment for why this isn't landed in the shared
    // global_interaction::populate / ff/kb31_septic_extension_t.hpp instead. Byte-equal to
    // global_interaction::populate (verified on a 3.7M-event capture).
    auto poseidon2_result = riscv_global_detail::populate(
      &cols.interaction, event.message, event.is_receive, true, event.kind, poseidon2_constants);

    // Byte lookup events matching CPU extra_record.
    add_u16_range_check<F>(byte_trace, static_cast<uint16_t>(event.message[0] & 0xFFFF));
    add_u16_range_check<F>(byte_trace, static_cast<uint16_t>(event.message[7]));
    // No kind<32 LTU event: that lookup is gone on the Rust side (constant sender kinds back the
    // bound, and both keygen paths assert it).
    // Offset u8 range check + y6_limbs range checks (mirror eval_single_digest).
    const uint8_t mid_bound = static_cast<uint8_t>(1u << (Y6_RANGE_BITS - F::TOP_BITS - Y6_LO_LIMB_BITS)); // 128
    const uint8_t top_bound = static_cast<uint8_t>((1u << F::TOP_BITS) - 1);       // 127
    // Merged U8 pair -- slot order must match the constraint: b = offset, c = the message[0]
    // 8-bit limb. A swap balances only by accident.
    add_u8_range_check<F>(byte_trace, static_cast<uint8_t>(cols.interaction.offset.as_canonical_u32()),
                          static_cast<uint8_t>((event.message[0] >> 16) & 0xFF));
    add_u16_range_check<F>(byte_trace, static_cast<uint16_t>(cols.interaction.y6_limbs[0].as_canonical_u32()));
    handle_byte_lookup_event<F>(
      byte_trace, ByteOpcode::LTU, static_cast<uint8_t>(cols.interaction.y6_limbs[1].as_canonical_u32()), mid_bound);
    handle_byte_lookup_event<F>(
      byte_trace, ByteOpcode::LTU, static_cast<uint8_t>(cols.interaction.y6_limbs[2].as_canonical_u32()), top_bound);

    if (poseidon2_result.tag == FfiOption<Poseidon2Event>::Tag::Some) poseidon2_events = poseidon2_result.some._0;

    septic_curve_t pt;
    for (int i = 0; i < 7; i++) {
      pt.x.value[i] = cols.interaction.x_coordinate._0[i];
      pt.y.value[i] = cols.interaction.y_coordinate._0[i];
    }
    return pt;
  }

  /// Kernel to convert events to trace with extra record.
  ///
  /// launch_bounds(256, 6): only a win together with the hoisted+inlined local populate() above
  /// (riscv_global_septic_local.cuh) -- that alone compiles to ~141 regs (1 block/SM); forcing
  /// 6 blocks/SM compresses to ~40 regs + some spills, the source of the ~1.85x measured for this
  /// kernel alone. launch_bounds on the stock (noinline) kernel is a regression -- never standalone.
  ///
  /// NOTE, and it bounds what that 1.85x is worth here: production does not reach this kernel.
  /// The dispatcher routes through `launch_events_to_trace_with_extra_record_landed`, which
  /// launches the `__top1` specialization at block=32 with no launch bounds, and this kernel is
  /// entered only on the `num_rows == 0` path. So two launch strategies coexist and the annotated
  /// one is the unused half. The memory read/write and poseidon2 kernels faced the same choice
  /// between two launch strategies and settled it by per-kernel profiling; this one has not
  /// been measured on these shapes, so it is left as-is rather than silently switched.
  template <class F>
  __global__ void __launch_bounds__(256, 6) events_to_trace_with_extra_record_kernel(
    const size_t event_size,
    const size_t trace_size,
    const GlobalInteractionEvent* events,
    F* trace_matrix,
    const size_t num_cols,
    const F* __restrict__ round_constants,
    septic_curve_t* d_data,
    F* byte_trace,
    Poseidon2Event* poseidon2_events)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= event_size) return;

    if (idx == 0) d_data[0] = septic_curve_t::start_point();

    Poseidon2Constants<F> poseidon2_constants = {
      WIDTH, alpha, internal_rounds, EXTERNAL_ROUNDS, round_constants};

    GlobalCols<F>* cols = reinterpret_cast<GlobalCols<F>*>(&trace_matrix[idx * num_cols]);

    d_data[idx + 1] =
      event_to_row_with_extra_record<F>(events[idx], *cols, poseidon2_constants, byte_trace, poseidon2_events[idx]);
  }

  // ===========================================================================
  // LANDED: events_to_trace_with_extra_record_kernel
  //
  // Measured at event_size 524288 / 1048576 / 4194304, the fastest launch is the same at all
  // three: the kernel with block=32 (one warp per block) instead of
  // production's block=256, recomputing grid to cover every event one-thread-per-event.
  // The win is finer block-retirement granularity for the data-dependent rejection
  // loop in global_interaction::populate — finished warps free their SM slot
  // immediately instead of a 256-thread block stranding on its slowest lane.
  //
  // The specialization body below is byte-identical to the production
  // events_to_trace_with_extra_record_kernel above. (Two `#pragma unroll` hints on the fixed-count
  // i<8 / i<7 stores, measured a no-op; omitted here to keep the body exactly equal to
  // good_default. The win is purely the block=32 launch geometry in the helper below, not the
  // body.) The body is therefore inherently generic: it computes identically to good_default for
  // any geometry that maps one thread per event over [0, event_size).
  //
  // The helper routes ALL real shapes (num_rows > 0) to the block=32 specialization
  // (grid derived from the SAME num_rows production uses — no magic constant), and
  // falls through to good_default (the unmodified production kernel, prod geometry) for
  // num_rows == 0. good_default is the correctness net for any shape the spec declines.
  // ===========================================================================

  /// Specialization: block=32 launch of the (byte-identical) production body.
  template <class F>
  __global__ void events_to_trace_with_extra_record_kernel_top1(
    const size_t event_size,
    const size_t trace_size,
    const GlobalInteractionEvent* events,
    F* trace_matrix,
    const size_t num_cols,
    const F* __restrict__ round_constants,
    septic_curve_t* d_data,
    F* byte_trace,
    Poseidon2Event* poseidon2_events)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= event_size) return;

    if (idx == 0) d_data[0] = septic_curve_t::start_point();

    Poseidon2Constants<F> poseidon2_constants = {
      WIDTH, alpha, internal_rounds, EXTERNAL_ROUNDS, round_constants};

    GlobalCols<F>* cols = reinterpret_cast<GlobalCols<F>*>(&trace_matrix[idx * num_cols]);

    d_data[idx + 1] =
      event_to_row_with_extra_record<F>(events[idx], *cols, poseidon2_constants, byte_trace, poseidon2_events[idx]);
  }

  /// Dispatcher: route shapes to the landed specialization, falling back to the
  /// unmodified production kernel (good_default) on any shape it does not serve.
  template <class F>
  static inline void launch_events_to_trace_with_extra_record_landed(
    cudaStream_t stream,
    dim3 prod_grid,
    dim3 prod_block,
    const size_t event_size,
    const size_t trace_size,
    const GlobalInteractionEvent* events,
    F* trace_matrix,
    const size_t num_cols,
    const F* round_constants,
    septic_curve_t* d_data,
    F* byte_trace,
    Poseidon2Event* poseidon2_events)
  {
    // num_rows = trace_size / num_cols is the app-level scalar that drives production's
    // own grid (block_size=256; grid=ceil(num_rows/256)). One thread per event.
    const size_t num_rows = (num_cols == 0) ? 0 : (trace_size / num_cols);

    if (num_rows > 0) {
      // block=32 (one warp), grid recomputed from num_rows to cover
      // every event. Broadly routed — the body is generic, so this is correct for any
      // shape; only speed (never correctness) is at stake outside the trained envelope.
      constexpr unsigned ETR_BLK = 32;
      dim3 lgrid((unsigned)((num_rows + ETR_BLK - 1) / ETR_BLK), 1, 1);
      dim3 lblock(ETR_BLK, 1, 1);
      events_to_trace_with_extra_record_kernel_top1<F><<<lgrid, lblock, 0, stream>>>(
        event_size, trace_size, events, trace_matrix, num_cols, round_constants,
        d_data, byte_trace, poseidon2_events);
      return;
    }

    // good_default: unmodified production kernel with production's grid/block unchanged.
    events_to_trace_with_extra_record_kernel<<<prod_grid, prod_block, 0, stream>>>(
      event_size, trace_size, events, trace_matrix, num_cols, round_constants,
      d_data, byte_trace, poseidon2_events);
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

    if (idx < event_count) {
      // partial_sums[idx] = START + sum_{j<idx} P_j (the scan ran in place, so P_idx is re-read
      // from the row's interaction columns, written by events_to_trace_with_extra_record).
      septic_curve_t point;
      for (int i = 0; i < 7; i++) {
        point.x.value[i] = cols->interaction.x_coordinate._0[i];
        point.y.value[i] = cols->interaction.y_coordinate._0[i];
      }
      global_accumulation::populate_real(cols->accumulation, partial_sums[idx], point);
    } else {
      global_interaction::populate_dummy(&(cols->interaction));
      global_accumulation::populate_dummy(cols->accumulation, final_digest);
    }
  }

  /// Seed for the no-event case: d_curve_points[0] = START, which is then both the (empty) prefix
  /// sum and the final digest, so every padding row carries cum = START exactly like the CPU
  /// tracegen (FIRST · (cum − START) must hold on an empty chunk too).
  template <class F>
  __global__ void global_seed_start_point_kernel(septic_curve_t* out)
  {
    if (blockIdx.x == 0 && threadIdx.x == 0) *out = septic_curve_t::start_point();
  }

  /// TPC_FULL prologue (1 block): build the 54-word dummy-row TEMPLATE on-device. Word c == the
  /// value the baseline global_accumulate_kernel leaves at column c of a dummy row (0 for columns
  /// it never writes — those are still 0 from this launcher's own cudaMemsetAsync, and
  /// events_to_trace_with_extra_record_kernel never touches rows >= event_size).
  /// REBUILT EVERY CALL: final_digest is per-launch data — never cache the template across
  /// launches (device-pool address reuse makes stale caching a real risk).
  template <class F>
  __global__ void global_accumulate_template_kernel(const septic_curve_t& final_digest, uint32_t* tmpl)
  {
    const int c = threadIdx.x;
    if (c >= 54) return;
    const uint32_t* fw = reinterpret_cast<const uint32_t*>(&final_digest);
    uint32_t v = 0u;                                   // cols 0..11, 12 (offset), 27..32, 33..39 (lambda): zero
    if (c >= 13 && c <= 19) v = reinterpret_cast<const uint32_t&>(septic_curve_t::dummy_x[c - 13]);
    else if (c >= 20 && c <= 26) v = reinterpret_cast<const uint32_t&>(septic_curve_t::dummy_y[c - 20]);
    else if (c >= 40 && c <= 53) v = fw[c - 40];       // cum = final_digest
    tmpl[c] = v;
  }

  /// Thread-per-row lambda witness for the real rows (TPC_FULL companion): lambda needs one septic
  /// inversion per row, so it is not a word copy like the rest of the accumulation cells. Reads
  /// cum = partial_sums[idx] and the row's interaction point (already in the trace) and writes the
  /// 7 lambda words (columns 33..39); disjoint from the cells the TPC kernel writes.
  template <class F>
  __global__ void global_lambda_kernel(
    const septic_curve_t* __restrict__ partial_sums, const size_t event_count, F* trace_matrix, const size_t num_cols)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= event_count) return;

    GlobalCols<F>* cols = reinterpret_cast<GlobalCols<F>*>(&trace_matrix[idx * num_cols]);
    septic_curve_t point;
    for (int i = 0; i < 7; i++) {
      point.x.value[i] = cols->interaction.x_coordinate._0[i];
      point.y.value[i] = cols->interaction.y_coordinate._0[i];
    }
    const septic_extension_t lambda = global_accumulation::slope(partial_sums[idx], point);
    for (int i = 0; i < 7; i++) {
      cols->accumulation.lambda[0]._0[i] = lambda.value[i];
    }
  }

  /// Thread-per-cell accumulation kernel (TPC_FULL): threads [0, real_cells) copy the real
  /// region — row = t/14, col = 40 + t%14, i.e. the 14 `cum` cells populate_real writes
  /// (cum = partial_sums[row]; lambda is written by global_lambda_kernel) — so partial_sums
  /// reads coalesce and the 56B/row span stores replace the baseline's 4B-at-216B-stride
  /// dribble. Threads [real_cells, real_cells+dummy_cells) fill the dummy region as ONE
  /// contiguous coalesced span (all 54 cols incl. the zero cols — byte-equal because the buffer
  /// arrives zeroed, see the memset argument on the template kernel above) from the smem-cached
  /// template.
  template <class F>
  __global__ void global_accumulate_tpc_kernel(
    const uint32_t* __restrict__ ps_words,
    const uint32_t* __restrict__ tmpl,
    const uint32_t real_cells,
    const uint32_t dummy_cells,
    const uint32_t event_count,
    uint32_t* trace)
  {
    __shared__ uint32_t s_tmpl[54];
    if (threadIdx.x < 54) s_tmpl[threadIdx.x] = __ldg(&tmpl[threadIdx.x]);
    __syncthreads();
    const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < real_cells) {
      const uint32_t row = t / 14u;
      const uint32_t c = t - row * 14u;
      trace[row * 54u + 40u + c] = __ldg(&ps_words[row * 14u + c]); // cum = partial_sums[row]
    } else {
      const uint32_t t2 = t - real_cells;
      if (t2 >= dummy_cells) return;
      // dummy region = one contiguous span: word w of the region is column w%54
      trace[event_count * 54u + t2] = s_tmpl[t2 % 54u];
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
    const size_t poseidon2_events_start_index,
    const F* round_constants)
  {
    try {
      CUDA_OK(cudaMemsetAsync(trace, 0, trace_size * sizeof(F), stream));

      const int block_size = 256;
      const size_t padded_event_size = trace_size / num_cols;

      // After the scan: d_curve_points[k] = START + sum_{j<k} P_j (row k's incoming sum) and
      // d_curve_points[event_size] is the chunk total (final digest).
      dev_ptr_t<septic_curve_t> d_curve_points{event_size + 1, stream};

      if (event_size == 0) {
        // No real rows: the whole trace is padding with cum = START (the empty sum), lambda = 0,
        // exactly the CPU tracegen's n = 0 output. Do NOT return early on the zero buffer — cum
        // would then be 0 and the FIRST-row constraint would fail.
        global_seed_start_point_kernel<F><<<1, 1, 0, stream>>>(&d_curve_points[0]);
        CUDA_OK(cudaGetLastError());
      } else {
        const int num_rows = trace_size / num_cols;
        const int grid_size = (num_rows + block_size - 1) / block_size;

        launch_events_to_trace_with_extra_record_landed<F>(stream, dim3(grid_size, 1, 1), dim3(block_size, 1, 1),
          event_size, trace_size, events, trace, num_cols, round_constants, &d_curve_points[0], byte_trace,
          &poseidon2_events[poseidon2_events_start_index]);

        CUDA_OK(cudaStreamSynchronize(stream));

        // Use CUB instead of Thrust for inclusive scan.
        RustError scan_result =
          cub_elliptic_curve_scan<F>(&d_curve_points[0], &d_curve_points[0], event_size + 1, stream);
        if (scan_result.code != cudaSuccess) return scan_result;
      }

      // Launch accumulation kernel.
      const int accumulate_grid_size = (padded_event_size + block_size - 1) / block_size;

      // Thread-per-cell fast path: only when the runtime width matches the compile-time
      // GlobalCols layout the TPC kernels hardcode (54 u32 cols) and the cell indices fit u32
      // arithmetic. Anything else falls back to the original thread-per-row kernel. Both paths
      // must produce byte-identical traces (the fallback is the reference).
      constexpr size_t kGlobalNumCols = sizeof(GlobalCols<F>) / sizeof(F);
      static_assert(kGlobalNumCols == 54, "riscv_global TPC kernels hardcode the 54-column GlobalCols layout");
      if (num_cols == kGlobalNumCols && kGlobalNumCols == 54 && sizeof(F) == sizeof(uint32_t) &&
          trace_size <= 0xFFFFFFFFull) {
        dev_ptr_t<uint32_t> d_tmpl{54, stream}; // 216B template scratch (freed like d_curve_points)
        global_accumulate_template_kernel<F><<<1, 64, 0, stream>>>(
          d_curve_points[event_size], &d_tmpl[0]);
        // lambda (7 words/row) needs a septic inversion per row: thread-per-row, real rows only.
        if (event_size > 0) {
          const int lambda_grid = (int)((event_size + block_size - 1) / block_size);
          global_lambda_kernel<F><<<lambda_grid, block_size, 0, stream>>>(
            &d_curve_points[0], event_size, trace, num_cols);
        }
        // cum (14 words/row) + the dummy rows: pure word copies, thread-per-cell.
        const uint32_t real_cells  = (uint32_t)(event_size * 14u);
        const uint32_t dummy_cells = (uint32_t)((padded_event_size - event_size) * 54u);
        const uint32_t total_cells = real_cells + dummy_cells;
        const int tpc_block = 512;
        const int tpc_grid = (int)((total_cells + tpc_block - 1) / tpc_block);
        global_accumulate_tpc_kernel<F><<<tpc_grid, tpc_block, 0, stream>>>(
          reinterpret_cast<const uint32_t*>(&d_curve_points[0]), &d_tmpl[0],
          real_cells, dummy_cells, (uint32_t)event_size,
          reinterpret_cast<uint32_t*>(trace));
      } else {
        global_accumulate_kernel<<<accumulate_grid_size, block_size, 0, stream>>>(
          d_curve_points[event_size], &d_curve_points[0], padded_event_size, event_size, trace, num_cols);
      }
      CUDA_OK(cudaGetLastError());

    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  }
} // namespace pico_gpu::riscv_global