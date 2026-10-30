// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0

#pragma once

#include "field/ff_config.hpp"
#include "tracegen/common/poseidon2.cuh"

using namespace ::pico_gpu::poseidon2;

namespace pico_gpu::riscv_poseidon2 {
  /// Row layout of one riscv permutation slot: rate-8 inputs only and the last ending full round
  /// committed on its 7 looked-up output lanes only (148 columns for KoalaBear).
  /// The Rust side passes `RISCV_NUM_POSEIDON2_VALUE_COLS` as `num_cols`; BOTH kernels below
  /// (baseline and staged) use this layout, so a width mismatch can never silently fall back to
  /// the full-width `Poseidon2ValueColsT` layout.
  using RiscvPoseidon2ValueColsT = RiscvPoseidon2ValueCols<field_t, half_external_rounds, internal_rounds>;

  __host__ __device__ inline void event_to_row(
    const Poseidon2Event& event, RiscvPoseidon2ValueColsT& cols, const field_t* __restrict__ round_constants)
  {
    cols.is_real = field_t::from_bool(true);
    // The permutation still runs on the full 16-lane state; only the rate part is committed (the
    // capacity lanes are zero for every event the Global chip emits and are AIR constants).
    field_t state[WIDTH];
    for (int i = 0; i < WIDTH; i++) {
      state[i] = field_t::from_canonical_u32(event.input[i]);
    }
    for (int i = 0; i < RISCV_POSEIDON2_NUM_INPUTS; i++) {
      cols.input[i] = state[i];
    }
    permute_and_populate_rounds_rate7(
      state,
      cols.beginning_full_rounds,
      cols.partial_rounds,
      cols.ending_full_rounds,
      cols.last_full_round.post,
      round_constants);
  }

  __global__ void events_to_rows_kernel(
    const Poseidon2Event* __restrict__ events,
    field_t* __restrict__ trace_matrix,
    size_t events_count,
    size_t num_cols,
    const field_t* __restrict__ round_constants)
  {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= events_count) return;

    RiscvPoseidon2ValueColsT* cols = reinterpret_cast<RiscvPoseidon2ValueColsT*>(&trace_matrix[idx * num_cols]);
    event_to_row(events[idx], *cols, round_constants);
  }

  // ---------------------------------------------------------------------------
  // Fast path (num_cols == sizeof(RiscvPoseidon2ValueColsT)/sizeof(field_t)).
  //
  // The baseline kernel dribbles 148 x 4B stores per thread at row stride. The
  // fast path instead (1) runs the permutation on a REGISTER state array,
  // (2) emits every row word in the original left-to-right write order with the
  // exact last-write-wins values into a 32-row per-warp shared-memory staging
  // buffer, and (3) flushes the warp's rows warp-linearly in TWO column phases
  // ([0,73) then [73,148)) so the stores coalesce into full sectors.
  // Byte-identical output to the baseline kernel.
  // ---------------------------------------------------------------------------
  namespace staged {
    constexpr size_t NUM_VALUE_COLS = sizeof(RiscvPoseidon2ValueColsT) / sizeof(field_t); // 148

    // Row word offsets (row layout = RiscvPoseidon2ValueColsT)
    constexpr int OFF_IS_REAL = 0;
    constexpr int OFF_INPUT = OFF_IS_REAL + 1;                           // 1
    constexpr int OFF_BEG = OFF_INPUT + (int)RISCV_POSEIDON2_NUM_INPUTS; // 9
    constexpr int OFF_PART = OFF_BEG + half_external_rounds * WIDTH;     // 73
    constexpr int OFF_END = OFF_PART + internal_rounds;                  // 93
    // ending_full_rounds[0..half-1) are full 16-lane rounds; last_full_round.post holds 7 lanes.
    static_assert(
      (int)(OFF_END + (half_external_rounds - 1) * WIDTH + RISCV_POSEIDON2_NUM_OUTPUTS) == (int)NUM_VALUE_COLS,
      "staged offsets must tile the row");

    // Two flush phases: cols [0, OFF_PART) then [OFF_PART, NUM_VALUE_COLS)
    constexpr int SEG0_WIDTH = OFF_PART;                             // 73
    constexpr int SEG1_WIDTH = (int)NUM_VALUE_COLS - OFF_PART;       // 75
    // Per-lane smem row stride: >= max segment width, odd -> bank-conflict-free
    constexpr int STRIDE = (SEG0_WIDTH > SEG1_WIDTH ? SEG0_WIDTH : SEG1_WIDTH) | 1; // 75
    constexpr int BLOCK_SIZE = 128;
    // 4 warps * 32 * 75 * 4B = 38.4KB dyn smem (< 48KB default cap, no opt-in)
    constexpr size_t SMEM_BYTES = (size_t)(BLOCK_SIZE / 32) * 32u * STRIDE * sizeof(uint32_t);

    // Run the whole permutation on a register state array, calling
    // `emit(col_index, value)` for every row word in the EXACT order/values the
    // baseline event_to_row writes them (is_real, the 8 committed inputs, then
    // the round witnesses).
    template <typename EmitFn>
    __device__ __forceinline__ void
    event_to_row_reg(const Poseidon2Event& event, const field_t* __restrict__ round_constants, EmitFn&& emit)
    {
      field_t state[WIDTH];

      emit(OFF_IS_REAL, field_t::from_bool(true));
#pragma unroll
      for (int i = 0; i < WIDTH; i++) {
        field_t v = field_t::from_canonical_u32(event.input[i]);
        state[i] = v;
        if (i < (int)RISCV_POSEIDON2_NUM_INPUTS) emit(OFF_INPUT + i, v); // only the rate part is committed
      }

      mds_light_permutation(state);

      size_t rc_offset = 0;
      for (unsigned int cnt = 0; cnt < half_external_rounds; cnt++) {
        add_round_constants(state, rc_offset, round_constants);
#pragma unroll
        for (int i = 0; i < WIDTH; i++) {
          state[i] = sbox_el(state[i]);
        }
        mds_light_permutation(state);
        rc_offset += WIDTH;
#pragma unroll
        for (int i = 0; i < WIDTH; i++) {
          emit(OFF_BEG + cnt * WIDTH + i, state[i]);
        }
      }

      for (unsigned int cnt = 0; cnt < internal_rounds; cnt++) {
        field_t element = state[0] + round_constants[rc_offset];
        element = sbox_el(element);
        emit(OFF_PART + cnt, element); // partial_rounds[cnt].post_sbox

        field_t part_sum = 0;
#pragma unroll
        for (int i = 1; i < WIDTH; i++) {
          part_sum = part_sum + state[i];
        }

        field_t full_sum = part_sum + element;
        state[0] = part_sum - element;
        internal_layer_mat_mul(state, full_sum);
        rc_offset += WIDTH;
      }

      for (unsigned int cnt = 0; cnt < half_external_rounds; cnt++) {
        add_round_constants(state, rc_offset, round_constants);
#pragma unroll
        for (int i = 0; i < WIDTH; i++) {
          state[i] = sbox_el(state[i]);
        }
        mds_light_permutation(state);
        rc_offset += WIDTH;
        // The last ending round commits only the looked-up output lanes (last_full_round.post,
        // which directly follows ending_full_rounds[half-2] in the row); its lane 6 lands on
        // column NUM_VALUE_COLS - 1, the second flush trigger.
        const int lanes = (cnt + 1 == half_external_rounds) ? (int)RISCV_POSEIDON2_NUM_OUTPUTS : WIDTH;
#pragma unroll
        for (int i = 0; i < WIDTH; i++) {
          if (i < lanes) emit(OFF_END + cnt * WIDTH + i, state[i]);
        }
      }
    }

    // Flush the warp's staged column segment [c0, c0+W) for rows_in_warp rows:
    // consecutive lanes cover consecutive words of each row segment, so the
    // stores coalesce to full sectors.
    template <int W>
    __device__ __forceinline__ void flush_seg(
      const uint32_t* __restrict__ buf,
      uint32_t* __restrict__ trace,
      size_t warp_row_base,
      size_t num_cols,
      int c0,
      unsigned rows_in_warp,
      int lane)
    {
      __syncwarp();
      unsigned total = rows_in_warp * W;
      for (unsigned w = lane; w < total; w += 32) {
        unsigned row = w / W;
        unsigned col = w - row * W;
        trace[(warp_row_base + row) * num_cols + (unsigned)c0 + col] = buf[row * STRIDE + col];
      }
      __syncwarp();
    }

    __global__ void events_to_rows_staged_kernel(
      const Poseidon2Event* __restrict__ events,
      field_t* __restrict__ trace_matrix,
      size_t events_count,
      size_t num_cols,
      const field_t* __restrict__ round_constants)
    {
      extern __shared__ uint32_t smem[];
      const int lane = threadIdx.x & 31;
      const int warp_in_block = threadIdx.x >> 5;
      uint32_t* buf = smem + (size_t)warp_in_block * 32u * STRIDE;
      uint32_t* my = buf + (size_t)lane * STRIDE;

      size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
      size_t warp_row_base = idx - lane;
      if (warp_row_base >= events_count) return; // whole warp out of range
      size_t rem = events_count - warp_row_base;
      unsigned rows_in_warp = rem < 32 ? (unsigned)rem : 32u;
      // Clamp: tail lanes past events_count compute a duplicate event; their smem
      // rows are never flushed. Keeps the warp convergent (uniform emit sequence
      // -> __syncwarp safe).
      size_t idx_c = idx < events_count ? idx : events_count - 1;
      uint32_t* trace_u32 = reinterpret_cast<uint32_t*>(trace_matrix);

      event_to_row_reg(events[idx_c], round_constants, [&](int c, field_t v) {
        if (c < OFF_PART) {
          my[c] = v.val;
          if (c == OFF_PART - 1) {
            flush_seg<SEG0_WIDTH>(buf, trace_u32, warp_row_base, num_cols, 0, rows_in_warp, lane);
          }
        } else {
          my[c - OFF_PART] = v.val;
          if (c == (int)NUM_VALUE_COLS - 1) {
            flush_seg<SEG1_WIDTH>(buf, trace_u32, warp_row_base, num_cols, OFF_PART, rows_in_warp, lane);
          }
        }
      });
    }
  } // namespace staged

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

  inline RustError generate_main(
    size_t num_cols,
    size_t event_size,
    size_t trace_size,
    const Poseidon2Event* events,
    field_t* trace,
    field_t* dummy_row,
    const field_t* round_constants,
    cudaStream_t stream)
  {
    try {

      // initialize trace matrix with zeros
      CUDA_OK(cudaMemsetAsync(trace, 0, trace_size * sizeof(field_t), stream));

      // launch kernel
      // generate real rows by events
      {
        // "staged" path: register-state permutation with phased warp flush.
        // Chosen over a per-__launch_bounds__ specialized launcher, which profiled ~2.6x slower.
        if (num_cols == staged::NUM_VALUE_COLS) {
          const int block_size = staged::BLOCK_SIZE;
          const int grid_size = (event_size + block_size - 1) / block_size;

          staged::events_to_rows_staged_kernel<<<grid_size, block_size, staged::SMEM_BYTES, stream>>>(
            events, trace, event_size, num_cols, round_constants);
        } else {
          const int block_size = 256;
          const int grid_size = (event_size + block_size - 1) / block_size;

          events_to_rows_kernel<<<grid_size, block_size, 0, stream>>>(
            events, trace, event_size, num_cols, round_constants);
        }
        CUDA_OK(cudaGetLastError());
      }

      // generate padding rows
      size_t num_total_rows = trace_size / num_cols;
      if (num_total_rows > event_size) {
        const int block_size = 256;
        size_t pad_size = num_total_rows - event_size;
        int grid_size = int((pad_size + block_size - 1) / block_size);

        size_t idx_dummy_start = event_size * num_cols;

        pad_dummy_rows_kernel<<<grid_size, block_size, 0, stream>>>(
          &trace[idx_dummy_start], pad_size, num_cols, &dummy_row[0]);
        CUDA_OK(cudaGetLastError());
      }
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  } // generate_main
} // namespace pico_gpu::riscv_poseidon2
