// Copyright (c) 2025-2026 Brevis. All Rights Reserved.
// Licensed under the Business Source License 1.1 (BUSL-1.1).
// See LICENSE.md for details.

#pragma once

#include <iostream>
#include "common/exception.cuh"
#include "common/rusterror.h"

#include "tracegen/common/types.cuh"
#include "tracegen/common/utils.cuh"
#include "tracegen/riscv/byte.cuh"


// ═══════════════════════════════════════════════════════════════════════
// GPU-side 4-way stable partition: split AluEventU64 by Opcode
// Two-kernel approach: histogram + scatter.  Single scan of the input,
// order-preserving, zero CUB dependency, minimal temp allocation.
// ═══════════════════════════════════════════════════════════════════════

namespace pico_gpu::add_sub_common {
  using namespace byte;

  static constexpr int PARTITION_BLOCK = 256;
  static constexpr int NUM_BUCKETS     = 4;
  static constexpr int NUM_WARPS       = PARTITION_BLOCK / 32;

  __device__ __forceinline__ int opcode_to_bucket(Opcode op) {
    switch (op) {
      case Opcode::ADD:  return 0;
      case Opcode::SUB:  return 1;
      case Opcode::ADDW: return 2;
      case Opcode::SUBW: return 3;
      default:           return -1;
    }
  }

  // Kernel 1: per-block histogram of the 4 opcodes.
  __global__ void partition_histogram_kernel(
    const AluEventU64* __restrict__ events,
    const size_t n,
    int* __restrict__ block_hist)          // [num_blocks * NUM_BUCKETS]
  {
    __shared__ int s_count[NUM_BUCKETS];
    if (threadIdx.x < NUM_BUCKETS) s_count[threadIdx.x] = 0;
    __syncthreads();

    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
      const int b = opcode_to_bucket(events[idx].opcode);
      if (b >= 0) atomicAdd(&s_count[b], 1);
    }
    __syncthreads();

    if (threadIdx.x < NUM_BUCKETS)
      block_hist[blockIdx.x * NUM_BUCKETS + threadIdx.x] = s_count[threadIdx.x];
  }

  // Parallel exclusive scan of block_hist ([num_blocks][4] ints) into block_offsets,
  // run between the histogram and scatter kernels. Implemented as a 3-kernel pipeline:
  // per-tile block reductions -> single-thread scan of the <=64 tile totals ->
  // per-tile rescan + base add. O(num_blocks); preserves output order and bytes.
  __global__ void hist_tile_sum_kernel(
    const int* __restrict__ block_hist, int4* __restrict__ tile_sums, uint32_t num_blocks)
  {
    __shared__ int4 smem[1024];
    const int tid = threadIdx.x;
    uint32_t i = blockIdx.x * 1024u + tid;
    int4 v = (i < num_blocks) ? reinterpret_cast<const int4*>(block_hist)[i]
                              : make_int4(0, 0, 0, 0);
    smem[tid] = v;
    __syncthreads();
#pragma unroll
    for (int off = 512; off > 0; off >>= 1) {
      if (tid < off) {
        smem[tid].x += smem[tid + off].x;
        smem[tid].y += smem[tid + off].y;
        smem[tid].z += smem[tid + off].z;
        smem[tid].w += smem[tid + off].w;
      }
      __syncthreads();
    }
    if (tid == 0) tile_sums[blockIdx.x] = smem[0];
  }

  __global__ void hist_tile_scan_kernel(int4* tile_sums, uint32_t num_tiles) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      int4 run = make_int4(0, 0, 0, 0);
      for (uint32_t i = 0; i < num_tiles; i++) {
        int4 v = tile_sums[i];
        tile_sums[i] = run;
        run.x += v.x; run.y += v.y; run.z += v.z; run.w += v.w;
      }
    }
  }

  __global__ void hist_tile_offsets_kernel(
    const int* __restrict__ block_hist, const int4* __restrict__ tile_sums,
    int* __restrict__ offsets, uint32_t num_blocks)
  {
    __shared__ int4 smem[1024];
    const int tid = threadIdx.x;
    uint32_t i = blockIdx.x * 1024u + tid;
    int4 v = (i < num_blocks) ? reinterpret_cast<const int4*>(block_hist)[i]
                              : make_int4(0, 0, 0, 0);
    smem[tid] = v;
    __syncthreads();
#pragma unroll
    for (int off = 1; off < 1024; off <<= 1) {
      int4 add = make_int4(0, 0, 0, 0);
      if (tid >= off) add = smem[tid - off];
      __syncthreads();
      smem[tid].x += add.x;
      smem[tid].y += add.y;
      smem[tid].z += add.z;
      smem[tid].w += add.w;
      __syncthreads();
    }
    int4 excl = (tid == 0) ? make_int4(0, 0, 0, 0) : smem[tid - 1];
    int4 base = tile_sums[blockIdx.x];
    if (i < num_blocks)
      reinterpret_cast<int4*>(offsets)[i] =
        make_int4(excl.x + base.x, excl.y + base.y, excl.z + base.z, excl.w + base.w);
  }

  // Kernel 2: stable scatter using warp-ballot prefix sums.
  __global__ void partition_scatter_kernel(
    const AluEventU64* __restrict__ events,
    const size_t n,
    const int* __restrict__ block_offsets,  // pre-scanned [num_blocks][4] exclusive base offsets
    AluEventU64* out0,   // ADD
    AluEventU64* out1,   // SUB
    AluEventU64* out2,   // ADDW
    AluEventU64* out3)   // SUBW
  {
    __shared__ int global_offset[NUM_BUCKETS];
    __shared__ int warp_counts[NUM_WARPS][NUM_BUCKETS];

    // Phase A – global write offset for this block: one int4 load from the
    // pre-scanned block_offsets (see hist_tile_*_kernel above).
    if (threadIdx.x < NUM_BUCKETS)
      global_offset[threadIdx.x] = block_offsets[blockIdx.x * NUM_BUCKETS + threadIdx.x];
    __syncthreads();

    // Phase B – intra-block stable offset via warp-ballot.
    const size_t idx     = blockIdx.x * blockDim.x + threadIdx.x;
    const int    warp_id = threadIdx.x >> 5;
    const int    lane_id = threadIdx.x & 31;

    int my_bucket = -1;
    if (idx < n) my_bucket = opcode_to_bucket(events[idx].opcode);

    int my_intra_warp = 0;
    for (int b = 0; b < NUM_BUCKETS; ++b) {
      unsigned mask = __ballot_sync(0xFFFFFFFF, my_bucket == b);
      if (my_bucket == b)
        my_intra_warp = __popc(mask & ((1u << lane_id) - 1));
      if (lane_id == 0)
        warp_counts[warp_id][b] = __popc(mask);
    }
    __syncthreads();

    int cross_warp = 0;
    if (my_bucket >= 0) {
      for (int w = 0; w < warp_id; ++w)
        cross_warp += warp_counts[w][my_bucket];
    }

    // Phase C – write to the correct output buffer.
    if (my_bucket >= 0 && idx < n) {
      const int pos = global_offset[my_bucket] + cross_warp + my_intra_warp;
      const AluEventU64 ev = events[idx];
      switch (my_bucket) {
        case 0: out0[pos] = ev; break;
        case 1: out1[pos] = ev; break;
        case 2: out2[pos] = ev; break;
        case 3: out3[pos] = ev; break;
      }
    }
  }

  /// Order-preserving 4-way partition of add_sub events into separate
  /// per-opcode output buffers.  Output pointers may be null when the
  /// corresponding count is zero.
  inline RustError partition_events(
    const AluEventU64* all_events,
    const size_t total_count,
    AluEventU64* out_add,
    AluEventU64* out_sub,
    AluEventU64* out_addw,
    AluEventU64* out_subw,
    cudaStream_t stream)
  {
    if (total_count == 0) return RustError{cudaSuccess};
    try {
      const int num_blocks = (total_count + PARTITION_BLOCK - 1) / PARTITION_BLOCK;
      const uint32_t num_tiles = ((uint32_t)num_blocks + 1023u) / 1024u;

      int* d_block_hist = nullptr;
      CUDA_OK(cudaMallocAsync(&d_block_hist,
                               num_blocks * NUM_BUCKETS * sizeof(int), stream));

      // Scratch for the parallel scan: [num_blocks][4] exclusive base offsets,
      // followed by the <=64-entry (num_tiles) tile-total scratch it's built from.
      int* d_scratch = nullptr;
      CUDA_OK(cudaMallocAsync(&d_scratch,
                               ((size_t)num_blocks + num_tiles) * NUM_BUCKETS * sizeof(int),
                               stream));
      int4* d_tile_sums = reinterpret_cast<int4*>(d_scratch + (size_t)num_blocks * NUM_BUCKETS);

      partition_histogram_kernel<<<num_blocks, PARTITION_BLOCK, 0, stream>>>(
        all_events, total_count, d_block_hist);
      CUDA_OK(cudaGetLastError());

      hist_tile_sum_kernel<<<num_tiles, 1024, 0, stream>>>(
        d_block_hist, d_tile_sums, (uint32_t)num_blocks);
      CUDA_OK(cudaGetLastError());

      hist_tile_scan_kernel<<<1, 32, 0, stream>>>(d_tile_sums, num_tiles);
      CUDA_OK(cudaGetLastError());

      hist_tile_offsets_kernel<<<num_tiles, 1024, 0, stream>>>(
        d_block_hist, d_tile_sums, d_scratch, (uint32_t)num_blocks);
      CUDA_OK(cudaGetLastError());

      partition_scatter_kernel<<<num_blocks, PARTITION_BLOCK, 0, stream>>>(
        all_events, total_count, d_scratch,
        out_add, out_sub, out_addw, out_subw);
      CUDA_OK(cudaGetLastError());

      CUDA_OK(cudaFreeAsync(d_block_hist, stream));
      CUDA_OK(cudaFreeAsync(d_scratch, stream));
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }
    return RustError{cudaSuccess};
  }
} // namespace pico_gpu::add_sub_common

// ═══════════════════════════════════════════════════════════════════════
// U64 Add-only chip (separate from the 32-bit AddSub chip above)
// ═══════════════════════════════════════════════════════════════════════

namespace pico_gpu::add_u64 {
  using namespace byte;

  static constexpr size_t U64_WORD_SIZE = 4;

  template <typename T>
  struct AddGadgetU64 {
    Word<T> value;
  };

  template <typename T>
  struct AddValueColsU64 {
    AddGadgetU64<T> add_operation;
    Word<T> operand_1;
    Word<T> operand_2;
    T is_add;
  };

  template <class F>
  __device__ inline void write_word_u64(Word<F>& word, uint64_t v)
  {
    word._0[0] = F::from_canonical_u16((uint16_t)(v & 0xFFFF));
    word._0[1] = F::from_canonical_u16((uint16_t)((v >> 16) & 0xFFFF));
    word._0[2] = F::from_canonical_u16((uint16_t)((v >> 32) & 0xFFFF));
    word._0[3] = F::from_canonical_u16((uint16_t)((v >> 48) & 0xFFFF));
  }

  template <class F>
  __device__ inline void event_to_row(
    const AluEventU64& event, AddValueColsU64<F>& cols, F* byte_trace)
  {
    cols.is_add = F::one();

    uint64_t result = event.b + event.c;
    write_word_u64(cols.add_operation.value, result);
    write_word_u64(cols.operand_1, event.b);
    write_word_u64(cols.operand_2, event.c);

    add_u16_range_check<F>(byte_trace, (uint16_t)(result & 0xFFFF));
    add_u16_range_check<F>(byte_trace, (uint16_t)((result >> 16) & 0xFFFF));
    add_u16_range_check<F>(byte_trace, (uint16_t)((result >> 32) & 0xFFFF));
    add_u16_range_check<F>(byte_trace, (uint16_t)((result >> 48) & 0xFFFF));
  }

  template <class F, size_t NumCols>
  __global__ void add_events_to_trace_kernel(
    const AluEventU64* events, const size_t count,
    F* trace_matrix, const size_t num_cols, F* byte_trace)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if constexpr (NumCols > 0 && NumCols == sizeof(AddValueColsU64<F>) / sizeof(F) &&
                  sizeof(F) == sizeof(uint32_t)) {
      // WSTAGE fast path (compile-time-width specialization only): the 32 rows of a warp
      // form ONE contiguous row-major span in the trace. Each thread builds its row in
      // dynamic shared memory (stride-13 staging writes are bank-conflict-free:
      // gcd(13,32)=1), then the warp flushes the whole span linearly = fully coalesced
      // 128B stores, removing the sector amplification of per-thread strided 52B row
      // stores (2.4x standalone). Byte-identical: the staged row is zero-initialized and
      // flushed in full, matching the pre-zeroed buffer; event_to_row and the byte-lookup
      // side effects run unmodified.
      extern __shared__ uint32_t wstage_smem[];
      const int warp = threadIdx.x >> 5;
      const int lane = threadIdx.x & 31;
      uint32_t* wrow = wstage_smem + (size_t)warp * 32 * NumCols; // this warp's 32xNumCols tile
      uint32_t* myrow = wrow + (size_t)lane * NumCols;            // this thread's staged row
#pragma unroll
      for (int i = 0; i < (int)NumCols; i++) myrow[i] = 0u;
      __syncwarp();
      if (idx < count) {
        AddValueColsU64<F>* cols = reinterpret_cast<AddValueColsU64<F>*>(myrow);
        event_to_row<F>(events[idx], *cols, byte_trace);
      }
      __syncwarp();
      const size_t warp_base = (size_t)blockIdx.x * blockDim.x + (size_t)warp * 32;
      if (warp_base < count) {
        const size_t rem = count - warp_base;
        const int words = (rem >= 32 ? 32 : (int)rem) * (int)NumCols;
        uint32_t* out = reinterpret_cast<uint32_t*>(&trace_matrix[warp_base * NumCols]);
        for (int k = lane; k < words; k += 32) out[k] = wrow[k];
      }
    } else {
      if (idx >= count) return;

      AddValueColsU64<F>* cols =
        reinterpret_cast<AddValueColsU64<F>*>(&trace_matrix[idx * num_cols]);
      event_to_row<F>(events[idx], *cols, byte_trace);
    }
  }

  template <class F>
  inline RustError generate_extra_record_and_main(
    const size_t num_cols,
    const size_t event_size,
    const size_t trace_size,
    const AluEventU64* events,
    F* trace,
    F* byte_trace,
    cudaStream_t stream)
  {
    try {
      CUDA_OK(cudaMemsetAsync(trace, 0, trace_size * sizeof(F), stream));
      if (event_size == 0) return RustError{cudaSuccess};

      const int block_size = 256;
      const int grid_size = (event_size + block_size - 1) / block_size;

      // Dispatch the WSTAGE fast path on the real compile-time struct width (13 on this
      // branch). Deriving it from sizeof(AddValueColsU64<F>) keeps the dispatch in sync
      // with the kernel's internal guard (NumCols == sizeof(...)/sizeof(F)); other
      // widths fall through to the generic (NumCols=0) path.
      constexpr size_t kAddU64Cols = sizeof(AddValueColsU64<F>) / sizeof(F);
      if (num_cols == kAddU64Cols) {
        // Block 128 = 4 warps x 32 rows x 13 cols x 4B = 6.5KB dynamic smem — under the
        // 48KB default, so no cudaFuncSetAttribute opt-in is needed.
        const int wstage_block = 128;
        const size_t wstage_shmem =
          (size_t)(wstage_block / 32) * 32 * kAddU64Cols * sizeof(F);
        const int wstage_grid = (event_size + wstage_block - 1) / wstage_block;
        add_events_to_trace_kernel<F, kAddU64Cols>
          <<<wstage_grid, wstage_block, wstage_shmem, stream>>>(
            events, event_size, trace, num_cols, byte_trace);
      } else {
        add_events_to_trace_kernel<F, 0><<<grid_size, block_size, 0, stream>>>(
          events, event_size, trace, num_cols, byte_trace);
      }
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  }
} // namespace pico_gpu::add_u64

// ═══════════════════════════════════════════════════════════════════════
// U64 Addw chip — ADDW: lower-32-bit add with sign extension
// ═══════════════════════════════════════════════════════════════════════

namespace pico_gpu::addw_u64 {
  using namespace byte;

  template <typename T>
  struct U16MSBGadgetU64 {
    T msb;
  };

  template <typename T>
  struct AddwGadgetU64 {
    T value[2];            // lower 32-bit result as 2 u16 limbs
    U16MSBGadgetU64<T> msb; // MSB of value[1]
  };

  template <typename T>
  struct AddwValueColsU64 {
    AddwGadgetU64<T> addw_operation;
    Word<T> operand_1;     // 4 u16 limbs
    Word<T> operand_2;     // 4 u16 limbs
    T is_addw;
  };

  template <class F>
  __device__ inline void write_word_u64(Word<F>& word, uint64_t v)
  {
    word._0[0] = F::from_canonical_u16((uint16_t)(v & 0xFFFF));
    word._0[1] = F::from_canonical_u16((uint16_t)((v >> 16) & 0xFFFF));
    word._0[2] = F::from_canonical_u16((uint16_t)((v >> 32) & 0xFFFF));
    word._0[3] = F::from_canonical_u16((uint16_t)((v >> 48) & 0xFFFF));
  }

  template <class F>
  __device__ inline void event_to_row(
    const AluEventU64& event, AddwValueColsU64<F>& cols, F* byte_trace)
  {
    cols.is_addw = F::one();

    uint32_t lower32 = (uint32_t)event.b + (uint32_t)event.c;
    uint16_t limb0 = (uint16_t)(lower32 & 0xFFFF);
    uint16_t limb1 = (uint16_t)((lower32 >> 16) & 0xFFFF);

    cols.addw_operation.value[0] = F::from_canonical_u16(limb0);
    cols.addw_operation.value[1] = F::from_canonical_u16(limb1);

    // u16 range checks for the two result limbs
    add_u16_range_check<F>(byte_trace, limb0);
    add_u16_range_check<F>(byte_trace, limb1);

    // MSB gadget: msb = (limb1 >> 15) & 1, range check limb1 * 2
    uint16_t msb = (limb1 >> 15) & 1;
    cols.addw_operation.msb.msb = F::from_canonical_u16(msb);
    add_u16_range_check<F>(byte_trace, (uint16_t)(limb1 * 2));

    write_word_u64(cols.operand_1, event.b);
    write_word_u64(cols.operand_2, event.c);
  }

  template <class F>
  __global__ void addw_events_to_trace_kernel(
    const AluEventU64* events, const size_t count,
    F* trace_matrix, const size_t num_cols, F* byte_trace)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    AddwValueColsU64<F>* cols =
      reinterpret_cast<AddwValueColsU64<F>*>(&trace_matrix[idx * num_cols]);
    event_to_row<F>(events[idx], *cols, byte_trace);
  }

  template <class F>
  inline RustError generate_extra_record_and_main(
    const size_t num_cols,
    const size_t event_size,
    const size_t trace_size,
    const AluEventU64* events,
    F* trace,
    F* byte_trace,
    cudaStream_t stream)
  {
    try {
      CUDA_OK(cudaMemsetAsync(trace, 0, trace_size * sizeof(F), stream));
      if (event_size == 0) return RustError{cudaSuccess};

      const int block_size = 256;
      const int grid_size = (event_size + block_size - 1) / block_size;

      addw_events_to_trace_kernel<<<grid_size, block_size, 0, stream>>>(
        events, event_size, trace, num_cols, byte_trace);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  }
} // namespace pico_gpu::addw_u64

// ═══════════════════════════════════════════════════════════════════════
// U64 Sub chip — SUB: 64-bit wrapping subtraction
// ═══════════════════════════════════════════════════════════════════════

namespace pico_gpu::sub_u64 {
  using namespace byte;

  template <typename T>
  struct SubGadgetU64 {
    Word<T> value;
  };

  template <typename T>
  struct SubValueColsU64 {
    SubGadgetU64<T> sub_operation;
    Word<T> operand_1;
    Word<T> operand_2;
    T is_sub;
  };

  template <class F>
  __device__ inline void write_word_u64(Word<F>& word, uint64_t v)
  {
    word._0[0] = F::from_canonical_u16((uint16_t)(v & 0xFFFF));
    word._0[1] = F::from_canonical_u16((uint16_t)((v >> 16) & 0xFFFF));
    word._0[2] = F::from_canonical_u16((uint16_t)((v >> 32) & 0xFFFF));
    word._0[3] = F::from_canonical_u16((uint16_t)((v >> 48) & 0xFFFF));
  }

  template <class F>
  __device__ inline void event_to_row(
    const AluEventU64& event, SubValueColsU64<F>& cols, F* byte_trace)
  {
    cols.is_sub = F::one();

    uint64_t result = event.b - event.c;
    write_word_u64(cols.sub_operation.value, result);
    write_word_u64(cols.operand_1, event.b);
    write_word_u64(cols.operand_2, event.c);

    add_u16_range_check<F>(byte_trace, (uint16_t)(result & 0xFFFF));
    add_u16_range_check<F>(byte_trace, (uint16_t)((result >> 16) & 0xFFFF));
    add_u16_range_check<F>(byte_trace, (uint16_t)((result >> 32) & 0xFFFF));
    add_u16_range_check<F>(byte_trace, (uint16_t)((result >> 48) & 0xFFFF));
  }

  template <class F>
  __global__ void sub_events_to_trace_kernel(
    const AluEventU64* events, const size_t count,
    F* trace_matrix, const size_t num_cols, F* byte_trace)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    SubValueColsU64<F>* cols =
      reinterpret_cast<SubValueColsU64<F>*>(&trace_matrix[idx * num_cols]);
    event_to_row<F>(events[idx], *cols, byte_trace);
  }

  template <class F>
  inline RustError generate_extra_record_and_main(
    const size_t num_cols,
    const size_t event_size,
    const size_t trace_size,
    const AluEventU64* events,
    F* trace,
    F* byte_trace,
    cudaStream_t stream)
  {
    try {
      CUDA_OK(cudaMemsetAsync(trace, 0, trace_size * sizeof(F), stream));
      if (event_size == 0) return RustError{cudaSuccess};

      const int block_size = 256;
      const int grid_size = (event_size + block_size - 1) / block_size;

      sub_events_to_trace_kernel<<<grid_size, block_size, 0, stream>>>(
        events, event_size, trace, num_cols, byte_trace);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  }
} // namespace pico_gpu::sub_u64

// ═══════════════════════════════════════════════════════════════════════
// U64 Subw chip — SUBW: lower-32-bit sub with sign extension
// ═══════════════════════════════════════════════════════════════════════

namespace pico_gpu::subw_u64 {
  using namespace byte;

  template <typename T>
  struct U16MSBGadgetU64 {
    T msb;
  };

  template <typename T>
  struct SubwGadgetU64 {
    T value[2];
    U16MSBGadgetU64<T> msb;
  };

  template <typename T>
  struct SubwValueColsU64 {
    SubwGadgetU64<T> subw_operation;
    Word<T> operand_1;
    Word<T> operand_2;
    T is_subw;
  };

  template <class F>
  __device__ inline void write_word_u64(Word<F>& word, uint64_t v)
  {
    word._0[0] = F::from_canonical_u16((uint16_t)(v & 0xFFFF));
    word._0[1] = F::from_canonical_u16((uint16_t)((v >> 16) & 0xFFFF));
    word._0[2] = F::from_canonical_u16((uint16_t)((v >> 32) & 0xFFFF));
    word._0[3] = F::from_canonical_u16((uint16_t)((v >> 48) & 0xFFFF));
  }

  template <class F>
  __device__ inline void event_to_row(
    const AluEventU64& event, SubwValueColsU64<F>& cols, F* byte_trace)
  {
    cols.is_subw = F::one();

    int32_t lower32 = (int32_t)event.b - (int32_t)event.c;
    uint64_t value = (uint64_t)(int64_t)lower32;
    uint16_t limb0 = (uint16_t)(value & 0xFFFF);
    uint16_t limb1 = (uint16_t)((value >> 16) & 0xFFFF);

    cols.subw_operation.value[0] = F::from_canonical_u16(limb0);
    cols.subw_operation.value[1] = F::from_canonical_u16(limb1);

    add_u16_range_check<F>(byte_trace, limb0);
    add_u16_range_check<F>(byte_trace, limb1);

    uint16_t msb = (limb1 >> 15) & 1;
    cols.subw_operation.msb.msb = F::from_canonical_u16(msb);
    add_u16_range_check<F>(byte_trace, (uint16_t)(limb1 * 2));

    write_word_u64(cols.operand_1, event.b);
    write_word_u64(cols.operand_2, event.c);
  }

  template <class F>
  __global__ void subw_events_to_trace_kernel(
    const AluEventU64* events, const size_t count,
    F* trace_matrix, const size_t num_cols, F* byte_trace)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    SubwValueColsU64<F>* cols =
      reinterpret_cast<SubwValueColsU64<F>*>(&trace_matrix[idx * num_cols]);
    event_to_row<F>(events[idx], *cols, byte_trace);
  }

  template <class F>
  inline RustError generate_extra_record_and_main(
    const size_t num_cols,
    const size_t event_size,
    const size_t trace_size,
    const AluEventU64* events,
    F* trace,
    F* byte_trace,
    cudaStream_t stream)
  {
    try {
      CUDA_OK(cudaMemsetAsync(trace, 0, trace_size * sizeof(F), stream));
      if (event_size == 0) return RustError{cudaSuccess};

      const int block_size = 256;
      const int grid_size = (event_size + block_size - 1) / block_size;

      subw_events_to_trace_kernel<<<grid_size, block_size, 0, stream>>>(
        events, event_size, trace, num_cols, byte_trace);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  }
} // namespace pico_gpu::subw_u64