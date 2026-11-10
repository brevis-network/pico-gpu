// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// What every kernel that reads the execution history shares with the others: the row a thread
// writes, the grid over a trace, the one opening of the fill per block, the warp staging of whole
// rows, and the launch an entry makes. history.cuh holds the fill and its loaders; each chip family's
// header (history_*.cuh) includes this one.
//
// A kernel opens the fill once per block (open_block): one thread reads the descriptor and, where the
// kernel needs it, the completion word, into shared memory, and every thread of the block reads that
// view. The opening is the kernel's first act, before any thread returns, and reaches every thread's
// barrier; a fill that cannot be opened sets the same bits of the error word whichever thread opens
// it. Each kernel keeps its readiness: the descriptor alone for the CPU slices, the CPU padding and
// the validation, the completion word too for every other reader.
//
// A grid covers its threads with one block at least (grid_for), so a launch is never of no blocks. A
// trace of no rows is the entry's to decide: the mul, bitwise, shift, ecall and syscall entries refuse
// it with the launch-configuration error (refuse_no_rows); every other entry launches over it, so a
// fill holding rows such a trace cannot hold still sets the error word.
//
// Each launch is checked where it is enqueued (check_launch), so an entry that enqueues no kernel
// reads no launch error, which would be another call's.
//
// Row compiles for the host too, with the *_row_of functions that return it; the rest only under
// nvcc.
#pragma once

#include "tracegen/riscv/history.cuh"

#if defined(__CUDACC__)
  #include "common/exception.cuh"
  #include "common/rusterror.h"
#endif

namespace pico_gpu::history {
  // What one thread of a trace kernel writes.
  enum class Row : uint32_t {
    // The thread writes nothing: it lies past what it builds, or the fill could not be read and the
    // error word says why.
    Nothing,
    Event,
    Padding,
  };
} // namespace pico_gpu::history

#if defined(__CUDACC__)

namespace pico_gpu::history {
  // The blocks of `block_size` threads that cover `threads` threads, one at least.
  inline unsigned int grid_for(size_t threads, int block_size)
  {
    const size_t blocks = (threads + block_size - 1) / block_size;
    return static_cast<unsigned int>(blocks > 0 ? blocks : 1);
  }

  // The fill `args` names, opened once for the whole block: thread 0 opens it into shared memory and,
  // after the barrier, every thread reads that one view. Null when the fill cannot be opened, the
  // error word set as open sets it. Every thread of the block calls it, at the kernel's entry and
  // before any thread returns, since the barriers wait for all of them. The view lives in one
  // shared slot per kernel, so the first barrier keeps a second opening in the same kernel from
  // overwriting it while a thread still reads the first.
  __device__ inline const History* open_block(const HistoryArgs& args, Readiness readiness)
  {
    __shared__ History fill;
    __shared__ bool opened;
    __syncthreads();
    if (threadIdx.x == 0) opened = open(args, readiness, fill);
    __syncthreads();
    return opened ? &fill : nullptr;
  }

  // A trace of NumCols 32-bit words per row, written a warp at a time: the warp stages its 32 rows in
  // shared memory and stores them as one contiguous span, coalesced 128B stores in place of
  // per-thread strided row stores. The warp's threads are the kernel's rows from
  // blockIdx.x * blockDim.x + 32 * warp, each stored at that row plus `first` of the trace, those
  // past `count` not at all. A staged row is zeroed, `build(row)` writes it where the thread's row
  // is `real`, and it is flushed whole, as the zeroed trace holds it, so the trace is that of a
  // per-thread store into the zeroed trace. Every thread of the warp calls it, with the same `first`
  // and `count`, since it reaches the warp's barriers. The kernel is launched with 32 * NumCols words
  // of dynamic shared memory per warp.
  template <size_t NumCols, class F, class Build>
  __device__ inline void stage_warp_rows(F* trace_matrix, size_t first, size_t count, bool real, Build build)
  {
    extern __shared__ uint32_t wstage_smem[];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    uint32_t* wrow = wstage_smem + (size_t)warp * 32 * NumCols;
    uint32_t* myrow = wrow + (size_t)lane * NumCols;
  #pragma unroll
    for (int i = 0; i < (int)NumCols; i++)
      myrow[i] = 0u;
    __syncwarp();
    if (real) build(myrow);
    __syncwarp();
    const size_t warp_base = (size_t)blockIdx.x * blockDim.x + (size_t)warp * 32;
    if (warp_base < count) {
      const size_t rem = count - warp_base;
      const int words = (rem >= 32 ? 32 : (int)rem) * (int)NumCols;
      uint32_t* out = reinterpret_cast<uint32_t*>(&trace_matrix[(first + warp_base) * NumCols]);
      for (int k = lane; k < words; k += 32)
        out[k] = wrow[k];
    }
  }

  // Throws the error of the launch just enqueued, if it failed. Called right after every launch, so
  // a failed launch enqueues nothing after it.
  inline void check_launch() { CUDA_OK(cudaGetLastError()); }

  // An entry's launch: `launches` enqueues the chip's kernels on its stream, calling check_launch
  // after each. A CUDA error it throws becomes the RustError returned.
  template <class Launches>
  inline RustError launch(Launches launches)
  {
    try {
      launches();
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }
    return RustError{cudaSuccess};
  }

  // launch on the chip's trace, zeroed on `stream` first.
  template <class F, class Launches>
  inline RustError launch_zeroed(F* trace, size_t trace_size, cudaStream_t stream, Launches launches)
  {
    return launch([&] {
      CUDA_OK(cudaMemsetAsync(trace, 0, trace_size * sizeof(F), stream));
      launches();
    });
  }

  // Refuses a trace of no rows with the error a launch of no blocks returns.
  inline void refuse_no_rows(size_t rows)
  {
    if (rows == 0) CUDA_OK(cudaErrorInvalidConfiguration);
  }
} // namespace pico_gpu::history

#endif // defined(__CUDACC__)
