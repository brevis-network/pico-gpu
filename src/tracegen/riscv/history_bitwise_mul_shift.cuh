// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// The mul, bitwise, shift-left and shift-right traces built from the execution history: each chip
// reads its list through history.cuh's loaders, so its entry takes no event pointer. The chips' own
// headers hold the row builders these kernels call. Included from one translation unit only,
// through the forwarders of pico's gpu-api/cuda/history/bitwise_mul_shift.cuh.
//
// A chip's rows are its list's, in list order, each built by its chip's event_to_row from the event
// the loaders rebuild, the builder that made the chip's rows before the history, so a row and its
// byte lookups are unchanged for the same event. The four traces are checked as multisets of rows,
// so list order proves the same trace whatever order the host's streams hold. The rows past the
// list are the chips' padding: zero for mul and bitwise, the shifts' padding row for the shifts.
//
// A launch is sized from the trace, not from a count: the list's length is the descriptor's, on the
// device. A list longer than the trace, or a list row whose step is not one of the chip's opcodes,
// sets the list bit of the error word, so a fill whose lists do not match the trace the host sized
// fails the chunk rather than drop or misfile a row.
//
// What a thread writes (row_of) compiles for the host too, so it is tested off the device; the
// kernels and their launchers only under nvcc.
#pragma once

#include "tracegen/riscv/history.cuh"

#if defined(__CUDACC__)
  #include "common/rusterror.h"
  #include "tracegen/riscv/bitwise.cuh"
  #include "tracegen/riscv/mul.cuh"
  #include "tracegen/riscv/sll.cuh"
  #include "tracegen/riscv/sr.cuh"
#endif

namespace pico_gpu::alu_history {
  // Whether `opcode` is one of the opcodes whose steps `list` holds.
  REPLAY_HD inline bool list_holds(uint32_t list, Opcode opcode)
  {
    switch (list) {
    case history::LIST_MUL:
      return opcode == Opcode::MUL || opcode == Opcode::MULH || opcode == Opcode::MULHU ||
             opcode == Opcode::MULHSU || opcode == Opcode::MULW;
    case history::LIST_BITWISE:
      return opcode == Opcode::XOR || opcode == Opcode::OR || opcode == Opcode::AND;
    case history::LIST_SHIFT_LEFT:
      return opcode == Opcode::SLL || opcode == Opcode::SLLW;
    case history::LIST_SHIFT_RIGHT:
      return opcode == Opcode::SRL || opcode == Opcode::SRA || opcode == Opcode::SRLW || opcode == Opcode::SRAW;
    default:
      return false;
    }
  }

  enum class Row : uint32_t {
    // The thread writes nothing: it lies past the trace, or the fill could not be read and the
    // error word says why.
    Nothing,
    Event,
    Padding,
  };

  // What thread `idx` of a kernel over the `rows` rows of `list`'s trace writes: its list row's
  // event, padding past the list, or nothing. `count` is the list's length, the same in every
  // thread, or 0 when the fill cannot be read or the list does not fit; `reports` names the one
  // thread that reports the second.
  REPLAY_HD inline Row row_of(
    const history::HistoryArgs& args,
    uint32_t list,
    size_t rows,
    size_t idx,
    bool reports,
    uint32_t& count,
    AluEventU64& event)
  {
    count = 0;
    history::History h;
    if (!history::open(args, history::Readiness::Complete, h)) return Row::Nothing;
    const uint32_t length = history::list_length(h, list);
    if (length > rows) {
      if (reports) history::fail(h.error, history::ERROR_LIST);
      return Row::Nothing;
    }
    count = length;
    if (idx >= rows) return Row::Nothing;
    if (idx >= count) return Row::Padding;
    const uint32_t row = static_cast<uint32_t>(idx);
    Opcode opcode;
    if (!history::entry_opcode(h, list, row, opcode)) return Row::Nothing;
    if (!list_holds(list, opcode)) {
      history::fail(h.error, history::ERROR_LIST);
      return Row::Nothing;
    }
    event = history::alu_event(h, list, row);
    return Row::Event;
  }
} // namespace pico_gpu::alu_history

#if defined(__CUDACC__)

namespace pico_gpu::alu_history {
  constexpr int BLOCK_SIZE = 256;

  inline unsigned int grid_for(size_t rows, int block_size)
  {
    return static_cast<unsigned int>((rows + block_size - 1) / block_size);
  }
} // namespace pico_gpu::alu_history

namespace pico_gpu::mul_history {
  template <class F>
  __global__ void mul_history_to_trace_kernel(
    const history::HistoryArgs args, const size_t rows, F* trace_matrix, const size_t num_cols, F* byte_trace)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t count;
    AluEventU64 event;
    if (alu_history::row_of(args, history::LIST_MUL, rows, idx, idx == 0, count, event) != alu_history::Row::Event)
      return;
    mul_u64::MulValueColsU64<F>* cols = reinterpret_cast<mul_u64::MulValueColsU64<F>*>(&trace_matrix[idx * num_cols]);
    mul_u64::event_to_row<F>(event, *cols, byte_trace);
  }

  template <class F>
  inline RustError generate_extra_record_and_main(
    const history::HistoryArgs& args,
    const size_t num_cols,
    const size_t trace_size,
    F* trace,
    F* byte_trace,
    cudaStream_t stream)
  {
    try {
      CUDA_OK(cudaMemsetAsync(trace, 0, trace_size * sizeof(F), stream));
      const size_t rows = trace_size / num_cols;
      mul_history_to_trace_kernel<<<
        alu_history::grid_for(rows, alu_history::BLOCK_SIZE), alu_history::BLOCK_SIZE, 0, stream>>>(
        args, rows, trace, num_cols, byte_trace);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  }
} // namespace pico_gpu::mul_history

namespace pico_gpu::bitwise_history {
  // Two shapes: at the struct's own width a warp stages its 32 rows in shared memory and stores them
  // as one contiguous span, coalesced 128B stores in place of per-thread strided 108B row stores
  // (the 27-word stride writes the staging tile without bank conflicts, gcd(27, 32) = 1); at any
  // other width each thread stores its own row. A staged row is zeroed and flushed whole, as the
  // zeroed trace holds it, so the trace is the same either way. Every thread of a warp reaches the
  // staging's barriers, since row_of's count is the same in all of them.
  template <class F, size_t NumCols>
  __global__ void bitwise_history_to_trace_kernel(
    const history::HistoryArgs args, const size_t rows, F* trace_matrix, const size_t num_cols, F* byte_trace)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t count;
    AluEventU64 event;
    const alu_history::Row row =
      alu_history::row_of(args, history::LIST_BITWISE, rows, idx, idx == 0, count, event);
    if constexpr (NumCols > 0 && NumCols == sizeof(bitwise_u64::BitwiseValueColsU64<F>) / sizeof(F) &&
                  sizeof(F) == sizeof(uint32_t)) {
      extern __shared__ uint32_t wstage_smem[];
      const int warp = threadIdx.x >> 5;
      const int lane = threadIdx.x & 31;
      uint32_t* wrow = wstage_smem + (size_t)warp * 32 * NumCols;
      uint32_t* myrow = wrow + (size_t)lane * NumCols;
#pragma unroll
      for (int i = 0; i < (int)NumCols; i++) myrow[i] = 0u;
      __syncwarp();
      if (row == alu_history::Row::Event)
        bitwise_u64::event_to_row<F>(event, *reinterpret_cast<bitwise_u64::BitwiseValueColsU64<F>*>(myrow), byte_trace);
      __syncwarp();
      const size_t warp_base = (size_t)blockIdx.x * blockDim.x + (size_t)warp * 32;
      if (warp_base < count) {
        const size_t rem = count - warp_base;
        const int words = (rem >= 32 ? 32 : (int)rem) * (int)NumCols;
        uint32_t* out = reinterpret_cast<uint32_t*>(&trace_matrix[warp_base * NumCols]);
        for (int k = lane; k < words; k += 32) out[k] = wrow[k];
      }
    } else {
      if (row != alu_history::Row::Event) return;
      bitwise_u64::BitwiseValueColsU64<F>* cols =
        reinterpret_cast<bitwise_u64::BitwiseValueColsU64<F>*>(&trace_matrix[idx * num_cols]);
      bitwise_u64::event_to_row<F>(event, *cols, byte_trace);
    }
  }

  template <class F>
  inline RustError generate_extra_record_and_main(
    const history::HistoryArgs& args,
    const size_t num_cols,
    const size_t trace_size,
    F* trace,
    F* byte_trace,
    cudaStream_t stream)
  {
    try {
      CUDA_OK(cudaMemsetAsync(trace, 0, trace_size * sizeof(F), stream));
      const size_t rows = trace_size / num_cols;
      constexpr size_t kBitwiseU64Cols = sizeof(bitwise_u64::BitwiseValueColsU64<F>) / sizeof(F);
      if (num_cols == kBitwiseU64Cols) {
        // 4 warps x 32 rows x 27 columns x 4B = 13.5KB of dynamic shared memory, under the 48KB
        // default, so no cudaFuncSetAttribute opt-in is needed.
        const int wstage_block = 128;
        const size_t wstage_shmem = (size_t)(wstage_block / 32) * 32 * kBitwiseU64Cols * sizeof(F);
        bitwise_history_to_trace_kernel<F, kBitwiseU64Cols>
          <<<alu_history::grid_for(rows, wstage_block), wstage_block, wstage_shmem, stream>>>(
            args, rows, trace, num_cols, byte_trace);
      } else {
        bitwise_history_to_trace_kernel<F, 0>
          <<<alu_history::grid_for(rows, alu_history::BLOCK_SIZE), alu_history::BLOCK_SIZE, 0, stream>>>(
            args, rows, trace, num_cols, byte_trace);
      }
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  }
} // namespace pico_gpu::bitwise_history

namespace pico_gpu::sll_history {
  template <class F>
  __global__ void sll_history_to_trace_kernel(
    const history::HistoryArgs args, const size_t rows, F* trace_matrix, const size_t num_cols, F* byte_trace)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t count;
    AluEventU64 event;
    const alu_history::Row row =
      alu_history::row_of(args, history::LIST_SHIFT_LEFT, rows, idx, idx == 0, count, event);
    if (row == alu_history::Row::Nothing) return;
    sll_u64::ShiftLeftValueColsU64<F>* cols =
      reinterpret_cast<sll_u64::ShiftLeftValueColsU64<F>*>(&trace_matrix[idx * num_cols]);
    if (row == alu_history::Row::Padding)
      sll_u64::assign_padding_row<F>(*cols);
    else
      sll_u64::event_to_row<F>(event, *cols, byte_trace);
  }

  template <class F>
  inline RustError generate_extra_record_and_main(
    const history::HistoryArgs& args,
    const size_t num_cols,
    const size_t trace_size,
    F* trace,
    F* byte_trace,
    cudaStream_t stream)
  {
    try {
      CUDA_OK(cudaMemsetAsync(trace, 0, trace_size * sizeof(F), stream));
      const size_t rows = trace_size / num_cols;
      sll_history_to_trace_kernel<<<
        alu_history::grid_for(rows, alu_history::BLOCK_SIZE), alu_history::BLOCK_SIZE, 0, stream>>>(
        args, rows, trace, num_cols, byte_trace);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  }
} // namespace pico_gpu::sll_history

namespace pico_gpu::sr_history {
  template <class F>
  __global__ void sr_history_to_trace_kernel(
    const history::HistoryArgs args, const size_t rows, F* trace_matrix, const size_t num_cols, F* byte_trace)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t count;
    AluEventU64 event;
    const alu_history::Row row =
      alu_history::row_of(args, history::LIST_SHIFT_RIGHT, rows, idx, idx == 0, count, event);
    if (row == alu_history::Row::Nothing) return;
    sr_u64::ShiftRightValueColsU64<F>* cols =
      reinterpret_cast<sr_u64::ShiftRightValueColsU64<F>*>(&trace_matrix[idx * num_cols]);
    if (row == alu_history::Row::Padding)
      sr_u64::assign_padding_row<F>(*cols);
    else
      sr_u64::event_to_row<F>(event, *cols, byte_trace);
  }

  template <class F>
  inline RustError generate_extra_record_and_main(
    const history::HistoryArgs& args,
    const size_t num_cols,
    const size_t trace_size,
    F* trace,
    F* byte_trace,
    cudaStream_t stream)
  {
    try {
      CUDA_OK(cudaMemsetAsync(trace, 0, trace_size * sizeof(F), stream));
      const size_t rows = trace_size / num_cols;
      sr_history_to_trace_kernel<<<
        alu_history::grid_for(rows, alu_history::BLOCK_SIZE), alu_history::BLOCK_SIZE, 0, stream>>>(
        args, rows, trace, num_cols, byte_trace);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  }
} // namespace pico_gpu::sr_history

#endif // defined(__CUDACC__)
