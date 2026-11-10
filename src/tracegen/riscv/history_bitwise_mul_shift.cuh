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

#include "tracegen/riscv/history_launch.cuh"

#if defined(__CUDACC__)
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

  using history::Row;

  // What thread `idx` of a kernel over the `rows` rows of `list`'s trace writes, `h` being the fill
  // opened for the whole history: its list row's event, padding past the list, or nothing. `count`
  // is the list's length, the same in every thread, or 0 when the list does not fit; `reports` names
  // the one thread that reports it. The row is fetched once, its opcode tested between the fetch's
  // halves; a row whose accesses cannot be read is the zero event's, the error word failing the
  // chunk.
  REPLAY_HD inline Row row_of(
    const history::History& h,
    uint32_t list,
    size_t rows,
    size_t idx,
    bool reports,
    uint32_t& count,
    AluEventU64& event)
  {
    count = 0;
    const uint32_t length = history::list_length(h, list);
    if (length > rows) {
      if (reports) history::fail(h.error, history::ERROR_LIST);
      return Row::Nothing;
    }
    count = length;
    if (idx >= rows) return Row::Nothing;
    if (idx >= count) return Row::Padding;
    uint32_t index = 0;
    history::Fetched fetched;
    if (!history::fetch_entry(h, list, static_cast<uint32_t>(idx), index, fetched)) return Row::Nothing;
    if (!list_holds(list, fetched.instruction.opcode)) {
      history::fail(h.error, history::ERROR_LIST);
      return Row::Nothing;
    }
    event = history::fetch_accesses(h, index, fetched) ? history::alu_event_of(fetched) : AluEventU64{};
    return Row::Event;
  }
} // namespace pico_gpu::alu_history

#if defined(__CUDACC__)

namespace pico_gpu::alu_history {
  constexpr int BLOCK_SIZE = 256;

  // The row thread `idx` writes, the fill opened once for its block; `count` 0 when the fill cannot
  // be opened. Every thread of the block calls it first, since the opening's barrier waits for all
  // of them.
  __device__ inline Row block_row_of(
    const history::HistoryArgs& args, uint32_t list, size_t rows, size_t idx, uint32_t& count, AluEventU64& event)
  {
    count = 0;
    const history::History* h = history::open_block(args, history::Readiness::Complete);
    return h ? row_of(*h, list, rows, idx, idx == 0, count, event) : Row::Nothing;
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
    if (alu_history::block_row_of(args, history::LIST_MUL, rows, idx, count, event) != alu_history::Row::Event) return;
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
    return history::launch_zeroed(trace, trace_size, stream, [&] {
      const size_t rows = trace_size / num_cols;
      history::refuse_no_rows(rows);
      mul_history_to_trace_kernel<<<
        history::grid_for(rows, alu_history::BLOCK_SIZE), alu_history::BLOCK_SIZE, 0, stream>>>(
        args, rows, trace, num_cols, byte_trace);
      history::check_launch();
    });
  }
} // namespace pico_gpu::mul_history

namespace pico_gpu::bitwise_history {
  // Two shapes: at the struct's own width a warp stages its 32 rows and stores them as one span
  // (history::stage_warp_rows; the 27-word stride writes the staging tile without bank conflicts,
  // gcd(27, 32) = 1); at any other width each thread stores its own row. Every thread of a warp
  // reaches the staging's barriers, since row_of's count is the same in all of them.
  template <class F, size_t NumCols>
  __global__ void bitwise_history_to_trace_kernel(
    const history::HistoryArgs args, const size_t rows, F* trace_matrix, const size_t num_cols, F* byte_trace)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t count;
    AluEventU64 event;
    const alu_history::Row row = alu_history::block_row_of(args, history::LIST_BITWISE, rows, idx, count, event);
    if constexpr (NumCols > 0 && NumCols == sizeof(bitwise_u64::BitwiseValueColsU64<F>) / sizeof(F) &&
                  sizeof(F) == sizeof(uint32_t)) {
      history::stage_warp_rows<NumCols>(trace_matrix, 0, count, row == alu_history::Row::Event, [&](uint32_t* staged) {
        bitwise_u64::event_to_row<F>(
          event, *reinterpret_cast<bitwise_u64::BitwiseValueColsU64<F>*>(staged), byte_trace);
      });
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
    return history::launch_zeroed(trace, trace_size, stream, [&] {
      const size_t rows = trace_size / num_cols;
      history::refuse_no_rows(rows);
      constexpr size_t kBitwiseU64Cols = sizeof(bitwise_u64::BitwiseValueColsU64<F>) / sizeof(F);
      if (num_cols == kBitwiseU64Cols) {
        // 4 warps x 32 rows x 27 columns x 4B = 13.5KB of dynamic shared memory, under the 48KB
        // default, so no cudaFuncSetAttribute opt-in is needed.
        const int wstage_block = 128;
        const size_t wstage_shmem = (size_t)(wstage_block / 32) * 32 * kBitwiseU64Cols * sizeof(F);
        bitwise_history_to_trace_kernel<F, kBitwiseU64Cols>
          <<<history::grid_for(rows, wstage_block), wstage_block, wstage_shmem, stream>>>(
            args, rows, trace, num_cols, byte_trace);
      } else {
        bitwise_history_to_trace_kernel<F, 0>
          <<<history::grid_for(rows, alu_history::BLOCK_SIZE), alu_history::BLOCK_SIZE, 0, stream>>>(
            args, rows, trace, num_cols, byte_trace);
      }
      history::check_launch();
    });
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
    const alu_history::Row row = alu_history::block_row_of(args, history::LIST_SHIFT_LEFT, rows, idx, count, event);
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
    return history::launch_zeroed(trace, trace_size, stream, [&] {
      const size_t rows = trace_size / num_cols;
      history::refuse_no_rows(rows);
      sll_history_to_trace_kernel<<<
        history::grid_for(rows, alu_history::BLOCK_SIZE), alu_history::BLOCK_SIZE, 0, stream>>>(
        args, rows, trace, num_cols, byte_trace);
      history::check_launch();
    });
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
    const alu_history::Row row = alu_history::block_row_of(args, history::LIST_SHIFT_RIGHT, rows, idx, count, event);
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
    return history::launch_zeroed(trace, trace_size, stream, [&] {
      const size_t rows = trace_size / num_cols;
      history::refuse_no_rows(rows);
      sr_history_to_trace_kernel<<<
        history::grid_for(rows, alu_history::BLOCK_SIZE), alu_history::BLOCK_SIZE, 0, stream>>>(
        args, rows, trace, num_cols, byte_trace);
      history::check_launch();
    });
  }
} // namespace pico_gpu::sr_history

#endif // defined(__CUDACC__)
