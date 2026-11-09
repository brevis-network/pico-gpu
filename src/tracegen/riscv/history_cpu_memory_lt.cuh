// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// The CPU, program, memory read/write and less-than traces built from the execution history: each
// chip reads its steps or its list through history.cuh's loaders, so its entry takes no event
// pointer. The CPU chip reads every step, slice by slice, and writes into no other chip's buffer;
// the less-than chip's rows end with one comparison per branch step. The chips' own headers hold the
// row builders these kernels call. Included from one translation unit only, through the forwarders
// of pico's gpu-api/cuda/history/cpu_memory_lt.cuh.
//
// A CPU row is written from its step's fetched rows (cpu_u64::fetched_to_row), with no event between.
// Every other row is built by its chip's event_to_row from the event the loaders rebuild, the builder
// that made the chip's rows before the history, so a row and its byte lookups are unchanged for the
// same event. The CPU trace is in order: row i is step i, whatever slice builds it. The less-than and
// memory read/write traces are checked as multisets of rows, so list order proves the same trace
// whatever order the host's streams hold; the program chip's rows are counts by pc, so no order
// arises.
//
// A launch is sized from the trace, never from a count: the step count and the list lengths are the
// descriptor's, on the device. A trace shorter than what the fill holds, a list row whose step is
// not of the chip's opcodes, a CPU launch that does not cover the fill's slices, or a pc outside the
// program sets a bit of the error word, so the chunk fails rather than drop or misfile a row. The
// program entry, whose kernels write one row per instruction, refuses an instruction count past its
// trace's rows before either launches.
//
// What a thread writes (the *_row_of functions) compiles for the host too, so it is tested off the
// device; the kernels and their launchers only under nvcc.
#pragma once

#include "tracegen/riscv/history_launch.cuh"

#if defined(__CUDACC__)
  #include "tracegen/riscv/cpu.cuh"
  #include "tracegen/riscv/lt.cuh"
  #include "tracegen/riscv/memory_read_write.cuh"
  #include "tracegen/riscv/program.cuh"
  // The stream-hold test kernel moved to its own header; this one still includes it, so an FFI file
  // that reaches `spin` through this header alone keeps building.
  #include "tracegen/riscv/history_spin.cuh"
#endif

namespace pico_gpu::cpu_memory_lt_history {
  using history::Row;

  // What thread `idx` of the kernel for CPU slice `slice` writes, `h` being the fill opened for its
  // descriptor alone: the row of the slice's step `idx`, whose rows `fetched` holds, at trace row
  // `first + idx`, or nothing.
  // The host launched `slices` such kernels of `covered` threads each over a trace of `rows` rows; a
  // fill cut otherwise, or holding more steps than the trace has rows, writes nothing and is
  // reported by the thread `reports` names. `first` and `count` are the slice's first step and step
  // count, the same in every thread, or 0 when the slice cannot be built. The slice's own copy, the
  // step after it included, must have landed. The CPU chip reads no memory access, so the step's is
  // left unread. A step the loaders cannot read sets its bit of the error word, which fails the
  // chunk; its row is then built from a zeroed fetch.
  template <history::ExceptionLookup Lookup = history::DEFAULT_EXCEPTION_LOOKUP>
  REPLAY_HD inline Row cpu_slice_row_of(
    const history::History& h,
    uint32_t slice,
    uint32_t slices,
    size_t covered,
    size_t rows,
    size_t idx,
    bool reports,
    uint32_t& first,
    uint32_t& count,
    history::Fetched& fetched)
  {
    first = 0;
    count = 0;
    if (h.step_count > rows) {
      if (reports) history::fail(h.error, history::ERROR_STEP);
      return Row::Nothing;
    }
    if (history::cpu_slices(h) != slices) {
      if (reports) history::fail(h.error, history::ERROR_SLICE);
      return Row::Nothing;
    }
    uint32_t from = 0, end = 0;
    if (!history::cpu_slice(h, slice, from, end)) return Row::Nothing;
    if (end - from > covered) {
      if (reports) history::fail(h.error, history::ERROR_SLICE);
      return Row::Nothing;
    }
    first = from;
    count = end - from;
    if (idx >= count) return Row::Nothing;
    if (!history::fetch<Lookup, history::MemorySlot::Skip>(h, from + static_cast<uint32_t>(idx), fetched))
      fetched = history::Fetched{};
    return Row::Event;
  }

  // What thread `idx` of the CPU pad kernel over a trace of `rows` rows writes, `h` being the fill
  // opened for its descriptor alone: the padding row past the fill's steps, or nothing. Only the
  // descriptor is read, so the pad kernel may run beside the slices.
  REPLAY_HD inline Row cpu_pad_row_of(const history::History& h, size_t rows, size_t idx, bool reports)
  {
    if (h.step_count > rows) {
      if (reports) history::fail(h.error, history::ERROR_STEP);
      return Row::Nothing;
    }
    if (idx >= rows || idx < h.step_count) return Row::Nothing;
    return Row::Padding;
  }

  // What thread `idx` of a kernel over the `rows` rows of the less-than trace writes, `h` being the
  // fill opened for the whole history: its own list's steps, then one comparison per branch step,
  // padding past them, or nothing. `count` is the two lists' length, the same in every thread, or 0
  // when the lists do not fit; `reports` names the one thread that reports it. The row is fetched
  // once, its opcode tested between the fetch's halves; a row whose accesses cannot be read is the
  // zero event's, the error word failing the chunk.
  template <history::ExceptionLookup Lookup = history::DEFAULT_EXCEPTION_LOOKUP>
  REPLAY_HD inline Row lt_row_of(
    const history::History& h, size_t rows, size_t idx, bool reports, uint32_t& count, AluEventU64& event)
  {
    count = 0;
    const uint32_t own = history::list_length(h, history::LIST_LT);
    const uint64_t length = uint64_t(own) + history::list_length(h, history::LIST_BRANCH);
    if (length > rows) {
      if (reports) history::fail(h.error, history::ERROR_LIST);
      return Row::Nothing;
    }
    count = static_cast<uint32_t>(length);
    if (idx >= rows) return Row::Nothing;
    if (idx >= count) return Row::Padding;
    const uint32_t row = static_cast<uint32_t>(idx);
    const bool compare = row >= own;
    uint32_t index = 0;
    history::Fetched fetched;
    if (!history::fetch_entry(
          h, compare ? history::LIST_BRANCH : history::LIST_LT, compare ? row - own : row, index, fetched))
      return Row::Nothing;
    const Opcode opcode = fetched.instruction.opcode;
    if (compare ? !history::is_branch(opcode) : opcode != Opcode::SLT && opcode != Opcode::SLTU) {
      history::fail(h.error, history::ERROR_LIST);
      return Row::Nothing;
    }
    if (!history::fetch_accesses<Lookup>(h, index, fetched))
      event = AluEventU64{};
    else
      event = compare ? history::compare_event_of(fetched) : history::alu_event_of(fetched);
    return Row::Event;
  }

  // What thread `idx` of a kernel over the `rows` rows of the memory read/write trace writes, `h`
  // being the fill opened for the whole history: the CPU event of its memory list row, padding past
  // the list, or nothing; `count` and the fetch as lt_row_of's.
  template <history::ExceptionLookup Lookup = history::DEFAULT_EXCEPTION_LOOKUP>
  REPLAY_HD inline Row memory_row_of(
    const history::History& h, size_t rows, size_t idx, bool reports, uint32_t& count, FfiCpuEventU64& event)
  {
    count = 0;
    const uint32_t length = history::list_length(h, history::LIST_MEMORY);
    if (length > rows) {
      if (reports) history::fail(h.error, history::ERROR_LIST);
      return Row::Nothing;
    }
    count = length;
    if (idx >= rows) return Row::Nothing;
    if (idx >= count) return Row::Padding;
    uint32_t index = 0;
    history::Fetched fetched;
    if (!history::fetch_entry(h, history::LIST_MEMORY, static_cast<uint32_t>(idx), index, fetched)) return Row::Nothing;
    if (!history::is_memory(fetched.instruction.opcode)) {
      history::fail(h.error, history::ERROR_LIST);
      return Row::Nothing;
    }
    event = history::fetch_accesses<Lookup>(h, index, fetched) ? history::cpu_event_of(h, fetched) : FfiCpuEventU64{};
    return Row::Event;
  }

  // The program trace row the step at `idx` executes, among the program's `instructions` rows, `h`
  // being the fill opened for the whole history: false past the fill's steps, or with the pc bit set
  // for a pc no row holds. A pc the program table cannot fetch is never a step of an honest fill, so
  // it is reported rather than skipped.
  REPLAY_HD inline bool program_row_of(const history::History& h, size_t instructions, size_t idx, size_t& row)
  {
    if (idx >= h.step_count) return false;
    history::Step step;
    if (!history::step_at(h, static_cast<uint32_t>(idx), step)) return false;
    const uint64_t pc = step.pc;
    const uint64_t base = h.program.pc_base;
    if (pc < base || (pc - base) % 4 != 0 || (pc - base) / 4 >= instructions) {
      history::fail(h.error, history::ERROR_PC);
      return false;
    }
    row = static_cast<size_t>((pc - base) / 4);
    return true;
  }
} // namespace pico_gpu::cpu_memory_lt_history

#if defined(__CUDACC__)

namespace pico_gpu::cpu_memory_lt_history {
  constexpr int BLOCK_SIZE = 256;
  // The row-staging kernels' block: 4 warps x 32 rows x the row width stays under the 48KB default
  // of dynamic shared memory for both staged traces, so no cudaFuncSetAttribute opt-in is needed.
  // Block 256 would need it, and asking near the ~99KB sm_120 cap fails with a sticky
  // invalid-argument error.
  constexpr int WSTAGE_BLOCK = 128;
} // namespace pico_gpu::cpu_memory_lt_history

namespace pico_gpu::cpu_history {
  using cpu_memory_lt_history::Row;

  // Two shapes: at the struct's own width a warp stages its 32 rows and stores them as one span
  // (history::stage_warp_rows), in place of per-thread strided row stores that touch about eight
  // times the sectors; at any other width each thread stores its own row. Every thread of a warp
  // reaches the staging's barriers, since the slice's `first` and `count` are the same in all of
  // them.
  template <class F, size_t NumCols>
  __global__ void cpu_history_slice_kernel(
    const history::HistoryArgs args,
    const uint32_t slice,
    const uint32_t slices,
    const size_t rows,
    F* __restrict__ trace_matrix,
    const size_t num_cols,
    F* __restrict__ byte_trace)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t covered = (size_t)gridDim.x * blockDim.x;
    const history::History* h = history::open_block(args, history::Readiness::Descriptor);
    uint32_t first = 0, count = 0;
    history::Fetched fetched;
    const Row row =
      h ? cpu_memory_lt_history::cpu_slice_row_of(*h, slice, slices, covered, rows, idx, idx == 0, first, count, fetched)
        : Row::Nothing;
    if constexpr (NumCols > 0 && NumCols == sizeof(CpuColsU64<F>) / sizeof(F) && sizeof(F) == sizeof(uint32_t)) {
      history::stage_warp_rows<NumCols>(trace_matrix, first, count, row == Row::Event, [&](uint32_t* staged) {
        cpu_u64::fetched_to_row<F>(*h, fetched, *reinterpret_cast<CpuColsU64<F>*>(staged), byte_trace);
      });
    } else {
      if (row != Row::Event) return;
      const size_t stride = (NumCols > 0) ? NumCols : num_cols;
      CpuColsU64<F>* cols = reinterpret_cast<CpuColsU64<F>*>(&trace_matrix[(first + idx) * stride]);
      cpu_u64::fetched_to_row<F>(*h, fetched, *cols, byte_trace);
    }
  }

  template <class F>
  __global__ void cpu_history_pad_kernel(
    const history::HistoryArgs args, const size_t rows, F* __restrict__ trace_matrix, const size_t num_cols)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const history::History* h = history::open_block(args, history::Readiness::Descriptor);
    if (!h || cpu_memory_lt_history::cpu_pad_row_of(*h, rows, idx, idx == 0) != Row::Padding) return;
    cpu_u64::assign_padding_row<F>(*reinterpret_cast<CpuColsU64<F>*>(&trace_matrix[idx * num_cols]));
  }

  // The rows of CPU slice `slice` of `slices`. `slice_steps` is the slice length the fill was cut
  // with, which sizes the grid; the kernel holds the fill's own cut to it. The caller zeroes the
  // trace before any slice runs.
  template <class F>
  inline RustError generate_extra_record_and_main(
    const history::HistoryArgs& args,
    const uint32_t slice,
    const uint32_t slices,
    const size_t slice_steps,
    const size_t num_cols,
    const size_t trace_size,
    F* trace,
    F* byte_trace,
    cudaStream_t stream)
  {
    if (slice_steps == 0) return RustError{cudaErrorInvalidValue};
    return history::launch([&] {
      const size_t rows = trace_size / num_cols;
      constexpr uint32_t kCpuCols = sizeof(CpuColsU64<F>) / sizeof(F);
      // Pinned: the Rust `NUM_CPU_COLS` is the same 58, and the host writes `is_last` into this
      // layout by column index after the kernels run, so a silent width change on either side
      // would put that write in the wrong column.
      static_assert(sizeof(CpuColsU64<F>) / sizeof(F) == 58, "CpuCols width drift vs Rust");
      if (num_cols == kCpuCols) {
        const int block = cpu_memory_lt_history::WSTAGE_BLOCK;
        const size_t shmem = (size_t)(block / 32) * 32 * kCpuCols * sizeof(F);
        cpu_history_slice_kernel<F, kCpuCols><<<history::grid_for(slice_steps, block), block, shmem, stream>>>(
          args, slice, slices, rows, trace, num_cols, byte_trace);
      } else {
        const int block = cpu_memory_lt_history::BLOCK_SIZE;
        cpu_history_slice_kernel<F, 0><<<history::grid_for(slice_steps, block), block, 0, stream>>>(
          args, slice, slices, rows, trace, num_cols, byte_trace);
      }
      history::check_launch();
    });
  }

  // The padding rows past the fill's steps, on a zeroed trace.
  template <class F>
  inline RustError pad_trace(
    const history::HistoryArgs& args, const size_t num_cols, const size_t trace_size, F* trace, cudaStream_t stream)
  {
    return history::launch([&] {
      const size_t rows = trace_size / num_cols;
      const int block = cpu_memory_lt_history::BLOCK_SIZE;
      cpu_history_pad_kernel<F><<<history::grid_for(rows, block), block, 0, stream>>>(args, rows, trace, num_cols);
      history::check_launch();
    });
  }
} // namespace pico_gpu::cpu_history

namespace pico_gpu::program_history {
  // The fill is opened before any thread returns: every thread of the block reaches the opening's
  // barrier.
  template <class F>
  __global__ void program_history_counts_kernel(
    const history::HistoryArgs args, const size_t steps, F* trace_matrix, const size_t num_cols, const size_t instructions)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const history::History* h = history::open_block(args, history::Readiness::Complete);
    if (idx >= steps || !h) return;
    size_t row;
    if (!cpu_memory_lt_history::program_row_of(*h, instructions, idx, row)) return;
    ProgramMultCols<uint32_t>* cols = reinterpret_cast<ProgramMultCols<uint32_t>*>(&trace_matrix[row * num_cols]);
    atomicAdd(&cols->multiplicity, 1);
  }

  // Each program row's execution count, from the fill's step pcs. One thread per step of the fill
  // the arguments name: the step count sizing the grid is the launch identity's, which every thread
  // holds the descriptor to. Both kernels write the first `instructions` rows of the trace, so an
  // instruction count past its rows is refused before either launches, never cut to fit.
  template <class F>
  inline RustError generate_main(
    const history::HistoryArgs& args,
    const size_t num_cols,
    const size_t trace_size,
    const size_t instructions,
    F* trace,
    cudaStream_t stream)
  {
    const size_t rows = num_cols == 0 ? 0 : trace_size / num_cols;
    if (instructions > rows) return RustError{cudaErrorInvalidValue};
    return history::launch_zeroed(trace, trace_size, stream, [&] {
      const size_t steps = args.expected.steps;
      const int block = cpu_memory_lt_history::BLOCK_SIZE;
      if (steps > 0) {
        program_history_counts_kernel<F>
          <<<history::grid_for(steps, block), block, 0, stream>>>(args, steps, trace, num_cols, instructions);
        history::check_launch();
      }
      if (instructions > 0) {
        program_u64::to_montgomery_kernel<F>
          <<<history::grid_for(instructions, block), block, 0, stream>>>(instructions, trace, num_cols);
        history::check_launch();
      }
    });
  }
} // namespace pico_gpu::program_history

namespace pico_gpu::memory_read_write_history {
  using cpu_memory_lt_history::Row;

  // The CPU kernel's two shapes, one thread per trace row: every thread of a warp reaches the
  // staging's barriers, since memory_row_of's count is the same in all of them. The rows past the
  // list stay as the zeroed trace leaves them, which is the chip's padding.
  template <class F, size_t NumCols>
  __global__ void memory_rw_history_kernel(
    const history::HistoryArgs args, const size_t rows, F* __restrict__ trace_matrix, const size_t num_cols,
    F* __restrict__ byte_trace)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const history::History* h = history::open_block(args, history::Readiness::Complete);
    uint32_t count = 0;
    FfiCpuEventU64 event;
    const Row row = h ? cpu_memory_lt_history::memory_row_of(*h, rows, idx, idx == 0, count, event) : Row::Nothing;
    // The code a store proves it misses: the program table's, `[pc_base, pc_base + 4n)`.
    const uint64_t code_start = args.program.pc_base;
    const uint64_t code_end = args.program.pc_base + 4 * args.program.count;
    if constexpr (NumCols > 0 && NumCols == sizeof(MemoryChipValueColsU64<F>) / sizeof(F) &&
                  sizeof(F) == sizeof(uint32_t)) {
      history::stage_warp_rows<NumCols>(trace_matrix, 0, count, row == Row::Event, [&](uint32_t* staged) {
        memory_read_write_u64::event_to_row_u64<F>(
          event, *reinterpret_cast<MemoryChipValueColsU64<F>*>(staged), byte_trace, code_start, code_end);
      });
    } else {
      if (row != Row::Event) return;
      memory_read_write_u64::event_to_row_u64<F>(
        event, *reinterpret_cast<MemoryChipValueColsU64<F>*>(&trace_matrix[idx * num_cols]), byte_trace,
        code_start, code_end);
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
      constexpr size_t kMemRwCols = sizeof(MemoryChipValueColsU64<F>) / sizeof(F);
      // Pinned: the Rust `NUM_MEMORY_CHIP_VALUE_COLS` asserts the same 57 (49 and the code range's 8),
      // so a field added or removed on one side only fails the build instead of shifting every
      // column.
      static_assert(kMemRwCols == 57, "MemoryChipValueCols width drift vs Rust");
      if (num_cols == kMemRwCols) {
        const int block = cpu_memory_lt_history::WSTAGE_BLOCK;
        const size_t shmem = (size_t)(block / 32) * 32 * kMemRwCols * sizeof(F);
        memory_rw_history_kernel<F, kMemRwCols>
          <<<history::grid_for(rows, block), block, shmem, stream>>>(args, rows, trace, num_cols, byte_trace);
      } else {
        const int block = cpu_memory_lt_history::BLOCK_SIZE;
        memory_rw_history_kernel<F, 0>
          <<<history::grid_for(rows, block), block, 0, stream>>>(args, rows, trace, num_cols, byte_trace);
      }
      history::check_launch();
    });
  }
} // namespace pico_gpu::memory_read_write_history

namespace pico_gpu::lt_history {
  using cpu_memory_lt_history::Row;

  template <class F>
  __global__ void lt_history_to_trace_kernel(
    const history::HistoryArgs args, const size_t rows, F* trace_matrix, const size_t num_cols, F* byte_trace)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const history::History* h = history::open_block(args, history::Readiness::Complete);
    uint32_t count;
    AluEventU64 event;
    if (!h || cpu_memory_lt_history::lt_row_of(*h, rows, idx, idx == 0, count, event) != Row::Event) return;
    lt_u64::LtValueColsU64<F>* cols = reinterpret_cast<lt_u64::LtValueColsU64<F>*>(&trace_matrix[idx * num_cols]);
    lt_u64::event_to_row<F>(event, *cols, byte_trace);
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
      const int block = cpu_memory_lt_history::BLOCK_SIZE;
      lt_history_to_trace_kernel<<<history::grid_for(rows, block), block, 0, stream>>>(
        args, rows, trace, num_cols, byte_trace);
      history::check_launch();
    });
  }
} // namespace pico_gpu::lt_history

#endif // defined(__CUDACC__)
