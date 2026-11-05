// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// The ecall and RISC-V syscall traces built from the execution history: the ecall chip reads the
// ecall list and its public-values tally from the operands the loaders rebuild, and the syscall
// chip reads the history's syscall rows, so neither entry takes an event pointer. The chips' own
// headers hold the row builders these kernels call. Included from one translation unit only,
// through the forwarders of pico's gpu-api/cuda/history/ecall_syscall.cuh.
//
// The ecall trace is compared in order: its rows are the ecall list's, in list order, which is
// execution order. Each row is built by the chip's event_to_row from the event the loaders rebuild,
// the builder that made the chip's rows before the history, so a row and its byte lookups are
// unchanged for the same event. Its COMMIT / COMMIT_DEFERRED_PROOFS tally columns are running sums
// down the trace, padding included, so the last row carries the chunk's; the scan stops at the
// list's length as the device reads it, and the rows past it take the last list row's sums.
//
// The syscall trace is a multiset of rows, but each row also writes one global-bus event at its own
// index past the host's start index, and the global trace is in order: the history's syscall rows
// are in the stream's order, so the same event lands at the same index. The host reserved exactly
// as many global slots as it counted syscall rows; a fill holding another count would leave slots
// unwritten or overrun the next chip's, so it writes nothing and sets the syscall bit. Only the
// row's named fields are read: the syscall row's padding bytes reach no column and no message.
//
// A launch is sized from the trace, not from a count: the list's length and the syscall count are
// the descriptor's, on the device. A list longer than the trace, or an ecall-list step that is no
// ECALL, sets the list bit of the error word, so a fill whose lists do not match the trace the host
// sized fails the chunk rather than drop or misfile a row.
//
// What a thread writes (ecall_row_of, syscall_row_of) and the tally's row count compile for the
// host too, so they are tested off the device; the kernels and their launchers only under nvcc.
#pragma once

#include "tracegen/riscv/history.cuh"

#if defined(__CUDACC__)
  #include <cub/cub.cuh>
  #include "common/rusterror.h"
  #include "tracegen/riscv/ecall.cuh"
  #include "tracegen/riscv/syscall.cuh"
#endif

namespace pico_gpu::ecall_syscall_history {
  enum class Row : uint32_t {
    // The thread writes nothing: it lies past the trace, or the fill could not be read and the
    // error word says why.
    Nothing,
    Event,
    Padding,
  };

  // Opens the fill and gives the ecall list's length in `count`, or 0 when the fill cannot be read
  // or the list does not fit the trace's `rows` rows; `reports` names the one thread that reports
  // the second.
  REPLAY_HD inline bool open_ecall_list(
    const history::HistoryArgs& args, size_t rows, bool reports, history::History& h, uint32_t& count)
  {
    count = 0;
    if (!history::open(args, history::Readiness::Complete, h)) return false;
    const uint32_t length = history::list_length(h, history::LIST_ECALL);
    if (length > rows) {
      if (reports) history::fail(h.error, history::ERROR_LIST);
      return false;
    }
    count = length;
    return true;
  }

  // What thread `idx` of a kernel over the ecall trace's `rows` rows writes: its list row's CPU
  // event, padding past the list, or nothing.
  REPLAY_HD inline Row ecall_row_of(
    const history::HistoryArgs& args, size_t rows, size_t idx, bool reports, uint32_t& count, FfiCpuEventU64& event)
  {
    history::History h;
    if (!open_ecall_list(args, rows, reports, h, count)) return Row::Nothing;
    if (idx >= rows) return Row::Nothing;
    if (idx >= count) return Row::Padding;
    const uint32_t row = static_cast<uint32_t>(idx);
    Opcode opcode;
    if (!history::entry_opcode(h, history::LIST_ECALL, row, opcode)) return Row::Nothing;
    if (opcode != Opcode::ECALL) {
      history::fail(h.error, history::ERROR_LIST);
      return Row::Nothing;
    }
    event = history::ecall_event(h, row);
    return Row::Event;
  }

  // The rows the tally scan runs over: the ecall list's, or none when the event kernel wrote none.
  REPLAY_HD inline uint32_t ecall_rows(const history::HistoryArgs& args, size_t rows)
  {
    history::History h;
    uint32_t count;
    open_ecall_list(args, rows, false, h, count);
    return count;
  }

  // What thread `idx` of a kernel over the syscall trace's `rows` rows writes, its global event
  // going to slot `idx` of the `global_slots` the host reserved.
  REPLAY_HD inline Row syscall_row_of(
    const history::HistoryArgs& args,
    size_t rows,
    size_t global_slots,
    size_t idx,
    bool reports,
    uint32_t& count,
    SyscallEventU64& event)
  {
    count = 0;
    history::History h;
    if (!history::open(args, history::Readiness::Complete, h)) return Row::Nothing;
    const uint32_t length = h.syscall_count;
    if (length > rows || length != global_slots) {
      if (reports) history::fail(h.error, history::ERROR_SYSCALL);
      return Row::Nothing;
    }
    count = length;
    if (idx >= rows) return Row::Nothing;
    if (idx >= count) return Row::Padding;
    event = history::syscall_event(h, static_cast<uint32_t>(idx));
    return Row::Event;
  }
} // namespace pico_gpu::ecall_syscall_history

#if defined(__CUDACC__)

namespace pico_gpu::ecall_syscall_history {
  constexpr int BLOCK_SIZE = 256;

  inline unsigned int grid_for(size_t rows, int block_size)
  {
    return static_cast<unsigned int>((rows + block_size - 1) / block_size);
  }
} // namespace pico_gpu::ecall_syscall_history

namespace pico_gpu::ecall_history {
  using ecall_syscall_history::Row;

  template <class F>
  __global__ void ecall_history_to_trace_kernel(
    const history::HistoryArgs args, const size_t rows, F* trace_matrix, const size_t num_cols, F* byte_trace)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t count;
    FfiCpuEventU64 event;
    if (ecall_syscall_history::ecall_row_of(args, rows, idx, idx == 0, count, event) != Row::Event) return;
    EcallChipColsU64<F>* cols = reinterpret_cast<EcallChipColsU64<F>*>(&trace_matrix[idx * num_cols]);
    ecall_u64::event_to_row<F>(event, *cols, byte_trace);
  }

  // One block walks the list rows a tile at a time, turning each row's own share of the tally into
  // the running sum down the trace: ecall rows are one per syscall, so one block keeps up and needs
  // no scratch memory. The row count is read on the device, the same in every thread, so every
  // thread reaches the scan's barriers.
  template <class F>
  __global__ void commit_tally_scan_kernel(
    const history::HistoryArgs args, const size_t rows, F* trace, const size_t num_cols)
  {
    using Tally = ecall_u64::CommitTally<F>;
    typedef cub::BlockScan<Tally, ecall_u64::COMMIT_TALLY_BLOCK> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;
    __shared__ Tally carry;
    const size_t count = ecall_syscall_history::ecall_rows(args, rows);
    if (threadIdx.x == 0) carry = Tally{F::zero(), F::zero()};
    __syncthreads();

    for (size_t base = 0; base < count; base += ecall_u64::COMMIT_TALLY_BLOCK) {
      const size_t row = base + threadIdx.x;
      EcallChipColsU64<F>* cols = reinterpret_cast<EcallChipColsU64<F>*>(&trace[row * num_cols]);
      Tally own{F::zero(), F::zero()};
      if (row < count) own = Tally{cols->commits, cols->commit_mask};
      Tally prefix;
      Tally tile_total;
      BlockScan(temp_storage).InclusiveScan(own, prefix, ecall_u64::CommitTallyAdd<F>(), tile_total);
      const Tally before = carry;
      if (row < count) {
        cols->commits = before.count + prefix.count;
        cols->commit_mask = before.mask + prefix.mask;
      }
      __syncthreads();
      if (threadIdx.x == 0) carry = ecall_u64::CommitTallyAdd<F>()(before, tile_total);
      __syncthreads();
    }
  }

  // The rows past the list take the last list row's sums; with no list row they stay zero.
  template <class F>
  __global__ void commit_tally_fill_kernel(
    const history::HistoryArgs args, const size_t rows, F* trace, const size_t num_cols)
  {
    const size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const size_t count = ecall_syscall_history::ecall_rows(args, rows);
    if (count == 0 || row < count) return;
    const EcallChipColsU64<F>* last = reinterpret_cast<const EcallChipColsU64<F>*>(&trace[(count - 1) * num_cols]);
    EcallChipColsU64<F>* cols = reinterpret_cast<EcallChipColsU64<F>*>(&trace[row * num_cols]);
    cols->commits = last->commits;
    cols->commit_mask = last->commit_mask;
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
      // The mirrored struct must stay field-for-field with the Rust `#[repr(C)]` `EcallChipCols`: a
      // width that disagrees means the two have drifted, and writing at this struct's offsets would
      // fill the wrong columns, so it is refused.
      constexpr size_t NUM_ECALL_CHIP_COLS = sizeof(EcallChipColsU64<F>) / sizeof(F);
      if (num_cols != NUM_ECALL_CHIP_COLS) return RustError{cudaErrorInvalidValue};

      CUDA_OK(cudaMemsetAsync(trace, 0, trace_size * sizeof(F), stream));
      const size_t rows = trace_size / num_cols;
      const unsigned int grid = ecall_syscall_history::grid_for(rows, ecall_syscall_history::BLOCK_SIZE);
      ecall_history_to_trace_kernel<F>
        <<<grid, ecall_syscall_history::BLOCK_SIZE, 0, stream>>>(args, rows, trace, num_cols, byte_trace);
      CUDA_OK(cudaGetLastError());
      commit_tally_scan_kernel<F>
        <<<1, ecall_u64::COMMIT_TALLY_BLOCK, 0, stream>>>(args, rows, trace, num_cols);
      CUDA_OK(cudaGetLastError());
      commit_tally_fill_kernel<F>
        <<<grid, ecall_syscall_history::BLOCK_SIZE, 0, stream>>>(args, rows, trace, num_cols);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  }
} // namespace pico_gpu::ecall_history

namespace pico_gpu::syscall_history {
  using ecall_syscall_history::Row;

  // `reserved` is the first of the host's `global_slots` slots. The history holds the RISC-V
  // chunk's own syscalls, which it sends: never a receive.
  template <class F>
  __global__ void syscall_history_to_trace_kernel(
    const history::HistoryArgs args,
    const size_t rows,
    F* trace_matrix,
    const size_t num_cols,
    GlobalInteractionEvent* reserved,
    const size_t global_slots,
    F* byte_trace)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t count;
    SyscallEventU64 event;
    if (ecall_syscall_history::syscall_row_of(args, rows, global_slots, idx, idx == 0, count, event) != Row::Event)
      return;
    SyscallColsU64<F>* cols = reinterpret_cast<SyscallColsU64<F>*>(&trace_matrix[idx * num_cols]);
    syscall_u64::event_to_row(event, cols, &reserved[idx], false, byte_trace);
  }

  template <class F>
  inline RustError generate_extra_record_and_main(
    const history::HistoryArgs& args,
    const size_t num_cols,
    const size_t trace_size,
    F* trace,
    F* byte_trace,
    GlobalInteractionEvent* global_events,
    const size_t global_events_start_index,
    const size_t global_slots,
    cudaStream_t stream)
  {
    try {
      // A width other than the mirrored struct's means the two have drifted.
      if (num_cols != sizeof(SyscallColsU64<F>) / sizeof(F)) return RustError{cudaErrorInvalidValue};
      // No slot to write may come with no buffer to write it in.
      if (global_slots > 0 && global_events == nullptr) return RustError{cudaErrorInvalidValue};

      CUDA_OK(cudaMemsetAsync(trace, 0, trace_size * sizeof(F), stream));
      const size_t rows = trace_size / num_cols;
      GlobalInteractionEvent* reserved =
        global_events == nullptr ? nullptr : &global_events[global_events_start_index];
      syscall_history_to_trace_kernel<F>
        <<<ecall_syscall_history::grid_for(rows, ecall_syscall_history::BLOCK_SIZE),
           ecall_syscall_history::BLOCK_SIZE, 0, stream>>>(
          args, rows, trace, num_cols, reserved, global_slots, byte_trace);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  }
} // namespace pico_gpu::syscall_history

#endif // defined(__CUDACC__)
