// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// The add, addw, sub, subw and div/rem traces built from the execution history: each chip reads its
// list through history.cuh's loaders, so no entry takes an event the host uploaded. The chips' own
// headers hold the row builders these kernels call. Included from one translation unit only,
// through the forwarders of pico's gpu-api/cuda/history/add_sub_divrem.cuh.
//
// The history keeps one list for the ADD and ADDW steps and one for the SUB and SUBW steps. Each of
// the four chips takes its opcode's rows of its list by a split on the device: a count of each
// block's rows of the opcode, a scan of those counts into each block's first rank, and a pass that
// writes each row at its rank, so a chip's rows keep list order. The split runs on a fixed grid, each
// block over its own consecutive range of the list, because the list's length is the descriptor's on
// the device and no host count may size the grid. The rows a chip finds must be the descriptor's
// count for it and fit the trace: a miss sets the list bit of the error word, never a cut.
//
// Div/rem's rows are its list's. A signed div/rem row with a negative divisor or remainder derives an
// ADD row for the add chip, so none, one or two per row, and the descriptor carries their total. The
// div/rem kernel stages each row's derived rows beside it; the same count, scan and placement then
// append them in row order to a buffer of exactly the descriptor's count, which the add chip reads
// after its own rows. An append of any other count sets the list bit and writes nothing past the
// buffer. Div/rem writes nothing into a buffer of the add chip's.
//
// The five traces are checked as multisets of rows, so list order proves the same traces whatever
// order the host's streams hold. The rows past a chip's own are the chips' padding: zero for the
// four add and sub chips, div/rem's padding row for div/rem.
//
// What a thread decides compiles for the host too, so it is tested off the device; the kernels and
// their launchers only under nvcc.
#pragma once

#include "tracegen/riscv/history_launch.cuh"

#if defined(__CUDACC__)
  #include <cub/cub.cuh>
  #include "tracegen/riscv/add_sub.cuh"
  #include "tracegen/riscv/divrem.cuh"
#endif

namespace pico_gpu::add_sub_divrem_history {
  // One chip's rows of a coarse list: those of its opcode, beside the list's other opcode.
  struct Split {
    uint32_t list;
    Opcode opcode;
    Opcode other;
  };

  // The descriptor of a fill `history::open` accepted.
  REPLAY_HD inline const history::FillDescriptor& descriptor(const history::HistoryArgs& args)
  {
    return *reinterpret_cast<const history::FillDescriptor*>(args.fill);
  }

  // The rows the descriptor counts for the chip of `opcode`, one of the coarse lists' four.
  REPLAY_HD inline uint32_t described_rows(const history::FillDescriptor& d, Opcode opcode)
  {
    switch (opcode) {
    case Opcode::ADD:
      return d.add_rows;
    case Opcode::ADDW:
      return d.addw_rows;
    case Opcode::SUB:
      return d.sub_rows;
    case Opcode::SUBW:
      return d.subw_rows;
    default:
      return 0;
    }
  }

  // Whether a row of the split's list whose step is of `opcode` is one of its chip's. A step of
  // neither of the list's opcodes is misfiled: it sets the list bit and is no chip's.
  REPLAY_HD inline bool opcode_is_mine(const history::History& h, const Split& split, Opcode opcode)
  {
    if (opcode == split.opcode) return true;
    if (opcode != split.other) history::fail(h.error, history::ERROR_LIST);
    return false;
  }

  // Whether row `row` of the split's list is one of its chip's, fetched as far as its instruction:
  // fetch_accesses on the same `index` and `fetched` reads the rest of a row that is.
  REPLAY_HD inline bool fetch_mine(
    const history::History& h, const Split& split, uint32_t row, uint32_t& index, history::Fetched& fetched)
  {
    return history::fetch_entry(h, split.list, row, index, fetched) &&
           opcode_is_mine(h, split, fetched.instruction.opcode);
  }

  // Whether row `row` of the split's list is one of its chip's: fetch_mine's answer, so the count
  // kernel counts exactly the rows the row kernel places.
  REPLAY_HD inline bool is_mine(const history::History& h, const Split& split, uint32_t row)
  {
    uint32_t index = 0;
    history::Fetched fetched;
    return fetch_mine(h, split, row, index, fetched);
  }

  // The rows [first, end) of `length` that block `block` of `blocks` covers: consecutive ranges in
  // block order, so ranking a block's rows after every earlier block's keeps the rows' order.
  REPLAY_HD inline void block_range(uint32_t length, uint32_t blocks, uint32_t block, uint32_t& first, uint32_t& end)
  {
    const uint64_t per = (uint64_t(length) + blocks - 1) / blocks;
    const uint64_t from = per * block;
    first = from < length ? static_cast<uint32_t>(from) : length;
    end = from + per < length ? static_cast<uint32_t>(from + per) : length;
  }

  // Whether a split found the descriptor's count of its chip's rows, and those rows, followed for the
  // add chip by div/rem's derived rows, fit the trace's `rows`.
  REPLAY_HD inline bool split_fits(
    const history::History& h, const history::FillDescriptor& d, const Split& split, uint64_t found, uint64_t rows)
  {
    const uint64_t own = described_rows(d, split.opcode);
    const uint64_t appended = split.opcode == Opcode::ADD ? d.divrem_add_rows : 0;
    if (found != own || own + appended > rows) {
      history::fail(h.error, history::ERROR_LIST);
      return false;
    }
    return true;
  }

  // Whether the buffer div/rem appended its derived rows to, `capacity` rows long, holds the
  // descriptor's count of them, and they fit the add trace's `rows` after the add list's own.
  REPLAY_HD inline bool derived_fit(
    const history::History& h, const history::FillDescriptor& d, uint64_t capacity, uint64_t rows)
  {
    if (d.divrem_add_rows != capacity || uint64_t(d.add_rows) + d.divrem_add_rows > rows) {
      history::fail(h.error, history::ERROR_LIST);
      return false;
    }
    return true;
  }

  // Whether div/rem appended the descriptor's count of derived rows, into a buffer of exactly that
  // many.
  REPLAY_HD inline bool append_fits(
    const history::History& h, const history::FillDescriptor& d, uint64_t appended, uint64_t capacity)
  {
    if (appended != d.divrem_add_rows || appended != capacity) {
      history::fail(h.error, history::ERROR_LIST);
      return false;
    }
    return true;
  }

  REPLAY_HD inline bool is_divrem(Opcode opcode)
  {
    switch (opcode) {
    case Opcode::DIV:
    case Opcode::DIVU:
    case Opcode::REM:
    case Opcode::REMU:
    case Opcode::DIVW:
    case Opcode::DIVUW:
    case Opcode::REMW:
    case Opcode::REMUW:
      return true;
    default:
      return false;
    }
  }

  using history::Row;

  // What thread `idx` of a kernel over the `rows` rows of div/rem's trace writes, `h` being the fill
  // opened for the whole history: its list row's event, padding past the list, or nothing. `count` is
  // the list's length, the same in every thread, or 0 when the list does not fit; `reports` names the
  // one thread that reports it. The row is fetched once, its opcode tested between the fetch's
  // halves; a row whose accesses cannot be read is the zero event's, the error word failing the
  // chunk.
  REPLAY_HD inline Row divrem_row_of(
    const history::History& h, size_t rows, size_t idx, bool reports, uint32_t& count, AluEventU64& event)
  {
    count = 0;
    const uint32_t length = history::list_length(h, history::LIST_DIVREM);
    if (length > rows) {
      if (reports) history::fail(h.error, history::ERROR_LIST);
      return Row::Nothing;
    }
    count = length;
    if (idx >= rows) return Row::Nothing;
    if (idx >= count) return Row::Padding;
    uint32_t index = 0;
    history::Fetched fetched;
    if (!history::fetch_entry(h, history::LIST_DIVREM, static_cast<uint32_t>(idx), index, fetched)) return Row::Nothing;
    if (!is_divrem(fetched.instruction.opcode)) {
      history::fail(h.error, history::ERROR_LIST);
      return Row::Nothing;
    }
    event = history::fetch_accesses(h, index, fetched) ? history::alu_event_of(fetched) : AluEventU64{};
    return Row::Event;
  }

  // An opcode no ALU row carries: a staged slot still holding it was not written.
  constexpr uint8_t UNWRITTEN = 0xff;

  // How many of a div/rem row's two staged derived rows its builder wrote, which it writes densely
  // from the first.
  REPLAY_HD inline uint32_t derived_written(const AluEventU64* staged)
  {
    if (static_cast<uint8_t>(staged[0].opcode) == UNWRITTEN) return 0;
    return static_cast<uint8_t>(staged[1].opcode) == UNWRITTEN ? 1 : 2;
  }
} // namespace pico_gpu::add_sub_divrem_history

#if defined(__CUDACC__)

namespace pico_gpu::add_sub_divrem_history {
  // The grid of every pass over a range of rows. Fixed, so that no count sizes it; 1024 blocks are
  // the one block of threads the scan of their counts runs in.
  constexpr uint32_t RANGE_BLOCKS = 1024;
  constexpr uint32_t RANGE_THREADS = 256;
  constexpr int ROW_THREADS = 256;

  using RangeReduce = cub::BlockReduce<uint32_t, RANGE_THREADS>;
  using RangeScan = cub::BlockScan<uint32_t, RANGE_THREADS>;
  using BlocksScan = cub::BlockScan<uint32_t, RANGE_BLOCKS>;

  // The sum over this block's range of `items(row)`, in thread 0 alone.
  template <class Items>
  __device__ uint32_t range_items(uint32_t first, uint32_t end, Items items)
  {
    __shared__ typename RangeReduce::TempStorage temp;
    uint32_t sum = 0;
    for (uint32_t row = first + threadIdx.x; row < end; row += blockDim.x)
      sum += items(row);
    return RangeReduce(temp).Sum(sum);
  }

  // Replaces each of the RANGE_BLOCKS block counts at `block_items` by the items of the blocks before
  // it, and returns their total to every thread. One block of RANGE_BLOCKS threads.
  __device__ inline uint32_t scan_blocks(uint32_t* block_items)
  {
    __shared__ typename BlocksScan::TempStorage temp;
    uint32_t before, total;
    BlocksScan(temp).ExclusiveSum(block_items[threadIdx.x], before, total);
    block_items[threadIdx.x] = before;
    return total;
  }

  // Calls `place(row, at, n)` for each row of this block's range with n = `items(row)` > 0, `at`
  // being the rank of its first item among every item of the rows before it, from `base` on. A thread
  // calls `place` for the row it last called `items` on, with no call between, so `place` may read
  // what `items` left for its row. Every thread of the block calls it with the same range: the tiles
  // step in lockstep.
  template <class Items, class Place>
  __device__ void place_range(uint32_t first, uint32_t end, uint32_t base, Items items, Place place)
  {
    __shared__ typename RangeScan::TempStorage temp;
    for (uint32_t tile = first; tile < end; tile += RANGE_THREADS) {
      const uint32_t row = tile + threadIdx.x;
      const uint32_t n = row < end ? items(row) : 0;
      uint32_t before, total;
      RangeScan(temp).ExclusiveSum(n, before, total);
      __syncthreads();
      if (n != 0) place(row, base + before, n);
      base += total;
    }
  }

  // ── the split of a coarse list ───────────────────────────────────────────────────────────────

  // The four chips' rows, each its chip's own event_to_row, so a row and its byte lookups are those
  // the chip has always built for the same event.
  struct AddRow {
    template <class F>
    __device__ static void write(const AluEventU64& event, F* row, F* byte_trace)
    {
      add_u64::event_to_row<F>(event, *reinterpret_cast<add_u64::AddValueColsU64<F>*>(row), byte_trace);
    }
  };
  struct AddwRow {
    template <class F>
    __device__ static void write(const AluEventU64& event, F* row, F* byte_trace)
    {
      addw_u64::event_to_row<F>(event, *reinterpret_cast<addw_u64::AddwValueColsU64<F>*>(row), byte_trace);
    }
  };
  struct SubRow {
    template <class F>
    __device__ static void write(const AluEventU64& event, F* row, F* byte_trace)
    {
      sub_u64::event_to_row<F>(event, *reinterpret_cast<sub_u64::SubValueColsU64<F>*>(row), byte_trace);
    }
  };
  struct SubwRow {
    template <class F>
    __device__ static void write(const AluEventU64& event, F* row, F* byte_trace)
    {
      subw_u64::event_to_row<F>(event, *reinterpret_cast<subw_u64::SubwValueColsU64<F>*>(row), byte_trace);
    }
  };

  // The fill, opened once for the block, and this block's range of the split's list; null and an
  // empty range when the fill cannot be read. Every thread of the block calls it first, since the
  // opening's barrier waits for all of them.
  __device__ inline const history::History* split_range(
    const history::HistoryArgs& args, const Split& split, uint32_t& first, uint32_t& end)
  {
    first = end = 0;
    const history::History* h = history::open_block(args, history::Readiness::Complete);
    if (h) block_range(history::list_length(*h, split.list), gridDim.x, blockIdx.x, first, end);
    return h;
  }

  __global__ void split_count_kernel(const history::HistoryArgs args, const Split split, uint32_t* block_rows)
  {
    uint32_t first, end;
    const history::History* h = split_range(args, split, first, end);
    const uint32_t mine = range_items(first, end, [&](uint32_t row) { return is_mine(*h, split, row) ? 1u : 0u; });
    if (threadIdx.x == 0) block_rows[blockIdx.x] = mine;
  }

  __global__ void split_scan_kernel(
    const history::HistoryArgs args, const Split split, const size_t rows, uint32_t* block_rows)
  {
    const uint32_t found = scan_blocks(block_rows);
    if (threadIdx.x != 0) return;
    history::History h;
    if (history::open(args, history::Readiness::Complete, h)) split_fits(h, descriptor(args), split, found, rows);
  }

  template <class Chip, class F>
  __global__ void split_rows_kernel(
    const history::HistoryArgs args,
    const Split split,
    const uint32_t* block_rows,
    const size_t rows,
    F* trace_matrix,
    const size_t num_cols,
    F* byte_trace)
  {
    uint32_t first, end;
    const history::History* h = split_range(args, split, first, end);
    // The row `items` fetched as far as its instruction, which `place` fetches the rest of.
    uint32_t index = 0;
    history::Fetched fetched;
    place_range(
      first,
      end,
      block_rows[blockIdx.x],
      [&](uint32_t row) { return fetch_mine(*h, split, row, index, fetched) ? 1u : 0u; },
      [&](uint32_t, uint32_t at, uint32_t) {
        // Past the trace the scan has already failed the chunk; the row goes nowhere.
        if (at >= rows) return;
        Chip::template write<F>(
          history::fetch_accesses(*h, index, fetched) ? history::alu_event_of(fetched) : AluEventU64{},
          &trace_matrix[size_t(at) * num_cols], byte_trace);
      });
  }

  // The chip's rows of the split's list, in list order, from the first row of the trace.
  template <class Chip, class F>
  inline void launch_split(
    const history::HistoryArgs& args,
    const Split split,
    const size_t rows,
    F* trace,
    const size_t num_cols,
    F* byte_trace,
    cudaStream_t stream)
  {
    uint32_t* block_rows = nullptr;
    CUDA_OK(cudaMallocAsync(&block_rows, RANGE_BLOCKS * sizeof(uint32_t), stream));
    split_count_kernel<<<RANGE_BLOCKS, RANGE_THREADS, 0, stream>>>(args, split, block_rows);
    history::check_launch();
    split_scan_kernel<<<1, RANGE_BLOCKS, 0, stream>>>(args, split, rows, block_rows);
    history::check_launch();
    split_rows_kernel<Chip, F><<<RANGE_BLOCKS, RANGE_THREADS, 0, stream>>>(
      args, split, block_rows, rows, trace, num_cols, byte_trace);
    history::check_launch();
    CUDA_OK(cudaFreeAsync(block_rows, stream));
  }

  template <class Chip, class F>
  inline RustError split_trace(
    const history::HistoryArgs& args,
    const Split split,
    const size_t num_cols,
    const size_t trace_size,
    F* trace,
    F* byte_trace,
    cudaStream_t stream)
  {
    return history::launch_zeroed(trace, trace_size, stream, [&] {
      launch_split<Chip, F>(args, split, trace_size / num_cols, trace, num_cols, byte_trace, stream);
    });
  }

  // ── div/rem's derived ADD rows ───────────────────────────────────────────────────────────────

  // The add trace's rows after the add list's own: div/rem's derived rows, in the order it appended
  // them.
  template <class F>
  __global__ void derived_rows_kernel(
    const history::HistoryArgs args,
    const AluEventU64* derived,
    const size_t capacity,
    const size_t rows,
    F* trace_matrix,
    const size_t num_cols,
    F* byte_trace)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const history::History* h = history::open_block(args, history::Readiness::Complete);
    if (!h) return;
    const history::FillDescriptor& d = descriptor(args);
    if (!derived_fit(*h, d, capacity, rows)) return;
    for (size_t j = idx; j < d.divrem_add_rows; j += size_t(gridDim.x) * blockDim.x)
      AddRow::write<F>(derived[j], &trace_matrix[(d.add_rows + j) * num_cols], byte_trace);
  }

  // Div/rem's rows, each with the derived rows its builder writes staged in its two slots and their
  // number in `derived_rows`.
  template <class F>
  __global__ void divrem_rows_kernel(
    const history::HistoryArgs args,
    const size_t rows,
    F* trace_matrix,
    const size_t num_cols,
    F* byte_trace,
    uint32_t* derived_rows,
    AluEventU64* staged)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const history::History* h = history::open_block(args, history::Readiness::Complete);
    uint32_t count = 0;
    AluEventU64 event;
    const Row row = h ? divrem_row_of(*h, rows, idx, idx == 0, count, event) : Row::Nothing;
    if (idx >= rows) return;
    uint32_t written = 0;
    divrem::DivRemValueColsU64<F>* cols =
      reinterpret_cast<divrem::DivRemValueColsU64<F>*>(&trace_matrix[idx * num_cols]);
    if (row == Row::Event) {
      AluEventU64* slots = &staged[2 * idx];
      slots[0].opcode = slots[1].opcode = static_cast<Opcode>(UNWRITTEN);
      divrem::event_to_row<F>(event, *cols, byte_trace, slots);
      written = derived_written(slots);
    } else if (row == Row::Padding) {
      divrem::assign_padding_row<F>(*cols);
    }
    derived_rows[idx] = written;
  }

  // This block's range of div/rem's `rows` rows.
  __device__ inline void divrem_range(const size_t rows, uint32_t& first, uint32_t& end)
  {
    block_range(static_cast<uint32_t>(rows), gridDim.x, blockIdx.x, first, end);
  }

  __global__ void append_count_kernel(const uint32_t* derived_rows, const size_t rows, uint32_t* block_rows)
  {
    uint32_t first, end;
    divrem_range(rows, first, end);
    const uint32_t derived = range_items(first, end, [&](uint32_t row) { return derived_rows[row]; });
    if (threadIdx.x == 0) block_rows[blockIdx.x] = derived;
  }

  __global__ void append_scan_kernel(
    const history::HistoryArgs args, uint32_t* block_rows, const size_t capacity, uint32_t* appended)
  {
    const uint32_t total = scan_blocks(block_rows);
    if (threadIdx.x != 0) return;
    *appended = total;
    history::History h;
    if (history::open(args, history::Readiness::Complete, h)) append_fits(h, descriptor(args), total, capacity);
  }

  __global__ void append_kernel(
    const uint32_t* derived_rows,
    const size_t rows,
    const uint32_t* block_rows,
    const AluEventU64* staged,
    AluEventU64* derived,
    const size_t capacity)
  {
    uint32_t first, end;
    divrem_range(rows, first, end);
    place_range(
      first,
      end,
      block_rows[blockIdx.x],
      [&](uint32_t row) { return derived_rows[row]; },
      [&](uint32_t row, uint32_t at, uint32_t n) {
        for (uint32_t k = 0; k < n; ++k)
          if (size_t(at) + k < capacity) derived[at + k] = staged[2 * size_t(row) + k];
      });
  }
} // namespace pico_gpu::add_sub_divrem_history

namespace pico_gpu::add_history {
  // The add list's ADD rows, then the `capacity` rows div/rem appended at `derived`.
  template <class F>
  inline RustError generate_extra_record_and_main(
    const history::HistoryArgs& args,
    const size_t num_cols,
    const size_t trace_size,
    F* trace,
    F* byte_trace,
    const AluEventU64* derived,
    const size_t capacity,
    cudaStream_t stream)
  {
    using namespace add_sub_divrem_history;
    return history::launch_zeroed(trace, trace_size, stream, [&] {
      const size_t rows = trace_size / num_cols;
      launch_split<AddRow, F>(
        args, Split{history::LIST_ADD, Opcode::ADD, Opcode::ADDW}, rows, trace, num_cols, byte_trace, stream);
      // One block at least: a descriptor that counts derived rows the host allocated none for must
      // still be seen.
      derived_rows_kernel<<<history::grid_for(capacity, ROW_THREADS), ROW_THREADS, 0, stream>>>(
        args, derived, capacity, rows, trace, num_cols, byte_trace);
      history::check_launch();
    });
  }
} // namespace pico_gpu::add_history

namespace pico_gpu::addw_history {
  template <class F>
  inline RustError generate_extra_record_and_main(
    const history::HistoryArgs& args,
    const size_t num_cols,
    const size_t trace_size,
    F* trace,
    F* byte_trace,
    cudaStream_t stream)
  {
    using namespace add_sub_divrem_history;
    return split_trace<AddwRow, F>(
      args, Split{history::LIST_ADD, Opcode::ADDW, Opcode::ADD}, num_cols, trace_size, trace, byte_trace, stream);
  }
} // namespace pico_gpu::addw_history

namespace pico_gpu::sub_history {
  template <class F>
  inline RustError generate_extra_record_and_main(
    const history::HistoryArgs& args,
    const size_t num_cols,
    const size_t trace_size,
    F* trace,
    F* byte_trace,
    cudaStream_t stream)
  {
    using namespace add_sub_divrem_history;
    return split_trace<SubRow, F>(
      args, Split{history::LIST_SUB, Opcode::SUB, Opcode::SUBW}, num_cols, trace_size, trace, byte_trace, stream);
  }
} // namespace pico_gpu::sub_history

namespace pico_gpu::subw_history {
  template <class F>
  inline RustError generate_extra_record_and_main(
    const history::HistoryArgs& args,
    const size_t num_cols,
    const size_t trace_size,
    F* trace,
    F* byte_trace,
    cudaStream_t stream)
  {
    using namespace add_sub_divrem_history;
    return split_trace<SubwRow, F>(
      args, Split{history::LIST_SUB, Opcode::SUBW, Opcode::SUB}, num_cols, trace_size, trace, byte_trace, stream);
  }
} // namespace pico_gpu::subw_history

namespace pico_gpu::divrem_history {
  // Div/rem's trace, and its derived ADD rows appended in row order to `derived`, `capacity` rows
  // long; `appended` receives how many it appended.
  template <class F>
  inline RustError generate_extra_record_and_main(
    const history::HistoryArgs& args,
    const size_t num_cols,
    const size_t trace_size,
    F* trace,
    F* byte_trace,
    AluEventU64* derived,
    const size_t capacity,
    uint32_t* appended,
    cudaStream_t stream)
  {
    using namespace add_sub_divrem_history;
    const size_t rows = trace_size / num_cols;
    if (rows > UINT32_MAX) return RustError{cudaErrorInvalidValue};
    return history::launch_zeroed(trace, trace_size, stream, [&] {
      // No early return on an empty list: a shape can list this chip with no rows, and its padding
      // row is not all zero.
      uint32_t* counts = nullptr;
      AluEventU64* staged = nullptr;
      CUDA_OK(cudaMallocAsync(&counts, (rows + RANGE_BLOCKS) * sizeof(uint32_t), stream));
      CUDA_OK(cudaMallocAsync(&staged, (2 * rows > 0 ? 2 * rows : 1) * sizeof(AluEventU64), stream));
      uint32_t* block_rows = counts + rows;
      divrem_rows_kernel<<<history::grid_for(rows, ROW_THREADS), ROW_THREADS, 0, stream>>>(
        args, rows, trace, num_cols, byte_trace, counts, staged);
      history::check_launch();
      append_count_kernel<<<RANGE_BLOCKS, RANGE_THREADS, 0, stream>>>(counts, rows, block_rows);
      history::check_launch();
      append_scan_kernel<<<1, RANGE_BLOCKS, 0, stream>>>(args, block_rows, capacity, appended);
      history::check_launch();
      append_kernel<<<RANGE_BLOCKS, RANGE_THREADS, 0, stream>>>(counts, rows, block_rows, staged, derived, capacity);
      history::check_launch();
      CUDA_OK(cudaFreeAsync(staged, stream));
      CUDA_OK(cudaFreeAsync(counts, stream));
    });
  }
} // namespace pico_gpu::divrem_history

#endif // defined(__CUDACC__)
