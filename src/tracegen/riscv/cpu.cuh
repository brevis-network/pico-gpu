// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0

#pragma once

#include <cstdint>
#include <iostream>
#include "common/rusterror.h"
#include "tracegen/common/field_word_range_checker.cuh"
#include "tracegen/riscv/instruction.cuh"
#include "tracegen/riscv/memory_read_write.cuh"
#include "tracegen/common/types.cuh"
#include "tracegen/common/utils.cuh"
#include "tracegen/riscv/byte.cuh"
#include "tracegen/riscv/divrem.cuh"
// Used directly by the ecall populate below. It used to arrive through
// field_word_range_checker.cuh, which no longer needs it.
#include "tracegen/common/is_zero.cuh"


// ─── U64 CPU chip ──────────────────────────────────────────────────────
// NOTE: `event_to_row` and the `populate_*` helpers it calls are hand-mirrored by
// `cpu_u64_packed::event_to_row_packed` in packed_tracegen.cuh, which fills the same columns from
// the packed event. Nothing makes the compiler check the two against each other — edit both, or
// MLE_PACKED_EVENTS=1 silently builds a different trace.
namespace pico_gpu::cpu_u64 {
  using namespace byte;
  using divrem::write_word_from_u64;

  // Two limbs (Rust `Pc::try_from`, which rejects a value at or above 2^32; the disassembler's
  // guest address bound rules that out, and this side does not check).
  template <class F>
  __device__ inline void write_pc_from_u64(Pc<F>& pc, uint64_t value)
  {
    pc._0[0] = F::from_canonical_u16(static_cast<uint16_t>(value & 0xFFFF));
    pc._0[1] = F::from_canonical_u16(static_cast<uint16_t>((value >> 16) & 0xFFFF));
  }

  template <class F>
  __device__ inline void populate_chunk_clk(const FfiCpuEventU64& event, CpuColsU64<F>& cols)
  {
    // No chunk column to fill: the AIR's messages carry the `execution_chunk` public value.
    const uint32_t compat_clk = static_cast<uint32_t>(event.clk);

    // One `clk` column, mirrors Rust populate_chunk_clk. The Rust side asserts clk < 2^25 (the
    // chunk-split budget); this side does not check.
    cols.clk = F::from_canonical_u32(compat_clk);

    // No byte event for clk: no such lookup exists on the Rust side (clk is bounded by the row
    // chain -- riscv_cpu/chunk_clk). Emitting one here would unbalance the multiset.
  }

  template <class F>
  __device__ inline void populate_opcode_selector(
    const InstructionU64& instruction, OpcodeSelectorColsU64<F>& cols)
  {
    cols.imm_b = F::from_bool(instruction.imm_b);
    cols.imm_c = F::from_bool(instruction.imm_c);

    // The ALU class is derived in the constraints (one-hot complement), so this branch
    // simply leaves every class selector at zero instead of committing is_alu.
    if (instruction::is_alu_instruction_u64(instruction)) {
      // no-op: ALU class is derived, not committed.
    } else if (instruction::is_ecall_instruction_u64(instruction)) {
      cols.is_ecall = F::one();
    } else if (instruction::is_memory_instruction_u64(instruction)) {
      // per-variant flags live in the MemoryReadWriteChip now.
      cols.is_memory = F::one();
      switch (instruction.opcode) {
      case Opcode::SB:
      case Opcode::SH:
      case Opcode::SW:
      case Opcode::SD:
        cols.is_store = F::one();
        break;
      default: break;
      }
    } else if (instruction::is_branch_instruction_u64(instruction)) {
      switch (instruction.opcode) {
      case Opcode::BEQ:  cols.is_beq = F::one(); break;
      case Opcode::BNE:  cols.is_bne = F::one(); break;
      case Opcode::BLT:  cols.is_blt = F::one(); break;
      case Opcode::BGE:  cols.is_bge = F::one(); break;
      case Opcode::BLTU: cols.is_bltu = F::one(); break;
      case Opcode::BGEU: cols.is_bgeu = F::one(); break;
      default: break;
      }
    } else if (instruction.opcode == Opcode::JAL) {
      cols.is_jal = F::ONE;
    } else if (instruction.opcode == Opcode::JALR) {
      cols.is_jalr = F::ONE;
    }
    // A trap instruction (UNIMP / EBREAK, Rust `Opcode::is_trap`) sets no class selector; the
    // emulator never emits an event for one.
  }

  template <class F>
  __device__ inline void populate_memory_access_u64(
    MemoryAccessLiteCols<F>& col,
    uint64_t value, uint32_t rec_chunk, uint32_t rec_timestamp,
    uint32_t rec_prev_chunk, uint32_t rec_prev_timestamp,
    F* byte_trace)
  {
    write_word_from_u64(col.value, value);
    // The predecessor is always in this chunk. When the address was last touched in an earlier
    // chunk there is no in-chunk timestamp to name, so the sentinel 0 stands in -- the same label
    // MemoryLocalChip puts on the state the chunk inherits. The emulator starts each chunk's clk
    // at 4, which keeps every real access above the sentinel. Against the sentinel the diff spans
    // the whole clk range, so it fits 16+9 only because the Rust split config caps a chunk's clk
    // at 2^25 - 1; `& 0x1FF` below would truncate silently past that.
    // The predecessor clk is not a column (Rust `MemoryAccessLiteCols`): the AIR rebuilds
    // it from the two committed difference limbs, so it is computed here only to difference from.
    const uint32_t prev_clk = (rec_prev_chunk == rec_chunk) ? rec_prev_timestamp : 0u;

    const uint32_t diff_minus_one = rec_timestamp - prev_clk - 1;

    col.diff_16bit_limb = F::from_canonical_u16(static_cast<uint16_t>(diff_minus_one & 0xFFFF));
    // Single 9-bit high limb (bits 16..25), BitRange(9)-checked.
    col.diff_hi9 = F::from_canonical_u16(static_cast<uint16_t>((diff_minus_one >> 16) & 0x1FF));

    add_u16_range_check<F>(byte_trace, static_cast<uint16_t>(diff_minus_one & 0xFFFF));
    add_bit_range_check<F>(byte_trace, static_cast<uint16_t>((diff_minus_one >> 16) & 0x1FF), 9);
  }

  template <class F>
  __device__ inline void populate_read_write_cols_u64(
    MemoryReadWriteLiteCols<F>& col, const MemoryRecordEnumU64& record,
    F* byte_trace)
  {
    if (record.tag == MemoryRecordEnumU64::Tag::Write) {
      write_word_from_u64(col.prev_value, record.write._0.prev_value);
    } else {
      write_word_from_u64(col.prev_value, record.read._0.value);
    }
    populate_memory_access_u64<F>(
      col.access, record.value(), record.chunk(), record.timestamp(),
      record.prev_chunk(), record.prev_timestamp(),
      byte_trace);
  }

  template <class F>
  __device__ inline void populate_read_cols_u64(
    MemoryReadLiteCols<F>& col, const MemoryReadRecordU64& record,
    F* byte_trace)
  {
    populate_memory_access_u64<F>(
      col.access, record.value, record.chunk, record.timestamp,
      record.prev_chunk, record.prev_timestamp,
      byte_trace);
  }

  template <class F>
  __device__ inline void populate_branch(
    const FfiCpuEventU64& event, CpuColsU64<F>& cols,
    AluEventU64* alu_lt_events, size_t& alu_lt_index)
  {
    if (instruction::is_branch_instruction_u64(event.instruction)) {
      const bool use_signed_comparison =
        event.instruction.opcode == Opcode::BLT || event.instruction.opcode == Opcode::BGE;

      bool a_lt_b, a_gt_b;
      if (use_signed_comparison) {
        a_lt_b = (int64_t)event.a < (int64_t)event.b;
        a_gt_b = (int64_t)event.a > (int64_t)event.b;
      } else {
        a_lt_b = event.a < event.b;
        a_gt_b = event.a > event.b;
      }

      const Opcode alu_op_code = use_signed_comparison ? Opcode::SLT : Opcode::SLTU;

      // One merged comparison event (mirrors Rust `populate_branch`): the Lt row pins both bits,
      // `a[0] = a_lt_b`, `a[1] = a_gt_b`. One lt slot per branch -- must stay in step with the
      // Rust `count_cpu_event` and the packed mirror in packed_tracegen.cuh.
      AluEventU64 cmp_event = {};
      cmp_event.clk = event.clk;
      cmp_event.opcode = alu_op_code;
      cmp_event.a = (a_lt_b ? 1ULL : 0ULL) | ((a_gt_b ? 1ULL : 0ULL) << 16);
      cmp_event.b = event.a;
      cmp_event.c = event.b;
      cmp_event.is_cmp = true;
      alu_lt_events[alu_lt_index++] = cmp_event;

      // `a_eq_b` is derived in the constraints (1 - a_lt_b - a_gt_b); no column, and neither is
      // the taken predicate any more -- the AIR builds it from these two bits and the opcode
      // selectors (`CpuChip::branch_taken_expr`). The column that held it is now `is_last`, which
      // the host writes. A taken branch emits no ADD event either: the AIR proves
      // `next_pc = pc + c` with one field equation over the sign-extended offset instead of
      // dispatching to the Add chip, and emitting one here would unbalance the ALU bus.
      cols.a_lt_b = F::from_bool(a_lt_b);
      cols.a_gt_b = F::from_bool(a_gt_b);
    }
  }

  // No populate_jump and no ADD event from any Cpu row: a JAL target is a field equation and a
  // JALR target is proven in-chip by four boolean carry expressions (Rust jump/constraints.rs).
  // No populate_auipc either: since X12 the transpiler re-encodes `AUIPC rd, imm` as the
  // two-immediate `ADD rd, imm, pc`, so that row's Add event arrives through the emulator's
  // generic ALU path. The host `count_cpu_event` reserves no ADD slot (`.add` is always 0), so
  // the only ALU events this kernel writes are the branch comparisons.

  // The Cpu kernel keeps one responsibility for ecall rows (the ecall decode gadget
  // itself, and the U16ToU8 u8-range byte lookups, live on the Ecall chip):
  //   1. commit the `is_halt` flag (certified by the Cpu→Ecall lookup).
  template <class F>
  __device__ inline bool populate_ecall(const FfiCpuEventU64& event, CpuColsU64<F>& cols, F* byte_trace)
  {
    bool is_halt = false;

    if (cols.opcode_selector.is_ecall == F::one()) {
      const uint32_t limb0 = cols.op_a_access.prev_value._0[0].as_canonical_u32();
      const uint8_t syscall_id_byte = static_cast<uint8_t>(limb0 & 0xFF);

      is_halt = syscall_id_byte == static_cast<uint8_t>(static_cast<uint32_t>(SyscallCode::HALT));
      cols.is_halt = F::from_bool(is_halt);
    }

    return is_halt;
  }

  template <class F>
  __device__ inline void assign_padding_row(CpuColsU64<F>& cols)
  {
    cols.opcode_selector.imm_b = F::one();
    cols.opcode_selector.imm_c = F::one();
  }

  template <class F>
  __device__ void event_to_row(
    const FfiCpuEventU64& event, CpuColsU64<F>& cols, F* byte_trace,
    AluEventU64* alu_lt_events)
  {
    populate_chunk_clk<F>(event, cols);

    write_pc_from_u64(cols.pc, event.pc);
    write_pc_from_u64(cols.next_pc, event.next_pc);

    // No pc or next_pc range-check events: both lookups are gone on the Rust side. pc's two
    // limbs are pinned element by element by the Program lookup to the preprocessed table's
    // canonical Pc(pc_base + 4i), which is stronger than a range check; next_pc is bound to
    // the successor row's pc, or to a public value the recursion chain pins. Emitting events
    // here would leave the Byte table with multiplicity no lookup claims, and even an honest
    // trace would fail the regional cumulative sum. Mirrors riscv_cpu/traces.rs.

    instruction::populate_instruction_u64<F>(event.instruction, cols.instruction);
    populate_opcode_selector<F>(event.instruction, cols.opcode_selector);

    write_word_from_u64(cols.op_a_access.value_mut(), event.a);
    write_word_from_u64(cols.op_b_access.value_mut(), event.b);
    write_word_from_u64(cols.op_c_access.value_mut(), event.c);

    // Populate memory accesses for operands
    if (event.a_record.tag == FfiOption<MemoryRecordEnumU64>::Tag::Some) {
      populate_read_write_cols_u64<F>(
        cols.op_a_access, event.a_record.some._0, byte_trace);
    }
    // No byte event for a jump row's return address: the AIR has no range check on op_a there
    // (limb 0 of pc+4 is canonical because pc_carry below is pinned to the Program table's
    // [pc0 >= 0xFFFC], Rust register/constraints.rs). Emitting one would unbalance the multiset.

    if (event.b_record.tag == FfiOption<MemoryRecordEnumU64>::Tag::Some &&
        event.b_record.some._0.tag == MemoryRecordEnumU64::Tag::Read) {
      populate_read_cols_u64<F>(
        cols.op_b_access, event.b_record.some._0.read._0, byte_trace);
    }
    if (event.c_record.tag == FfiOption<MemoryRecordEnumU64>::Tag::Some &&
        event.c_record.some._0.tag == MemoryRecordEnumU64::Tag::Read) {
      populate_read_cols_u64<F>(
        cols.op_c_access, event.c_record.some._0.read._0, byte_trace);
    }

    size_t alu_lt_index = 0;

    populate_branch<F>(event, cols, alu_lt_events, alu_lt_index);
    populate_ecall(event, cols, byte_trace);

    // num_extra_clk: for ecall instructions, extract 3rd byte of syscall code
    // (bits 16-23 of op_a prev_value as u64). In u16-limb layout, this is
    // the low 8 bits of _0[1] (limb 1 = bits[16:31]).
    // For non-ecall, remains 0 (pre-zeroed).
    if (instruction::is_ecall_instruction_u64(event.instruction)) {
      const uint32_t limb1 = cols.op_a_access.prev_value._0[1].as_canonical_u32();
      cols.num_extra_clk = F::from_canonical_u8(static_cast<uint8_t>(limb1 & 0xFF));
    }

    // pc_carry = [pc0 >= 0xFFFC] on every real row (mirrors Rust riscv_cpu/traces.rs). The
    // Program message carries the column, so it has to equal the preprocessed table's value for
    // this pc (program.cuh writes the same function of the row's pc). On a jump row with
    // rd != x0 it is the carry out of limb 0 in pc+4=op_a: a table pc is 4-aligned, so pc0+4
    // overflows exactly when pc0 == 0xFFFC. Only limb 0 can carry (a guest address stays below
    // 0x7800_0000, so limb 1 absorbs it and op_a's limb 2 is zero); the AIR checks those two as
    // plain equalities. Padding rows leave it at 0 (assign_padding_row).
    cols.pc_carry = F::from_bool((event.pc & 0xFFFF) >= 0xFFFC);

    cols.is_real = F::one();

    // Dispatch multiplicity, zero when the destination is `x0`. Mirrors the host side
    // (riscv_cpu/traces.rs): the CPU AIR gates the ALU lookup on this, so a row written here
    // without it would ask for a lookup the AIR never emits.
    const F not_x0 = F::from_bool(event.instruction.op_a != 0);
    cols.is_alu_not_x0 =
        F::from_bool(instruction::is_alu_instruction_u64(event.instruction)) * not_x0;

    // Bit 0 of the unmasked JALR target, which the emulator masked out of `next_pc`.
    cols.jalr_lsb = (event.instruction.opcode == Opcode::JALR)
                        ? F::from_canonical_u32(static_cast<uint32_t>((event.b + event.c) & 1ull))
                        : F::zero();
  }

  template <class F, size_t NumCols>
  __global__ void cpu_events_to_trace_kernel(
    const size_t event_size,
    const FfiCpuEventU64* __restrict__ events,
    F* __restrict__ trace_matrix,
    const size_t num_cols,
    const CpuExtraEventIndices* __restrict__ extra_event_indices,
    F* __restrict__ byte_trace,
    // The ADD/SUB buffer and its start index stay in the kernel and FFI signature (the host
    // hands them over for every chip) but no Cpu row writes an ADD event, so neither is read;
    // `extra_event_indices[idx].add` is always 0.
    AluEventU64* /*alu_add_sub_events*/,
    const size_t /*alu_add_sub_events_start_index*/,
    AluEventU64* alu_lt_events,
    const size_t alu_lt_events_start_index)
  {
    // No bulk chunk U16 event here: the chunk lookup is gone on both sides. If it ever comes
    // back, emit it from exactly one place -- this prologue with multiplicity event_size, or a
    // per-row copy in populate_chunk_clk, never both. Keeping the prologue while deleting only
    // the per-row copy is what unbalanced the byte multiset before.

    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if constexpr (NumCols > 0 && NumCols == sizeof(CpuColsU64<F>) / sizeof(F) &&
                  sizeof(F) == sizeof(uint32_t)) {
      // WSTAGE fast path (compile-time-width specialization only): the 32 rows of a warp form
      // ONE contiguous row-major span in the trace. Each thread builds its row in dynamic
      // shared memory, then the warp flushes the whole span linearly = fully coalesced 128B
      // stores, removing the ~8x sector amplification of per-thread strided row stores
      // (4.4x standalone). Byte-identical: the staged row is zero-initialized and flushed in
      // full, matching the pre-zeroed buffer; event_to_row and the byte/ALU side effects run
      // unmodified.
      extern __shared__ uint32_t wstage_smem[];
      const int warp = threadIdx.x >> 5;
      const int lane = threadIdx.x & 31;
      uint32_t* wrow = wstage_smem + (size_t)warp * 32 * NumCols; // this warp's 32xNumCols tile
      uint32_t* myrow = wrow + (size_t)lane * NumCols;            // this thread's staged row
#pragma unroll
      for (int i = 0; i < (int)NumCols; i++) myrow[i] = 0u;
      __syncwarp();
      if (idx < event_size) {
        CpuColsU64<F>* cols = reinterpret_cast<CpuColsU64<F>*>(myrow);
        event_to_row<F>(
          events[idx], *cols, byte_trace,
          &alu_lt_events[alu_lt_events_start_index + extra_event_indices[idx].lt]);
      }
      __syncwarp();
      const size_t warp_base = (size_t)blockIdx.x * blockDim.x + (size_t)warp * 32;
      if (warp_base < event_size) {
        const size_t rem = event_size - warp_base;
        const int words = (rem >= 32 ? 32 : (int)rem) * (int)NumCols;
        uint32_t* out = reinterpret_cast<uint32_t*>(&trace_matrix[warp_base * NumCols]);
        for (int k = lane; k < words; k += 32) out[k] = wrow[k];
      }
    } else {
      if (idx < event_size) {
        const size_t stride = (NumCols > 0) ? NumCols : num_cols;
        CpuColsU64<F>* cols = reinterpret_cast<CpuColsU64<F>*>(&trace_matrix[idx * stride]);
        event_to_row<F>(
          events[idx], *cols, byte_trace,
          &alu_lt_events[alu_lt_events_start_index + extra_event_indices[idx].lt]);
      }
    }
  }

  template <class F>
  __global__ void cpu_pad_trace_kernel(
    const size_t num_cols, const size_t num_rows_to_pad,
    F* __restrict__ trace_matrix_start_of_padding)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_rows_to_pad) {
      CpuColsU64<F>* cols = reinterpret_cast<CpuColsU64<F>*>(&trace_matrix_start_of_padding[idx * num_cols]);
      assign_padding_row<F>(*cols);
    }
  }

  template <class F>
  inline RustError pad_trace(
    const size_t num_cols, const size_t start_row, const size_t end_row,
    F* trace, cudaStream_t stream)
  {
    try {
      const int block_size = 256;

      // Pad rows [start_row, end_row) with zeros + padding pattern
      if (start_row < end_row) {
        const size_t num_rows_to_pad = end_row - start_row;
        F* trace_ptr = &trace[start_row * num_cols];
        CUDA_OK(cudaMemsetAsync(trace_ptr, 0, num_rows_to_pad * num_cols * sizeof(F), stream));
        const int grid_size = (num_rows_to_pad + block_size - 1) / block_size;
        cpu_pad_trace_kernel<F><<<grid_size, block_size, 0, stream>>>(num_cols, num_rows_to_pad, trace_ptr);
      }

    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }
    return RustError{cudaSuccess};
  }

  template <class F>
  inline RustError generate_extra_record_and_main(
    const size_t num_cols,
    const size_t event_size,
    const FfiCpuEventU64* events,
    F* trace,
    cudaStream_t stream,
    const CpuExtraEventIndices* extra_event_indices,
    F* byte_trace,
    AluEventU64* alu_add_sub_events,
    const size_t alu_add_sub_events_start_index,
    AluEventU64* alu_lt_events,
    const size_t alu_lt_events_start_index)
  {
    try {
      if (event_size == 0) return RustError{cudaSuccess};

      const int block_size = 256;
      const int grid_size = (event_size + block_size - 1) / block_size;

      // NOTE: this pins the dispatch against CpuColsU64 only. CpuColsU64 is a hand-maintained
      // mirror of the Rust CpuCols — drift between the two is invisible to the compiler, and only
      // MLE_TRACE_COMPARE catches it.
      // Dispatch the WSTAGE fast path on the real compile-time struct width -- a hardcoded
      // one goes stale on any column repack and silently leaves the fast path
      // dead. Deriving it from sizeof(CpuColsU64<F>) keeps the dispatch in sync with the
      // kernel's internal guard (NumCols == sizeof(CpuColsU64<F>)/sizeof(F)); other widths
      // fall through to the generic (NumCols=0) path.
      constexpr uint32_t kCpuCols = sizeof(CpuColsU64<F>) / sizeof(F);
      // Pin the width: the Rust `NUM_CPU_COLS` is the same 58, and the host writes `is_last`
      // into this layout by column index after the kernels run. A silent width change on either
      // side would put that write in the wrong column.
      static_assert(sizeof(CpuColsU64<F>) / sizeof(F) == 58, "CpuCols width drift vs Rust");
      if (num_cols == kCpuCols) {
        // WSTAGE launch shape: block 128 keeps the dynamic smem at
        // 4 warps x 32 rows x kCpuCols x 4B, under the 48KB default —
        // no cudaFuncSetAttribute opt-in needed (requesting near the ~99KB sm_120
        // cap fails with a sticky invalid-argument error; block 256 would need it).
        const int wstage_block = 128;
        const int wstage_grid = (event_size + wstage_block - 1) / wstage_block;
        const size_t wstage_shmem =
          (size_t)(wstage_block / 32) * 32 * kCpuCols * sizeof(F);
        cpu_events_to_trace_kernel<F, kCpuCols><<<wstage_grid, wstage_block, wstage_shmem, stream>>>(
          event_size, events, trace, num_cols, extra_event_indices, byte_trace,
          alu_add_sub_events, alu_add_sub_events_start_index,
          alu_lt_events, alu_lt_events_start_index);
      } else {
        cpu_events_to_trace_kernel<F, 0><<<grid_size, block_size, 0, stream>>>(
          event_size, events, trace, num_cols, extra_event_indices, byte_trace,
          alu_add_sub_events, alu_add_sub_events_start_index,
          alu_lt_events, alu_lt_events_start_index);
      }
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }
    return RustError{cudaSuccess};
  }

} // namespace pico_gpu::cpu_u64
