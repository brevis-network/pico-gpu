// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Packed-event (MLE_PACKED_EVENTS=1) trace-gen kernels: cpu / memory_read_write / program / lt /
// add / sub / addw / subw / partition / divrem.
//
// The packed kernels come in three shapes, and which one a chip uses decides whether an edit to its
// production body reaches the packed path:
//
//   - lt / add / sub / addw / subw / divrem reconstruct the fat event from the packed stream (exact
//     by the packing invariants, see packed_events.cuh) and call the ORIGINAL event_to_row body, so
//     they follow edits to it automatically.
//   - cpu and memory_read_write are hand-written direct-fill MIRRORS of their fat bodies: they read
//     packed fields straight into columns, because reconstructing the 304B fat event puts its
//     unions in thread-local memory. An edit to cpu.cuh or memory_read_write.cuh does NOT reach
//     them — the mirror must be updated in step, or the two paths silently disagree.
//   - partition and program share no body with the fat path: they read one or two fields (.opcode,
//     .pc) off the packed struct directly.
//
// divrem's original body emits extra ALU events into a small local AluEventU64 scratch pre-filled
// with a sentinel opcode; its wrapper then converts the written slots to the packed ALU format
// (writes are dense from slot 0, so slot order is preserved). The cpu mirror writes only
// LtEventPacked (the branch comparison) and no ADD event -- a JALR target is proven in-chip; the
// memrw mirror emits no ALU events (the address addition is proven in-chip) and only carries the
// slot pointer.
//
// Nothing in this repo compiles these headers: the translation units that include them are
// pico's gpu-api/cuda/chips_group1.cu (this file) and chips_group2.cu (packed_ecall.cuh), so
// the ABI static_asserts in packed_events.cuh only fire over there. Include this from
// chips_group1.cu only — it pulls in headers with non-template __global__ kernels, which must stay
// single-TU.

#pragma once

#include "common/exception.cuh"
#include "common/rusterror.h"

#include "tracegen/riscv/packed_events.cuh"
#include "tracegen/riscv/cpu.cuh"
#include "tracegen/riscv/memory_read_write.cuh"
#include "tracegen/riscv/lt.cuh"
#include "tracegen/riscv/add_sub.cuh"
#include "tracegen/riscv/divrem.cuh"
#include "tracegen/riscv/program.cuh"

// ─── CPU chip ───────────────────────────────────────────────────────────────────────────────
namespace pico_gpu::cpu_u64_packed {
  using namespace pico_gpu::cpu_u64;

  // Direct-fill specialization: reads the packed fields straight into the columns instead of
  // reconstructing the 304B fat event (whose unions land in thread-local memory). Line-for-line
  // mirror of cpu_u64::event_to_row on the kernel path; the derived
  // quantities (record chunk == chunk, record ts == clk + {3,2,1}, and a/b/c holding the record's
  // committed value) are guaranteed per event by the Rust packer, which falls back to the fat path
  // on any violation.
  template <class F>
  __device__ inline void event_to_row_packed(
    const FfiCpuEventPacked& pe, const uint32_t chunk, CpuColsU64<F>& cols, F* byte_trace,
    LtEventPacked* alu_lt_events)
  {
    // populate_chunk_clk mirror: one `clk` column. No chunk column (the AIR's messages carry the
    // `execution_chunk` public value) and no clk byte event on either path: no such lookup
    // exists on the Rust side (clk is bounded by the row chain, riscv_cpu/chunk_clk), so
    // emitting one here would unbalance the byte multiset.
    cols.clk = F::from_canonical_u32(pe.clk);

    // pc / next_pc (u32 in the packed stream): two u16 limbs each, the Rust `Pc`.
    const uint64_t pc = pe.pc;
    const uint64_t next_pc = pe.next_pc;
    write_pc_from_u64(cols.pc, pc);
    write_pc_from_u64(cols.next_pc, next_pc);
    // No pc or next_pc range-check events here either (mirrors the fat path): both lookups are
    // gone on the Rust side, pc's limbs being pinned element by element by the Program lookup.
    // Emitting them would unbalance the byte multiset.

    // Instruction: a plain 32B union-free local, so it stays in registers (SROA), feeding the
    // unmodified instruction/selector helpers.
    InstructionU64 instr;
    instr.opcode = static_cast<Opcode>(pe.opcode);
    instr.op_a = pe.op_a;
    instr.op_b = static_cast<uint64_t>(static_cast<int64_t>(pe.op_b));
    instr.op_c = static_cast<uint64_t>(static_cast<int64_t>(pe.op_c));
    instr.imm_b = (pe.flags & PK_IMM_B) != 0;
    instr.imm_c = (pe.flags & PK_IMM_C) != 0;
    instruction::populate_instruction_u64<F>(instr, cols.instruction);
    populate_opcode_selector<F>(instr, cols.opcode_selector);

    write_word_from_u64(cols.op_a_access.value_mut(), pe.a);
    write_word_from_u64(cols.op_b_access.value_mut(), pe.b);
    write_word_from_u64(cols.op_c_access.value_mut(), pe.c);

    // Operand memory accesses. populate_read_write_cols_u64 / populate_read_cols_u64 mirrors:
    // pe.a/b/c already hold the record's committed value (see packed_events.cuh), timestamps are
    // clk + {3,2,1}, record chunk == chunk; b/c records are always reads.
    if (pe.flags & PK_A_SOME) {
      write_word_from_u64(
        cols.op_a_access.prev_value,
        (pe.flags & PK_A_WRITE) ? pe.a_prev_value : pe.a);
      populate_memory_access_u64<F>(
        cols.op_a_access.access, pe.a, chunk, pe.clk + 3,
        pe.a_prev_chunk, pe.a_prev_ts, byte_trace);
    }
    if (pe.flags & PK_B_SOME) {
      populate_memory_access_u64<F>(
        cols.op_b_access.access, pe.b, chunk, pe.clk + 2,
        pe.b_prev_chunk, pe.b_prev_ts, byte_trace);
    }
    if (pe.flags & PK_C_SOME) {
      populate_memory_access_u64<F>(
        cols.op_c_access.access, pe.c, chunk, pe.clk + 1,
        pe.c_prev_chunk, pe.c_prev_ts, byte_trace);
    }

    // No byte event for a jump row's return address (mirrors the fat path): the AIR has no range
    // check on op_a there; limb 0 of pc+4 is canonical because pc_carry below is pinned to the
    // Program table's [pc0 >= 0xFFFC].

    // populate_branch mirror with a direct packed Lt write. No ADD event from any row: a JAL
    // target is a field equation and a JALR target is proven in-chip (Rust jump/constraints.rs).
    // (The fat events' clk fields are not part of the packed ALU formats -- no consumer of these
    // buffers reads them. The lt format does carry `.a` (both comparison bits) and `is_cmp`.)
    const Opcode op = instr.opcode;
    const bool is_branch = instruction::is_branch_instruction_u64(instr);
    if (is_branch) {
      const bool use_signed_comparison = (op == Opcode::BLT || op == Opcode::BGE);
      bool a_lt_b, a_gt_b;
      if (use_signed_comparison) {
        a_lt_b = (int64_t)pe.a < (int64_t)pe.b;
        a_gt_b = (int64_t)pe.a > (int64_t)pe.b;
      } else {
        a_lt_b = pe.a < pe.b;
        a_gt_b = pe.a > pe.b;
      }
      const uint8_t alu_op_code =
        static_cast<uint8_t>(use_signed_comparison ? Opcode::SLT : Opcode::SLTU);

      // One merged comparison event per branch (mirrors cpu.cuh populate_branch and the Rust
      // count of one lt slot per branch).
      LtEventPacked lt_ev = {};
      lt_ev.opcode = alu_op_code;
      lt_ev.a = (a_lt_b ? 1ULL : 0ULL) | ((a_gt_b ? 1ULL : 0ULL) << 16);
      lt_ev.b = pe.a;
      lt_ev.c = pe.b;
      lt_ev.is_cmp = 1;
      alu_lt_events[0] = lt_ev;

      // Neither the equality bit nor the taken predicate is a column (see cpu.cuh); the slot
      // that held the latter is `is_last`, written by the host. No ADD event on a taken branch.
      cols.a_lt_b = F::from_bool(a_lt_b);
      cols.a_gt_b = F::from_bool(a_gt_b);
    }
    // No AUIPC arm: since X12 the transpiler re-encodes `AUIPC rd, imm` as the two-immediate
    // `ADD rd, imm, pc`, so that row's Add event arrives through the emulator's generic ALU path
    // and no slot is reserved for it here.

    // populate_ecall mirror (reads only cols filled above).
    bool is_halt = false;
    if (cols.opcode_selector.is_ecall == F::one()) {
      const uint32_t limb0 = cols.op_a_access.prev_value._0[0].as_canonical_u32();
      const uint8_t syscall_id_byte = static_cast<uint8_t>(limb0 & 0xFF);
      is_halt = syscall_id_byte == static_cast<uint8_t>(static_cast<uint32_t>(SyscallCode::HALT));
      cols.is_halt = F::from_bool(is_halt);
    }

    // num_extra_clk mirror.
    if (instruction::is_ecall_instruction_u64(instr)) {
      const uint32_t limb1 = cols.op_a_access.prev_value._0[1].as_canonical_u32();
      cols.num_extra_clk = F::from_canonical_u8(static_cast<uint8_t>(limb1 & 0xFF));
    }

    // pc_carry mirror: [pc0 >= 0xFFFC] on every real row, the Program table's value for this pc;
    // see the fat body for the single-witness argument.
    cols.pc_carry = F::from_bool((pc & 0xFFFF) >= 0xFFFC);

    // The two dispatch-multiplicity / JALR columns the fat body fills. This mirror is
    // hand-written, so they do not arrive with an edit to cpu.cuh — and they are live columns
    // the AIR reads, so leaving them zero fails the dispatch lookups on the packed path, which
    // is the default (MLE_PACKED_EVENTS=1).
    {
      const F not_x0 = F::from_bool(pe.op_a != 0);
      cols.is_alu_not_x0 = F::from_bool(instruction::is_alu_instruction_u64(instr)) * not_x0;
      // Bit 0 of the unmasked JALR target, which the emulator masked out of `next_pc`.
      cols.jalr_lsb = (instr.opcode == Opcode::JALR)
                        ? F::from_canonical_u32(static_cast<uint32_t>(
                            (static_cast<uint64_t>(pe.b) + static_cast<uint64_t>(pe.c)) & 1ull))
                        : F::zero();
    }
    cols.is_real = F::one();
  }

  template <class F, size_t NumCols>
  __global__ void cpu_events_to_trace_kernel_packed(
    const size_t event_size,
    const FfiCpuEventPacked* __restrict__ events,
    const uint32_t chunk,
    F* __restrict__ trace_matrix,
    const size_t num_cols,
    const CpuExtraEventIndices* __restrict__ extra_event_indices,
    F* __restrict__ byte_trace,
    // Kept in the kernel and FFI signature, never read: no Cpu row writes an ADD event (see the
    // fat kernel), and `extra_event_indices[idx].add` is always 0.
    AddSubEventPacked* /*alu_add_sub_events*/,
    const size_t /*alu_add_sub_events_start_index*/,
    LtEventPacked* alu_lt_events,
    const size_t alu_lt_events_start_index)
  {
    // No bulk chunk U16 event here either -- this prologue mirrored the fat kernel's
    // first-thread bump, and THAT (not the per-row copy) was the live emitter on both paths.

    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if constexpr (NumCols > 0 && NumCols == sizeof(CpuColsU64<F>) / sizeof(F) &&
                  sizeof(F) == sizeof(uint32_t)) {
      // WSTAGE fast path (see the fat kernel for the full rationale).
      extern __shared__ uint32_t wstage_smem_p[];
      const int warp = threadIdx.x >> 5;
      const int lane = threadIdx.x & 31;
      uint32_t* wrow = wstage_smem_p + (size_t)warp * 32 * NumCols;
      uint32_t* myrow = wrow + (size_t)lane * NumCols;
#pragma unroll
      for (int i = 0; i < (int)NumCols; i++) myrow[i] = 0u;
      __syncwarp();
      if (idx < event_size) {
        CpuColsU64<F>* cols = reinterpret_cast<CpuColsU64<F>*>(myrow);
        event_to_row_packed<F>(
          events[idx], chunk, *cols, byte_trace,
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
        event_to_row_packed<F>(
          events[idx], chunk, *cols, byte_trace,
          &alu_lt_events[alu_lt_events_start_index + extra_event_indices[idx].lt]);
      }
    }
  }

  template <class F>
  inline RustError generate_extra_record_and_main_packed(
    const size_t num_cols,
    const size_t event_size,
    const FfiCpuEventPacked* events,
    const uint32_t chunk,
    F* trace,
    cudaStream_t stream,
    const CpuExtraEventIndices* extra_event_indices,
    F* byte_trace,
    AddSubEventPacked* alu_add_sub_events,
    const size_t alu_add_sub_events_start_index,
    LtEventPacked* alu_lt_events,
    const size_t alu_lt_events_start_index)
  {
    try {
      if (event_size == 0) return RustError{cudaSuccess};

      const int block_size = 256;
      const int grid_size = (event_size + block_size - 1) / block_size;

      constexpr uint32_t kCpuCols = sizeof(CpuColsU64<F>) / sizeof(F);
      // Pin the width, exactly as cpu.cuh does for the fat path. This mirror is a second
      // hand-written copy of the Rust CpuCols layout, and the host writes `is_last` into the
      // trace by column index after these kernels run, so a silent width change on either side
      // would put that write in the wrong column on the packed path -- which is the default
      // (MLE_PACKED_EVENTS=1). The Rust `NUM_CPU_COLS` asserts the same 58.
      static_assert(sizeof(CpuColsU64<F>) / sizeof(F) == 58, "CpuCols width drift vs Rust");
      if (num_cols == kCpuCols) {
        const int wstage_block = 128;
        const int wstage_grid = (event_size + wstage_block - 1) / wstage_block;
        const size_t wstage_shmem =
          (size_t)(wstage_block / 32) * 32 * kCpuCols * sizeof(F);
        cpu_events_to_trace_kernel_packed<F, kCpuCols>
          <<<wstage_grid, wstage_block, wstage_shmem, stream>>>(
            event_size, events, chunk, trace, num_cols, extra_event_indices, byte_trace,
            alu_add_sub_events, alu_add_sub_events_start_index,
            alu_lt_events, alu_lt_events_start_index);
      } else {
        cpu_events_to_trace_kernel_packed<F, 0><<<grid_size, block_size, 0, stream>>>(
          event_size, events, chunk, trace, num_cols, extra_event_indices, byte_trace,
          alu_add_sub_events, alu_add_sub_events_start_index,
          alu_lt_events, alu_lt_events_start_index);
      }
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }
    return RustError{cudaSuccess};
  }
} // namespace pico_gpu::cpu_u64_packed

// ─── MemoryReadWrite chip ───────────────────────────────────────────────────────────────────
namespace pico_gpu::memory_read_write_u64_packed {
  using namespace pico_gpu::memory_read_write_u64;

  // Direct-fill specialization: line-for-line mirror of event_to_row_u64 +
  // populate_memory_access_rw_u64 + mem_inst_populate_u64 + populate_memory_u64, reading the
  // packed fields directly. The fat path's "write raw operand, then overwrite with the record
  // value" dance collapses to single writes because the packed stream already carries the
  // committed values (verified per event by the Rust packer, fat fallback otherwise).
  template <class F>
  __device__ inline void event_to_row_u64_packed(
    const FfiCpuEventPacked& pe, const FfiMemRecordPacked& mrec, const uint32_t chunk,
    MemoryChipValueColsU64<F>& col, F* byte_trace)
  {
    col.chunk = F::from_canonical_u32(chunk);
    col.clk = F::from_canonical_u32(pe.clk);

    const bool m_some = (pe.flags & PK_M_SOME) != 0;
    const bool m_write = (pe.flags & PK_M_WRITE) != 0;

    // populate_memory_access_rw_u64 mirror (record chunk == chunk, timestamp == clk).
    if (m_some) {
      if (m_write) {
        write_word_from_u64(col.memory_access.prev_value, mrec.prev_value);
        populate_access_lite_u64<F>(col.memory_access.access, chunk, pe.clk, mrec.value,
                                    mrec.prev_chunk, mrec.prev_ts, mrec.prev_value, byte_trace);
      } else {
        write_word_from_u64(col.memory_access.prev_value, mrec.value);
        populate_access_lite_u64<F>(col.memory_access.access, chunk, pe.clk, mrec.value,
                                    mrec.prev_chunk, mrec.prev_ts, mrec.value, byte_trace);
      }
    }

    // mem_inst_populate_u64 mirror.
    const Opcode opcode = static_cast<Opcode>(pe.opcode);
    col.instruction.op_a_0 = F::from_bool(pe.op_a == 0);
    switch (opcode) {
    case Opcode::LB:  col.instruction.is_lb = F::one(); break;
    case Opcode::LBU: col.instruction.is_lbu = F::one(); break;
    case Opcode::LH:  col.instruction.is_lh = F::one(); break;
    case Opcode::LHU: col.instruction.is_lhu = F::one(); break;
    case Opcode::LW:  col.instruction.is_lw = F::one(); break;
    case Opcode::LWU: col.instruction.is_lwu = F::one(); break;
    case Opcode::LD:  col.instruction.is_ld = F::one(); break;
    case Opcode::SB:  col.instruction.is_sb = F::one(); break;
    case Opcode::SH:  col.instruction.is_sh = F::one(); break;
    case Opcode::SW:  col.instruction.is_sw = F::one(); break;
    case Opcode::SD:  col.instruction.is_sd = F::one(); break;
    default: break;
    }
    // Committed operand words (packed a/b/c are the record-committed values already).
    write_word_from_u64(col.instruction.op_a_value, pe.a);
    write_word_from_u64(col.instruction.op_b_value, pe.b);
    write_word_from_u64(col.instruction.op_c_value, pe.c);

    // populate_memory_u64 mirror.
    const uint64_t memory_addr = pe.b + pe.c;
    col.addr_word[0] = F::from_canonical_u16(static_cast<uint16_t>(memory_addr & 0xFFFF));
    col.addr_word[1] = F::from_canonical_u16(static_cast<uint16_t>((memory_addr >> 16) & 0xFFFF));
    // Two limbs only (an address at or above 2^32 is unprovable; the Rust side asserts it). No
    // ALU ADD event: the AIR proves the addition in-chip (see the fat path in
    // memory_read_write.cuh).

    const uint16_t addr0 = static_cast<uint16_t>(memory_addr & 0xFFFF);
    const uint16_t addr1 = static_cast<uint16_t>((memory_addr >> 16) & 0xFFFF);
    // Mirror of the fat path: one U16Range event for addr_word[1] (limb 0 is bounded by the
    // BitRange(13) offset lookup below), and the stack guard inverse 1 / addr_word[1].
    add_u16_range_check<F>(byte_trace, addr1);
    if (addr1 != 0) {
      col.addr_limb1_inv = F::from_canonical_u16(addr1).reciprocal();
    }

    const uint8_t offset = static_cast<uint8_t>(addr0 & 0x07);
    col.offset_bit[0] = F::from_bool(offset & 1);
    col.offset_bit[1] = F::from_bool(offset & 2);
    col.offset_bit[2] = F::from_bool(offset & 4);

    const int bit0 = offset & 1;
    const int bit1 = (offset >> 1) & 1;
    const int bit2 = (offset >> 2) & 1;
    // The one-hot limb selector is not committed: the AIR forms it from the offset bits.
    const int limb_index = 2 * bit2 + bit1;

    const uint16_t aligned_div8 = static_cast<uint16_t>((addr0 - offset) / 8);
    handle_bit_range_check_event<F>(byte_trace,
      static_cast<uint8_t>(aligned_div8 >> 8),
      static_cast<uint8_t>(aligned_div8 & 0xFF), 13);

    uint64_t mem_value_u64;
    if (m_some) {
      mem_value_u64 = m_write ? mrec.prev_value : mrec.value;
    } else {
      return;
    }

    const uint16_t mem_limbs[4] = {
      static_cast<uint16_t>(mem_value_u64 & 0xFFFF),
      static_cast<uint16_t>((mem_value_u64 >> 16) & 0xFFFF),
      static_cast<uint16_t>((mem_value_u64 >> 32) & 0xFFFF),
      static_cast<uint16_t>((mem_value_u64 >> 48) & 0xFFFF),
    };

    const uint16_t selected_limb = mem_limbs[limb_index];
    col.selected_limb = F::from_canonical_u16(selected_limb);

    const bool is_load = (opcode == Opcode::LB || opcode == Opcode::LBU ||
                          opcode == Opcode::LH || opcode == Opcode::LHU ||
                          opcode == Opcode::LW || opcode == Opcode::LWU ||
                          opcode == Opcode::LD);

    if (is_load) {
      switch (opcode) {
      case Opcode::LB:
      case Opcode::LBU: {
        const uint8_t low_byte = static_cast<uint8_t>(selected_limb & 0xFF);
        const uint8_t high_byte = static_cast<uint8_t>((selected_limb >> 8) & 0xFF);
        // The low byte goes into mem_limb_low_byte, the slot the SB rows use for the same split.
        col.mem_limb_low_byte = F::from_canonical_u8(low_byte);
        const uint8_t sel_byte = bit0 ? high_byte : low_byte;
        col.selected_byte = F::from_canonical_u8(sel_byte);
        // The memory-limb split U8Range lookup shared with SB rows, (low, high) of selected_limb.
        // This mirror is hand-written and edits to memory_read_write.cuh do not reach it (see this
        // file's header), and MLE_PACKED_EVENTS=1 is the default, so it is the copy that runs.
        add_u8_range_check(byte_trace, low_byte, high_byte);

        uint8_t msb = 0;
        if (opcode == Opcode::LB) {
          msb = (sel_byte >> 7) & 1;
          // LB rides the LH/LW U16MSB lookup: lhw_val = 256 * byte, operand 2 * lhw_val - msb * 2^16.
          const uint16_t lhw = static_cast<uint16_t>(256u * sel_byte);
          col.lhw_val = F::from_canonical_u16(lhw);
          const uint16_t diff = static_cast<uint16_t>(lhw * 2u - msb * 65536u);
          add_u16_range_check(byte_trace, diff);
        }
        col.msb = F::from_canonical_u8(msb);
        break;
      }
      case Opcode::LH:
      case Opcode::LHU: {
        uint16_t msb = 0;
        if (opcode == Opcode::LH) {
          msb = (selected_limb >> 15) & 1;
          uint16_t diff = static_cast<uint16_t>(selected_limb * 2 - msb * 65536u);
          add_u16_range_check(byte_trace, diff);
        }
        col.msb = F::from_canonical_u16(msb);
        col.lhw_val = F::from_canonical_u16(selected_limb);
        break;
      }
      case Opcode::LW:
      case Opcode::LWU: {
        // High half of the selected word (selected_limb is the low half).
        const uint16_t sw1 = bit2 ? mem_limbs[3] : mem_limbs[1];

        uint16_t msb = 0;
        if (opcode == Opcode::LW) {
          msb = (sw1 >> 15) & 1;
          uint16_t diff = static_cast<uint16_t>(sw1 * 2 - msb * 65536u);
          add_u16_range_check(byte_trace, diff);
        }
        col.msb = F::from_canonical_u16(msb);
        // Written on LW and LWU alike; the AIR pins it on both.
        col.lhw_val = F::from_canonical_u16(sw1);
        break;
      }
      case Opcode::LD: {
        // `unsigned_mem_val` is de-materialized; nothing extra to populate.
        break;
      }
      default: break;
      }
    }

    // SB needs the byte split plus `shift` (the byte increment), SH needs only `shift` (the limb
    // replacement); SW/SD populate nothing and leave `shift` at zero (store_value is
    // de-materialized). A store's op_a is a register READ, so packed a == the raw register value.
    if (opcode == Opcode::SB) {
      const uint16_t a_limb0 = static_cast<uint16_t>(pe.a & 0xFFFF);
      const uint16_t old_limb =
        static_cast<uint16_t>((mem_value_u64 >> (16 * limb_index)) & 0xFFFF);
      const uint8_t old_low = static_cast<uint8_t>(old_limb & 0xFF);
      const uint8_t old_high = static_cast<uint8_t>((old_limb >> 8) & 0xFF);
      const uint8_t reg_low = static_cast<uint8_t>(a_limb0 & 0xFF);
      const uint8_t reg_high = static_cast<uint8_t>((a_limb0 >> 8) & 0xFF);

      col.register_low_byte = F::from_canonical_u8(reg_low);
      col.mem_limb_low_byte = F::from_canonical_u8(old_low);

      // The byte increment, which is the SB row's `shift`.
      int32_t increment;
      if (bit0 == 0)
        increment = (int32_t)reg_low - (int32_t)old_low;
      else
        increment = (int32_t)reg_low * 256 - (int32_t)old_limb + (int32_t)old_low;

      if (increment >= 0)
        col.shift = F::from_canonical_u32(static_cast<uint32_t>(increment));
      else
        col.shift = F::zero() - F::from_canonical_u32(static_cast<uint32_t>(-increment));

      add_u8_range_check(byte_trace, reg_low, reg_high);
      add_u8_range_check(byte_trace, old_low, old_high);
    } else if (opcode == Opcode::SH) {
      // Mirror of the fat path's SH arm: shift = op_a[0] - selected_limb, field-negative when the
      // stored half-word is below the old one.
      const uint16_t a_limb0 = static_cast<uint16_t>(pe.a & 0xFFFF);
      col.shift = F::from_canonical_u16(a_limb0) - F::from_canonical_u16(selected_limb);
    }
  }

  template <class F, size_t NumCols>
  __global__ void memory_rw_events_to_trace_kernel_packed(
    const FfiCpuEventPacked* __restrict__ events,
    const FfiMemRecordPacked* __restrict__ mem_records,
    const uint32_t chunk,
    const size_t* __restrict__ event_indices,
    const size_t count,
    F* __restrict__ trace_matrix,
    const size_t num_cols,
    F* __restrict__ byte_trace)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if constexpr (NumCols > 0 && NumCols == sizeof(MemoryChipValueColsU64<F>) / sizeof(F) &&
                  sizeof(F) == sizeof(uint32_t)) {
      extern __shared__ uint32_t wstage_smem_mp[];
      const int warp = threadIdx.x >> 5;
      const int lane = threadIdx.x & 31;
      uint32_t* wrow = wstage_smem_mp + (size_t)warp * 32 * NumCols;
      uint32_t* myrow = wrow + (size_t)lane * NumCols;
#pragma unroll
      for (int i = 0; i < (int)NumCols; i++) myrow[i] = 0u;
      __syncwarp();
      if (idx < count) {
        MemoryChipValueColsU64<F>* col = reinterpret_cast<MemoryChipValueColsU64<F>*>(myrow);
        event_to_row_u64_packed<F>(
          events[event_indices[idx]], mem_records[idx], chunk, *col, byte_trace);
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
      MemoryChipValueColsU64<F>* col =
        reinterpret_cast<MemoryChipValueColsU64<F>*>(&trace_matrix[idx * num_cols]);
      event_to_row_u64_packed<F>(
        events[event_indices[idx]], mem_records[idx], chunk, *col, byte_trace);
    }
  }

  // Persistent grid-collapse + WSTAGE variant (mirror of the landed fat dispatch geometry).
  template <class F, size_t NumCols>
  __launch_bounds__(128, 2)
  __global__ void memory_rw_events_to_trace_kernel_persist_wstage_packed(
    const FfiCpuEventPacked* __restrict__ events,
    const FfiMemRecordPacked* __restrict__ mem_records,
    const uint32_t chunk,
    const size_t* __restrict__ event_indices,
    const size_t count,
    F* __restrict__ trace_matrix,
    const size_t num_cols,
    F* __restrict__ byte_trace)
  {
    static_assert(sizeof(F) == sizeof(uint32_t),
                  "row staging assumes a 32-bit field element");
    extern __shared__ uint32_t wstage_smem_mpp[];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    uint32_t* wrow = wstage_smem_mpp + (size_t)warp * 32 * NumCols;
    uint32_t* myrow = wrow + (size_t)lane * NumCols;

    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t warp_base = (size_t)blockIdx.x * blockDim.x + (size_t)warp * 32;
         warp_base < count; warp_base += stride) {
      const size_t idx = warp_base + (size_t)lane;
#pragma unroll
      for (int i = 0; i < (int)NumCols; i++) myrow[i] = 0u;
      __syncwarp();
      if (idx < count) {
        MemoryChipValueColsU64<F>* col = reinterpret_cast<MemoryChipValueColsU64<F>*>(myrow);
        event_to_row_u64_packed<F>(
          events[event_indices[idx]], mem_records[idx], chunk, *col, byte_trace);
      }
      __syncwarp();
      const size_t rem = count - warp_base;
      const int words = (rem >= 32 ? 32 : (int)rem) * (int)NumCols;
      uint32_t* out = reinterpret_cast<uint32_t*>(&trace_matrix[warp_base * NumCols]);
      for (int k = lane; k < words; k += 32) out[k] = wrow[k];
      __syncwarp();
    }
  }

  template <class F>
  inline RustError generate_extra_record_and_main_packed(
    const size_t num_cols,
    const size_t event_size,
    const size_t trace_size,
    const FfiCpuEventPacked* events,
    const FfiMemRecordPacked* mem_records,
    const uint32_t chunk,
    const size_t* event_indices,
    F* trace,
    cudaStream_t stream,
    F* byte_trace)
  {
    try {
      CUDA_OK(cudaMemsetAsync(trace, 0, trace_size * sizeof(F), stream));
      if (event_size == 0) return RustError{cudaSuccess};

      const int block_size = 256;
      const int grid_size = (event_size + block_size - 1) / block_size;

      constexpr size_t kMemRwCols = sizeof(MemoryChipValueColsU64<F>) / sizeof(F);
      // Pin the width, exactly as memory_read_write.cuh does for the fat path; this mirror is a
      // second hand-written copy of the Rust layout.
      static_assert(kMemRwCols == 49, "MemoryChipValueCols width drift vs Rust");
      int sm_count = 0;
      const bool have_sm =
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0) == cudaSuccess
        && sm_count > 0;
      if (num_cols == kMemRwCols && have_sm) {
        const unsigned int blk = 128u;
        unsigned int g = (unsigned int)((sm_count * 9 + 5) / 10); // round(0.9*SM)
        const size_t full = (event_size + (size_t)blk - 1) / (size_t)blk;
        if ((size_t)g > full) g = (unsigned int)full;
        if (g < 1u) g = 1u;
        const size_t shmem = (size_t)(blk / 32) * 32 * kMemRwCols * sizeof(F);
        memory_rw_events_to_trace_kernel_persist_wstage_packed<F, kMemRwCols>
          <<<dim3(g, 1, 1), dim3(blk, 1, 1), shmem, stream>>>(
            events, mem_records, chunk, event_indices, event_size, trace, num_cols, byte_trace);
      } else {
        memory_rw_events_to_trace_kernel_packed<F, 0><<<grid_size, block_size, 0, stream>>>(
          events, mem_records, chunk, event_indices, event_size, trace, num_cols, byte_trace);
      }
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }
    return RustError{cudaSuccess};
  }
} // namespace pico_gpu::memory_read_write_u64_packed

// ─── Lt chip ────────────────────────────────────────────────────────────────────────────────
namespace pico_gpu::lt_u64_packed {
  using namespace pico_gpu::lt_u64;

  template <class F>
  __global__ void lt_events_to_trace_kernel_packed(
    const LtEventPacked* events, const size_t count,
    F* trace_matrix, const size_t num_cols, F* byte_trace)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    LtValueColsU64<F>* cols =
      reinterpret_cast<LtValueColsU64<F>*>(&trace_matrix[idx * num_cols]);
    const AluEventU64 e = unpack_lt(events[idx]);
    event_to_row<F>(e, *cols, byte_trace);
  }

  template <class F>
  inline RustError generate_extra_record_and_main_packed(
    const size_t num_cols,
    const size_t event_size,
    const size_t trace_size,
    const LtEventPacked* events,
    F* trace,
    F* byte_trace,
    cudaStream_t stream)
  {
    try {
      CUDA_OK(cudaMemsetAsync(trace, 0, trace_size * sizeof(F), stream));
      if (event_size == 0) return RustError{cudaSuccess};

      const int block_size = 256;
      const int grid_size = (event_size + block_size - 1) / block_size;

      lt_events_to_trace_kernel_packed<<<grid_size, block_size, 0, stream>>>(
        events, event_size, trace, num_cols, byte_trace);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }
    return RustError{cudaSuccess};
  }
} // namespace pico_gpu::lt_u64_packed

// ─── Add / Addw / Sub / Subw chips ──────────────────────────────────────────────────────────
// One shared kernel shape: unpack the 24B event and run the chip's unmodified event_to_row, behind
// the WSTAGE fast path. Among the unpacked kernels only add_u64 stages its rows this way, so for
// addw/sub/subw this is a new staging path rather than a mirror of an existing one,
// and its correctness rests on the `NumCols == sizeof(COLS<F>)/sizeof(F)` dispatch guard below.
#define PICO_PACKED_ALU_CHIP(NS, COLS)                                                        \
  namespace pico_gpu::NS##_packed {                                                           \
    using namespace pico_gpu::NS;                                                             \
                                                                                              \
    template <class F, size_t NumCols>                                                        \
    __global__ void events_to_trace_kernel_packed(                                            \
      const AddSubEventPacked* events, const size_t count,                                    \
      F* trace_matrix, const size_t num_cols, F* byte_trace)                                  \
    {                                                                                         \
      const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;                               \
      if constexpr (NumCols > 0 && NumCols == sizeof(COLS<F>) / sizeof(F) &&                  \
                    sizeof(F) == sizeof(uint32_t)) {                                          \
        extern __shared__ uint32_t wstage_smem_alu[];                                         \
        const int warp = threadIdx.x >> 5;                                                    \
        const int lane = threadIdx.x & 31;                                                    \
        uint32_t* wrow = wstage_smem_alu + (size_t)warp * 32 * NumCols;                       \
        uint32_t* myrow = wrow + (size_t)lane * NumCols;                                      \
        _Pragma("unroll") for (int i = 0; i < (int)NumCols; i++) myrow[i] = 0u;               \
        __syncwarp();                                                                         \
        if (idx < count) {                                                                    \
          COLS<F>* cols = reinterpret_cast<COLS<F>*>(myrow);                                  \
          const AluEventU64 e = unpack_add_sub(events[idx]);                                  \
          event_to_row<F>(e, *cols, byte_trace);                                              \
        }                                                                                     \
        __syncwarp();                                                                         \
        const size_t warp_base = (size_t)blockIdx.x * blockDim.x + (size_t)warp * 32;         \
        if (warp_base < count) {                                                              \
          const size_t rem = count - warp_base;                                               \
          const int words = (rem >= 32 ? 32 : (int)rem) * (int)NumCols;                       \
          uint32_t* out = reinterpret_cast<uint32_t*>(&trace_matrix[warp_base * NumCols]);    \
          for (int k = lane; k < words; k += 32) out[k] = wrow[k];                            \
        }                                                                                     \
      } else {                                                                                \
        if (idx >= count) return;                                                             \
        COLS<F>* cols = reinterpret_cast<COLS<F>*>(&trace_matrix[idx * num_cols]);            \
        const AluEventU64 e = unpack_add_sub(events[idx]);                                    \
        event_to_row<F>(e, *cols, byte_trace);                                                \
      }                                                                                       \
    }                                                                                         \
                                                                                              \
    template <class F>                                                                        \
    inline RustError generate_extra_record_and_main_packed(                                   \
      const size_t num_cols,                                                                  \
      const size_t event_size,                                                                \
      const size_t trace_size,                                                                \
      const AddSubEventPacked* events,                                                        \
      F* trace,                                                                               \
      F* byte_trace,                                                                          \
      cudaStream_t stream)                                                                    \
    {                                                                                         \
      try {                                                                                   \
        CUDA_OK(cudaMemsetAsync(trace, 0, trace_size * sizeof(F), stream));                   \
        if (event_size == 0) return RustError{cudaSuccess};                                   \
                                                                                              \
        const int block_size = 256;                                                           \
        const int grid_size = (event_size + block_size - 1) / block_size;                     \
                                                                                              \
        constexpr size_t kCols = sizeof(COLS<F>) / sizeof(F);                                 \
        if (num_cols == kCols) {                                                              \
          const int wstage_block = 128;                                                       \
          const size_t wstage_shmem =                                                         \
            (size_t)(wstage_block / 32) * 32 * kCols * sizeof(F);                             \
          const int wstage_grid = (event_size + wstage_block - 1) / wstage_block;             \
          events_to_trace_kernel_packed<F, kCols>                                             \
            <<<wstage_grid, wstage_block, wstage_shmem, stream>>>(                            \
              events, event_size, trace, num_cols, byte_trace);                               \
        } else {                                                                              \
          events_to_trace_kernel_packed<F, 0><<<grid_size, block_size, 0, stream>>>(          \
            events, event_size, trace, num_cols, byte_trace);                                 \
        }                                                                                     \
        CUDA_OK(cudaGetLastError());                                                          \
      } catch (const cuda_error& e) {                                                         \
        return RustError{e.code()};                                                           \
      }                                                                                       \
      return RustError{cudaSuccess};                                                          \
    }                                                                                         \
  }

PICO_PACKED_ALU_CHIP(add_u64, AddValueColsU64)
PICO_PACKED_ALU_CHIP(addw_u64, AddwValueColsU64)
PICO_PACKED_ALU_CHIP(sub_u64, SubValueColsU64)
PICO_PACKED_ALU_CHIP(subw_u64, SubwValueColsU64)

#undef PICO_PACKED_ALU_CHIP

// ─── ALU 4-way partition (packed events) ────────────────────────────────────────────────────
namespace pico_gpu::add_sub_common_packed {
  using namespace pico_gpu::add_sub_common;

  __global__ void partition_histogram_kernel_packed(
    const AddSubEventPacked* __restrict__ events,
    const size_t n,
    int* __restrict__ block_hist)
  {
    __shared__ int s_count[NUM_BUCKETS];
    if (threadIdx.x < NUM_BUCKETS) s_count[threadIdx.x] = 0;
    __syncthreads();

    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
      const int b = opcode_to_bucket(static_cast<Opcode>(events[idx].opcode));
      if (b >= 0) atomicAdd(&s_count[b], 1);
    }
    __syncthreads();

    if (threadIdx.x < NUM_BUCKETS)
      block_hist[blockIdx.x * NUM_BUCKETS + threadIdx.x] = s_count[threadIdx.x];
  }

  __global__ void partition_scatter_kernel_packed(
    const AddSubEventPacked* __restrict__ events,
    const size_t n,
    const int* __restrict__ block_offsets,
    AddSubEventPacked* out0,   // ADD
    AddSubEventPacked* out1,   // SUB
    AddSubEventPacked* out2,   // ADDW
    AddSubEventPacked* out3)   // SUBW
  {
    __shared__ int global_offset[NUM_BUCKETS];
    __shared__ int warp_counts[NUM_WARPS][NUM_BUCKETS];

    if (threadIdx.x < NUM_BUCKETS)
      global_offset[threadIdx.x] = block_offsets[blockIdx.x * NUM_BUCKETS + threadIdx.x];
    __syncthreads();

    const size_t idx     = blockIdx.x * blockDim.x + threadIdx.x;
    const int    warp_id = threadIdx.x >> 5;
    const int    lane_id = threadIdx.x & 31;

    int my_bucket = -1;
    if (idx < n) my_bucket = opcode_to_bucket(static_cast<Opcode>(events[idx].opcode));

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

    if (my_bucket >= 0 && idx < n) {
      const int pos = global_offset[my_bucket] + cross_warp + my_intra_warp;
      const AddSubEventPacked ev = events[idx];
      switch (my_bucket) {
        case 0: out0[pos] = ev; break;
        case 1: out1[pos] = ev; break;
        case 2: out2[pos] = ev; break;
        case 3: out3[pos] = ev; break;
      }
    }
  }

  inline RustError partition_events_packed(
    const AddSubEventPacked* all_events,
    const size_t total_count,
    AddSubEventPacked* out_add,
    AddSubEventPacked* out_sub,
    AddSubEventPacked* out_addw,
    AddSubEventPacked* out_subw,
    cudaStream_t stream)
  {
    if (total_count == 0) return RustError{cudaSuccess};
    try {
      const int num_blocks = (total_count + PARTITION_BLOCK - 1) / PARTITION_BLOCK;
      const uint32_t num_tiles = ((uint32_t)num_blocks + 1023u) / 1024u;

      int* d_block_hist = nullptr;
      CUDA_OK(cudaMallocAsync(&d_block_hist,
                               num_blocks * NUM_BUCKETS * sizeof(int), stream));

      int* d_scratch = nullptr;
      CUDA_OK(cudaMallocAsync(&d_scratch,
                               ((size_t)num_blocks + num_tiles) * NUM_BUCKETS * sizeof(int),
                               stream));
      int4* d_tile_sums = reinterpret_cast<int4*>(d_scratch + (size_t)num_blocks * NUM_BUCKETS);

      partition_histogram_kernel_packed<<<num_blocks, PARTITION_BLOCK, 0, stream>>>(
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

      partition_scatter_kernel_packed<<<num_blocks, PARTITION_BLOCK, 0, stream>>>(
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
} // namespace pico_gpu::add_sub_common_packed

// ─── DivRem chip (packed add_sub extras; the divrem input events stay fat AluEventU64) ─────
namespace pico_gpu::divrem_packed {
  using namespace pico_gpu::divrem;

  template <class F>
  __global__ void events_to_trace_kernel_packed(
    const size_t event_size,
    const size_t trace_size,
    const AluEventU64* events,
    F* trace_matrix,
    const size_t num_cols,
    const DivremExtraEventIndices* extra_event_indices,
    F* byte_trace,
    AddSubEventPacked* alu_add_sub_events,
    const size_t alu_add_sub_events_start_index)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx * num_cols < trace_size) {
      DivRemValueColsU64<F>* cols =
        reinterpret_cast<DivRemValueColsU64<F>*>(&trace_matrix[idx * num_cols]);

      if (idx < event_size) {
        // The original body writes up to two add_sub extras, densely from slot 0.
        AluEventU64 sadd[2];
        sadd[0].opcode = sadd[1].opcode = static_cast<Opcode>(PACKED_ALU_SENTINEL);
        event_to_row<F>(events[idx], *cols, byte_trace, sadd);
        AddSubEventPacked* out =
          &alu_add_sub_events[alu_add_sub_events_start_index + extra_event_indices[idx].add];
#pragma unroll
        for (int i = 0; i < 2; i++)
          if (static_cast<uint8_t>(sadd[i].opcode) != PACKED_ALU_SENTINEL)
            out[i] = pack_add_sub(sadd[i]);
      } else {
        assign_padding_row<F>(*cols);
      }
    }
  }

  template <class F>
  inline RustError generate_extra_record_and_main_packed(
    const size_t num_cols,
    const size_t event_size,
    const size_t trace_size,
    const AluEventU64* events,
    F* trace,
    cudaStream_t stream,
    const DivremExtraEventIndices* extra_event_indices,
    F* byte_trace,
    AddSubEventPacked* alu_add_sub_events,
    const size_t alu_add_sub_events_start_index)
  {
    try {
      CUDA_OK(cudaMemsetAsync(trace, 0, trace_size * sizeof(F), stream));
      // No early return on zero events: a whitelist shape can list this chip with no events,
      // and its padding row is not all-zero, so the kernel runs over every row and writes it.

      const int block_size = 256;
      const int num_rows = trace_size / num_cols;
      const int grid_size = (num_rows + block_size - 1) / block_size;

      events_to_trace_kernel_packed<<<grid_size, block_size, 0, stream>>>(
        event_size, trace_size, events, trace, num_cols, extra_event_indices,
        byte_trace,
        alu_add_sub_events, alu_add_sub_events_start_index);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }
    return RustError{cudaSuccess};
  }
} // namespace pico_gpu::divrem_packed

// ─── Program chip (multiplicity counting; reads only event.pc) ─────────────────────────────
namespace pico_gpu::program_u64_packed {
  using namespace pico_gpu::program_u64;

  template <class F>
  __global__ void cpu_pcs_to_trace_kernel_packed(
    const FfiCpuEventPacked* events, uint64_t pc_base, size_t count,
    F* trace_matrix, size_t num_cols, size_t num_instrs)
  {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    uint64_t pc = events[idx].pc;
    uint64_t offset = pc - pc_base;
    if (offset % 4 != 0) return;
    uint64_t row_index = offset / 4;
    if (row_index >= num_instrs) return;

    ProgramMultCols<uint32_t>* cols = reinterpret_cast<ProgramMultCols<uint32_t>*>(
      &trace_matrix[row_index * num_cols]);
    atomicAdd(&cols->multiplicity, 1);
  }

  template <class F>
  inline RustError generate_main_packed(
    size_t num_cols,
    size_t event_size,
    size_t trace_size,
    uint64_t pc_base,
    const FfiCpuEventPacked* events,
    F* trace,
    cudaStream_t stream,
    size_t num_instrs = 0)
  {
    try {
      CUDA_OK(cudaMemsetAsync(trace, 0, trace_size * sizeof(F), stream));
      if (event_size == 0) { return RustError{cudaSuccess}; }

      const int block_size = 256;
      size_t convert_rows = (num_instrs > 0) ? num_instrs : (trace_size / num_cols);

      int grid_size = (event_size + block_size - 1) / block_size;
      cpu_pcs_to_trace_kernel_packed<<<grid_size, block_size, 0, stream>>>(
        events, pc_base, event_size, trace, num_cols, convert_rows);
      CUDA_OK(cudaGetLastError());

      grid_size = (convert_rows + block_size - 1) / block_size;
      to_montgomery_kernel<F><<<grid_size, block_size, 0, stream>>>(convert_rows, trace, num_cols);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  }
} // namespace pico_gpu::program_u64_packed
