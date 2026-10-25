// Copyright (c) 2025-2026 Brevis. All Rights Reserved.
// Licensed under the Business Source License 1.1 (BUSL-1.1).
// See LICENSE.md for details.

#pragma once

#include <iostream>
#include "common/rusterror.h"
#include "tracegen/common/types.cuh"
#include "tracegen/common/utils.cuh"


namespace pico_gpu::program_preprocess_u64 {
  // U64 version of program preprocess kernel
  // Uses InstructionU64 (u8 op_a, u64 op_b/c) and ProgramPreprocessedColsU64 with Pc<F>
  // (2 u16 limbs) for pc, SignExtImm<F> (2 limbs + sign) for op_b/op_c,
  // OpcodeSelectorColsU64 (13 class / immediate selectors) for selectors and the pc_carry bit
  // [pc0 >= 0xFFFC] the Cpu rows are pinned to through the Program lookup.

  // A trap instruction (Rust `Opcode::is_trap`): the table holds no row for its pc, see below.
  __device__ inline bool is_trap(Opcode opcode)
  {
    return opcode == Opcode::UNIMP || opcode == Opcode::EBREAK;
  }

  template <class F>
  __global__ void program_preprocess_kernel_u64(
    const InstructionU64* instrs, const size_t count, const size_t padded_count,
    const uint64_t pc_base, F* trace_matrix, const size_t num_cols)
  {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= padded_count) { return; }

    // Padding rows repeat instruction 0 rather than staying zero. A zero row is a valid entry in
    // this table -- it reads as "pc = 0 holds an instruction with no opcode selected" -- and the
    // multiplicity column is prover-supplied and ungated, so a CPU row could claim to execute it.
    // A trap instruction (UNIMP / EBREAK) is replaced by instruction 0 the same way: with no trap
    // selector, a trap row in the table would be claimable as a no-op, while a pc that is absent
    // from the table cannot be claimed at all. The host does both the same way
    // (riscv_program/traces.rs; it also asserts that instruction 0 is not a trap, which this
    // kernel relies on); the two preprocessed tables have to agree or their commitments differ.
    const size_t src = (idx < count && !is_trap(instrs[idx].opcode)) ? idx : 0;

    const InstructionU64& inst = instrs[src];
    const uint64_t pc = pc_base + (src << 2);
    // Pin the preprocessed width: this is a raw reinterpret_cast onto the host-allocated row, so
    // any drift between ProgramPreprocessedColsU64 and the Rust ProgramPreprocessedCols writes
    // past (or short of) the row. The Rust side is
    // `Pc(2) + InstructionCols(9) + OpcodeSelectorCols(13) + pc_carry(1) = 25`.
    static_assert(sizeof(ProgramPreprocessedColsU64<F>) / sizeof(F) == 25,
                  "ProgramPreprocessedCols width drift vs Rust");
    ProgramPreprocessedColsU64<F>* cols =
      reinterpret_cast<ProgramPreprocessedColsU64<F>*>(&trace_matrix[idx * num_cols]);

    // pc as 2 u16 limbs (Pc<F>). The Rust `Pc::try_from` rejects a pc at or above 2^32 at
    // keygen; this side does not check.
    cols->pc._0[0] = F::from_canonical_u16((uint16_t)(pc & 0xFFFF));
    cols->pc._0[1] = F::from_canonical_u16((uint16_t)((pc >> 16) & 0xFFFF));
    // The carry out of limb 0 in pc+4, a function of the row's own pc (a trap row therefore
    // carries row 0's). The Cpu rows write the same [pc0 >= 0xFFFC] (cpu.cuh) and the Program
    // message carries it; the host writes the same value (riscv_program/traces.rs).
    cols->pc_carry = F::from_bool((pc & 0xFFFF) >= 0xFFFC);

    // instruction fields
    cols->instruction.opcode = F::from_canonical_u8((uint8_t)inst.opcode);
    // op_a is a u8 register index, a single limb
    cols->instruction.op_a = F::from_canonical_u16((uint16_t)inst.op_a);
    // op_b / op_c: the two low u16 limbs plus the sign fill of bits 32..64 (Rust
    // `SignExtImm::from_u64`, order limb[0], limb[1], sign). The Rust populate asserts bits
    // 32..64 are all zero or all one; this side takes bit 63 as the sign and does not assert.
    // Same split as instruction::populate_instruction_u64, which the Cpu rows use.
    cols->instruction.op_b.limb[0] = F::from_canonical_u16((uint16_t)(inst.op_b & 0xFFFF));
    cols->instruction.op_b.limb[1] = F::from_canonical_u16((uint16_t)((inst.op_b >> 16) & 0xFFFF));
    cols->instruction.op_b.sign = F::from_bool((inst.op_b >> 63) != 0);
    cols->instruction.op_c.limb[0] = F::from_canonical_u16((uint16_t)(inst.op_c & 0xFFFF));
    cols->instruction.op_c.limb[1] = F::from_canonical_u16((uint16_t)((inst.op_c >> 16) & 0xFFFF));
    cols->instruction.op_c.sign = F::from_bool((inst.op_c >> 63) != 0);
    cols->instruction.op_a_0 = inst.op_a == 0 ? F::one() : F::zero();

    // selectors (OpcodeSelectorColsU64)
    cols->selectors.imm_b = inst.imm_b ? F::one() : F::zero();
    cols->selectors.imm_c = inst.imm_c ? F::one() : F::zero();
    switch ((uint8_t)inst.opcode) {
    case (uint8_t)Opcode::ADD:
    case (uint8_t)Opcode::SUB:
    case (uint8_t)Opcode::XOR:
    case (uint8_t)Opcode::OR:
    case (uint8_t)Opcode::AND:
    case (uint8_t)Opcode::SLL:
    case (uint8_t)Opcode::SRL:
    case (uint8_t)Opcode::SRA:
    case (uint8_t)Opcode::SLT:
    case (uint8_t)Opcode::SLTU:
    case (uint8_t)Opcode::MUL:
    case (uint8_t)Opcode::MULH:
    case (uint8_t)Opcode::MULHU:
    case (uint8_t)Opcode::MULHSU:
    case (uint8_t)Opcode::DIV:
    case (uint8_t)Opcode::DIVU:
    case (uint8_t)Opcode::REM:
    case (uint8_t)Opcode::REMU:
    case (uint8_t)Opcode::ADDW:
    case (uint8_t)Opcode::SUBW:
    case (uint8_t)Opcode::SLLW:
    case (uint8_t)Opcode::SRLW:
    case (uint8_t)Opcode::SRAW:
    case (uint8_t)Opcode::MULW:
    case (uint8_t)Opcode::DIVW:
    case (uint8_t)Opcode::DIVUW:
    case (uint8_t)Opcode::REMW:
    case (uint8_t)Opcode::REMUW:
      // ALU class is derived in the constraints (one-hot complement); no is_alu column.
      break;
    case (uint8_t)Opcode::ECALL:
      cols->selectors.is_ecall = F::one();
      break;
    // per-variant memory flags live in the MemoryReadWriteChip now.
    case (uint8_t)Opcode::LB:
    case (uint8_t)Opcode::LBU:
    case (uint8_t)Opcode::LH:
    case (uint8_t)Opcode::LHU:
    case (uint8_t)Opcode::LW:
    case (uint8_t)Opcode::LWU:
    case (uint8_t)Opcode::LD:
      cols->selectors.is_memory = F::one();
      break;
    case (uint8_t)Opcode::SB:
    case (uint8_t)Opcode::SH:
    case (uint8_t)Opcode::SW:
    case (uint8_t)Opcode::SD:
      cols->selectors.is_memory = F::one();
      cols->selectors.is_store = F::one();
      break;
    case (uint8_t)Opcode::BEQ:
      cols->selectors.is_beq = F::one();
      break;
    case (uint8_t)Opcode::BNE:
      cols->selectors.is_bne = F::one();
      break;
    case (uint8_t)Opcode::BLT:
      cols->selectors.is_blt = F::one();
      break;
    case (uint8_t)Opcode::BGE:
      cols->selectors.is_bge = F::one();
      break;
    case (uint8_t)Opcode::BLTU:
      cols->selectors.is_bltu = F::one();
      break;
    case (uint8_t)Opcode::BGEU:
      cols->selectors.is_bgeu = F::one();
      break;
    case (uint8_t)Opcode::JAL:
      cols->selectors.is_jal = F::one();
      break;
    case (uint8_t)Opcode::JALR:
      cols->selectors.is_jalr = F::one();
      break;
    // No AUIPC case: since X12 the transpiler re-encodes `AUIPC rd, imm` as the two-immediate
    // `ADD rd, imm, pc`, and the selector column is gone. An AUIPC instruction reaching this
    // table (hand-built Program) falls through to `default` and sets no class selector, exactly
    // like the Rust `OpcodeSelectorCols::populate`; the Cpu row then lands in the derived ALU
    // class, so the proof is rejected rather than under-constrained. With `rd != x0` what
    // rejects it is the Cpu AIR's definitional constraint
    // `is_alu_not_x0 == is_alu_instruction * (1 - op_a_0)`, which the honest trace violates
    // (tracegen writes 0 there because AUIPC is not an ALU opcode, while the AIR reads 1);
    // forging that column only moves the failure to the Alu bus, where nothing receives opcode
    // 27. `auipc x0, imm` proves, and correctly so -- it is a no-op.
    // No UNIMP / EBREAK case: `src` is never a trap instruction (see the filler rule above).
    default:
      break;
    }
  }

  template <class F>
  inline RustError generate_preprocessed(
    const InstructionU64* instrs,
    const size_t num_instrs,
    const size_t padded_num_rows,
    const uint64_t pc_base,
    F* trace,
    const size_t num_cols,
    cudaStream_t stream)
  {
    try {
      if (num_instrs == 0) { return RustError{cudaSuccess}; }
      const int block_size = 256;
      int grid_size = (padded_num_rows + block_size - 1) / block_size;
      program_preprocess_kernel_u64<<<grid_size, block_size, 0, stream>>>(
        instrs, num_instrs, padded_num_rows, pc_base, trace, num_cols);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }
    return RustError{cudaSuccess};
  }
} // namespace pico_gpu::program_preprocess_u64

namespace pico_gpu::program_u64 {

  template <class F>
  __global__ void cpu_pcs_to_trace_kernel(
    const FfiCpuEventU64* events, uint64_t pc_base, size_t count,
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
  __global__ void to_montgomery_kernel(size_t num_instrs, F* trace_matrix, size_t num_cols)
  {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_instrs) {
      ProgramMultCols<uint32_t>* cols = reinterpret_cast<ProgramMultCols<uint32_t>*>(
        &trace_matrix[idx * num_cols]);
      cols->multiplicity = ((uint64_t)cols->multiplicity << F::MONTY_BITS) % F::MOD;
    }
  }

  template <class F>
  inline RustError generate_main(
    size_t num_cols,
    size_t event_size,
    size_t trace_size,
    uint64_t pc_base,
    const FfiCpuEventU64* events,
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
      cpu_pcs_to_trace_kernel<<<grid_size, block_size, 0, stream>>>(
        events, pc_base, event_size, trace, num_cols, convert_rows);
      CUDA_OK(cudaGetLastError());

      grid_size = (convert_rows + block_size - 1) / block_size;
      to_montgomery_kernel<<<grid_size, block_size, 0, stream>>>(convert_rows, trace, num_cols);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  } // generate_main

} // namespace pico_gpu::program_u64
