#pragma once

#include <iostream>
#include <util/rusterror.h>
#include "types.hpp"
#include "utils.hpp"

namespace pico_gpu::program {
  // Kernel implementation
  template <class F>
  __global__ void program_preprocess_kernel(
    const Instruction* instrs, const size_t count, const uint32_t pc_base, F* trace_matrix, const size_t num_cols)
  {
    // Calculate thread index
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) { return; }

    const Instruction& inst = instrs[idx];
    const uint32_t pc = pc_base + (idx << 2);
    ProgramPreprocessedCols<F>* cols = reinterpret_cast<ProgramPreprocessedCols<F>*>(&trace_matrix[idx * num_cols]);

    cols->pc = F::from_canonical_u32(pc);
    cols->instruction.opcode = F::from_canonical_u8((uint8_t)inst.opcode);
    cols->instruction.op_a._0[0] = F::from_canonical_u8((inst.op_a >> 0) & 0xff);
    cols->instruction.op_a._0[1] = F::from_canonical_u8((inst.op_a >> 8) & 0xff);
    cols->instruction.op_a._0[2] = F::from_canonical_u8((inst.op_a >> 16) & 0xff);
    cols->instruction.op_a._0[3] = F::from_canonical_u8((inst.op_a >> 24) & 0xff);
    cols->instruction.op_b._0[0] = F::from_canonical_u8((inst.op_b >> 0) & 0xff);
    cols->instruction.op_b._0[1] = F::from_canonical_u8((inst.op_b >> 8) & 0xff);
    cols->instruction.op_b._0[2] = F::from_canonical_u8((inst.op_b >> 16) & 0xff);
    cols->instruction.op_b._0[3] = F::from_canonical_u8((inst.op_b >> 24) & 0xff);
    cols->instruction.op_c._0[0] = F::from_canonical_u8((inst.op_c >> 0) & 0xff);
    cols->instruction.op_c._0[1] = F::from_canonical_u8((inst.op_c >> 8) & 0xff);
    cols->instruction.op_c._0[2] = F::from_canonical_u8((inst.op_c >> 16) & 0xff);
    cols->instruction.op_c._0[3] = F::from_canonical_u8((inst.op_c >> 24) & 0xff);
    cols->instruction.op_a_0 = inst.op_a == 0 ? F::one() : F::zero();
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
      cols->selectors.is_alu = F::one();
      break;
    case (uint8_t)Opcode::ECALL:
      cols->selectors.is_ecall = F::one();
      break;
    case (uint8_t)Opcode::LB:
      cols->selectors.is_lb = F::one();
      break;
    case (uint8_t)Opcode::LBU:
      cols->selectors.is_lbu = F::one();
      break;
    case (uint8_t)Opcode::LH:
      cols->selectors.is_lh = F::one();
      break;
    case (uint8_t)Opcode::LHU:
      cols->selectors.is_lhu = F::one();
      break;
    case (uint8_t)Opcode::LW:
      cols->selectors.is_lw = F::one();
      break;
    case (uint8_t)Opcode::SB:
      cols->selectors.is_sb = F::one();
      break;
    case (uint8_t)Opcode::SH:
      cols->selectors.is_sh = F::one();
      break;
    case (uint8_t)Opcode::SW:
      cols->selectors.is_sw = F::one();
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
    case (uint8_t)Opcode::AUIPC:
      cols->selectors.is_auipc = F::one();
      break;
    case (uint8_t)Opcode::UNIMP:
      cols->selectors.is_unimpl = F::one();
      break;
    default:
      break;
    }
  }

  template <class F>
  __global__ void cpu_events_to_trace_kernel(
    const FfiCpuEvent* events, const uint32_t pc_base, size_t count, F* trace_matrix, size_t num_cols)
  {
    // Calculate thread index
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    const FfiCpuEvent& e = events[idx];
    // let pc = input.program.pc_base + (i as u32 * 4);
    uint32_t row_index = e.pc - pc_base;
    if (row_index & ~0x3 != 0) { return; }
    row_index = row_index / 4;
    ProgramMultCols<uint32_t>* cols = reinterpret_cast<ProgramMultCols<uint32_t>*>(&trace_matrix[row_index * num_cols]);

    atomicAdd(&cols->multiplicity, 1);
  }

  template <class F>
  __global__ void program_kernel(size_t trace_size, F* trace_matrix, size_t num_cols)
  {
    // Calculate thread index
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx * num_cols < trace_size) {
      // TODO: we may compute the min and max row numbers by cpu events, then pass into gpu function and ignore other
      // rows
      ProgramMultCols<uint32_t>* cols = reinterpret_cast<ProgramMultCols<uint32_t>*>(&trace_matrix[idx * num_cols]);

      // each call to program_kernel should have a unique idx, so we can unsafely modify cols->mult
      cols->multiplicity = ((uint64_t)cols->multiplicity << F::MONTY_BITS) % F::MOD;
    }
  }
  template <class F>
  inline RustError generate_preprocessed(
    const Instruction* instrs,
    const size_t num_instrs,
    const uint32_t pc_base,
    F* trace,
    const size_t num_cols,
    cudaStream_t stream)
  {
    try {
      if (num_instrs == 0) { return RustError{cudaSuccess}; }

      // launch kernel
      const int block_size = 256;
      int grid_size = (num_instrs + block_size - 1) / block_size;
      program_preprocess_kernel<<<grid_size, block_size, 0, stream>>>(instrs, num_instrs, pc_base, trace, num_cols);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  } // generate_preprocessed

  template <class F>
  inline RustError generate_main(
    size_t num_cols,
    size_t event_size,
    size_t trace_size,
    uint32_t pc_base,
    const FfiCpuEvent* events,
    F* trace,
    cudaStream_t stream)
  {
    try {
      if (event_size == 0) { return RustError{cudaSuccess}; }

      // launch kernel
      const int block_size = 256;
      int grid_size = (event_size + block_size - 1) / block_size;
      cpu_events_to_trace_kernel<<<grid_size, block_size, 0, stream>>>(events, pc_base, event_size, trace, num_cols);
      CUDA_OK(cudaGetLastError());
      grid_size = (trace_size / num_cols + block_size - 1) / block_size;
      program_kernel<<<grid_size, block_size, 0, stream>>>(trace_size, trace, num_cols);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  } // generate_main
} // namespace pico_gpu::program
