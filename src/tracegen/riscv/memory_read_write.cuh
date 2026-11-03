// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0

#pragma once

#include <cstddef>
#include <iostream>
#include "common/rusterror.h"
#include "tracegen/common/field_word_range_checker.cuh"
#include "tracegen/common/prelude.cuh"
#include "tracegen/common/utils.cuh"
#include "tracegen/riscv/byte.cuh"

// ─── U64 Memory Read/Write ─────────────────────────────────────────
namespace pico_gpu::memory_read_write_u64 {
  using namespace byte;
  using namespace pico_gpu;

  template <class F>
  __device__ inline void write_word_from_u64(Word<F>& word, uint64_t value) {
    word._0[0] = F::from_canonical_u16(static_cast<uint16_t>(value & 0xFFFF));
    word._0[1] = F::from_canonical_u16(static_cast<uint16_t>((value >> 16) & 0xFFFF));
    word._0[2] = F::from_canonical_u16(static_cast<uint16_t>((value >> 32) & 0xFFFF));
    word._0[3] = F::from_canonical_u16(static_cast<uint16_t>((value >> 48) & 0xFFFF));
  }

  // Legacy access layout, which still commits prev_clk. Called by the precompile chips
  // (fp_op.cuh, curve_op.cuh, keccak_control.cuh, precompile_sha256_{compress,extend}.hpp) on
  // their MemoryAccessCols; signature and body unchanged. The Cpu / MemoryReadWrite rows use
  // populate_access_lite_u64 below.
  template <class F>
  __device__ void populate_access_u64(
    MemoryAccessCols<F>& col,
    uint32_t cur_chunk, uint32_t cur_timestamp, uint64_t cur_value,
    uint32_t prev_chunk, uint32_t prev_timestamp, uint64_t prev_value,
    F* byte_trace)
  {
    write_word_from_u64(col.value, cur_value);
    // The predecessor is always in this chunk. When the address was last touched in an earlier
    // chunk there is no in-chunk timestamp to name, so the sentinel 0 stands in -- the same label
    // MemoryLocalChip puts on the state the chunk inherits. The emulator starts each chunk's clk
    // at 4, which keeps every real access above the sentinel. Against the sentinel the diff spans
    // the whole clk range, so it fits 16+9 only because the Rust split config caps a chunk's clk
    // at 2^25 - 1; `& 0x1FF` below would truncate silently past that.
    const uint32_t prev_clk = (prev_chunk == cur_chunk) ? prev_timestamp : 0u;
    col.prev_clk = F::from_canonical_u32(prev_clk);

    const uint64_t diff_minus_one = (uint64_t)cur_timestamp - (uint64_t)prev_clk - 1;
    const uint16_t diff_16bit = static_cast<uint16_t>(diff_minus_one & 0xFFFF);
    const uint16_t diff_hi9 = static_cast<uint16_t>((diff_minus_one >> 16) & 0x1FF);
    col.diff_16bit_limb = F::from_canonical_u16(diff_16bit);
    // Single 9-bit high limb (bits 16..25), BitRange(9)-checked.
    col.diff_hi9 = F::from_canonical_u16(diff_hi9);

    add_u16_range_check(byte_trace, diff_16bit);
    add_bit_range_check(byte_trace, diff_hi9, 9);
  }

  // Lite access layout (Rust `MemoryAccessLiteCols`): same value and difference limbs as above,
  // no prev_clk column. The AIR rebuilds the predecessor clk from the two committed difference
  // limbs, so it is computed here only to difference from. Used by the MemoryReadWrite rows on
  // both the fat path (populate_memory_access_rw_u64) and the packed mirror.
  template <class F>
  __device__ void populate_access_lite_u64(
    MemoryAccessLiteCols<F>& col,
    uint32_t cur_chunk, uint32_t cur_timestamp, uint64_t cur_value,
    uint32_t prev_chunk, uint32_t prev_timestamp, uint64_t prev_value,
    F* byte_trace)
  {
    write_word_from_u64(col.value, cur_value);
    // Sentinel-or-real predecessor clk, exactly as in populate_access_u64.
    const uint32_t prev_clk = (prev_chunk == cur_chunk) ? prev_timestamp : 0u;

    const uint64_t diff_minus_one = (uint64_t)cur_timestamp - (uint64_t)prev_clk - 1;
    const uint16_t diff_16bit = static_cast<uint16_t>(diff_minus_one & 0xFFFF);
    const uint16_t diff_hi9 = static_cast<uint16_t>((diff_minus_one >> 16) & 0x1FF);
    col.diff_16bit_limb = F::from_canonical_u16(diff_16bit);
    // Single 9-bit high limb (bits 16..25), BitRange(9)-checked.
    col.diff_hi9 = F::from_canonical_u16(diff_hi9);

    add_u16_range_check(byte_trace, diff_16bit);
    add_bit_range_check(byte_trace, diff_hi9, 9);
  }

  template <class F>
  __device__ void populate_memory_access_rw_u64(
    MemoryReadWriteLiteCols<F>& col,
    const MemoryRecordEnumU64& record,
    F* byte_trace)
  {
    if (record.tag == MemoryRecordEnumU64::Tag::Read) {
      auto& r = record.read._0;
      write_word_from_u64(col.prev_value, r.value);
      populate_access_lite_u64(col.access, r.chunk, r.timestamp, r.value,
                               r.prev_chunk, r.prev_timestamp, r.value, byte_trace);
    } else {
      auto& w = record.write._0;
      write_word_from_u64(col.prev_value, w.prev_value);
      populate_access_lite_u64(col.access, w.chunk, w.timestamp, w.value,
                               w.prev_chunk, w.prev_timestamp, w.prev_value, byte_trace);
    }
  }

  template <class F>
  __device__ void mem_inst_populate_u64(
    MemoryInstructionColsU64<F>& col,
    const FfiCpuEventU64& event,
    F* byte_trace)
  {
    const auto opcode = event.instruction.opcode;
    // `opcode` column is de-materialized (dispatch message uses the flag-weighted sum).
    col.op_a_0 = F::from_bool(event.instruction.op_a == 0);

    switch (opcode) {
    case Opcode::LB:  col.is_lb = F::one(); break;
    case Opcode::LBU: col.is_lbu = F::one(); break;
    case Opcode::LH:  col.is_lh = F::one(); break;
    case Opcode::LHU: col.is_lhu = F::one(); break;
    case Opcode::LW:  col.is_lw = F::one(); break;
    case Opcode::LWU: col.is_lwu = F::one(); break;
    case Opcode::LD:  col.is_ld = F::one(); break;
    case Opcode::SB:  col.is_sb = F::one(); break;
    case Opcode::SH:  col.is_sh = F::one(); break;
    case Opcode::SW:  col.is_sw = F::one(); break;
    case Opcode::SD:  col.is_sd = F::one(); break;
    default: break;
    }

    write_word_from_u64(col.op_a_value, event.a);
    write_word_from_u64(col.op_b_value, event.b);
    write_word_from_u64(col.op_c_value, event.c);

    // Overwrite with the memory-record values for a, b, and c (must commit the
    // same record-derived values as the Cpu chip's dispatch-lookup operands).
    if (event.a_record.tag == FfiOption<MemoryRecordEnumU64>::Tag::Some) {
      uint64_t val = event.a_record.some._0.value();
      write_word_from_u64(col.op_a_value, val);
    }
    if (event.b_record.tag == FfiOption<MemoryRecordEnumU64>::Tag::Some &&
        event.b_record.some._0.tag == MemoryRecordEnumU64::Tag::Read) {
      write_word_from_u64(col.op_b_value, event.b_record.some._0.read._0.value);
    }
    if (event.c_record.tag == FfiOption<MemoryRecordEnumU64>::Tag::Some &&
        event.c_record.some._0.tag == MemoryRecordEnumU64::Tag::Read) {
      write_word_from_u64(col.op_c_value, event.c_record.some._0.read._0.value);
    }
  }

  template <class F>
  __device__ void populate_memory_u64(
    MemoryChipValueColsU64<F>& col,
    const FfiCpuEventU64& event,
    F* byte_trace)
  {
    const uint64_t memory_addr = event.b + event.c;
    // `addr_word` holds only the low 2 u16 limbs; the AIR proves the addition in-chip with a
    // carry chain whose 3rd and 4th result limbs are literal 0s (an address at or above 2^32 is
    // unprovable; the Rust trace generator asserts the bound, this kernel does not). No ALU ADD
    // event is emitted for it.
    col.addr_word[0] = F::from_canonical_u16(static_cast<uint16_t>(memory_addr & 0xFFFF));
    col.addr_word[1] = F::from_canonical_u16(static_cast<uint16_t>((memory_addr >> 16) & 0xFFFF));

    const uint16_t addr0 = static_cast<uint16_t>(memory_addr & 0xFFFF);
    const uint16_t addr1 = static_cast<uint16_t>((memory_addr >> 16) & 0xFFFF);
    // The AIR U16Range-checks addr_word[1] in-chip (count is_mem); limb 0 is bounded by the
    // BitRange(13) offset lookup below and gets no event of its own.
    add_u16_range_check<F>(byte_trace, addr1);
    // Stack guard inverse: 1 / addr_word[1] on real rows (limb 1 is non-zero above the register
    // space); the memset leaves 0 on padding rows.
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
    if (event.memory_record.tag == FfiOption<MemoryRecordEnumU64>::Tag::Some) {
      auto& rec = event.memory_record.some._0;
      if (rec.tag == MemoryRecordEnumU64::Tag::Read)
        mem_value_u64 = rec.read._0.value;
      else
        mem_value_u64 = rec.write._0.prev_value;
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

    const auto opcode = event.instruction.opcode;
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
        // The low byte goes into mem_limb_low_byte, the slot the SB rows use for the same split
        // of the memory limb they modify (Rust read_write/traces.rs).
        col.mem_limb_low_byte = F::from_canonical_u8(low_byte);
        const uint8_t sel_byte = bit0 ? high_byte : low_byte;
        col.selected_byte = F::from_canonical_u8(sel_byte);
        // The memory-limb split U8Range lookup shared with SB rows: (low, high) of selected_limb.
        // LBU bounds the high byte nowhere else.
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
        // Shared LH/LW/LB mux operand.
        col.lhw_val = F::from_canonical_u16(selected_limb);
        break;
      }
      case Opcode::LW:
      case Opcode::LWU: {
        // The selected word is [selected_limb, sw1] with sw1 chosen by bit2 (word alignment
        // makes bit1 = 0, so selected_limb already is the low half).
        const uint16_t sw1 = bit2 ? mem_limbs[3] : mem_limbs[1];

        uint16_t msb = 0;
        if (opcode == Opcode::LW) {
          msb = (sw1 >> 15) & 1;
          uint16_t diff = static_cast<uint16_t>(sw1 * 2 - msb * 65536u);
          add_u16_range_check(byte_trace, diff);
        }
        col.msb = F::from_canonical_u16(msb);
        // Shared LH/LW/LB mux operand; the AIR pins it on LWU rows too (it is the limb-1 load value
        // there), so it is written on both variants.
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

    // `store_value` is de-materialized; the unified store equation is asserted directly on
    // memory_access.value on the AIR side. SB needs the byte split plus `shift` (the byte
    // increment), SH needs only `shift` (the limb replacement); SW/SD need no extra witnesses
    // and leave `shift` at zero, as its ungated definition requires (the row is pre-zeroed).
    if (opcode == Opcode::SB) {
      const uint16_t a_limb0 = static_cast<uint16_t>(event.a & 0xFFFF);
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
      // The SH row's `shift` is the limb replacement op_a[0] - selected_limb, a field-negative
      // value whenever the stored half-word is below the old one.
      const uint16_t a_limb0 = static_cast<uint16_t>(event.a & 0xFFFF);
      col.shift = F::from_canonical_u16(a_limb0) - F::from_canonical_u16(selected_limb);
    }
  }

  template <class F>
  __device__ void event_to_row_u64(
    const FfiCpuEventU64& event,
    MemoryChipValueColsU64<F>& col,
    F* byte_trace)
  {
    col.chunk = F::from_canonical_u32(event.chunk);
    col.clk = F::from_canonical_u32(static_cast<uint32_t>(event.clk));

    if (event.memory_record.tag == FfiOption<MemoryRecordEnumU64>::Tag::Some) {
      populate_memory_access_rw_u64(col.memory_access, event.memory_record.some._0, byte_trace);
    }

    mem_inst_populate_u64(col.instruction, event, byte_trace);
    populate_memory_u64(col, event, byte_trace);
  }
} // namespace pico_gpu::memory_read_write_u64

// ─── Legacy 32-bit memory_read_write namespace (used by precompile chips) ────
namespace pico_gpu::memory_read_write {
  using namespace pico_gpu::byte;

  template <class F>
  __device__ void populate(
    MemoryAccessCols<F>& col,
    uint32_t cur_value, uint32_t cur_chunk, uint32_t cur_timestamp,
    uint32_t prev_chunk, uint32_t prev_timestamp,
    F* byte_trace)
  {
    col.value._0[0] = F::from_canonical_u32(cur_value & 0xFF);
    col.value._0[1] = F::from_canonical_u32((cur_value >> 8) & 0xFF);
    col.value._0[2] = F::from_canonical_u32((cur_value >> 16) & 0xFF);
    col.value._0[3] = F::from_canonical_u32((cur_value >> 24) & 0xFF);
    // The predecessor is always in this chunk. When the address was last touched in an earlier
    // chunk there is no in-chunk timestamp to name, so the sentinel 0 stands in -- the same label
    // MemoryLocalChip puts on the state the chunk inherits. The emulator starts each chunk's clk
    // at 4, which keeps every real access above the sentinel. Against the sentinel the diff spans
    // the whole clk range, so it fits 16+9 only because the Rust split config caps a chunk's clk
    // at 2^25 - 1; `& 0x1FF` below would truncate silently past that.
    const uint32_t prev_clk = (prev_chunk == cur_chunk) ? prev_timestamp : 0u;
    col.prev_clk = F::from_canonical_u32(prev_clk);

    const uint32_t diff_minus_one = cur_timestamp - prev_clk - 1;
    const uint16_t diff_16bit = static_cast<uint16_t>(diff_minus_one & 0xFFFF);
    const uint16_t diff_hi9 = static_cast<uint16_t>((diff_minus_one >> 16) & 0x1FF);
    col.diff_16bit_limb = F::from_canonical_u16(diff_16bit);
    // Single 9-bit high limb (bits 16..25), BitRange(9)-checked.
    col.diff_hi9 = F::from_canonical_u16(diff_hi9);

    if (byte_trace) {
      add_u16_range_check(byte_trace, diff_16bit);
      add_bit_range_check(byte_trace, diff_hi9, 9);
    }
  }

  /// Populate from MemoryReadRecord (read access col).
  template <class F>
  __device__ void populate(MemoryReadCols<F>& col, const MemoryReadRecord& record, F* byte_trace)
  {
    populate(col.access, record.value, record.chunk, record.timestamp,
             record.prev_chunk, record.prev_timestamp, byte_trace);
  }

  /// Populate from MemoryWriteRecord (write access col).
  template <class F>
  __device__ void populate(MemoryWriteCols<F>& col, const MemoryWriteRecord& record, F* byte_trace)
  {
    col.prev_value._0[0] = F::from_canonical_u32(record.prev_value & 0xFF);
    col.prev_value._0[1] = F::from_canonical_u32((record.prev_value >> 8) & 0xFF);
    col.prev_value._0[2] = F::from_canonical_u32((record.prev_value >> 16) & 0xFF);
    col.prev_value._0[3] = F::from_canonical_u32((record.prev_value >> 24) & 0xFF);
    populate(col.access, record.value, record.chunk, record.timestamp,
             record.prev_chunk, record.prev_timestamp, byte_trace);
  }

  /// Populate from MemoryReadRecord (read-write access col, used for read).
  template <class F>
  __device__ void populate(MemoryReadWriteCols<F>& col, const MemoryReadRecord& record, F* byte_trace)
  {
    col.prev_value._0[0] = F::from_canonical_u32(record.value & 0xFF);
    col.prev_value._0[1] = F::from_canonical_u32((record.value >> 8) & 0xFF);
    col.prev_value._0[2] = F::from_canonical_u32((record.value >> 16) & 0xFF);
    col.prev_value._0[3] = F::from_canonical_u32((record.value >> 24) & 0xFF);
    populate(col.access, record.value, record.chunk, record.timestamp,
             record.prev_chunk, record.prev_timestamp, byte_trace);
  }

  /// Populate from MemoryWriteRecord (read-write access col, used for write).
  template <class F>
  __device__ void populate(MemoryReadWriteCols<F>& col, const MemoryWriteRecord& record, F* byte_trace)
  {
    col.prev_value._0[0] = F::from_canonical_u32(record.prev_value & 0xFF);
    col.prev_value._0[1] = F::from_canonical_u32((record.prev_value >> 8) & 0xFF);
    col.prev_value._0[2] = F::from_canonical_u32((record.prev_value >> 16) & 0xFF);
    col.prev_value._0[3] = F::from_canonical_u32((record.prev_value >> 24) & 0xFF);
    populate(col.access, record.value, record.chunk, record.timestamp,
             record.prev_chunk, record.prev_timestamp, byte_trace);
  }
} // namespace pico_gpu::memory_read_write