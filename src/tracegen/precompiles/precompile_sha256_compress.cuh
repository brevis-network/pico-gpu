// Copyright (c) 2025-2026 Brevis. All Rights Reserved.
// Licensed under the Business Source License 1.1 (BUSL-1.1).
// See LICENSE.md for details.

#pragma once

#include "tracegen/riscv/memory_read_write.cuh"
#include "tracegen/common/utils.cuh"
#include "tracegen/common/types.cuh"
#include "common/exception.cuh"
#include "common/rusterror.h"
#include "tracegen/riscv/byte.cuh"
#include <chrono>
#include <iostream>
#include <ostream>
#include <cstring>
#include "tracegen/precompiles/sha256_operations.cuh"
#include "field/ff_config.hpp"
#include "tracegen/riscv/syscall_addr.cuh"

namespace pico_gpu::precompile_sha_compress {
  using namespace sha256_operations;
  using namespace memory_read_write_u64;
  using namespace byte;

  template <typename T>
  struct Word32 {
    T _0[2];
  };

  template <typename T>
  struct AddrWord {
    T _0[3];
  };

  template <typename T>
  struct AddrAddGadget {
    T value[3];
    __device__ uint64_t populate(field_t* byte_trace, uint64_t base, uint64_t offset) {
        uint64_t expected = base + offset;
        uint16_t limbs[3] = {
            static_cast<uint16_t>(expected & 0xFFFF),
            static_cast<uint16_t>((expected >> 16) & 0xFFFF),
            static_cast<uint16_t>((expected >> 32) & 0xFFFF)
        };
        value[0] = T::from_canonical_u16(limbs[0]);
        value[1] = T::from_canonical_u16(limbs[1]);
        value[2] = T::from_canonical_u16(limbs[2]);
        // Range check all three limbs, matching Rust's add_u16_range_checks
        pico_gpu::byte::add_u16_range_check(byte_trace, limbs[0]);
        pico_gpu::byte::add_u16_range_check(byte_trace, limbs[1]);
        pico_gpu::byte::add_u16_range_check(byte_trace, limbs[2]);
        return expected;
    }
  };

  struct MemoryReadRecordU64 {
    uint64_t value;
    uint32_t chunk;
    uint32_t timestamp;
    uint32_t prev_chunk;
    uint32_t prev_timestamp;
  };

  struct MemoryWriteRecordU64 {
    uint64_t value;
    uint32_t chunk;
    uint32_t timestamp;
    uint64_t prev_value;
    uint32_t prev_chunk;
    uint32_t prev_timestamp;
  };

   struct ShaCompressFfiEvent {
        uint32_t chunk;
        uint32_t _padding;
        uint64_t clk;
        uint64_t w_ptr;
        uint64_t h_ptr;
        uint32_t h[8];
        uint32_t w[64];
        MemoryReadRecordU64 h_read_records[8];
        MemoryReadRecordU64 w_i_read_records[64];
        MemoryWriteRecordU64 h_write_records[8];
   };

  /// SHA256 round constants.
  __device__ static const uint32_t SHA_COMPRESS_K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

  template <typename T>
  struct FixedRotateRightU32Gadget {
    T value[2];
    T higher_limb[2];

    __device__ uint32_t populate(uint32_t input, uint32_t rotation, field_t* byte_trace) {
        uint32_t expected = (input >> rotation) | (input << (32 - rotation));
        value[0] = T::from_canonical_u16(static_cast<uint16_t>(expected & 0xFFFF));
        value[1] = T::from_canonical_u16(static_cast<uint16_t>((expected >> 16) & 0xFFFF));
        uint16_t input_limbs[2] = {static_cast<uint16_t>(input & 0xFFFF), static_cast<uint16_t>((input >> 16) & 0xFFFF)};
        uint32_t nb_limbs_to_rotate = rotation / 16;
        uint32_t nb_bits_to_rotate = rotation % 16;
        uint16_t input_limbs_rotated[2] = {input_limbs[nb_limbs_to_rotate % 2], input_limbs[(1 + nb_limbs_to_rotate) % 2]};
        for (int i = 0; i < 2; i++) {
            uint16_t limb = input_limbs_rotated[i];
            uint16_t lw = limb & ((1 << nb_bits_to_rotate) - 1);
            uint16_t hw = limb >> nb_bits_to_rotate;
            higher_limb[i] = T::from_canonical_u16(hw);
            pico_gpu::byte::handle_bit_range_check_event(byte_trace, (uint8_t)(lw >> 8), (uint8_t)(lw & 0xFF), (uint8_t)nb_bits_to_rotate);
            pico_gpu::byte::handle_bit_range_check_event(byte_trace, (uint8_t)(hw >> 8), (uint8_t)(hw & 0xFF), (uint8_t)(16 - nb_bits_to_rotate));
        }
        return expected;
    }
  };

  template <typename T>
  struct U32toU8Gadget {
    T low_bytes[2];
    __device__ void populate(uint32_t val) {
        uint16_t limb0 = static_cast<uint16_t>(val & 0xFFFF);
        uint16_t limb1 = static_cast<uint16_t>((val >> 16) & 0xFFFF);
        low_bytes[0] = T::from_canonical_u8(static_cast<uint8_t>(limb0 & 0xFF));
        low_bytes[1] = T::from_canonical_u8(static_cast<uint8_t>(limb1 & 0xFF));
    }
  };

  template <typename T>
  struct XorU32Gadget {
    U32toU8Gadget<T> b_low_bytes;
    U32toU8Gadget<T> c_low_bytes;
    T value[4];
    __device__ uint32_t populate(uint32_t a, uint32_t b, field_t* byte_trace) {
        uint32_t expected = a ^ b;
        b_low_bytes.populate(a);
        c_low_bytes.populate(b);
        for (int i = 0; i < 4; i++) {
            uint8_t ab = (a >> (i * 8)) & 0xFF;
            uint8_t bb = (b >> (i * 8)) & 0xFF;
            uint8_t x = ab ^ bb;
            value[i] = T::from_canonical_u8(x);
            pico_gpu::byte::handle_byte_lookup_event(byte_trace, ByteOpcode::XOR, ab, bb);
        }
        return expected;
    }
  };

  template <typename T>
  struct AndU32Gadget {
    U32toU8Gadget<T> b_low_bytes;
    U32toU8Gadget<T> c_low_bytes;
    T value[4];
    __device__ uint32_t populate(uint32_t a, uint32_t b, field_t* byte_trace) {
        uint32_t expected = a & b;
        b_low_bytes.populate(a);
        c_low_bytes.populate(b);
        for (int i = 0; i < 4; i++) {
            uint8_t ab = (a >> (i * 8)) & 0xFF;
            uint8_t bb = (b >> (i * 8)) & 0xFF;
            uint8_t x = ab & bb;
            value[i] = T::from_canonical_u8(x);
            pico_gpu::byte::handle_byte_lookup_event(byte_trace, ByteOpcode::AND, ab, bb);
        }
        return expected;
    }
  };

  template <typename T>
  struct NotU32Gadget {
    T value[2];
    __device__ uint32_t populate(uint32_t a) {
        uint32_t expected = ~a;
        value[0] = T::from_canonical_u16(static_cast<uint16_t>(expected & 0xFFFF));
        value[1] = T::from_canonical_u16(static_cast<uint16_t>((expected >> 16) & 0xFFFF));
        return expected;
    }
  };

  template <typename T>
  struct AddU32Gadget {
    T value[2];
    __device__ uint32_t populate(uint32_t a, uint32_t b, field_t* byte_trace) {
        uint32_t expected = a + b;
        uint16_t limbs[2] = {static_cast<uint16_t>(expected & 0xFFFF), static_cast<uint16_t>((expected >> 16) & 0xFFFF)};
        value[0] = T::from_canonical_u16(limbs[0]);
        value[1] = T::from_canonical_u16(limbs[1]);
        pico_gpu::byte::add_u16_range_check(byte_trace, limbs[0]);
        pico_gpu::byte::add_u16_range_check(byte_trace, limbs[1]);
        return expected;
    }
  };

  template <typename T>
  struct Add5U32Gadget {
    T value[2];
    __device__ uint32_t populate(uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t e, field_t* byte_trace) {
        uint32_t expected = a + b + c + d + e;
        uint16_t limbs[2] = {static_cast<uint16_t>(expected & 0xFFFF), static_cast<uint16_t>((expected >> 16) & 0xFFFF)};
        value[0] = T::from_canonical_u16(limbs[0]);
        value[1] = T::from_canonical_u16(limbs[1]);
        pico_gpu::byte::add_u16_range_check(byte_trace, limbs[0]);
        pico_gpu::byte::add_u16_range_check(byte_trace, limbs[1]);

        // Carry range checks matching Rust
        uint32_t base = 1u << 16;
        uint32_t carry = 0;
        uint16_t a_l[2] = {static_cast<uint16_t>(a & 0xFFFF), static_cast<uint16_t>((a >> 16) & 0xFFFF)};
        uint16_t b_l[2] = {static_cast<uint16_t>(b & 0xFFFF), static_cast<uint16_t>((b >> 16) & 0xFFFF)};
        uint16_t c_l[2] = {static_cast<uint16_t>(c & 0xFFFF), static_cast<uint16_t>((c >> 16) & 0xFFFF)};
        uint16_t d_l[2] = {static_cast<uint16_t>(d & 0xFFFF), static_cast<uint16_t>((d >> 16) & 0xFFFF)};
        uint16_t e_l[2] = {static_cast<uint16_t>(e & 0xFFFF), static_cast<uint16_t>((e >> 16) & 0xFFFF)};
        uint8_t carry_limbs[2];
        for (int i = 0; i < 2; i++) {
            carry = ((uint32_t)a_l[i] + (uint32_t)b_l[i] + (uint32_t)c_l[i] + (uint32_t)d_l[i] + (uint32_t)e_l[i] + carry - (uint32_t)limbs[i]) / base;
            carry_limbs[i] = (uint8_t)carry;
        }
        pico_gpu::byte::add_u8_range_check(byte_trace, carry_limbs[0], carry_limbs[1]);

        return expected;
    }
  };

  template <typename T>
  struct ShaCompressCols {
    T chunk; T clk; AddrWord<T> w_ptr; AddrWord<T> h_ptr; T index;
    T octet[8]; T octet_num[10]; MemoryReadWriteCols<T> mem; T mem_value[2];
    AddrWord<T> mem_addr; AddrAddGadget<T> mem_addr_init; AddrAddGadget<T> mem_addr_compress; AddrAddGadget<T> mem_addr_finalize;
    Word32<T> a, b, c, d, e, f, g, h, k;
    FixedRotateRightU32Gadget<T> e_rr_6, e_rr_11, e_rr_25;
    XorU32Gadget<T> s1_intermediate, s1;
    AndU32Gadget<T> e_and_f; NotU32Gadget<T> e_not; AndU32Gadget<T> e_not_and_g; XorU32Gadget<T> ch;
    Add5U32Gadget<T> temp1;
    FixedRotateRightU32Gadget<T> a_rr_2, a_rr_13, a_rr_22;
    XorU32Gadget<T> s0_intermediate, s0;
    AndU32Gadget<T> a_and_b, a_and_c, b_and_c;
    XorU32Gadget<T> maj_intermediate, maj;
    AddU32Gadget<T> temp2, d_add_temp1, temp1_add_temp2;
    Word32<T> finalized_operand; AddU32Gadget<T> finalize_add;
    T is_initialize, is_compression, is_finalize, is_real;
    __device__ ShaCompressCols() {}
  };

  // Helper to set mem_value as two u16 limbs from a u32
  template <typename T>
  __device__ inline void set_mem_value(T (&mv)[2], uint32_t val) {
    mv[0] = T::from_canonical_u16(static_cast<uint16_t>(val & 0xFFFF));
    mv[1] = T::from_canonical_u16(static_cast<uint16_t>((val >> 16) & 0xFFFF));
  }

  template <typename T>
  __device__ inline void
  event_to_rows(const ShaCompressFfiEvent& event, ShaCompressCols<T>* rows, size_t& row_count, field_t* byte_trace)
  {
    size_t current_row = 0;
    const T f_chunk = T::from_canonical_u32(event.chunk);
    const T f_clk = T::from_canonical_u32(static_cast<uint32_t>(event.clk));
    const T f_one = T::one();
    auto split_addr = [&](AddrWord<T>& aw, uint64_t addr) {
        aw._0[0] = T::from_canonical_u16(static_cast<uint16_t>(addr & 0xFFFF));
        aw._0[1] = T::from_canonical_u16(static_cast<uint16_t>((addr >> 16) & 0xFFFF));
        aw._0[2] = T::from_canonical_u16(static_cast<uint16_t>((addr >> 32) & 0xFFFF));
    };
    auto set_w32 = [&](Word32<T>& w, uint32_t v) {
        w._0[0] = T::from_canonical_u16(static_cast<uint16_t>(v & 0xFFFF));
        w._0[1] = T::from_canonical_u16(static_cast<uint16_t>((v >> 16) & 0xFFFF));
    };

    // Phase 1: Initialize (8 rows)
    for (size_t j = 0; j < 8; j++) {
      ShaCompressCols<T>& cols = rows[current_row];
      cols.chunk = f_chunk; cols.clk = f_clk;
      split_addr(cols.w_ptr, event.w_ptr); split_addr(cols.h_ptr, event.h_ptr);
      cols.octet[j] = f_one; cols.octet_num[0] = f_one; cols.is_initialize = f_one;
      cols.index = T::from_canonical_u32(static_cast<uint32_t>(j));

      // Memory: read h[j]
      auto& rec = event.h_read_records[j];
      uint64_t init_addr = cols.mem_addr_init.populate(byte_trace, event.h_ptr, static_cast<uint64_t>(j) * 8);
      write_word_from_u64(cols.mem.prev_value, rec.value);
      populate_access_u64(cols.mem.access, rec.chunk, rec.timestamp, rec.value, rec.prev_chunk, rec.prev_timestamp, rec.value, byte_trace);
      set_mem_value(cols.mem_value, static_cast<uint32_t>(rec.value));
      split_addr(cols.mem_addr, init_addr);

      // Set a-h from h_read_records values (matching Rust)
      set_w32(cols.a, static_cast<uint32_t>(event.h_read_records[0].value));
      set_w32(cols.b, static_cast<uint32_t>(event.h_read_records[1].value));
      set_w32(cols.c, static_cast<uint32_t>(event.h_read_records[2].value));
      set_w32(cols.d, static_cast<uint32_t>(event.h_read_records[3].value));
      set_w32(cols.e, static_cast<uint32_t>(event.h_read_records[4].value));
      set_w32(cols.f, static_cast<uint32_t>(event.h_read_records[5].value));
      set_w32(cols.g, static_cast<uint32_t>(event.h_read_records[6].value));
      set_w32(cols.h, static_cast<uint32_t>(event.h_read_records[7].value));

      cols.is_real = f_one; current_row++;
    }

    // Phase 2: Compression (64 rows)
    uint32_t h_a[8] = {event.h[0], event.h[1], event.h[2], event.h[3], event.h[4], event.h[5], event.h[6], event.h[7]};
    uint32_t o_num_idx = 0;
    for (size_t j = 0; j < 64; j++) {
      if (j % 8 == 0) o_num_idx++;
      ShaCompressCols<T>& cols = rows[current_row];
      cols.chunk = f_chunk; cols.clk = f_clk;
      split_addr(cols.w_ptr, event.w_ptr); split_addr(cols.h_ptr, event.h_ptr);
      set_w32(cols.k, SHA_COMPRESS_K[j]);
      cols.octet[j % 8] = f_one; cols.octet_num[o_num_idx] = f_one; cols.is_compression = f_one;
      cols.index = T::from_canonical_u32(static_cast<uint32_t>(j + 8));

      // Memory: read w[j]
      auto& rec = event.w_i_read_records[j];
      uint64_t comp_addr = cols.mem_addr_compress.populate(byte_trace, event.w_ptr, static_cast<uint64_t>(j) * 8);
      write_word_from_u64(cols.mem.prev_value, rec.value);
      populate_access_u64(cols.mem.access, rec.chunk, rec.timestamp, rec.value, rec.prev_chunk, rec.prev_timestamp, rec.value, byte_trace);
      set_mem_value(cols.mem_value, static_cast<uint32_t>(rec.value));
      split_addr(cols.mem_addr, comp_addr);

      // Set working variables
      uint32_t a = h_a[0], b = h_a[1], c = h_a[2], d = h_a[3], e = h_a[4], f = h_a[5], g = h_a[6], h = h_a[7];
      set_w32(cols.a, a); set_w32(cols.b, b); set_w32(cols.c, c); set_w32(cols.d, d);
      set_w32(cols.e, e); set_w32(cols.f, f); set_w32(cols.g, g); set_w32(cols.h, h);

      // Sigma1
      uint32_t e_rr6 = cols.e_rr_6.populate(e, 6, byte_trace);
      uint32_t e_rr11 = cols.e_rr_11.populate(e, 11, byte_trace);
      uint32_t e_rr25 = cols.e_rr_25.populate(e, 25, byte_trace);
      uint32_t s1_int = cols.s1_intermediate.populate(e_rr6, e_rr11, byte_trace);
      uint32_t s1 = cols.s1.populate(s1_int, e_rr25, byte_trace);

      // Ch
      uint32_t e_and_f = cols.e_and_f.populate(e, f, byte_trace);
      uint32_t e_not = cols.e_not.populate(e);
      uint32_t e_not_and_g = cols.e_not_and_g.populate(e_not, g, byte_trace);
      uint32_t ch = cols.ch.populate(e_and_f, e_not_and_g, byte_trace);

      // Temp1
      uint32_t temp1 = cols.temp1.populate(h, s1, ch, event.w[j], SHA_COMPRESS_K[j], byte_trace);

      // Sigma0
      uint32_t a_rr2 = cols.a_rr_2.populate(a, 2, byte_trace);
      uint32_t a_rr13 = cols.a_rr_13.populate(a, 13, byte_trace);
      uint32_t a_rr22 = cols.a_rr_22.populate(a, 22, byte_trace);
      uint32_t s0_int = cols.s0_intermediate.populate(a_rr2, a_rr13, byte_trace);
      uint32_t s0 = cols.s0.populate(s0_int, a_rr22, byte_trace);

      // Maj
      uint32_t a_and_b = cols.a_and_b.populate(a, b, byte_trace);
      uint32_t a_and_c = cols.a_and_c.populate(a, c, byte_trace);
      uint32_t b_and_c = cols.b_and_c.populate(b, c, byte_trace);
      uint32_t maj_int = cols.maj_intermediate.populate(a_and_b, a_and_c, byte_trace);
      uint32_t maj = cols.maj.populate(maj_int, b_and_c, byte_trace);

      // Temp2 and new working vars
      uint32_t temp2 = cols.temp2.populate(s0, maj, byte_trace);
      uint32_t d_add_t1 = cols.d_add_temp1.populate(d, temp1, byte_trace);
      uint32_t t1_add_t2 = cols.temp1_add_temp2.populate(temp1, temp2, byte_trace);

      h_a[7]=g; h_a[6]=f; h_a[5]=e; h_a[4]=d_add_t1; h_a[3]=c; h_a[2]=b; h_a[1]=a; h_a[0]=t1_add_t2;
      cols.is_real = f_one; current_row++;
    }

    // Phase 3: Finalize (8 rows)
    uint32_t og_h[8] = {event.h[0], event.h[1], event.h[2], event.h[3], event.h[4], event.h[5], event.h[6], event.h[7]};
    o_num_idx++;
    // v tracks the finalized state, matching Rust's v[j] = h_array[j] then v[j] = finalized
    uint32_t v[8];
    for (int k = 0; k < 8; k++) v[k] = h_a[k];

    for (size_t j = 0; j < 8; j++) {
      ShaCompressCols<T>& cols = rows[current_row];
      cols.chunk = f_chunk; cols.clk = f_clk;
      split_addr(cols.w_ptr, event.w_ptr); split_addr(cols.h_ptr, event.h_ptr);
      cols.octet[j] = f_one; cols.octet_num[o_num_idx] = f_one; cols.is_finalize = f_one;
      cols.index = T::from_canonical_u32(static_cast<uint32_t>(j + 72));

      // Memory: write h[j]
      auto& rec = event.h_write_records[j];
      uint64_t fin_addr = cols.mem_addr_finalize.populate(byte_trace, event.h_ptr, static_cast<uint64_t>(j) * 8);
      write_word_from_u64(cols.mem.prev_value, rec.prev_value);
      populate_access_u64(cols.mem.access, rec.chunk, rec.timestamp, rec.value, rec.prev_chunk, rec.prev_timestamp, rec.prev_value, byte_trace);
      set_mem_value(cols.mem_value, static_cast<uint32_t>(rec.value));
      split_addr(cols.mem_addr, fin_addr);

      // finalize_add: og_h[j] + h_a[j]
      cols.finalize_add.populate(og_h[j], h_a[j], byte_trace);

      // Rust: v[j] = h_array[j]; then sets a-h from v
      // But actually Rust does: v[j] = h_array[j] first, THEN sets cols.a = v[0], etc.
      // Wait, let me re-read Rust more carefully...
      // In Rust trace.rs:
      //   v[j] = h_array[j];  (but h_array IS v initially, so this is a no-op for j==0)
      //   cols.a = u32_to_half_word(v[0]); ... etc
      // Actually v is initialized from h_array, so v == h_a at this point.
      // Then v[j] = h_array[j] — this doesn't change anything since v was copy of h_a.
      // Wait no, re-reading: v is declared as v = h_array.collect(), then for each j:
      //   v[j] = h_array[j]; <-- this IS h_a[j], same
      // Hmm, but finalized_operand uses cols.a/b/c etc which are set from v[].

      set_w32(cols.a, v[0]); set_w32(cols.b, v[1]); set_w32(cols.c, v[2]); set_w32(cols.d, v[3]);
      set_w32(cols.e, v[4]); set_w32(cols.f, v[5]); set_w32(cols.g, v[6]); set_w32(cols.h, v[7]);

      // Match Rust's finalized_operand assignment
      switch (j) {
        case 0: cols.finalized_operand = cols.a; break;
        case 1: cols.finalized_operand = cols.b; break;
        case 2: cols.finalized_operand = cols.c; break;
        case 3: cols.finalized_operand = cols.d; break;
        case 4: cols.finalized_operand = cols.e; break;
        case 5: cols.finalized_operand = cols.f; break;
        case 6: cols.finalized_operand = cols.g; break;
        case 7: cols.finalized_operand = cols.h; break;
      }

      cols.is_real = f_one; current_row++;
    }
    row_count = current_row;
  }

  template <typename T>
  __device__ void initialize_padded_row(ShaCompressCols<T>& cols, const size_t pad_index)
  {
    const int octet = pad_index % 8;
    const int octet_num = (pad_index / 8) % 10;
    const T f_one = T::one();
    cols.octet[octet] = f_one;
    cols.octet_num[octet_num] = f_one;
    cols.index = T::from_canonical_u32(static_cast<uint32_t>(8 * octet_num + octet));
    if (octet_num != 0 && octet_num != 9) {
      const int k_idx = (octet_num - 1) * 8 + octet;
      if (k_idx < 64) {
          uint32_t v = SHA_COMPRESS_K[k_idx];
          cols.k._0[0] = T::from_canonical_u16(static_cast<uint16_t>(v & 0xFFFF));
          cols.k._0[1] = T::from_canonical_u16(static_cast<uint16_t>((v >> 16) & 0xFFFF));
      }
    }
  }

  template <typename T>
  __global__ void extra_events_to_rows_kernel(const ShaCompressFfiEvent* __restrict__ events, T* __restrict__ trace_matrix, field_t* byte_trace, const size_t events_count, const size_t num_cols)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= events_count) return;
    ShaCompressCols<T>* cols = reinterpret_cast<ShaCompressCols<T>*>(&trace_matrix[idx * num_cols * 80]);
    size_t row_count;
    event_to_rows(events[idx], cols, row_count, byte_trace);
  }

  template <typename T>
  __global__ void pad_dummy_rows_kernel(T* __restrict__ trace_matrix, const size_t pad_count, const size_t num_cols, const size_t events_count)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pad_count) return;
    const size_t row_idx = events_count * 80 + idx;
    ShaCompressCols<T>* cols = reinterpret_cast<ShaCompressCols<T>*>(&trace_matrix[row_idx * num_cols]);
    memset(cols, 0, sizeof(ShaCompressCols<T>));
    initialize_padded_row(*cols, idx);
  }

  template <class F>
  inline RustError generate_extra_record_and_main(const ShaCompressFfiEvent* events, const size_t events_count, const size_t event_size, F* trace, const size_t total_rows, const size_t num_cols, F* byte_trace, cudaStream_t stream)
  {
    try {
      if (events_count == 0) return RustError{cudaSuccess};
      CUDA_OK(cudaMemsetAsync(trace, 0, (total_rows * num_cols) * sizeof(F), stream));
      const int block_size = 256;
      if (total_rows > events_count * 80) {
        const size_t pad_count = total_rows - (events_count * 80);
        const int pad_grid = (pad_count + block_size - 1) / block_size;
        pad_dummy_rows_kernel<<<pad_grid, block_size, 0, stream>>>(trace, pad_count, num_cols, events_count);
        CUDA_OK(cudaGetLastError());
      }
      if (byte_trace != nullptr) {
        const int grid_size = (events_count + block_size - 1) / block_size;
        extra_events_to_rows_kernel<<<grid_size, block_size, 0, stream>>>(events, trace, byte_trace, events_count, num_cols);
        CUDA_OK(cudaGetLastError());
      }
    } catch (const cuda_error& e) { return RustError{e.code()}; }
    return RustError{cudaSuccess};
  }

  // ==========================================
  // ShaCompressControlChip GPU Implementation
  // ==========================================

  template <typename T>
  struct ShaCompressControlCols {
    // Leads the ShaCompress state bus so the state machine is pinned to one chunk. clk restarts
    // at zero in every chunk, so without this the bus cannot tell two chunks apart.
    T chunk;
    T clk;
    pico_gpu::SyscallAddrGadget<T> w_ptr;
    pico_gpu::SyscallAddrGadget<T> h_ptr;
    AddrAddGadget<T> w_slice_end;
    AddrAddGadget<T> h_slice_end;
    T is_real;
    Word32<T> initial_state[8];
    Word32<T> final_state[8];
  };

  template <typename T>
  __device__ inline void
  event_to_row_control(const ShaCompressFfiEvent& event, ShaCompressControlCols<T>* cols, field_t* byte_trace)
  {
      cols->chunk = T::from_canonical_u32(event.chunk);
      cols->clk = T::from_canonical_u32(static_cast<uint32_t>(event.clk));
      cols->w_ptr.populate(byte_trace, event.w_ptr, 512);
      cols->h_ptr.populate(byte_trace, event.h_ptr, 64);
      cols->w_slice_end.populate(byte_trace, event.w_ptr, 63 * 8);
      cols->h_slice_end.populate(byte_trace, event.h_ptr, 7 * 8);
      cols->is_real = T::one();
      for (int i = 0; i < 8; i++) {
          uint32_t prev_value = event.h[i];
          uint32_t value = static_cast<uint32_t>(event.h_write_records[i].value);
          uint32_t diff = value - prev_value;

          cols->initial_state[i]._0[0] = T::from_canonical_u16(static_cast<uint16_t>(prev_value & 0xFFFF));
          cols->initial_state[i]._0[1] = T::from_canonical_u16(static_cast<uint16_t>((prev_value >> 16) & 0xFFFF));

          cols->final_state[i]._0[0] = T::from_canonical_u16(static_cast<uint16_t>(diff & 0xFFFF));
          cols->final_state[i]._0[1] = T::from_canonical_u16(static_cast<uint16_t>((diff >> 16) & 0xFFFF));
      }
  }

  template <typename T>
  __global__ void control_events_to_rows_kernel(
    const ShaCompressFfiEvent* __restrict__ events,
    T* __restrict__ trace_matrix,
    field_t* byte_trace,
    const size_t events_count,
    const size_t num_cols)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= events_count) return;

    ShaCompressControlCols<T>* cols = reinterpret_cast<ShaCompressControlCols<T>*>(&trace_matrix[idx * num_cols]);
    event_to_row_control(events[idx], cols, byte_trace);
  }

  template <typename T>
  __global__ void control_pad_dummy_rows_kernel(
    T* __restrict__ trace_matrix, const size_t pad_count, const size_t num_cols, const size_t events_count)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pad_count) return;

    const size_t row_idx = events_count + idx;
    ShaCompressControlCols<T>* cols = reinterpret_cast<ShaCompressControlCols<T>*>(&trace_matrix[row_idx * num_cols]);

    memset(cols, 0, sizeof(ShaCompressControlCols<T>));
  }

  template <class F>
  inline RustError generate_control_compress_extra_record_and_main(
    const ShaCompressFfiEvent* events,
    const size_t events_count,
    F* trace,
    const size_t total_rows,
    const size_t num_cols,
    F* byte_trace,
    cudaStream_t stream)
  {
    try {
      if (events_count == 0) return RustError{cudaSuccess};

      CUDA_OK(cudaMemsetAsync(trace, 0, (total_rows * num_cols) * sizeof(F), stream));

      const int block_size = 256;
      const int grid_size = (events_count + block_size - 1) / block_size;

      if (total_rows > events_count) {
        const size_t pad_count = total_rows - events_count;
        const int pad_size = (pad_count + block_size - 1) / block_size;
        control_pad_dummy_rows_kernel<<<pad_size, block_size, 0, stream>>>(trace, pad_count, num_cols, events_count);
        CUDA_OK(cudaGetLastError());
      }

      if (byte_trace != nullptr) {
        control_events_to_rows_kernel<<<grid_size, block_size, 0, stream>>>(events, trace, byte_trace, events_count, num_cols);
        CUDA_OK(cudaGetLastError());
      }
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  }

} // namespace pico_gpu::precompile_sha_compress