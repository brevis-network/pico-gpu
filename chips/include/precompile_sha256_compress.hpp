#pragma once

#include "memory_read_write.hpp"
#include "utils.hpp"
#include "types.hpp"
#include "util/exception.cuh"
#include "util/rusterror.h"
#include "byte.hpp"
#include <chrono>
#include <iostream>
#include <ostream>
#include <cstring>
#include "sha256_operations.hpp"
#include "../../ff/ff_config.hpp"

namespace pico_gpu::precompile_sha_compress {
  using namespace sha256_operations;
  using namespace memory_read_write;
  using namespace byte;

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

  /// Rotate right operation.
  __device__ inline uint32_t rotr(uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); }

  /// Addition gadget for two 32-bit values with carry tracking.
  template <typename T>
  struct AddGadget {
    Word<T> value;
    T carry[3];

    __device__ AddGadget() {}

    __device__ uint32_t populate(uint32_t a_u32, uint32_t b_u32, field_t* byte_trace)
    {
      uint32_t expected = a_u32 + b_u32;
      write_word_from_u32_v2(value, expected);

      // Extract bytes in little endian order.
      uint8_t a[4] = {
        (uint8_t)(a_u32 & 0xFF), (uint8_t)((a_u32 >> 8) & 0xFF), (uint8_t)((a_u32 >> 16) & 0xFF),
        (uint8_t)((a_u32 >> 24) & 0xFF)};
      uint8_t b[4] = {
        (uint8_t)(b_u32 & 0xFF), (uint8_t)((b_u32 >> 8) & 0xFF), (uint8_t)((b_u32 >> 16) & 0xFF),
        (uint8_t)((b_u32 >> 24) & 0xFF)};

      // Initialize carries.
      carry[0] = T::zero();
      carry[1] = T::zero();
      carry[2] = T::zero();

      uint8_t carry_vals[3] = {0, 0, 0};

      // Calculate carries byte by byte.
      if ((uint32_t)a[0] + (uint32_t)b[0] > 255) {
        carry_vals[0] = 1;
        carry[0] = T::one();
      }
      if ((uint32_t)a[1] + (uint32_t)b[1] + (uint32_t)carry_vals[0] > 255) {
        carry_vals[1] = 1;
        carry[1] = T::one();
      }
      if ((uint32_t)a[2] + (uint32_t)b[2] + (uint32_t)carry_vals[1] > 255) {
        carry_vals[2] = 1;
        carry[2] = T::one();
      }

      // Add byte range check events.
      for (int i = 0; i < 4; i += 2) {
        add_u8_range_check(byte_trace, a[i], a[i + 1]);
        add_u8_range_check(byte_trace, b[i], b[i + 1]);
      }
      array_t<uint8_t, WORD_SIZE> expected_bytes = u32_to_le_bytes(expected);
      add_u8_range_checks(byte_trace, expected_bytes);

      return expected;
    }
  };

  /// Bitwise AND operation.
  template <typename T>
  struct AndOperation {
    Word<T> value;

    __device__ AndOperation() {}

    __device__ uint32_t populate(uint32_t x, uint32_t y, field_t* byte_trace)
    {
      uint32_t expected = x & y;

      // Extract bytes in little endian order.
      uint8_t x_bytes[4] = {
        (uint8_t)(x & 0xFF), (uint8_t)((x >> 8) & 0xFF), (uint8_t)((x >> 16) & 0xFF), (uint8_t)((x >> 24) & 0xFF)};
      uint8_t y_bytes[4] = {
        (uint8_t)(y & 0xFF), (uint8_t)((y >> 8) & 0xFF), (uint8_t)((y >> 16) & 0xFF), (uint8_t)((y >> 24) & 0xFF)};

      // Populate result byte by byte.
      for (int i = 0; i < WORD_SIZE; i++) {
        uint8_t and_result = x_bytes[i] & y_bytes[i];
        value._0[i] = T::from_canonical_u8(and_result);
        handle_byte_lookup_event(byte_trace, ByteOpcode::AND, x_bytes[i], y_bytes[i]);
      }

      return expected;
    }
  };

  /// Bitwise NOT operation.
  template <typename T>
  struct NotOperation {
    Word<T> value;

    __device__ NotOperation() {}

    __device__ uint32_t populate(uint32_t x, field_t* byte_trace)
    {
      uint32_t expected = ~x;

      // Extract bytes in little endian order.
      uint8_t x_bytes[4] = {
        (uint8_t)(x & 0xFF), (uint8_t)((x >> 8) & 0xFF), (uint8_t)((x >> 16) & 0xFF), (uint8_t)((x >> 24) & 0xFF)};

      // Populate result byte by byte.
      for (int i = 0; i < WORD_SIZE; i++) {
        value._0[i] = T::from_canonical_u8(~x_bytes[i]);
      }
      for (int i = 0; i < 4; i += 2) {
        add_u8_range_check(byte_trace, x_bytes[i], x_bytes[i + 1]);
      }

      return expected;
    }
  };

  /// Five-way addition operation with carry tracking.
  template <typename T>
  struct Add5Operation {
    Word<T> value;
    Word<T> is_carry_0;
    Word<T> is_carry_1;
    Word<T> is_carry_2;
    Word<T> is_carry_3;
    Word<T> is_carry_4;
    Word<T> carry;

    __device__ Add5Operation() {}

    // Version with byte lookup for extra record.
    __device__ uint32_t
    populate(uint32_t a_u32, uint32_t b_u32, uint32_t c_u32, uint32_t d_u32, uint32_t e_u32, field_t* byte_trace)
    {
      uint32_t expected = ((((a_u32 + b_u32) + c_u32) + d_u32) + e_u32);
      write_word_from_u32_v2(value, expected);

      // Extract bytes.
      uint8_t a[4], b[4], c[4], d[4], e[4];
      for (int i = 0; i < 4; i++) {
        a[i] = (a_u32 >> (i * 8)) & 0xFF;
        b[i] = (b_u32 >> (i * 8)) & 0xFF;
        c[i] = (c_u32 >> (i * 8)) & 0xFF;
        d[i] = (d_u32 >> (i * 8)) & 0xFF;
        e[i] = (e_u32 >> (i * 8)) & 0xFF;
      }

      const uint32_t base = 256;
      uint8_t carry_vals[5] = {0, 0, 0, 0, 0};

      // Calculate carries and set indication fields.
      for (int i = 0; i < WORD_SIZE; i++) {
        uint32_t res = (uint32_t)a[i] + (uint32_t)b[i] + (uint32_t)c[i] + (uint32_t)d[i] + (uint32_t)e[i];
        if (i > 0) res += (uint32_t)carry_vals[i - 1];
        carry_vals[i] = (uint8_t)(res / base);

        is_carry_0._0[i] = (carry_vals[i] == 0) ? T::one() : T::zero();
        is_carry_1._0[i] = (carry_vals[i] == 1) ? T::one() : T::zero();
        is_carry_2._0[i] = (carry_vals[i] == 2) ? T::one() : T::zero();
        is_carry_3._0[i] = (carry_vals[i] == 3) ? T::one() : T::zero();
        is_carry_4._0[i] = (carry_vals[i] == 4) ? T::one() : T::zero();

        carry._0[i] = T::from_canonical_u8(carry_vals[i]);
      }

      // Add byte range check events.
      for (int i = 0; i < 4; i += 2) {
        add_u8_range_check(byte_trace, a[i], a[i + 1]);
        add_u8_range_check(byte_trace, b[i], b[i + 1]);
        add_u8_range_check(byte_trace, c[i], c[i + 1]);
        add_u8_range_check(byte_trace, d[i], d[i + 1]);
        add_u8_range_check(byte_trace, e[i], e[i + 1]);
      }
      array_t<uint8_t, WORD_SIZE> expected_bytes = u32_to_le_bytes(expected);
      add_u8_range_checks(byte_trace, expected_bytes);

      return expected;
    }

    // Version without byte lookup for main trace.
    __device__ uint32_t populate(uint32_t a_u32, uint32_t b_u32, uint32_t c_u32, uint32_t d_u32, uint32_t e_u32)
    {
      uint32_t expected = ((((a_u32 + b_u32) + c_u32) + d_u32) + e_u32);
      write_word_from_u32_v2(value, expected);

      // Extract bytes.
      uint8_t a[4], b[4], c[4], d[4], e[4];
      for (int i = 0; i < 4; i++) {
        a[i] = (a_u32 >> (i * 8)) & 0xFF;
        b[i] = (b_u32 >> (i * 8)) & 0xFF;
        c[i] = (c_u32 >> (i * 8)) & 0xFF;
        d[i] = (d_u32 >> (i * 8)) & 0xFF;
        e[i] = (e_u32 >> (i * 8)) & 0xFF;
      }

      const uint32_t base = 256;
      uint8_t carry_vals[5] = {0, 0, 0, 0, 0};

      for (int i = 0; i < WORD_SIZE; i++) {
        uint32_t res = (uint32_t)a[i] + (uint32_t)b[i] + (uint32_t)c[i] + (uint32_t)d[i] + (uint32_t)e[i];
        if (i > 0) res += (uint32_t)carry_vals[i - 1];
        carry_vals[i] = (uint8_t)(res / base);

        is_carry_0._0[i] = (carry_vals[i] == 0) ? T::one() : T::zero();
        is_carry_1._0[i] = (carry_vals[i] == 1) ? T::one() : T::zero();
        is_carry_2._0[i] = (carry_vals[i] == 2) ? T::one() : T::zero();
        is_carry_3._0[i] = (carry_vals[i] == 3) ? T::one() : T::zero();
        is_carry_4._0[i] = (carry_vals[i] == 4) ? T::one() : T::zero();

        carry._0[i] = T::from_canonical_u8(carry_vals[i]);
      }

      return expected;
    }
  };

  /// SHA256 compress columns structure.
  template <typename T>
  struct ShaCompressCols {
    // Basic inputs.
    T chunk;
    T clk;
    T w_ptr;
    T h_ptr;
    T start;

    // State indicators.
    T octet[8];
    T octet_num[10];

    // Memory access.
    MemoryReadWriteCols<T> mem;
    T mem_addr;

    // Hash state variables.
    Word<T> a, b, c, d, e, f, g, h;

    // Current K value.
    Word<T> k;

    // Sigma1 calculation gadgets.
    FixedRotateRightOperation<T> e_rr_6;
    FixedRotateRightOperation<T> e_rr_11;
    FixedRotateRightOperation<T> e_rr_25;
    XorOperation<T> s1_intermediate;
    XorOperation<T> s1;

    // Choice function gadgets.
    AndOperation<T> e_and_f;
    NotOperation<T> e_not;
    AndOperation<T> e_not_and_g;
    XorOperation<T> ch;

    // Temp1 calculation.
    Add5Operation<T> temp1;

    // Sigma0 calculation gadgets.
    FixedRotateRightOperation<T> a_rr_2;
    FixedRotateRightOperation<T> a_rr_13;
    FixedRotateRightOperation<T> a_rr_22;
    XorOperation<T> s0_intermediate;
    XorOperation<T> s0;

    // Majority function gadgets.
    AndOperation<T> a_and_b;
    AndOperation<T> a_and_c;
    AndOperation<T> b_and_c;
    XorOperation<T> maj_intermediate;
    XorOperation<T> maj;

    // Temp2 and final additions.
    AddGadget<T> temp2;
    AddGadget<T> d_add_temp1;
    AddGadget<T> temp1_add_temp2;

    // Finalization.
    Word<T> finalized_operand;
    AddGadget<T> finalize_add;

    // Phase indicators.
    T is_initialize;
    T is_compression;
    T is_finalize;
    T is_last_row;
    T is_real;

    __device__ ShaCompressCols() {}
  };

  /// Convert SHA256 compress event to trace rows.
  template <typename T>
  __device__ inline void
  event_to_rows(const ShaCompressFfiEvent& event, ShaCompressCols<T>* rows, size_t& row_count, field_t* byte_trace)
  {
    size_t current_row = 0;

    // Phase 1: Initialize (8 rows) - Load initial hash values.
    for (size_t j = 0; j < 8; j++) {
      ShaCompressCols<T>& cols = rows[current_row];

      cols.chunk = T::from_canonical_u32(event.chunk);
      cols.clk = T::from_canonical_u32(event.clk);
      cols.w_ptr = T::from_canonical_u32(event.w_ptr);
      cols.h_ptr = T::from_canonical_u32(event.h_ptr);

      cols.octet[j] = T::one();
      cols.octet_num[0] = T::one();
      cols.is_initialize = T::one();

      populate_read(cols.mem, event.h_read_records[j], byte_trace);
      cols.mem_addr = T::from_canonical_u32(event.h_ptr + (j * 4));

      write_word_from_u32_v2(cols.a, event.h_read_records[0].value);
      write_word_from_u32_v2(cols.b, event.h_read_records[1].value);
      write_word_from_u32_v2(cols.c, event.h_read_records[2].value);
      write_word_from_u32_v2(cols.d, event.h_read_records[3].value);
      write_word_from_u32_v2(cols.e, event.h_read_records[4].value);
      write_word_from_u32_v2(cols.f, event.h_read_records[5].value);
      write_word_from_u32_v2(cols.g, event.h_read_records[6].value);
      write_word_from_u32_v2(cols.h, event.h_read_records[7].value);

      cols.is_real = T::one();
      cols.start = cols.is_real * cols.octet_num[0] * cols.octet[0];

      current_row++;
    }

    // Phase 2: Compression (64 rounds).
    uint32_t h_array[8] = {event.h[0], event.h[1], event.h[2], event.h[3],
                           event.h[4], event.h[5], event.h[6], event.h[7]};
    uint32_t octet_num_idx = 0;

    for (size_t j = 0; j < 64; j++) {
      if (j % 8 == 0) octet_num_idx++;

      ShaCompressCols<T>& cols = rows[current_row];

      write_word_from_u32_v2(cols.k, SHA_COMPRESS_K[j]);
      cols.is_compression = T::one();
      cols.octet[j % 8] = T::one();
      cols.octet_num[octet_num_idx] = T::one();

      cols.chunk = T::from_canonical_u32(event.chunk);
      cols.clk = T::from_canonical_u32(event.clk);
      cols.w_ptr = T::from_canonical_u32(event.w_ptr);
      cols.h_ptr = T::from_canonical_u32(event.h_ptr);

      populate_read(cols.mem, event.w_i_read_records[j], byte_trace);
      cols.mem_addr = T::from_canonical_u32(event.w_ptr + (j * 4));

      uint32_t a = h_array[0], b = h_array[1], c = h_array[2], d = h_array[3];
      uint32_t e = h_array[4], f = h_array[5], g = h_array[6], h = h_array[7];

      write_word_from_u32_v2(cols.a, a);
      write_word_from_u32_v2(cols.b, b);
      write_word_from_u32_v2(cols.c, c);
      write_word_from_u32_v2(cols.d, d);
      write_word_from_u32_v2(cols.e, e);
      write_word_from_u32_v2(cols.f, f);
      write_word_from_u32_v2(cols.g, g);
      write_word_from_u32_v2(cols.h, h);

      // Populate all compression gadgets.
      uint32_t e_rr_6 = cols.e_rr_6.populate(e, 6, byte_trace);
      uint32_t e_rr_11 = cols.e_rr_11.populate(e, 11, byte_trace);
      uint32_t e_rr_25 = cols.e_rr_25.populate(e, 25, byte_trace);
      uint32_t s1_intermediate = cols.s1_intermediate.populate(e_rr_6, e_rr_11, byte_trace);
      uint32_t s1 = cols.s1.populate(s1_intermediate, e_rr_25, byte_trace);

      uint32_t e_and_f = cols.e_and_f.populate(e, f, byte_trace);
      uint32_t e_not = cols.e_not.populate(e, byte_trace);
      uint32_t e_not_and_g = cols.e_not_and_g.populate(e_not, g, byte_trace);
      uint32_t ch = cols.ch.populate(e_and_f, e_not_and_g, byte_trace);

      uint32_t temp1 = cols.temp1.populate(h, s1, ch, event.w[j], SHA_COMPRESS_K[j], byte_trace);

      uint32_t a_rr_2 = cols.a_rr_2.populate(a, 2, byte_trace);
      uint32_t a_rr_13 = cols.a_rr_13.populate(a, 13, byte_trace);
      uint32_t a_rr_22 = cols.a_rr_22.populate(a, 22, byte_trace);
      uint32_t s0_intermediate = cols.s0_intermediate.populate(a_rr_2, a_rr_13, byte_trace);
      uint32_t s0 = cols.s0.populate(s0_intermediate, a_rr_22, byte_trace);

      uint32_t a_and_b = cols.a_and_b.populate(a, b, byte_trace);
      uint32_t a_and_c = cols.a_and_c.populate(a, c, byte_trace);
      uint32_t b_and_c = cols.b_and_c.populate(b, c, byte_trace);
      uint32_t maj_intermediate = cols.maj_intermediate.populate(a_and_b, a_and_c, byte_trace);
      uint32_t maj = cols.maj.populate(maj_intermediate, b_and_c, byte_trace);

      uint32_t temp2 = cols.temp2.populate(s0, maj, byte_trace);
      uint32_t d_add_temp1 = cols.d_add_temp1.populate(d, temp1, byte_trace);
      uint32_t temp1_add_temp2 = cols.temp1_add_temp2.populate(temp1, temp2, byte_trace);

      // Update hash state.
      h_array[7] = g;
      h_array[6] = f;
      h_array[5] = e;
      h_array[4] = d_add_temp1;
      h_array[3] = c;
      h_array[2] = b;
      h_array[1] = a;
      h_array[0] = temp1_add_temp2;

      cols.is_real = T::one();
      cols.start = cols.is_real * cols.octet_num[0] * cols.octet[0];

      current_row++;
    }

    // Phase 3: Finalize (8 rows).
    uint32_t v[8];
    for (int i = 0; i < 8; i++)
      v[i] = h_array[i];
    octet_num_idx += 1;

    for (size_t j = 0; j < 8; j++) {
      ShaCompressCols<T>& cols = rows[current_row];

      cols.chunk = T::from_canonical_u32(event.chunk);
      cols.clk = T::from_canonical_u32(event.clk);
      cols.w_ptr = T::from_canonical_u32(event.w_ptr);
      cols.h_ptr = T::from_canonical_u32(event.h_ptr);

      cols.octet[j] = T::one();
      cols.octet_num[octet_num_idx] = T::one();
      cols.is_finalize = T::one();

      cols.finalize_add.populate(event.h[j], v[j], byte_trace);
      populate_write(cols.mem, event.h_write_records[j], byte_trace);
      cols.mem_addr = T::from_canonical_u32(event.h_ptr + (j * 4));

      write_word_from_u32_v2(cols.a, v[0]);
      write_word_from_u32_v2(cols.b, v[1]);
      write_word_from_u32_v2(cols.c, v[2]);
      write_word_from_u32_v2(cols.d, v[3]);
      write_word_from_u32_v2(cols.e, v[4]);
      write_word_from_u32_v2(cols.f, v[5]);
      write_word_from_u32_v2(cols.g, v[6]);
      write_word_from_u32_v2(cols.h, v[7]);

      switch (j) {
      case 0:
        cols.finalized_operand = cols.a;
        break;
      case 1:
        cols.finalized_operand = cols.b;
        break;
      case 2:
        cols.finalized_operand = cols.c;
        break;
      case 3:
        cols.finalized_operand = cols.d;
        break;
      case 4:
        cols.finalized_operand = cols.e;
        break;
      case 5:
        cols.finalized_operand = cols.f;
        break;
      case 6:
        cols.finalized_operand = cols.g;
        break;
      case 7:
        cols.finalized_operand = cols.h;
        break;
      }

      cols.is_real = T::one();
      cols.is_last_row = cols.octet[7] * cols.octet_num[9];
      cols.start = cols.is_real * cols.octet_num[0] * cols.octet[0];

      current_row++;
    }

    row_count = current_row;
  }

  /// Initialize a padded dummy row.
  template <typename T>
  __device__ void initialize_padded_row(ShaCompressCols<T>& cols, const size_t pad_index)
  {
    const int octet = pad_index % 8;
    const int octet_num = (pad_index / 8) % 10;

    cols.octet[octet] = T::one();
    cols.octet_num[octet_num] = T::one();

    if (octet_num != 0 && octet_num != 9) {
      const int compression_idx = octet_num - 1;
      const int k_idx = compression_idx * 8 + octet;
      if (k_idx < 64) write_word_from_u32_v2(cols.k, SHA_COMPRESS_K[k_idx]);
    }

    cols.is_last_row = cols.octet[7] * cols.octet_num[9];
  }

  /// Rotate right for SHA256.
  __device__ uint32_t ror(uint32_t x, uint32_t bits) { return (x >> bits) | (x << (32 - bits)); }

  /// SHA256 compression function.
  __device__ void compress(uint32_t* hash, const uint32_t* w)
  {
    uint32_t a, b, c, d, e, f, g, h;
    a = hash[0];
    b = hash[1];
    c = hash[2];
    d = hash[3];
    e = hash[4];
    f = hash[5];
    g = hash[6];
    h = hash[7];

    for (int i = 0; i < 64; i++) {
      uint32_t s1 = ror(e, 6) ^ ror(e, 11) ^ ror(e, 25);
      uint32_t ch = (e & f) ^ (~e & g);
      uint32_t t1 = h + s1 + ch + SHA_COMPRESS_K[i] + w[i];
      uint32_t s0 = ror(a, 2) ^ ror(a, 13) ^ ror(a, 22);
      uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      uint32_t t2 = s0 + maj;
      h = g;
      g = f;
      f = e;
      e = d + t1;
      d = c;
      c = b;
      b = a;
      a = t1 + t2;
    }

    hash[0] = a;
    hash[1] = b;
    hash[2] = c;
    hash[3] = d;
    hash[4] = e;
    hash[5] = f;
    hash[6] = g;
    hash[7] = h;
  }

  /// Kernel to convert events to trace rows with byte lookup.
  template <typename T>
  __global__ void extra_events_to_rows_kernel(
    const ShaCompressFfiEvent* __restrict__ events,
    T* __restrict__ trace_matrix,
    field_t* byte_trace,
    const size_t events_count,
    const size_t num_cols)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= events_count) return;

    ShaCompressCols<T>* cols = reinterpret_cast<ShaCompressCols<T>*>(&trace_matrix[idx * num_cols * 80]);
    size_t row_count;
    event_to_rows(events[idx], cols, row_count, byte_trace);
  }

  /// Kernel to pad trace with dummy rows.
  template <typename T>
  __global__ void pad_dummy_rows_kernel(
    T* __restrict__ trace_matrix, const size_t pad_count, const size_t num_cols, const size_t events_count)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pad_count) return;

    const size_t row_idx = events_count * 80 + idx;
    ShaCompressCols<T>* cols = reinterpret_cast<ShaCompressCols<T>*>(&trace_matrix[row_idx * num_cols]);

    memset(cols, 0, sizeof(ShaCompressCols<T>));
    initialize_padded_row(*cols, idx);
  }

  /// Generate trace from SHA256 compress events.
  template <class F>
  inline RustError generate_extra_record_and_main(
    const ShaCompressFfiEvent* events,
    const size_t events_count,
    const size_t event_size,
    F* trace,
    const size_t total_rows,
    const size_t num_cols,
    F* byte_trace,
    cudaStream_t stream)
  {
    try {
      if (events_count == 0) return RustError{cudaSuccess};

      // Initialize trace matrix with zeros.
      CUDA_OK(cudaMemsetAsync(trace, 0, (total_rows * num_cols) * sizeof(F), stream));

      const int block_size = 256;

      // Pad dummy rows if needed.
      if (total_rows > events_count * 80) {
        const size_t pad_count = total_rows - (events_count * 80);
        const int pad_size = (pad_count + block_size - 1) / block_size;

        pad_dummy_rows_kernel<<<pad_size, block_size, 0, stream>>>(trace, pad_count, num_cols, events_count);
        CUDA_OK(cudaGetLastError());
      }

      // Launch extra record kernel with byte lookup.
      if (byte_trace != nullptr) {
        const int extra_grid_size = (events_count + block_size - 1) / block_size;

        extra_events_to_rows_kernel<<<extra_grid_size, block_size, 0, stream>>>(
          events, trace, byte_trace, events_count, num_cols);
        CUDA_OK(cudaGetLastError());
      }
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  }
} // namespace pico_gpu::precompile_sha_compress