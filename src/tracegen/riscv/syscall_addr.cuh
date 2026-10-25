// Copyright (c) 2025-2026 Brevis. All Rights Reserved.
// Licensed under the Business Source License 1.1 (BUSL-1.1).
// See LICENSE.md for details.

#pragma once

#include "tracegen/common/types.cuh"
#include "tracegen/common/is_zero.cuh"
#include "tracegen/riscv/byte.cuh"
#include "field/ff_config.hpp"

// Define member functions for SyscallAddrGadget and AddrAddGadget.
// These are defined here (not in types.cuh) to avoid circular includes with byte.cuh.

namespace pico_gpu {

  template <typename T>
  __device__ inline void SyscallAddrGadget<T>::populate(T* byte_trace, uint64_t addr_val, uint64_t len) {
      uint16_t addr_limbs[3] = {
          static_cast<uint16_t>(addr_val & 0xFFFF),
          static_cast<uint16_t>((addr_val >> 16) & 0xFFFF),
          static_cast<uint16_t>((addr_val >> 32) & 0xFFFF)
      };
      T sum_top_two_limb = T::from_canonical_u16(addr_limbs[1]) + T::from_canonical_u16(addr_limbs[2]);
      addr[0] = T::from_canonical_u16(addr_limbs[0]);
      addr[1] = T::from_canonical_u16(addr_limbs[1]);
      addr[2] = T::from_canonical_u16(addr_limbs[2]);

      top_two_limb_min = sum_top_two_limb.is_zero() ? T::zero() : sum_top_two_limb.reciprocal();

      T max_diff = sum_top_two_limb - T::from_canonical_u16(0xFFFF) * T::from_canonical_u8(2);
      bool is_max = pico_gpu::is_zero::populate_from_field_element(top_two_limb_max, max_diff);

      if (byte_trace) {
        if (is_max) {
            // (addr[0] + len) / 8, not addr[0] + len / 8. The AIR looks up
            // (addr[0] + is_max * len) / 8, so the two only agree with the sum inside the
            // division; addr[0] is a multiple of 8, which made the wrong form differ by 8x.
            uint16_t val = static_cast<uint16_t>(addr_limbs[0] + static_cast<uint16_t>(len)) / 8;
            pico_gpu::byte::handle_bit_range_check_event(byte_trace, (uint8_t)(val >> 8), (uint8_t)(val & 0xFF), 13);
        } else {
            uint16_t val = addr_limbs[0] / 8;
            pico_gpu::byte::handle_bit_range_check_event(byte_trace, (uint8_t)(val >> 8), (uint8_t)(val & 0xFF), 13);
        }

        // Pair the slice_range_check_u16 the AIR now emits over the three address limbs. Without
        // these the byte table is under-supplied on every syscall row. The sibling AddrAddGadget
        // below has always emitted them.
        pico_gpu::byte::add_u16_range_check(byte_trace, addr_limbs[0]);
        pico_gpu::byte::add_u16_range_check(byte_trace, addr_limbs[1]);
        pico_gpu::byte::add_u16_range_check(byte_trace, addr_limbs[2]);
      }
  }

  template <typename T>
  __device__ inline uint64_t AddrAddGadget<T>::populate(T* byte_trace, uint64_t base, uint64_t offset) {
      uint64_t expected = base + offset;
      uint16_t limbs[3] = {
          static_cast<uint16_t>(expected & 0xFFFF),
          static_cast<uint16_t>((expected >> 16) & 0xFFFF),
          static_cast<uint16_t>((expected >> 32) & 0xFFFF)
      };
      value[0] = T::from_canonical_u16(limbs[0]);
      value[1] = T::from_canonical_u16(limbs[1]);
      value[2] = T::from_canonical_u16(limbs[2]);
      if (byte_trace) {
        pico_gpu::byte::add_u16_range_check(byte_trace, limbs[0]);
        pico_gpu::byte::add_u16_range_check(byte_trace, limbs[1]);
        pico_gpu::byte::add_u16_range_check(byte_trace, limbs[2]);
      }
      return expected;
  }

} // namespace pico_gpu
