// Copyright (c) 2025-2026 Brevis. All Rights Reserved.
// Licensed under the Business Source License 1.1 (BUSL-1.1).
// See LICENSE.md for details.

#pragma once

#include <cassert>
#include "field/kb31_t.hpp"
#include "common/rusterror.h"

#include "tracegen/riscv/byte.cuh"
#include "tracegen/common/prelude.cuh"
#include "tracegen/common/types.cuh"
#include "tracegen/common/utils.cuh"

namespace pico_gpu::field_word_range_checker {
  /// (p - 1) >> 16 for the KoalaBear modulus 2^31 - 2^24 + 1, the largest value the most
  /// significant u16 limb of a canonical field element may take. This build only carries
  /// KoalaBear (see ff/), so the value is fixed rather than selected per field like the host's
  /// top_limb().
  constexpr uint16_t TOP_LIMB = 0x7F00;

  /// Mirrors FieldWordRangeChecker::populate on the host
  /// (vm/src/chips/gadgets/field_range_check/word_range.rs).
  ///
  /// The gadget used to follow the RV32 shape -- decompose bits 24..31 of a u32 and compare the
  /// recomposition against value[3]. Word is 4 x u16 here, so value[3] is bits 48..63 and that
  /// comparison meant nothing. The host now decides value < p from the top u16 limb alone.
  template <class F>
  __PICO_HOSTDEV__ inline void populate(
      FieldWordRangeChecker<F>& gadget, F* byte_trace, const uint64_t value)
  {
    const uint16_t ms_limb = static_cast<uint16_t>((value >> 16) & 0xFFFF);

    const bool lt = ms_limb < TOP_LIMB;
    gadget.ms_limb_lt_top = F::from_bool(lt);

    // ms_limb - TOP_LIMB + lt * 2^16 as a u16; the wrapping subtraction gives exactly that in
    // both branches, and the AIR range checks the same expression.
    byte::add_u16_range_check<F>(byte_trace, static_cast<uint16_t>(ms_limb - TOP_LIMB));

    // Only Mersenne31 needs a witness here, and this build has no Mersenne31. The host pins the
    // column to zero on the other two fields, so match that.
    gadget.low_limb_not_max_inv = F::zero();
  }
} // namespace pico_gpu::field_word_range_checker
