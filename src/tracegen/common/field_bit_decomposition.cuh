// Copyright (c) 2025-2026 Brevis. All Rights Reserved.
// Licensed under the Business Source License 1.1 (BUSL-1.1).
// See LICENSE.md for details.

#pragma once

#include <cassert>
#include "field/kb31_t.hpp"
#include "common/rusterror.h"

#include "tracegen/common/is_zero.cuh"
#include "tracegen/common/prelude.cuh"
#include "tracegen/common/types.cuh"
#include "tracegen/common/utils.cuh"

namespace pico_gpu::field_bit_decomposition {

  template <class F>
  __PICO_HOSTDEV__ inline void populate(FieldBitDecomposition<F>& gadget, const uint32_t value)
  {
    for (int i = 0; i < 32; ++i) {
      gadget.bits[i] = F::from_canonical_u32((value >> i) & 1);
    }
    
    uint32_t one_start_idx = 0u;

    F sum = F::zero();
    for (int i = 24 + one_start_idx; i < 32; ++i) {
      sum = sum + gadget.bits[i];
    }

    F adjustment = F::from_canonical_u32(7 - one_start_idx);
    F final_value = sum - adjustment;

    is_zero::populate_from_field_element(gadget.upper_all_one, final_value);
  }

} // namespace pico_gpu::field_bit_decomposition
