#pragma once

#include <cassert>
#include <ff/bb31_t.hpp>
#include <ff/kb31_t.hpp>
#include <util/rusterror.h>

#include "is_zero.hpp"
#include "prelude.hpp"
#include "types.hpp"
#include "utils.hpp"

namespace pico_gpu::field_bit_decomposition {

  template <class F>
  __PICO_HOSTDEV__ inline void populate(FieldBitDecomposition<F>& gadget, const uint32_t value)
  {
    for (int i = 0; i < 32; ++i) {
      gadget.bits[i] = F::from_canonical_u32((value >> i) & 1);
    }
    // For babybear field, one_start_idx = 3. For koalabear field, one_start_idx = 0.
    static_assert(std::is_same<F, bb31_t>::value || std::is_same<F, kb31_t>::value, "F must be bb31_t or kb31_t");
    uint32_t one_start_idx = std::is_same<F, bb31_t>::value ? 3u : 0u;

    F sum = F::zero();
    for (int i = 24 + one_start_idx; i < 32; ++i) {
      sum = sum + gadget.bits[i];
    }

    F adjustment = F::from_canonical_u32(7 - one_start_idx);
    F final_value = sum - adjustment;

    is_zero::populate_from_field_element(gadget.upper_all_one, final_value);
  }

} // namespace pico_gpu::field_bit_decomposition
