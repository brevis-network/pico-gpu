#pragma once

#include <cassert>
#include <ff/bb31_t.hpp>
#include <ff/kb31_t.hpp>
#include <util/rusterror.h>

#include "is_zero.hpp"
#include "prelude.hpp"
#include "types.hpp"
#include "utils.hpp"

namespace pico_gpu::field_word_range_checker {
  template <class F>
  __PICO_HOSTDEV__ inline void populate(FieldWordRangeChecker<F>& gadget, const uint32_t value)
  {
    // Decompose bits 24..31
    for (int i = 0; i < 8; ++i) {
      bool bit = (value & (1u << (i + 24))) != 0;
      gadget.most_sig_byte_decomp[i] = F::from_bool(bit);
    }
    // For babybear field, one_start_idx = 3. For koalabear field, one_start_idx = 0.
    static_assert(std::is_same<F, bb31_t>::value || std::is_same<F, kb31_t>::value, "F must be bb31_t or kb31_t");
    uint32_t one_start_idx = std::is_same<F, bb31_t>::value ? 3u : 0u;
    F sum = F::zero();
    for (size_t i = one_start_idx; i < 8; ++i) {
      sum = sum + gadget.most_sig_byte_decomp[i];
    }
    F tmp = F::from_canonical_u32(7 - one_start_idx);
    is_zero::populate_from_field_element(gadget.upper_all_one, sum - tmp);
  }
} // namespace pico_gpu::field_word_range_checker
