#pragma once

#include <cassert>
#include <util/rusterror.h>

#include "prelude.hpp"
#include "types.hpp"
#include "utils.hpp"

namespace pico_gpu::is_zero {

  template <class F>
  __PICO_HOSTDEV__ inline bool populate_from_field_element(IsZeroOperation<F>& gadget, F a)
  {
    if (a.is_zero()) {
      gadget.inverse = F::zero();
      gadget.result = F::one();
    } else {
      gadget.inverse = a.reciprocal();
      gadget.result = F::zero();
    }
    F prod = gadget.inverse * a;
    assert(prod == F::one() || prod == F::zero());
    return a.is_zero();
  } // populate_from_field_element

  template <class F>
  __PICO_HOSTDEV__ inline bool populate(IsZeroOperation<F>& gadget, const uint32_t a_u32)
  {
    F a = F::from_canonical_u32(a_u32);
    return populate_from_field_element(gadget, a);
  } // populate

} // namespace pico_gpu::is_zero
