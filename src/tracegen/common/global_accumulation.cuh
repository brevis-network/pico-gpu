// Copyright (c) 2025-2026 Brevis. All Rights Reserved.
// Licensed under the Business Source License 1.1 (BUSL-1.1).
// See LICENSE.md for details.

#pragma once

#include "field/ff_config.hpp"
#include "common/rusterror.h"

#include "tracegen/common/prelude.cuh"
#include "tracegen/common/types.cuh"
namespace pico_gpu::global_accumulation {
  /// Slope of the line through `cum` and `point`: the chord slope when the x-coordinates differ,
  /// the tangent slope (doubling) when point == cum. Mirrors Rust
  /// `GlobalAccumulationOperation::slope` branch for branch. `reciprocal()` has no zero guard, so
  /// the x-equality test MUST come first. point == -cum never occurs in an honest trace (the prefix
  /// scan would have produced the point at infinity) and is rejected by the constraints.
  __PICO_HOSTDEV__ inline septic_extension_t slope(const septic_curve_t& cum, const septic_curve_t& point)
  {
    septic_extension_t x_diff = point.x - cum.x;
    if (x_diff == septic_extension_t::zero()) {
      // Tangent: (3 x^2 + 2) / (2 y), the curve y^2 = x^3 + 2x + 26 z^5.
      septic_extension_t x2 = cum.x * cum.x;
      return (x2 + x2 + x2 + field_t::two()) / (cum.y + cum.y);
    }
    return (point.y - cum.y) / x_diff;
  }

  /// Real row: cum = the running sum before this row's point, lambda = slope(cum, point).
  __PICO_HOSTDEV__ inline void populate_real(
    GlobalAccumulationOperation<field_t, 1>& col, const septic_curve_t& cum, const septic_curve_t& point)
  {
    const septic_extension_t lambda = slope(cum, point);
    for (int j = 0; j < 7; j++) {
      col.lambda[0]._0[j] = lambda.value[j];
      col.cum[0][0]._0[j] = cum.x.value[j];
      col.cum[0][1]._0[j] = cum.y.value[j];
    }
  }

  /// Padding row: cum = final_digest (the chunk total, held constant), lambda = 0.
  __PICO_HOSTDEV__ inline void
  populate_dummy(GlobalAccumulationOperation<field_t, 1>& col, const septic_curve_t& final_digest)
  {
    for (int j = 0; j < 7; j++) {
      col.lambda[0]._0[j] = field_t::zero();
      col.cum[0][0]._0[j] = final_digest.x.value[j];
      col.cum[0][1]._0[j] = final_digest.y.value[j];
    }
  }
} // namespace pico_gpu::global_accumulation
