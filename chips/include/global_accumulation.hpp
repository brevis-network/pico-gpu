#pragma once

#include <ff/ff_config.hpp>
#include <util/rusterror.h>

#include "prelude.hpp"
#include "types.hpp"
namespace pico_gpu::global_accumulation {
  __PICO_HOSTDEV__ inline void populate_real(
    GlobalAccumulationOperation<field_t, 1>& col,
    const septic_curve_t* sums, // array of curve points
    const septic_curve_t& final_digest,
    const septic_extension_t& final_sum_checker)
  {
    int len = 2;
    for (int i = 0; i < 7; i++) {
      col.initial_digest[0]._0[i] = sums[0].x.value[i];
      col.initial_digest[1]._0[i] = sums[0].y.value[i];
    }

    for (int i = 0; i < 1; i++) {
      if (len >= i + 2) {
        // Case: valid point
        for (int j = 0; j < 7; j++) {
          col.sum_checker[i]._0[j] = field_t::zero(); // explicitly zero
          col.cumulative_sum[i][0]._0[j] = sums[i + 1].x.value[j];
          col.cumulative_sum[i][1]._0[j] = sums[i + 1].y.value[j];
        }
      } else {
        for (int j = 0; j < 7; j++) {
          col.sum_checker[i]._0[j] = final_sum_checker.value[j];
          col.cumulative_sum[i][0]._0[j] = final_digest.x.value[j];
          col.cumulative_sum[i][1]._0[j] = final_digest.y.value[j];
        }
      }
    }
  }

  __PICO_HOSTDEV__ inline void populate_dummy(
    GlobalAccumulationOperation<field_t, 1>& col,
    const septic_curve_t& final_digest,
    const septic_extension_t& final_sum_checker)
  {
    // Set initial_digest from final_digest
    for (int j = 0; j < 7; j++) {
      col.initial_digest[0]._0[j] = final_digest.x.value[j];
      col.initial_digest[1]._0[j] = final_digest.y.value[j];
    }

    // Set cumulative_sum and sum_checker
    for (int i = 0; i < 1; i++) {
      for (int j = 0; j < 7; j++) {
        col.sum_checker[i]._0[j] = final_sum_checker.value[j];
        col.cumulative_sum[i][0]._0[j] = final_digest.x.value[j];
        col.cumulative_sum[i][1]._0[j] = final_digest.y.value[j];
      }
    }
  }
} // namespace pico_gpu::global_accumulation
