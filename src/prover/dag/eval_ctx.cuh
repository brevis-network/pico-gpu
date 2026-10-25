// Copyright (c) 2025-2026 Brevis. All Rights Reserved.
// Licensed under the Business Source License 1.1 (BUSL-1.1).
// See LICENSE.md for details.

#pragma once

#include "field/fp.h"
#include "field/fpext.h"
#include "prover/matrix.cuh"

#include <cstdint>

namespace logup_zerocheck_mle {

// Device pointers every constraint-evaluation kernel needs to read leaf values.
struct EvalCoreCtx {
    const FpExt *__restrict__ d_selectors;
    const MainMatrixPtrs<FpExt> d_preprocessed;
    const MainMatrixPtrs<FpExt> *__restrict__ d_main;
    const Fp *__restrict__ d_public;
};

// Which AIR a block works on, and the block's index within that AIR's slice of the grid.
struct BlockCtx {
    uint32_t local_block_idx_x;
    uint32_t air_idx;
};

} // namespace logup_zerocheck_mle
