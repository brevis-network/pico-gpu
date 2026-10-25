// Copyright (c) 2025-2026 Brevis. All Rights Reserved.
// Licensed under the Business Source License 1.1 (BUSL-1.1).
// See LICENSE.md for details.

#pragma once

#include <cstdint>

template <typename T> struct MainMatrixPtrs {
    const T *__restrict__ data;
    uint32_t air_width;
};
