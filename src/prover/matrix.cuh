// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0

#pragma once

#include <cstdint>

template <typename T> struct MainMatrixPtrs {
    const T *__restrict__ data;
    uint32_t air_width;
};
