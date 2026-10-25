// Copyright (c) 2025-2026 Brevis. All Rights Reserved.
// Licensed under the Business Source License 1.1 (BUSL-1.1).
// See LICENSE.md for details.

#ifndef UTIL_EXCEPTION_CUH
#define UTIL_EXCEPTION_CUH

#include "exception.hpp"

#define CUDA_OK(expr)                                                                                                  \
  do {                                                                                                                 \
    cudaError_t code = expr;                                                                                           \
    if (code != cudaSuccess) {                                                                                         \
                                                                           \
      auto str = fmt("%s@%s:%d failed: \"%s\"", #expr, __FILE__, __LINE__, cudaGetErrorString(code));    \
      throw cuda_error{-code, str};                                                                                    \
    }                                                                                                                  \
  } while (0)

#endif
