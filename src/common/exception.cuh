// Modified by Brevis
// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

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
