// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifndef WARP_SZ
  #define WARP_SZ 32
#endif

// A simple way to allocate a temporary device pointer without having to
// care about freeing it.
template <typename T>
class dev_ptr_t
{
  T* d_ptr;
  cudaStream_t stream;

public:
  dev_ptr_t(size_t nelems) : d_ptr(nullptr), stream(nullptr)
  {
    if (nelems) {
      size_t n = (nelems + WARP_SZ - 1) & ((size_t)0 - WARP_SZ);
      CUDA_OK(cudaMalloc(&d_ptr, n * sizeof(T)));
    }
  }
  dev_ptr_t(size_t nelems, const cudaStream_t s) : d_ptr(nullptr), stream(s)
  {
    if (nelems) {
      size_t n = (nelems + WARP_SZ - 1) & ((size_t)0 - WARP_SZ);
      CUDA_OK(cudaMallocAsync(&d_ptr, n * sizeof(T), s));
    }
  }
  dev_ptr_t(const dev_ptr_t& r) = delete;
  dev_ptr_t& operator=(const dev_ptr_t& r) = delete;
  ~dev_ptr_t()
  {
    if (d_ptr) {
      if (stream)
        (void)cudaFreeAsync((void*)d_ptr, stream);
      else
        (void)cudaFree((void*)d_ptr);
    }
  }

  inline operator const T*() const { return d_ptr; }
  inline operator T*() const { return d_ptr; }
  inline operator void*() const { return (void*)d_ptr; }
  inline const T& operator[](size_t i) const { return d_ptr[i]; }
  inline T& operator[](size_t i) { return d_ptr[i]; }
};
