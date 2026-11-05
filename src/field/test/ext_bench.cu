// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0

// KoalaBear^5 extension-field cost on the GPU.
//
// Measures kb31_5_t, the production type, against alternative KoalaBear^5 arithmetic and
// layouts, on the two shapes the MLE prover's extension-field work comes in:
//   - compute-bound: dependent chains of ext*ext and base*ext multiplies;
//   - memory-bound:  the sumcheck/GKR fold  out[i] = a[2i] + r * (a[2i+1] - a[2i]).
//
// Fields (all values stored in Montgomery form, like kb31_t):
//   E5   KoalaBear^5 = F[X]/(X^5 + X^2 - 1)   (Plonky3 QuinticTrinomialExtensionField), generic
//   E5o  KoalaBear^5 with the +-1 reduction folded into the accumulation (5 reductions)
//   E5k  KoalaBear^5 as the library type kb31_5_t (field/koala_bear_quintic.hpp, PTX)
// KoalaBear^5 memory layouts for the fold: packed 20 B (E5), padded to 32 B (E5p), SoA planes.
//
// Correctness: every multiply is checked against Plonky3 vectors written by
// Pico's `mle-core/examples/e3_ext_vectors.rs` before anything is timed.
//
//   ./ext_bench <gpu> <vector_dir>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "field/fp.h"  // kb31_t
#include "field/koala_bear_quintic.hpp"  // kb31_5_t

#define CK(x)                                                                              \
    do {                                                                                   \
        cudaError_t e_ = (x);                                                              \
        if (e_ != cudaSuccess) {                                                           \
            fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
            exit(1);                                                                       \
        }                                                                                  \
    } while (0)

static constexpr uint32_t MOD = 0x7F000001u;
static constexpr uint32_t MPRIME = 0x7EFFFFFFu;  // -(p^-1) mod 2^32 (= kb31_t::M0; NOT fp.h Fp::M, which is p^-1)

// ---------------------------------------------------------------------------------------------
// Generic Montgomery helpers (used by E5, E5o, E5p)
// ---------------------------------------------------------------------------------------------
__host__ __device__ __forceinline__ uint32_t mont_reduce(uint64_t acc) {
    // acc < MOD * 2^32
    uint32_t m = (uint32_t)acc * MPRIME;
    acc += (uint64_t)m * MOD;
    uint32_t r = (uint32_t)(acc >> 32);
    return r >= MOD ? r - MOD : r;
}
__device__ __forceinline__ void acc_mul(uint64_t& acc, uint32_t a, uint32_t b) {
    acc += (uint64_t)a * b;  // a, b < MOD < 2^31 -> product < 2^62
    if ((uint32_t)(acc >> 32) >= MOD) acc -= ((uint64_t)MOD << 32);
}
__device__ __forceinline__ uint32_t addm(uint32_t a, uint32_t b) {
    uint32_t s = a + b;
    return s >= MOD ? s - MOD : s;
}
__device__ __forceinline__ uint32_t subm(uint32_t a, uint32_t b) {
    return a >= b ? a - b : a + MOD - b;
}
__device__ __forceinline__ uint32_t mmul(uint32_t a, uint32_t b) {
    return mont_reduce((uint64_t)a * b);
}

// ---------------------------------------------------------------------------------------------
// Field types
// ---------------------------------------------------------------------------------------------
struct E5  { uint32_t u[5]; };
struct alignas(32) E5p { uint32_t u[8]; };  // KB5 padded to 32 B; u[5..7] unused

static_assert(sizeof(E5) == 20 && sizeof(E5p) == 32, "sizes");

// ---- E5: X^5 = 1 - X^2
//   X^5 = 1 - X^2, X^6 = X - X^3, X^7 = X^2 - X^4, X^8 = X^3 + X^2 - 1
__device__ __forceinline__ void e5mul_raw(const uint32_t* a, const uint32_t* b, uint32_t* r) {
    uint32_t c[9];
#pragma unroll
    for (int k = 0; k < 9; k++) {
        uint64_t acc = 0;
#pragma unroll
        for (int i = 0; i < 5; i++) {
            int j = k - i;
            if (j >= 0 && j < 5) acc_mul(acc, a[i], b[j]);
        }
        c[k] = mont_reduce(acc);
    }
    r[0] = subm(addm(c[0], c[5]), c[8]);
    r[1] = addm(c[1], c[6]);
    r[2] = addm(subm(c[2], c[5]), addm(c[7], c[8]));
    r[3] = addm(subm(c[3], c[6]), c[8]);
    r[4] = subm(c[4], c[7]);
}
__device__ __forceinline__ E5 emul(const E5& a, const E5& b) { E5 r; e5mul_raw(a.u, b.u, r.u); return r; }
__device__ __forceinline__ E5 eadd(const E5& a, const E5& b) { E5 r; for (int i = 0; i < 5; i++) r.u[i] = addm(a.u[i], b.u[i]); return r; }
__device__ __forceinline__ E5 esub(const E5& a, const E5& b) { E5 r; for (int i = 0; i < 5; i++) r.u[i] = subm(a.u[i], b.u[i]); return r; }
__device__ __forceinline__ E5 esmul(const E5& a, uint32_t s) { E5 r; for (int i = 0; i < 5; i++) r.u[i] = mmul(a.u[i], s); return r; }

// ---- E5o: KoalaBear^5 with the reduction folded into the accumulation.
// The trinomial's reduction coefficients are all +-1, so each output coefficient is a signed sum
// of products and needs ONE Montgomery reduction (5 total, vs 9 in E5 and 7 in the KB4 PTX):
//   r0 = c0 + c5 - c8          r1 = c1 + c6          r2 = c2 - c5 + c7 + c8
//   r3 = c3 - c6 + c8          r4 = c4 - c7
// acc_msub keeps the accumulator invariant acc < MOD * 2^32 by adding MOD * 2^32 on underflow.
__device__ __forceinline__ void acc_msub(uint64_t& acc, uint32_t a, uint32_t b) {
    uint64_t t = (uint64_t)a * b;
    if (acc < t) acc += ((uint64_t)MOD << 32);
    acc -= t;
}
__device__ __forceinline__ void e5mul_opt(const uint32_t* a, const uint32_t* b, uint32_t* r) {
    uint64_t acc;
    acc = 0;  // r0 = c0 + c5 - c8
    acc_mul(acc, a[0], b[0]); acc_mul(acc, a[1], b[4]); acc_mul(acc, a[2], b[3]);
    acc_mul(acc, a[3], b[2]); acc_mul(acc, a[4], b[1]); acc_msub(acc, a[4], b[4]);
    r[0] = mont_reduce(acc);
    acc = 0;  // r1 = c1 + c6
    acc_mul(acc, a[0], b[1]); acc_mul(acc, a[1], b[0]); acc_mul(acc, a[2], b[4]);
    acc_mul(acc, a[3], b[3]); acc_mul(acc, a[4], b[2]);
    r[1] = mont_reduce(acc);
    acc = 0;  // r2 = c2 - c5 + c7 + c8
    acc_mul(acc, a[0], b[2]); acc_mul(acc, a[1], b[1]); acc_mul(acc, a[2], b[0]);
    acc_mul(acc, a[3], b[4]); acc_mul(acc, a[4], b[3]); acc_mul(acc, a[4], b[4]);
    acc_msub(acc, a[1], b[4]); acc_msub(acc, a[2], b[3]); acc_msub(acc, a[3], b[2]); acc_msub(acc, a[4], b[1]);
    r[2] = mont_reduce(acc);
    acc = 0;  // r3 = c3 - c6 + c8
    acc_mul(acc, a[0], b[3]); acc_mul(acc, a[1], b[2]); acc_mul(acc, a[2], b[1]);
    acc_mul(acc, a[3], b[0]); acc_mul(acc, a[4], b[4]);
    acc_msub(acc, a[2], b[4]); acc_msub(acc, a[3], b[3]); acc_msub(acc, a[4], b[2]);
    r[3] = mont_reduce(acc);
    acc = 0;  // r4 = c4 - c7
    acc_mul(acc, a[0], b[4]); acc_mul(acc, a[1], b[3]); acc_mul(acc, a[2], b[2]);
    acc_mul(acc, a[3], b[1]); acc_mul(acc, a[4], b[0]);
    acc_msub(acc, a[3], b[4]); acc_msub(acc, a[4], b[3]);
    r[4] = mont_reduce(acc);
}
struct E5o { uint32_t u[5]; };
__device__ __forceinline__ E5o emul(const E5o& a, const E5o& b) { E5o r; e5mul_opt(a.u, b.u, r.u); return r; }
__device__ __forceinline__ E5o eadd(const E5o& a, const E5o& b) { E5o r; for (int i = 0; i < 5; i++) r.u[i] = addm(a.u[i], b.u[i]); return r; }
__device__ __forceinline__ E5o esub(const E5o& a, const E5o& b) { E5o r; for (int i = 0; i < 5; i++) r.u[i] = subm(a.u[i], b.u[i]); return r; }
__device__ __forceinline__ E5o esmul(const E5o& a, uint32_t s) { E5o r; for (int i = 0; i < 5; i++) r.u[i] = mmul(a.u[i], s); return r; }

__device__ __forceinline__ E5p emul(const E5p& a, const E5p& b) { E5p r; e5mul_raw(a.u, b.u, r.u); r.u[5] = r.u[6] = r.u[7] = 0; return r; }
__device__ __forceinline__ E5p eadd(const E5p& a, const E5p& b) { E5p r; for (int i = 0; i < 5; i++) r.u[i] = addm(a.u[i], b.u[i]); r.u[5] = r.u[6] = r.u[7] = 0; return r; }
__device__ __forceinline__ E5p esub(const E5p& a, const E5p& b) { E5p r; for (int i = 0; i < 5; i++) r.u[i] = subm(a.u[i], b.u[i]); r.u[5] = r.u[6] = r.u[7] = 0; return r; }

// ---- E5k: the library type kb31_5_t
struct E5k { kb31_5_t v; };
static_assert(sizeof(E5k) == 20, "sizes");
__device__ __forceinline__ E5k emul(const E5k& a, const E5k& b) { E5k r; r.v = a.v * b.v; return r; }
__device__ __forceinline__ E5k eadd(const E5k& a, const E5k& b) { E5k r; r.v = a.v; r.v += b.v; return r; }
__device__ __forceinline__ E5k esub(const E5k& a, const E5k& b) { E5k r; r.v = a.v; r.v -= b.v; return r; }
__device__ __forceinline__ E5k esmul(const E5k& a, uint32_t s) { E5k r; r.v = a.v * kb31_t(s); return r; }

// ---------------------------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------------------------
template <class E>
__global__ void k_mul_vec(const E* a, const E* b, E* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = emul(a[i], b[i]);
}

// 4 independent dependent chains per thread: x = x*y + c.
template <class E>
__global__ void k_mul_chain(const E* a, const E* b, E* out, int n, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    E y = b[i], c = a[i];
    E x0 = a[i], x1 = b[i], x2 = eadd(a[i], b[i]), x3 = eadd(x2, a[i]);
    for (int it = 0; it < iters; it++) {
        x0 = eadd(emul(x0, y), c);
        x1 = eadd(emul(x1, y), c);
        x2 = eadd(emul(x2, y), c);
        x3 = eadd(emul(x3, y), c);
    }
    out[i] = eadd(eadd(x0, x1), eadd(x2, x3));
}

// base*ext chains: x = x*s + c, s a base-field element.
template <class E>
__global__ void k_smul_chain(const E* a, const E* b, E* out, int n, int iters, uint32_t s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    E c = a[i];
    E x0 = a[i], x1 = b[i], x2 = eadd(a[i], b[i]), x3 = eadd(x2, a[i]);
    for (int it = 0; it < iters; it++) {
        x0 = eadd(esmul(x0, s), c);
        x1 = eadd(esmul(x1, s), c);
        x2 = eadd(esmul(x2, s), c);
        x3 = eadd(esmul(x3, s), c);
    }
    out[i] = eadd(eadd(x0, x1), eadd(x2, x3));
}

// fold, AoS: out[i] = a[2i] + r*(a[2i+1] - a[2i])
template <class E>
__global__ void k_fold(const E* __restrict__ in, E* __restrict__ out, size_t n, E r) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        E lo = in[2 * i], hi = in[2 * i + 1];
        out[i] = eadd(lo, emul(r, esub(hi, lo)));
    }
}

// fold, KB5 SoA: five planes of 2n inputs, five planes of n outputs
__global__ void k_fold_soa5(const uint32_t* __restrict__ in, uint32_t* __restrict__ out, size_t n, E5 r) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        E5 lo, hi;
#pragma unroll
        for (int k = 0; k < 5; k++) {
            lo.u[k] = in[k * 2 * n + 2 * i];
            hi.u[k] = in[k * 2 * n + 2 * i + 1];
        }
        E5 v = eadd(lo, emul(r, esub(hi, lo)));
#pragma unroll
        for (int k = 0; k < 5; k++) out[k * n + i] = v.u[k];
    }
}

// ---------------------------------------------------------------------------------------------
// Host
// ---------------------------------------------------------------------------------------------
static uint32_t to_mont(uint32_t x) { return (uint32_t)(((uint64_t)x << 32) % MOD); }
static uint32_t from_mont(uint32_t m) { return mont_reduce((uint64_t)m); }

struct Vectors { uint32_t n = 0, d = 0; std::vector<uint32_t> a, b, c; };

static Vectors load_vectors(const std::string& path) {
    Vectors v;
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
    if (fread(&v.n, 4, 1, f) != 1 || fread(&v.d, 4, 1, f) != 1) exit(1);
    size_t m = (size_t)v.n * v.d;
    v.a.resize(m); v.b.resize(m); v.c.resize(m);
    if (fread(v.a.data(), 4, m, f) != m || fread(v.b.data(), 4, m, f) != m || fread(v.c.data(), 4, m, f) != m) exit(1);
    fclose(f);
    return v;
}

// Plonky3 basis -> device layout (Montgomery), coefficient k in word k.
static std::vector<uint32_t> to_layout(const std::vector<uint32_t>& src, uint32_t d, int words) {
    size_t n = src.size() / d;
    std::vector<uint32_t> out(n * words, 0);
    for (size_t i = 0; i < n; i++)
        for (uint32_t k = 0; k < d; k++) {
            out[i * words + k] = to_mont(src[i * d + k]);
        }
    return out;
}
static uint32_t from_layout_coef(const std::vector<uint32_t>& buf, size_t i, uint32_t k, int words) {
    return from_mont(buf[i * words + k]);
}

template <class E>
static bool check_mul(const char* name, const Vectors& v) {
    int words = sizeof(E) / 4;
    auto ha = to_layout(v.a, v.d, words), hb = to_layout(v.b, v.d, words);
    E *da, *db, *dc;
    size_t bytes = (size_t)v.n * sizeof(E);
    CK(cudaMalloc(&da, bytes)); CK(cudaMalloc(&db, bytes)); CK(cudaMalloc(&dc, bytes));
    CK(cudaMemcpy(da, ha.data(), bytes, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(db, hb.data(), bytes, cudaMemcpyHostToDevice));
    k_mul_vec<E><<<(v.n + 255) / 256, 256>>>(da, db, dc, v.n);
    CK(cudaDeviceSynchronize());
    std::vector<uint32_t> hc((size_t)v.n * words);
    CK(cudaMemcpy(hc.data(), dc, bytes, cudaMemcpyDeviceToHost));
    size_t bad = 0;
    for (size_t i = 0; i < v.n; i++)
        for (uint32_t k = 0; k < v.d; k++)
            if (from_layout_coef(hc, i, k, words) != v.c[i * v.d + k]) bad++;
    printf("check %-4s : %s (%u products, %zu wrong coefficients)\n", name, bad ? "FAIL" : "ok", v.n, bad);
    CK(cudaFree(da)); CK(cudaFree(db)); CK(cudaFree(dc));
    return bad == 0;
}

template <class F>
static float time_median(F launch, int reps = 7) {
    cudaEvent_t s, e;
    CK(cudaEventCreate(&s)); CK(cudaEventCreate(&e));
    launch();  // warm-up
    CK(cudaDeviceSynchronize());
    std::vector<float> t;
    for (int r = 0; r < reps; r++) {
        CK(cudaEventRecord(s));
        launch();
        CK(cudaEventRecord(e));
        CK(cudaEventSynchronize(e));
        float ms; CK(cudaEventElapsedTime(&ms, s, e));
        t.push_back(ms);
    }
    CK(cudaEventDestroy(s)); CK(cudaEventDestroy(e));
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

static void fill_random_mont(std::vector<uint32_t>& h) {
    uint64_t x = 0x9e3779b97f4a7c15ull;
    for (auto& w : h) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; w = to_mont((uint32_t)(x % MOD)); }
}

template <class E>
static void bench_chains(const char* name, int sms, double base_mul_ms, double base_smul_ms, double* out_mul, double* out_smul) {
    const int threads = 256, blocks = sms * 8, n = threads * blocks, iters = 1024;
    std::vector<uint32_t> h((size_t)n * (sizeof(E) / 4));
    fill_random_mont(h);
    for (size_t i = 0; i < h.size(); i++) if (sizeof(E) == 32 && std::string(name) == "E5p" && (i % 8) >= 5) h[i] = 0;
    E *da, *db, *dout;
    size_t bytes = (size_t)n * sizeof(E);
    CK(cudaMalloc(&da, bytes)); CK(cudaMalloc(&db, bytes)); CK(cudaMalloc(&dout, bytes));
    CK(cudaMemcpy(da, h.data(), bytes, cudaMemcpyHostToDevice));
    std::reverse(h.begin(), h.end());
    CK(cudaMemcpy(db, h.data(), bytes, cudaMemcpyHostToDevice));
    float ms = time_median([&] { k_mul_chain<E><<<blocks, threads>>>(da, db, dout, n, iters); });
    double gmul = (double)n * iters * 4 / (ms * 1e-3) / 1e9;
    *out_mul = ms;
    float ms2 = time_median([&] { k_smul_chain<E><<<blocks, threads>>>(da, db, dout, n, iters, to_mont(12345)); });
    double gsm = (double)n * iters * 4 / (ms2 * 1e-3) / 1e9;
    *out_smul = ms2;
    printf("chain %-4s : ext*ext %8.2f Gmul/s  (%.2fx E5k time) | base*ext %8.2f Gmul/s  (%.2fx E5k time)\n",
           name, gmul, base_mul_ms > 0 ? ms / base_mul_ms : 1.0, gsm, base_smul_ms > 0 ? ms2 / base_smul_ms : 1.0);
    CK(cudaFree(da)); CK(cudaFree(db)); CK(cudaFree(dout));
}

template <class E>
static float bench_fold(const char* name, int sms, size_t n, int value_bytes, double base_ms) {
    size_t words = sizeof(E) / 4;
    std::vector<uint32_t> h(2 * n * words);
    fill_random_mont(h);
    E *din, *dout;
    CK(cudaMalloc(&din, 2 * n * sizeof(E))); CK(cudaMalloc(&dout, n * sizeof(E)));
    CK(cudaMemcpy(din, h.data(), 2 * n * sizeof(E), cudaMemcpyHostToDevice));
    E r; memcpy(&r, h.data(), sizeof(E));
    int threads = 256, blocks = sms * 16;
    float ms = time_median([&] { k_fold<E><<<blocks, threads>>>(din, dout, n, r); });
    double gbs = (double)n * 3 * sizeof(E) / (ms * 1e-3) / 1e9;
    printf("fold  %-6s: %7.3f ms  %7.1f GB/s moved (%2zu B/elem stored, %2d B/elem payload)  %.2fx E5k time\n",
           name, ms, gbs, sizeof(E), value_bytes, base_ms > 0 ? ms / base_ms : 1.0);
    CK(cudaFree(din)); CK(cudaFree(dout));
    return ms;
}

static float bench_fold_soa5(int sms, size_t n, double base_ms) {
    std::vector<uint32_t> h(5 * 2 * n);
    fill_random_mont(h);
    uint32_t *din, *dout;
    CK(cudaMalloc(&din, h.size() * 4)); CK(cudaMalloc(&dout, 5 * n * 4));
    CK(cudaMemcpy(din, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    E5 r; for (int k = 0; k < 5; k++) r.u[k] = h[k];
    int threads = 256, blocks = sms * 16;
    float ms = time_median([&] { k_fold_soa5<<<blocks, threads>>>(din, dout, n, r); });
    double gbs = (double)n * 3 * 20 / (ms * 1e-3) / 1e9;
    printf("fold  %-6s: %7.3f ms  %7.1f GB/s moved (20 B/elem, 5 planes)                 %.2fx E5k time\n",
           "E5soa", ms, gbs, ms / base_ms);
    CK(cudaFree(din)); CK(cudaFree(dout));
    return ms;
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <gpu> <vector_dir>\n", argv[0]); return 2; }
    int gpu = atoi(argv[1]);
    std::string dir = argv[2];
    CK(cudaSetDevice(gpu));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, gpu));
    int sms = prop.multiProcessorCount;
    printf("device %d: %s, %d SMs\n\n", gpu, prop.name, sms);

    // 1. Correctness against Plonky3.
    Vectors v5 = load_vectors(dir + "/kb5_mul.bin");
    bool ok = true;
    ok &= check_mul<E5k>("E5k", v5);
    ok &= check_mul<E5>("E5", v5);
    ok &= check_mul<E5o>("E5o", v5);
    ok &= check_mul<E5p>("E5p", v5);
    if (!ok) { printf("\ncorrectness FAILED -- not timing\n"); return 1; }

    // 2. Compute-bound chains.
    printf("\n");
    double mk, sk, m, s;
    bench_chains<E5k>("E5k", sms, 0, 0, &mk, &sk);
    bench_chains<E5>("E5", sms, mk, sk, &m, &s);
    bench_chains<E5o>("E5o", sms, mk, sk, &m, &s);

    // 3. Memory-bound fold, 2^24 outputs.
    printf("\n");
    size_t n = (size_t)1 << 24;
    float fk = bench_fold<E5k>("E5k", sms, n, 20, 0);
    bench_fold<E5>("E5", sms, n, 20, fk);
    bench_fold<E5o>("E5o", sms, n, 20, fk);
    bench_fold<E5p>("E5p", sms, n, 20, fk);
    bench_fold_soa5(sms, n, fk);
    return 0;
}
