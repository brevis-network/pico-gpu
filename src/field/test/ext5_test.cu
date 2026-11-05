// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Differential test of kb31_5_t (field/koala_bear_quintic.hpp), the GPU KoalaBear^5 =
// F[X]/(X^5 + X^2 - 1), against a host reference.
//
// Oracle chain:
//   1. Plonky3 -> host reference: the reference product (plain `% p` arithmetic, no Montgomery)
//      must reproduce every product in vectors/kb5_mul.bin, written by Pico's
//      `mle-core/examples/e3_ext_vectors.rs` from QuinticTrinomialExtensionField<KoalaBear>.
//   2. host reference -> GPU: every kb31_5_t operation is checked on two input sets, the Plonky3
//      random inputs and an edge set (coefficients 0, 1, p-1, p-2, (p+-1)/2, 2^30, ... and
//      all-max vectors), the inputs that drive the 64-bit accumulators to their bounds.
// Squaring, Frobenius and inversion have no Plonky3 vectors of their own; they are checked
// against the reference product: sqr(a) == a*a, frob(a) == a^p, a * inv(a) == 1 (inv(0) == 0).
//
//   ./ext5_test <gpu> <vector_dir>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "ff/koala_bear.hpp"
#include "field/koala_bear_quintic.hpp"

#define CK(x)                                                                              \
    do {                                                                                   \
        cudaError_t e_ = (x);                                                              \
        if (e_ != cudaSuccess) {                                                           \
            fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
            exit(1);                                                                       \
        }                                                                                  \
    } while (0)

static constexpr uint64_t P = 0x7F000001u;
using E = kb31_5_t;
static_assert(sizeof(E) == 20 && alignof(E) == 4, "kb31_5_t must be packed 20 bytes");

// ---------------------------------------------------------------------------------------------
// Host reference, canonical form
// ---------------------------------------------------------------------------------------------
struct R5 { uint32_t v[5]; };

static R5 r_mul(const R5& a, const R5& b) {
    uint64_t c[9] = {0};
    for (int i = 0; i < 5; i++)
        for (int j = 0; j < 5; j++) c[i + j] = (c[i + j] + (uint64_t)a.v[i] * b.v[j]) % P;
    R5 r;
    r.v[0] = (uint32_t)((c[0] + c[5] + P - c[8]) % P);
    r.v[1] = (uint32_t)((c[1] + c[6]) % P);
    r.v[2] = (uint32_t)((c[2] + P - c[5] + c[7] + c[8]) % P);
    r.v[3] = (uint32_t)((c[3] + P - c[6] + c[8]) % P);
    r.v[4] = (uint32_t)((c[4] + P - c[7]) % P);
    return r;
}
static R5 r_pow(R5 a, uint64_t e) {
    R5 r = {{1, 0, 0, 0, 0}};
    for (; e; e >>= 1, a = r_mul(a, a))
        if (e & 1) r = r_mul(r, a);
    return r;
}
static R5 r_add(const R5& a, const R5& b) { R5 r; for (int i = 0; i < 5; i++) r.v[i] = (uint32_t)((a.v[i] + (uint64_t)b.v[i]) % P); return r; }
static R5 r_sub(const R5& a, const R5& b) { R5 r; for (int i = 0; i < 5; i++) r.v[i] = (uint32_t)((a.v[i] + P - b.v[i]) % P); return r; }
static R5 r_smul(const R5& a, uint32_t s) { R5 r; for (int i = 0; i < 5; i++) r.v[i] = (uint32_t)((uint64_t)a.v[i] * s % P); return r; }
static bool r_eq(const R5& a, const R5& b) { for (int i = 0; i < 5; i++) if (a.v[i] != b.v[i]) return false; return true; }
static bool r_is_zero(const R5& a) { return r_eq(a, R5{{0, 0, 0, 0, 0}}); }
static bool r_is_one(const R5& a) { return r_eq(a, R5{{1, 0, 0, 0, 0}}); }

static uint32_t to_mont(uint32_t x) { return (uint32_t)(((uint64_t)x << 32) % P); }
static uint32_t from_mont(uint32_t m) {
    // m * 2^-32 mod p; 2^-32 = (2^32)^(p-2)
    static uint64_t rinv = [] { uint64_t b = (1ull << 32) % P, r = 1; for (uint64_t e = P - 2; e; e >>= 1, b = b * b % P) if (e & 1) r = r * b % P; return r; }();
    return (uint32_t)(m * rinv % P);
}

// ---------------------------------------------------------------------------------------------
// Device kernels, one per operation
// ---------------------------------------------------------------------------------------------
enum Op { MUL, SQR, INV, FROB, SMUL, ADD, SUB, POW, NUM_OPS };
static const char* OP_NAME[NUM_OPS] = {"mul", "sqr", "inv", "frobenius", "base*ext", "add", "sub", "pow"};

__global__ void k_op(int op, const E* a, const E* b, E* out, int n, uint32_t s, uint32_t e) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    E x = a[i], y = b[i];
    switch (op) {
    case MUL:  out[i] = x * y; break;
    case SQR:  out[i] = x; out[i].sqr(); break;
    case INV:  out[i] = x.reciprocal(); break;
    case FROB: out[i] = x.frobenius(); break;
    case SMUL: out[i] = x * kb31_t{s}; break;
    case ADD:  out[i] = x + y; break;
    case SUB:  out[i] = x - y; break;
    case POW:  out[i] = x ^ e; break;
    }
}

// ---------------------------------------------------------------------------------------------
// Host driver
// ---------------------------------------------------------------------------------------------
static std::vector<R5> load_kb5(const std::string& path, std::vector<R5>& b, std::vector<R5>& c) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
    uint32_t n, d;
    if (fread(&n, 4, 1, f) != 1 || fread(&d, 4, 1, f) != 1 || d != 5) { fprintf(stderr, "bad header in %s\n", path.c_str()); exit(1); }
    std::vector<R5> a(n);
    b.resize(n); c.resize(n);
    if (fread(a.data(), 20, n, f) != n || fread(b.data(), 20, n, f) != n || fread(c.data(), 20, n, f) != n) exit(1);
    fclose(f);
    return a;
}

static std::vector<R5> edge_inputs(size_t n, uint64_t seed) {
    const uint32_t EDGE[] = {0, 1, 2, (uint32_t)P - 1, (uint32_t)P - 2, (uint32_t)P - 3, (uint32_t)(P - 1) / 2,
                             (uint32_t)(P + 1) / 2, 1u << 30, (1u << 30) + 1, 1u << 24, (uint32_t)P - (1u << 24)};
    const int NE = sizeof(EDGE) / sizeof(EDGE[0]);
    std::vector<R5> v(n);
    uint64_t x = seed;
    for (size_t i = 0; i < n; i++)
        for (int k = 0; k < 5; k++) {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            v[i].v[k] = EDGE[x % NE];
        }
    // Fixed corners: zero, one, all p-1, single p-1 in each position.
    v[0] = {{0, 0, 0, 0, 0}};
    v[1] = {{1, 0, 0, 0, 0}};
    v[2] = {{(uint32_t)P - 1, (uint32_t)P - 1, (uint32_t)P - 1, (uint32_t)P - 1, (uint32_t)P - 1}};
    for (int k = 0; k < 5; k++) { v[3 + k] = {{0, 0, 0, 0, 0}}; v[3 + k].v[k] = (uint32_t)P - 1; }
    return v;
}

struct Device {
    E *a, *b, *out;
    int n;
    Device(const std::vector<R5>& ha, const std::vector<R5>& hb) : n((int)ha.size()) {
        size_t bytes = (size_t)n * sizeof(E);
        std::vector<uint32_t> ma(n * 5), mb(n * 5);
        for (int i = 0; i < n; i++)
            for (int k = 0; k < 5; k++) { ma[i * 5 + k] = to_mont(ha[i].v[k]); mb[i * 5 + k] = to_mont(hb[i].v[k]); }
        CK(cudaMalloc(&a, bytes)); CK(cudaMalloc(&b, bytes)); CK(cudaMalloc(&out, bytes));
        CK(cudaMemcpy(a, ma.data(), bytes, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(b, mb.data(), bytes, cudaMemcpyHostToDevice));
    }
    ~Device() { cudaFree(a); cudaFree(b); cudaFree(out); }
    std::vector<R5> run(int op, uint32_t s_canon, uint32_t e) {
        k_op<<<(n + 255) / 256, 256>>>(op, a, b, out, n, to_mont(s_canon), e);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        std::vector<uint32_t> h((size_t)n * 5);
        CK(cudaMemcpy(h.data(), out, h.size() * 4, cudaMemcpyDeviceToHost));
        std::vector<R5> r(n);
        for (int i = 0; i < n; i++)
            for (int k = 0; k < 5; k++) r[i].v[k] = from_mont(h[i * 5 + k]);
        return r;
    }
};

static bool check_set(const char* set, const std::vector<R5>& a, const std::vector<R5>& b) {
    Device dev(a, b);
    const uint32_t S = (uint32_t)P - 12345, POW_E = 0xDEADBEEF;
    bool ok = true;
    for (int op = 0; op < NUM_OPS; op++) {
        std::vector<R5> got = dev.run(op, S, POW_E);
        size_t bad = 0, first = SIZE_MAX;
        for (size_t i = 0; i < a.size(); i++) {
            bool good;
            switch (op) {
            case MUL:  good = r_eq(got[i], r_mul(a[i], b[i])); break;
            case SQR:  good = r_eq(got[i], r_mul(a[i], a[i])); break;
            case INV:  good = r_is_zero(a[i]) ? r_is_zero(got[i]) : r_is_one(r_mul(a[i], got[i])); break;
            case FROB: good = r_eq(got[i], r_pow(a[i], P)); break;
            case SMUL: good = r_eq(got[i], r_smul(a[i], S)); break;
            case ADD:  good = r_eq(got[i], r_add(a[i], b[i])); break;
            case SUB:  good = r_eq(got[i], r_sub(a[i], b[i])); break;
            case POW:  good = r_eq(got[i], r_pow(a[i], POW_E)); break;
            }
            if (!good && bad++ == 0) first = i;
        }
        printf("  %-7s %-9s : %s (%zu inputs, %zu wrong", set, OP_NAME[op], bad ? "FAIL" : "ok", a.size(), bad);
        if (bad) printf(", first at %zu", first);
        printf(")\n");
        ok &= bad == 0;
    }
    return ok;
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <gpu> <vector_dir>\n", argv[0]); return 2; }
    CK(cudaSetDevice(atoi(argv[1])));
    std::string dir = argv[2];

    // 1. Plonky3 -> host reference.
    std::vector<R5> pb, pc;
    std::vector<R5> pa = load_kb5(dir + "/kb5_mul.bin", pb, pc);
    size_t ref_bad = 0;
    for (size_t i = 0; i < pa.size(); i++) ref_bad += !r_eq(r_mul(pa[i], pb[i]), pc[i]);
    printf("host reference vs Plonky3: %s (%zu products, %zu wrong)\n", ref_bad ? "FAIL" : "ok", pa.size(), ref_bad);
    if (ref_bad) return 1;

    // 2. host reference -> GPU.
    bool ok = check_set("plonky3", pa, pb);
    std::vector<R5> ea = edge_inputs(1 << 16, 0x9e3779b97f4a7c15ull), eb = edge_inputs(1 << 16, 0xd1b54a32d192ed03ull);
    ok &= check_set("edge", ea, eb);
    // edge x random, so a max-coefficient operand meets arbitrary ones too
    ok &= check_set("mixed", ea, pb);

    printf("%s\n", ok ? "ALL OK" : "FAILED");
    return ok ? 0 : 1;
}
