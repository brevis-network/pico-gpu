// Modified by Brevis
// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

/*
 * KoalaBear degree-5 extension field: kb31_5_t
 *
 * KoalaBear^5 = KoalaBear[x] / (x^5 + x^2 - 1)
 * Plonky3's QuinticTrinomialExtensionField<KoalaBear>, same basis order: element = sum c[i] x^i.
 *
 * Reduction identities (all coefficients +-1, so no multiply by a non-residue):
 *   x^5 = 1 - x^2,  x^6 = x - x^3,  x^7 = x^2 - x^4,  x^8 = x^3 + x^2 - 1
 *
 * Packed 20 bytes, 4-byte aligned, like Rust's `[KoalaBear; 5]`.
 *
 * Multiplication: each product coefficient c[k] = sum_{i+j=k} a[i]*b[j] is accumulated in a
 * 64-bit register pair and Montgomery-reduced once (hand-written PTX, see dotN), then the
 * reduction identities are applied with modular adds. Formulas follow Plonky3 da6ca12
 * field/src/extension/quintic_extension.rs.
 */

#ifndef __SPPARK_FF_KOALA_BEAR_QUINTIC_HPP__
#define __SPPARK_FF_KOALA_BEAR_QUINTIC_HPP__

#ifdef __CUDACC__
# include <cassert>
# include "ff/koala_bear.hpp"
# define inline __device__ __forceinline__

class kb31_5_t {
    union { kb31_t c[5]; uint32_t u[5]; };

    static const uint32_t MOD = 0x7F000001;   // KoalaBear prime
    static const uint32_t M   = 0x7EFFFFFF;   // -p^{-1} mod 2^32

public:
    static const uint32_t degree = 5;
    using mem_t = kb31_5_t;

    inline kb31_t& operator[](size_t i)             { return c[i]; }
    inline const kb31_t& operator[](size_t i) const { return c[i]; }
    inline size_t len() const                       { return 5; }

    inline kb31_5_t()           {}
    inline kb31_5_t(kb31_t a)   { c[0] = a; u[1] = u[2] = u[3] = u[4] = 0; }
    __host__ __device__ __forceinline__ kb31_5_t(int a)
    {   c[0] = kb31_t{a}; u[1] = u[2] = u[3] = u[4] = 0;   }
    __host__ __device__ __forceinline__ kb31_5_t(int d, int f, int g, int h, int k)
    {   c[0] = kb31_t{d}; c[1] = kb31_t{f}; c[2] = kb31_t{g}; c[3] = kb31_t{h}; c[4] = kb31_t{k};   }

private:
    // dotN: the Montgomery-reduced sum of N <= 5 products a_i * b_i, values in [0, MOD).
    // A product is below MOD^2 < 0.985 * 2^62, so two of them sum with the high word still below
    // MOD, and four sum in 64 bits with the high word below 2 * MOD; one conditional subtraction
    // of MOD from the high word ("trim") then restores high < MOD, which the reduction needs.
    // The five-product sum is trimmed after its fourth product as well.
# if defined(__CUDA_ARCH__) && !defined(__clang_analyzer__)
    // PTX: products accumulate in a (hi, lo) register pair through mad.lo.cc / madc.hi, and
    // every check and the reduction work on the 32-bit high word.
    // Operands: %0 result, %1 MOD, %2 M, then a_i = %(3 + 2i), b_i = %(4 + 2i).
#  define KB5_MUL(a, b) "mul.lo.u32 %lo, %" #a ", %" #b "; mul.hi.u32 %hi, %" #a ", %" #b ";\n\t"
#  define KB5_MAD(a, b) "mad.lo.cc.u32 %lo, %" #a ", %" #b ", %lo; madc.hi.u32 %hi, %" #a ", %" #b ", %hi;\n\t"
#  define KB5_TRIM      "setp.ge.u32 %p, %hi, %1; @%p sub.u32 %hi, %hi, %1;\n\t"
#  define KB5_REDC      "mul.lo.u32 %m, %lo, %2;\n\t"                                      \
                        "mad.lo.cc.u32 %lo, %m, %1, %lo; madc.hi.u32 %0, %m, %1, %hi;\n\t" \
                        "setp.ge.u32 %p, %0, %1; @%p sub.u32 %0, %0, %1;\n\t"
#  define KB5_DOT(body) "{ .reg.b32 %lo, %hi, %m; .reg.pred %p;\n\t" body "}"

    static inline uint32_t dot1(uint32_t a0, uint32_t b0)
    {
        uint32_t r;
        asm(KB5_DOT(KB5_MUL(3, 4) KB5_REDC)
            : "=r"(r) : "r"(MOD), "r"(M), "r"(a0), "r"(b0));
        return r;
    }
    static inline uint32_t dot2(uint32_t a0, uint32_t b0, uint32_t a1, uint32_t b1)
    {
        uint32_t r;
        asm(KB5_DOT(KB5_MUL(3, 4) KB5_MAD(5, 6) KB5_REDC)
            : "=r"(r) : "r"(MOD), "r"(M), "r"(a0), "r"(b0), "r"(a1), "r"(b1));
        return r;
    }
    static inline uint32_t dot3(uint32_t a0, uint32_t b0, uint32_t a1, uint32_t b1,
                                uint32_t a2, uint32_t b2)
    {
        uint32_t r;
        asm(KB5_DOT(KB5_MUL(3, 4) KB5_MAD(5, 6) KB5_MAD(7, 8) KB5_TRIM KB5_REDC)
            : "=r"(r) : "r"(MOD), "r"(M), "r"(a0), "r"(b0), "r"(a1), "r"(b1), "r"(a2), "r"(b2));
        return r;
    }
    static inline uint32_t dot4(uint32_t a0, uint32_t b0, uint32_t a1, uint32_t b1,
                                uint32_t a2, uint32_t b2, uint32_t a3, uint32_t b3)
    {
        uint32_t r;
        asm(KB5_DOT(KB5_MUL(3, 4) KB5_MAD(5, 6) KB5_MAD(7, 8) KB5_MAD(9, 10) KB5_TRIM KB5_REDC)
            : "=r"(r) : "r"(MOD), "r"(M), "r"(a0), "r"(b0), "r"(a1), "r"(b1), "r"(a2), "r"(b2),
                        "r"(a3), "r"(b3));
        return r;
    }
    static inline uint32_t dot5(uint32_t a0, uint32_t b0, uint32_t a1, uint32_t b1,
                                uint32_t a2, uint32_t b2, uint32_t a3, uint32_t b3,
                                uint32_t a4, uint32_t b4)
    {
        uint32_t r;
        asm(KB5_DOT(KB5_MUL(3, 4) KB5_MAD(5, 6) KB5_MAD(7, 8) KB5_MAD(9, 10) KB5_TRIM
                    KB5_MAD(11, 12) KB5_TRIM KB5_REDC)
            : "=r"(r) : "r"(MOD), "r"(M), "r"(a0), "r"(b0), "r"(a1), "r"(b1), "r"(a2), "r"(b2),
                        "r"(a3), "r"(b3), "r"(a4), "r"(b4));
        return r;
    }
#  undef KB5_MUL
#  undef KB5_MAD
#  undef KB5_TRIM
#  undef KB5_REDC
#  undef KB5_DOT
# else
    static inline void mac(uint64_t& acc, uint32_t a, uint32_t b)
    {   acc += a * (uint64_t)b;   }
    static inline void trim(uint64_t& acc)
    {   if ((uint32_t)(acc >> 32) >= MOD) acc -= (uint64_t)MOD << 32;   }
    static inline uint32_t redc(uint64_t acc)
    {
        trim(acc);
        acc += ((uint32_t)acc * M) * (uint64_t)MOD;
        uint32_t r = (uint32_t)(acc >> 32);
        return r >= MOD ? r - MOD : r;
    }
    static inline uint32_t dot1(uint32_t a0, uint32_t b0)
    {   uint64_t acc = 0; mac(acc, a0, b0); return redc(acc);   }
    static inline uint32_t dot2(uint32_t a0, uint32_t b0, uint32_t a1, uint32_t b1)
    {   uint64_t acc = 0; mac(acc, a0, b0); mac(acc, a1, b1); return redc(acc);   }
    static inline uint32_t dot3(uint32_t a0, uint32_t b0, uint32_t a1, uint32_t b1,
                                uint32_t a2, uint32_t b2)
    {   uint64_t acc = 0; mac(acc, a0, b0); mac(acc, a1, b1); mac(acc, a2, b2); return redc(acc);   }
    static inline uint32_t dot4(uint32_t a0, uint32_t b0, uint32_t a1, uint32_t b1,
                                uint32_t a2, uint32_t b2, uint32_t a3, uint32_t b3)
    {
        uint64_t acc = 0;
        mac(acc, a0, b0); mac(acc, a1, b1); mac(acc, a2, b2); mac(acc, a3, b3);
        return redc(acc);
    }
    static inline uint32_t dot5(uint32_t a0, uint32_t b0, uint32_t a1, uint32_t b1,
                                uint32_t a2, uint32_t b2, uint32_t a3, uint32_t b3,
                                uint32_t a4, uint32_t b4)
    {
        uint64_t acc = 0;
        mac(acc, a0, b0); mac(acc, a1, b1); mac(acc, a2, b2); mac(acc, a3, b3);
        trim(acc);
        mac(acc, a4, b4);
        return redc(acc);
    }
# endif

    // r = c0..c8 folded by x^5 = 1 - x^2 (see the identities above):
    //   r0 = c0 + c5 - c8   r1 = c1 + c6   r2 = c2 - c5 + c7 + c8   r3 = c3 - c6 + c8   r4 = c4 - c7
    static inline kb31_5_t fold(const kb31_t (&p)[9])
    {
        kb31_5_t ret;
        ret.c[0] = p[0] + p[5] - p[8];
        ret.c[1] = p[1] + p[6];
        ret.c[2] = p[2] - p[5] + p[7] + p[8];
        ret.c[3] = p[3] - p[6] + p[8];
        ret.c[4] = p[4] - p[7];
        return ret;
    }

    // Polynomial multiplication modulo x^5 + x^2 - 1: schoolbook, 25 base multiplications,
    // c[k] = sum_{i+j=k} a[i]*b[j].
    inline kb31_5_t& mul(const kb31_5_t& b)
    {
        const uint32_t* a = u;
        const uint32_t* v = b.u;
        kb31_t p[9];
        p[0] = kb31_t{dot1(a[0], v[0])};
        p[1] = kb31_t{dot2(a[0], v[1], a[1], v[0])};
        p[2] = kb31_t{dot3(a[0], v[2], a[1], v[1], a[2], v[0])};
        p[3] = kb31_t{dot4(a[0], v[3], a[1], v[2], a[2], v[1], a[3], v[0])};
        p[4] = kb31_t{dot5(a[0], v[4], a[1], v[3], a[2], v[2], a[3], v[1], a[4], v[0])};
        p[5] = kb31_t{dot4(a[1], v[4], a[2], v[3], a[3], v[2], a[4], v[1])};
        p[6] = kb31_t{dot3(a[2], v[4], a[3], v[3], a[4], v[2])};
        p[7] = kb31_t{dot2(a[3], v[4], a[4], v[3])};
        p[8] = kb31_t{dot1(a[4], v[4])};
        return *this = fold(p);
    }

public:
    // Polynomial squaring modulo x^5 + x^2 - 1: 15 base multiplications (Plonky3 quintic_square).
    inline kb31_5_t& sqr()
    {
        const uint32_t* a = u;
        uint32_t d[5];  // 2*a[i]
        #pragma unroll
        for (int i = 0; i < 5; i++) d[i] = *(c[i] + c[i]);

        kb31_t p[9];
        p[0] = kb31_t{dot1(a[0], a[0])};
        p[1] = kb31_t{dot1(a[0], d[1])};
        p[2] = kb31_t{dot2(a[0], d[2], a[1], a[1])};
        p[3] = kb31_t{dot2(a[0], d[3], a[1], d[2])};
        p[4] = kb31_t{dot3(a[0], d[4], a[1], d[3], a[2], a[2])};
        p[5] = kb31_t{dot2(a[1], d[4], a[2], d[3])};
        p[6] = kb31_t{dot2(a[2], d[4], a[3], a[3])};
        p[7] = kb31_t{dot1(a[3], d[4])};
        p[8] = kb31_t{dot1(a[4], a[4])};
        return *this = fold(p);
    }

# ifdef __CUDACC_RDC__
    friend __device__ __noinline__ kb31_5_t operator*(kb31_5_t a, kb31_5_t b)
    {   return a.mul(b);   }
    inline kb31_5_t& operator*=(const kb31_5_t& b)
    {   return *this = *this * b;   }

    friend __device__ __noinline__ kb31_5_t operator*(kb31_5_t a, kb31_t b)
    {
        kb31_5_t ret;
        for (size_t i = 0; i < 5; i++)
            ret[i] = a[i] * b;
        return ret;
    }
    friend inline kb31_5_t operator*(kb31_t b, const kb31_5_t& a)
    {   return a * b;   }
    inline kb31_5_t& operator*=(kb31_t b)
    {   return *this = *this * b;   }
# else
    friend inline kb31_5_t operator*(kb31_5_t a, const kb31_5_t& b)
    {   return a.mul(b);   }
    inline kb31_5_t& operator*=(const kb31_5_t& b)
    {   return mul(b);   }

    inline kb31_5_t& operator*=(kb31_t b)
    {
        for (size_t i = 0; i < 5; i++)
            c[i] *= b;
        return *this;
    }
    friend inline kb31_5_t operator*(kb31_5_t a, kb31_t b)
    {   return a *= b;   }
    friend inline kb31_5_t operator*(kb31_t b, kb31_5_t a)
    {   return a *= b;   }
# endif

    friend inline kb31_5_t operator+(const kb31_5_t& a, const kb31_5_t& b)
    {
        kb31_5_t ret;
        for (size_t i = 0; i < 5; i++)
            ret[i] = a[i] + b[i];
        return ret;
    }
    inline kb31_5_t& operator+=(const kb31_5_t& b)
    {   return *this = *this + b;   }

    friend inline kb31_5_t operator+(const kb31_5_t& a, kb31_t b)
    {
        kb31_5_t ret = a;
        ret[0] = a[0] + b;
        return ret;
    }
    friend inline kb31_5_t operator+(kb31_t b, const kb31_5_t& a)
    {   return a + b;   }
    inline kb31_5_t& operator+=(kb31_t b)
    {   c[0] += b; return *this;   }

    friend inline kb31_5_t operator-(const kb31_5_t& a, const kb31_5_t& b)
    {
        kb31_5_t ret;
        for (size_t i = 0; i < 5; i++)
            ret[i] = a[i] - b[i];
        return ret;
    }
    inline kb31_5_t& operator-=(const kb31_5_t& b)
    {   return *this = *this - b;   }

    friend inline kb31_5_t operator-(const kb31_5_t& a, kb31_t b)
    {
        kb31_5_t ret = a;
        ret[0] = a[0] - b;
        return ret;
    }
    friend inline kb31_5_t operator-(kb31_t b, const kb31_5_t& a)
    {
        kb31_5_t ret;
        ret[0] = b - a[0];
        for (size_t i = 1; i < 5; i++)
            ret[i] = -a[i];
        return ret;
    }
    inline kb31_5_t& operator-=(kb31_t b)
    {   c[0] -= b; return *this;   }

    // x -> x^p. x^{kp} for k = 1..4 are fixed field elements (Plonky3's KoalaBear
    // FROBENIUS_COEFFS, here in Montgomery form), so phi(a) = a0 + sum_{k=1..4} a_k * x^{kp}.
    inline kb31_5_t frobenius() const
    {
        const uint32_t FC[4][5] = {
            {0x2D993495, 0x0E0C91E5, 0x21A60922, 0x4BFF57A3, 0x41B550BE},  // x^p
            {0x0A1C252C, 0x46D57E39, 0x30149A86, 0x3B2EE89F, 0x3181252B},  // x^{2p}
            {0x35FCAE15, 0x182D4E41, 0x4C4C1CDF, 0x074FCCCB, 0x4A813FF9},  // x^{3p}
            {0x291099B3, 0x3A76433C, 0x21B1646D, 0x6EC65580, 0x378F06CD},  // x^{4p}
        };
        kb31_5_t ret;
        #pragma unroll
        for (int j = 0; j < 5; j++)
            ret.u[j] = dot4(u[1], FC[0][j], u[2], FC[1][j], u[3], FC[2][j], u[4], FC[3][j]);
        ret.c[0] += c[0];
        return ret;
    }

    // Inverse by the Frobenius norm (Plonky3 pseudo_inv): with
    //   prod_conj = a^{p + p^2 + p^3 + p^4},  Norm(a) = a * prod_conj  (a base-field element),
    // a^{-1} = prod_conj / Norm(a). Zero maps to zero.
    inline kb31_5_t reciprocal() const
    {
        kb31_5_t a_p = frobenius();
        kb31_5_t t = (*this * a_p).frobenius();             // a^{p + p^2}
        kb31_5_t prod_conj = t * t.frobenius().frobenius(); // a^{p + p^2 + p^3 + p^4}

        // Only the constant coefficient of a * prod_conj: c0 + c5 - c8.
        const kb31_5_t& b = prod_conj;
        kb31_t norm = kb31_t{dot4(u[1], b.u[4], u[2], b.u[3], u[3], b.u[2], u[4], b.u[1])}
                    + c[0] * b.c[0] - c[4] * b.c[4];

        return prod_conj * norm.reciprocal();
    }
    friend inline kb31_5_t operator/(int one, const kb31_5_t& a)
    {   assert(one == 1); return a.reciprocal();   }
    friend inline kb31_5_t operator/(const kb31_5_t& a, const kb31_5_t& b)
    {   return a * b.reciprocal();   }
    friend inline kb31_5_t operator/(kb31_t a, const kb31_5_t& b)
    {   return b.reciprocal() * a;   }
    friend inline kb31_5_t operator/(const kb31_5_t& a, kb31_t b)
    {   return a * b.reciprocal();   }
    inline kb31_5_t& operator/=(const kb31_5_t& a)
    {   return *this *= a.reciprocal();   }
    inline kb31_5_t& operator/=(kb31_t a)
    {   return *this *= a.reciprocal();   }

    inline bool is_one() const
    {   return c[0].is_one() & u[1]==0 & u[2]==0 & u[3]==0 & u[4]==0;   }
    inline bool is_zero() const
    {   return u[0]==0 & u[1]==0 & u[2]==0 & u[3]==0 & u[4]==0;   }

    // raise to a variable power
    inline kb31_5_t& operator^=(uint32_t p)
    {
        kb31_5_t s = *this;

        if (!(p&1)) {
            c[0] = kb31_t{1};
            u[1] = u[2] = u[3] = u[4] = 0;
        }

        #pragma unroll 1
        while (p >>= 1) {
            s.sqr();
            if (p&1)
                mul(s);
        }

        return *this;
    }
    friend inline kb31_5_t operator^(kb31_5_t a, uint32_t p)
    {   return a ^= p;   }
    inline kb31_5_t operator()(uint32_t p)
    {   return *this^p;   }

    // raise to a constant power
    inline kb31_5_t& operator^=(int p)
    {
        assert(p >= 2);

        kb31_5_t s = *this;
        if ((p&1) == 0) {
            do {
                s.sqr();
                p >>= 1;
            } while ((p&1) == 0);
            *this = s;
        }
        for (p >>= 1; p; p >>= 1) {
            s.sqr();
            if (p&1)
                mul(s);
        }

        return *this;
    }
    friend inline kb31_5_t operator^(kb31_5_t a, int p)
    {   return a ^= p;   }
    inline kb31_5_t operator()(int p)
    {   return *this^p;   }

    friend inline bool operator==(const kb31_5_t& a, const kb31_5_t& b)
    {   return a.u[0]==b.u[0] & a.u[1]==b.u[1] & a.u[2]==b.u[2] & a.u[3]==b.u[3] & a.u[4]==b.u[4];   }
    friend inline bool operator!=(const kb31_5_t& a, const kb31_5_t& b)
    {   return a.u[0]!=b.u[0] | a.u[1]!=b.u[1] | a.u[2]!=b.u[2] | a.u[3]!=b.u[3] | a.u[4]!=b.u[4];   }

# undef inline
# if defined(_GLIBCXX_IOSTREAM) || defined(_IOSTREAM_)
    friend std::ostream& operator<<(std::ostream& os, const kb31_5_t& a)
    {
        os << "[" << a.c[0] << ", " << a.c[1] << ", " << a.c[2] << ", " << a.c[3] << ", " << a.c[4] << "]";
        return os;
    }
# endif
};

#endif  // __CUDACC__
#endif  // __SPPARK_FF_KOALA_BEAR_QUINTIC_HPP__
