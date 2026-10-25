/*
 * KoalaBear field implementation using mont32_t template
 * 
 * KoalaBear prime: p = 2^31 - 2^24 + 1 = 2130706433 = 0x7F000001
 * 2-adicity: 24 (p-1 = 2^24 * 127)
 * 
 * Montgomery constants:
 *   MOD = 0x7F000001  (the prime)
 *   M0  = 0x7EFFFFFF  (-p^{-1} mod 2^32)
 *   RR  = 0x17F7EFE4  (R^2 mod p, where R = 2^32)
 *   ONE = 0x01FFFFFE  (R mod p, i.e., 1 in Montgomery form)
 * 
 * NOTE: The optimized sqr_n path in mont32_t has a bug that affects KoalaBear.
 * We provide a corrected sqr_n here and use it in reciprocal().
 */

#ifndef __SPPARK_FF_KOALA_BEAR_HPP__
#define __SPPARK_FF_KOALA_BEAR_HPP__

#ifdef __CUDACC__
# include <cassert>
# include "ff/mont32_t.cuh"
# define inline __device__ __forceinline__

using kb31_base = mont32_t<31, 0x7F000001, 0x7EFFFFFF, 0x17F7EFE4, 0x01FFFFFE>;

struct kb31_t : public kb31_base {
    using mem_t = kb31_t;
    static constexpr uint32_t MOD = 0x7F000001;
    static constexpr uint32_t M0  = 0x7EFFFFFF;

    inline kb31_t() {}
    inline kb31_t(const kb31_base& a) : kb31_base(a) {}
    inline kb31_t(const uint32_t *p)  : kb31_base(p) {}
    // this is used in constant declaration, e.g. as kb31_t{11}
    __host__ __device__ constexpr kb31_t(int a)      : kb31_base(a) {}
    __host__ __device__ constexpr kb31_t(uint32_t a) : kb31_base(a) {}

    // Corrected squaring loop (avoids buggy mont32_t::sqr_n path)
    static inline kb31_t sqr_n(kb31_t s, uint32_t n)
    {
# if defined(__CUDA_ARCH__) && !defined(__clang_analyzer__)
#  ifdef __GNUC__
#   define KB_ASM __asm__ __volatile__
#  else
#   define KB_ASM asm volatile
#  endif
        while (n--) {
            uint32_t tmp0, tmp1, red;
            KB_ASM("mul.lo.u32 %0, %2, %2; mul.hi.u32 %1, %2, %2;"
                : "=r"(tmp0), "=r"(tmp1)
                : "r"(*s));
            KB_ASM("mul.lo.u32 %0, %1, %2;" : "=r"(red) : "r"(tmp0), "r"(M0));
            KB_ASM("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.u32 %1, %2, %3, %4;"
                : "+r"(tmp0), "=r"(tmp1)
                : "r"(red), "r"(MOD), "r"(tmp1));
            if (tmp1 >= MOD) tmp1 -= MOD;
            *s = tmp1;
        }
#  undef KB_ASM
# else
        while (n--) {
            s = s * s;
        }
# endif
        return s;
    }
    
    static inline kb31_t sqr_n_mul(kb31_t s, uint32_t n, kb31_t m)
    {
        s = sqr_n(s, n);
        return s * m;
    }
    
    // Compute x^(p-2) = x^{-1} using a shorter addition chain.
    //
    // Addition chain ported from pico-gpu's own `src/field/kb31_t.hpp`. It uses 36 field-mults vs
    // the previous 56 (~36% fewer), computing the identical value x^(p-2) (bit-exact — verified
    // numerically against the old chain). p - 2 = 0x7EFFFFFF = 2130706431 = 127 * 2^24 - 1.
    //
    // Soundness: this is pure repeated Montgomery squaring/multiply. Every
    // intermediate is fully reduced by the unconditional final_sub baked into
    // sqr_n (the same fix cuda-mle applies to mont32_t's buggy KB sqr_n), so
    // NO x^2*x < R*MOD overflow boundary is touched (unlike an sbox that skips
    // an intermediate final_sub). The chain is a straight swap of this function
    // body; sqr_n / sqr_n_mul and all call sites are unchanged.
    inline kb31_t reciprocal() const
    {
        kb31_t p11, p1100, p1111, p111111, p1111110000, p1111111111, ret = *this;

        p11          = sqr_n_mul(ret, 1, ret);       // ret^3
        p1100        = sqr_n(p11, 2);                // (ret^3)^(2^2) = ret^12
        p1111        = p1100 * p11;                  // ret^12 * ret^3 = ret^15
        // (ret^12)^(2^2) * ret^15 = ret^48 * ret^15 = ret^63
        p111111      = sqr_n_mul(p1100, 2, p1111);
        p1111110000  = sqr_n(p111111, 4);            // (ret^63)^(2^4) = ret^1008
        p1111111111  = p1111110000 * p1111;          // ret^1008 * ret^15 = ret^1023
        ret          = p1111111111 * p1111110000;    // ret^1023 * ret^1008 = ret^2031
        // (ret^2031)^(2^10) * ret^1023 = ret^2080767
        ret          = sqr_n_mul(ret, 10, p1111111111);
        // (ret^2080767)^(2^10) * ret^1023 = ret^2130706431 = ret^(p-2)
        ret          = sqr_n_mul(ret, 10, p1111111111);

        return ret;
    }

    inline kb31_t& operator^=(int p)
    {
        kb31_base::operator^=(static_cast<uint32_t>(p));
        return *this;
    }
    
    friend inline kb31_t operator/(int one, kb31_t a)
    {   assert(one == 1); return a.reciprocal();   }
    friend inline kb31_t operator/(kb31_t a, kb31_t b)
    {   return a * b.reciprocal();   }
    inline kb31_t& operator/=(const kb31_t a)
    {   *this *= a.reciprocal(); return *this;   }
};

# undef inline
#endif // __CUDACC__

#endif // __SPPARK_FF_KOALA_BEAR_HPP__
