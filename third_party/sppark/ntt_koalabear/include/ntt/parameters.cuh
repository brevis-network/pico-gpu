/*
 * KoalaBear NTT parameters, compiled into the combined `cuda_kb_all` static library.
 *
 * The supra NTT sources include "ntt/parameters.cuh"; mle/supra_kb/include comes first on
 * the include path, so they get this file. It:
 *   1. Defines the _kb_ symbol-prefix macros the Rust bindings name.
 *   2. Selects kb31_t as fr_t.
 *   3. Provides KoalaBear-specific NTT root-of-unity tables.
 */

// Rename exported symbols to the _kb_ names the Rust bindings use.
#define _ct_mixed_radix_narrow   _kb_ct_mixed_radix_narrow
#define _generate_all_twiddles   _kb_generate_all_twiddles
#define _generate_partial_twiddles _kb_generate_partial_twiddles
#define _bit_rev                 _kb_bit_rev
#define _bit_rev_ext             _kb_bit_rev_ext
#define _bit_rev_frac_ext        _kb_bit_rev_frac_ext
#define _bit_rev_frac_ext_build_k2 _kb_bit_rev_frac_ext_build_k2

#ifndef __SPPARK_NTT_PARAMETERS_CUH__
#define __SPPARK_NTT_PARAMETERS_CUH__

#include "ff/koala_bear.hpp"
using fr_t = kb31_t;

#define WARP_SIZE 32
#define MAX_LG_DOMAIN_SIZE 24
#define LG_WINDOW_SIZE ((MAX_LG_DOMAIN_SIZE + 4) / 5)
#define WINDOW_SIZE (1 << LG_WINDOW_SIZE)
#define WINDOW_NUM ((MAX_LG_DOMAIN_SIZE + LG_WINDOW_SIZE - 1) / LG_WINDOW_SIZE)

// KoalaBear NTT roots of unity (Montgomery form).
// P = 2130706433 = 0x7F000001, g = 3, R = 2^32.
// forward_roots[i] = g^((P-1)/2^i) * R mod P.
const fr_t forward_roots_of_unity[MAX_LG_DOMAIN_SIZE + 1] = {
    fr_t(0x01fffffeu),
    fr_t(0x7d000003u),
    fr_t(0x7b020407u),
    fr_t(0x60f5ef4du),
    fr_t(0x6d249c01u),
    fr_t(0x788529f3u),
    fr_t(0x07f7373eu),
    fr_t(0x6fe91d3cu),
    fr_t(0x3fd49211u),
    fr_t(0x1e056392u),
    fr_t(0x6d969babu),
    fr_t(0x439600ccu),
    fr_t(0x150276fcu),
    fr_t(0x68cacc36u),
    fr_t(0x42336c40u),
    fr_t(0x019b1972u),
    fr_t(0x34e52f6du),
    fr_t(0x1c2eb437u),
    fr_t(0x7cb65829u),
    fr_t(0x29306faeu),
    fr_t(0x351c7fa7u),
    fr_t(0x6e3e9a00u),
    fr_t(0x47c2bdf7u),
    fr_t(0x0c895820u),
    fr_t(0x13c85195u),
};

const fr_t inverse_roots_of_unity[MAX_LG_DOMAIN_SIZE + 1] = {
    fr_t(0x01fffffeu),
    fr_t(0x7d000003u),
    fr_t(0x03fdfbfau),
    fr_t(0x4bfa6163u),
    fr_t(0x52605cfeu),
    fr_t(0x19b8de8du),
    fr_t(0x29a9eda0u),
    fr_t(0x7c319486u),
    fr_t(0x6be0a64fu),
    fr_t(0x119f6035u),
    fr_t(0x78c55038u),
    fr_t(0x5c627d99u),
    fr_t(0x498aeddeu),
    fr_t(0x27052f97u),
    fr_t(0x7bf75488u),
    fr_t(0x2f8a590cu),
    fr_t(0x1dac17b7u),
    fr_t(0x4678e204u),
    fr_t(0x157bdbf0u),
    fr_t(0x74ca2cd0u),
    fr_t(0x06ee8434u),
    fr_t(0x16c4aa06u),
    fr_t(0x4aee72abu),
    fr_t(0x77640e35u),
    fr_t(0x452f7763u),
};

const fr_t domain_size_inverse[MAX_LG_DOMAIN_SIZE + 1] = {
    fr_t(0x01fffffeu),
    fr_t(0x00ffffffu),
    fr_t(0x40000000u),
    fr_t(0x20000000u),
    fr_t(0x10000000u),
    fr_t(0x08000000u),
    fr_t(0x04000000u),
    fr_t(0x02000000u),
    fr_t(0x01000000u),
    fr_t(0x00800000u),
    fr_t(0x00400000u),
    fr_t(0x00200000u),
    fr_t(0x00100000u),
    fr_t(0x00080000u),
    fr_t(0x00040000u),
    fr_t(0x00020000u),
    fr_t(0x00010000u),
    fr_t(0x00008000u),
    fr_t(0x00004000u),
    fr_t(0x00002000u),
    fr_t(0x00001000u),
    fr_t(0x00000800u),
    fr_t(0x00000400u),
    fr_t(0x00000200u),
    fr_t(0x00000100u),
};

typedef unsigned int index_t;

#define TWIDDLES_SIZE (32 + 64 + 128 + 256 + 512)

extern __constant__ fr_t FORWARD_TWIDDLES[TWIDDLES_SIZE];
extern __constant__ fr_t INVERSE_TWIDDLES[TWIDDLES_SIZE];
extern __constant__ fr_t FORWARD_PARTIAL_TWIDDLES[WINDOW_NUM][WINDOW_SIZE];
extern __constant__ fr_t INVERSE_PARTIAL_TWIDDLES[WINDOW_NUM][WINDOW_SIZE];

#endif /* __SPPARK_NTT_PARAMETERS_CUH__ */
