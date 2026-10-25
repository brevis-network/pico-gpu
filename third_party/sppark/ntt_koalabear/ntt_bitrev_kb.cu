/*
 * KoalaBear bit-reversal kernel.
 *
 * Full port of ntt_bitrev.cu (sppark) with fr_t = kb31_t, including the
 * shared-memory tiled `bit_rev_permutation_z` variant and its adaptive
 * dispatch. The earlier KB copy only had the naive per-element swap kernel
 * (scattered 4-byte reads/writes, ~0.5 TB/s); the `_z` kernel stages
 * Z_COUNT x Z_COUNT tiles through shared memory so both global reads and
 * writes are coalesced (analysis.md §12.3).
 * _bit_rev_frac_ext permutes (EF, EF) pairs as opaque 32-byte words. _bit_rev_ext and
 * _bit_rev_frac_ext_build_k2 are stubbed out since the KB extension field type
 * kb31_4_t is not yet available in this library (the KB build_k2 lives in gkr.cu).
 */

#include <cstdint>
#include "launcher.cuh"
#include "ntt/ntt.cuh"

template<unsigned int Z_COUNT>
__device__ __forceinline__ unsigned subgroup_sync_mask(uint32_t idx)
{
    if constexpr (Z_COUNT >= WARP_SIZE) {
        return 0xffffffffu;
    } else {
        // This kernel uses 1D thread blocks, so the hardware warp lane matches threadIdx.x.
        // Callers must keep each logical Z-group contiguous within threadIdx.x.
        uint32_t lane = threadIdx.x & (WARP_SIZE - 1);
        uint32_t subgroup_base = lane - idx;
        return (((uint32_t)1 << Z_COUNT) - 1u) << subgroup_base;
    }
}

// Permutes the data in an array such that data[i] = data[bit_reverse(i)]
// and data[bit_reverse(i)] = data[i]
template<typename T>
__launch_bounds__(1024) __global__
void bit_rev_permutation(T* d_out, const T *d_in, uint32_t lg_domain_size,
                         uint32_t padded_poly_size, uint32_t poly_count)
{
    const uint32_t poly_idx = blockIdx.y + blockIdx.z * gridDim.y;
    if (poly_idx >= poly_count)
        return;
    d_out += static_cast<size_t>(poly_idx) * padded_poly_size;
    d_in += static_cast<size_t>(poly_idx) * padded_poly_size;

    if (gridDim.x == 1 && blockDim.x == (1u << lg_domain_size)) {
        uint32_t idx = threadIdx.x;
        uint32_t rev = bit_rev(idx, lg_domain_size);

        T t = d_in[idx];
        if (d_out == d_in)
            __syncthreads();
        d_out[rev] = t;
    } else {
        index_t idx = threadIdx.x + blockDim.x * (index_t)blockIdx.x;
        index_t rev = bit_rev(idx, lg_domain_size);
        bool copy = d_out != d_in && idx == rev;

        if (idx < rev || copy) {
            T t0 = d_in[idx];
            if (!copy) {
                T t1 = d_in[rev];
                d_out[idx] = t1;
            }
            d_out[rev] = t0;
        }
    }
}

template<typename T, unsigned int Z_COUNT>
__launch_bounds__(192, 2) __global__
void bit_rev_permutation_z(T* out, const T* in, uint32_t lg_domain_size,
                           uint32_t padded_poly_size, uint32_t poly_count)
{
    const uint32_t poly_idx = blockIdx.y + blockIdx.z * gridDim.y;
    if (poly_idx >= poly_count)
        return;
    out += static_cast<size_t>(poly_idx) * padded_poly_size;
    in += static_cast<size_t>(poly_idx) * padded_poly_size;

    const uint32_t LG_Z_COUNT = 31 - __clz(Z_COUNT);

    // Use byte array for extern shared memory to avoid symbol conflicts across template
    // instantiations
    extern __shared__ unsigned char xchg_raw[];
    T (*xchg)[Z_COUNT][Z_COUNT] = reinterpret_cast<T (*)[Z_COUNT][Z_COUNT]>(xchg_raw);

    uint32_t gid = threadIdx.x / Z_COUNT;
    uint32_t idx = threadIdx.x % Z_COUNT;
    uint32_t rev = bit_rev(idx, LG_Z_COUNT);
    unsigned subgroup_mask = subgroup_sync_mask<Z_COUNT>(idx);

    index_t step = (index_t)1 << (lg_domain_size - LG_Z_COUNT);
    index_t tid = threadIdx.x + blockDim.x * (index_t)blockIdx.x;

    #pragma unroll 1
    do {
        index_t group_idx = tid >> LG_Z_COUNT;
        index_t group_rev = bit_rev(group_idx, lg_domain_size - 2*LG_Z_COUNT);

        if (group_idx > group_rev)
            continue;

        index_t base_idx = group_idx * Z_COUNT + idx;
        index_t base_rev = group_rev * Z_COUNT + idx;

        T regs[Z_COUNT];

        #pragma unroll
        for (uint32_t i = 0; i < Z_COUNT; i++) {
            xchg[gid][i][rev] = (regs[i] = in[i * step + base_idx]);
            if (group_idx != group_rev)
                regs[i] = in[i * step + base_rev];
        }

        if constexpr (Z_COUNT > WARP_SIZE)
            __syncthreads();
        else
            __syncwarp(subgroup_mask);

        #pragma unroll
        for (uint32_t i = 0; i < Z_COUNT; i++)
            out[i * step + base_rev] = xchg[gid][rev][i];

        if (group_idx == group_rev)
            continue;

        if constexpr (Z_COUNT > WARP_SIZE)
            __syncthreads();
        else
            __syncwarp(subgroup_mask);

        #pragma unroll
        for (uint32_t i = 0; i < Z_COUNT; i++)
            xchg[gid][i][rev] = regs[i];

        if constexpr (Z_COUNT > WARP_SIZE)
            __syncthreads();
        else
            __syncwarp(subgroup_mask);

        #pragma unroll
        for (uint32_t i = 0; i < Z_COUNT; i++)
            out[i * step + base_idx] = xchg[gid][rev][i];

    } while (Z_COUNT <= WARP_SIZE && (tid += blockDim.x*gridDim.x) < step);
    // without "Z_COUNT <= WARP_SIZE" compiler spills 128 bytes to stack
}

template<typename T>
static int bit_rev_impl(T* d_out, const T* d_inp,
    uint32_t lg_domain_size, uint32_t padded_poly_size, uint32_t poly_count)
{
    size_t domain_size = (size_t)1 << lg_domain_size;
    // aim to read 4 cache lines of consecutive data per read
    const uint32_t Z_COUNT = 256 / sizeof(T);
    const uint32_t bsize = Z_COUNT > WARP_SIZE ? Z_COUNT : WARP_SIZE;

    if (poly_count == 0)
        return cudaSuccess;

    const uint32_t MAX_Y = 65535;
    uint32_t grid_y = poly_count < MAX_Y ? poly_count : MAX_Y;
    uint32_t grid_z = (poly_count + grid_y - 1) / grid_y;

    if (domain_size <= 1024)
        bit_rev_permutation<T><<<dim3(1u, grid_y, grid_z), domain_size>>>
                            (d_out, d_inp, lg_domain_size, padded_poly_size, poly_count);
    else if (domain_size < bsize * Z_COUNT)
        bit_rev_permutation<T><<<dim3(domain_size / WARP_SIZE, grid_y, grid_z), WARP_SIZE>>>
                            (d_out, d_inp, lg_domain_size, padded_poly_size, poly_count);
    else if (Z_COUNT > WARP_SIZE || lg_domain_size <= 32)
        bit_rev_permutation_z<T, Z_COUNT><<<dim3(domain_size / Z_COUNT / bsize, grid_y, grid_z), bsize,
                                            bsize * Z_COUNT * sizeof(T)>>>
                            (d_out, d_inp, lg_domain_size, padded_poly_size, poly_count);
    else {
        // Those GPUs that can reserve 96KB of shared memory can
        // schedule 2 blocks to each SM...
        int device;
        cudaGetDevice(&device);
        int sm_count;
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);

        bit_rev_permutation_z<T, Z_COUNT><<<dim3(sm_count * 2, grid_y, grid_z), 192,
                                            192 * Z_COUNT * sizeof(T)>>>
                                (d_out, d_inp, lg_domain_size, padded_poly_size, poly_count);
    }

    return CHECK_KERNEL();
}

// ── Exported entry points ─────────────────────────────────────────────────────

// RS-encode fast-path handshake state, owned by rs_ntt_fastpath_kb.cu: when the
// fused iNTT+zeta launcher already wrote its result as the dense bit-reversed
// half into the per-thread scratch, this bit_rev call has nothing left to do.
extern thread_local fr_t* kb_rs_dense_scratch;
extern thread_local bool kb_rs_dense_prepared;
extern thread_local bool kb_rs_dense_pending;

extern "C" int _kb_bit_rev(fr_t* d_out, const fr_t* d_inp,
    uint32_t lg_domain_size, uint32_t padded_poly_size, uint32_t poly_count)
{
    if (kb_rs_dense_prepared) {
        kb_rs_dense_prepared = false;
        if (d_out == d_inp && lg_domain_size == 24 &&
            padded_poly_size == (1u << lg_domain_size)) {
            kb_rs_dense_pending = true;
            return cudaSuccess;
        }
        // Pipeline contract violated: the staged scratch holds the only copy
        // of the iNTT result, so silently falling through would corrupt data.
        if (kb_rs_dense_scratch != nullptr) {
            cudaFreeAsync(kb_rs_dense_scratch, cudaStreamPerThread);
            kb_rs_dense_scratch = nullptr;
        }
        return cudaErrorInvalidValue;
    }
    return bit_rev_impl(d_out, d_inp, lg_domain_size, padded_poly_size, poly_count);
}

// KB extension-field bit-reversal is not yet implemented.
// This stub satisfies the linker; the Rust side never calls it for KB.
extern "C" int _kb_bit_rev_ext(void*, const void*, uint32_t, uint32_t, uint32_t)
{
    return cudaErrorNotSupported;
}

// One (numerator, denominator) pair of degree-4 extension elements: 8 words, 32 bytes, 4-byte
// aligned -- the layout of Rust's `(EF, EF)`. The permutation only moves elements, so it needs the
// size and nothing of the field; words stay in Montgomery form.
struct kb_frac_ext_words_t {
    uint32_t w[8];
};
static_assert(sizeof(kb_frac_ext_words_t) == 32, "frac ext pair must be 32 bytes");

extern "C" int _kb_bit_rev_frac_ext(kb_frac_ext_words_t* d_out, const kb_frac_ext_words_t* d_inp,
    uint32_t lg_domain_size, uint32_t padded_poly_size, uint32_t poly_count)
{
    return bit_rev_impl(d_out, d_inp, lg_domain_size, padded_poly_size, poly_count);
}

extern "C" int _kb_bit_rev_frac_ext_build_k2(void*, uint32_t, fr_t)
{
    return cudaErrorNotSupported;
}
