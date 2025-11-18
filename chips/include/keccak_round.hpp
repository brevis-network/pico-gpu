#pragma once

#include "keccak_consts.hpp"
#include "utils.hpp"
#include "prelude.hpp"

namespace pico_gpu::keccak {

  /// Debug utility to print 5x5 u64 matrix.
  template <typename T>
  __device__ inline void print_matrix_u64(const char* name, T mat[5][5])
  {
    printf("%s =\n", name);
    for (int x = 0; x < 5; ++x) {
      for (int y = 0; y < 5; ++y)
        printf("0x%016llx ", (unsigned long long)mat[x][y]);
      printf("\n");
    }
  }

  /// Generate trace row for a single Keccak round.
  /// Implements the θ, ρ, π, χ, and ι steps of Keccak-f[1600].
  template <class F>
  __device__ inline void generate_trace_row_for_round(KeccakCols<F>& row, const int round, uint64_t state[5][5])
  {
    // Mark the current round.
    row.step_flags[round] = F::one();

    // Step θ: Compute parity columns C[x] = ⊕ A[x][y].
    uint64_t C[5];
    for (int x = 0; x < 5; ++x)
      C[x] = state[x][0] ^ state[x][1] ^ state[x][2] ^ state[x][3] ^ state[x][4];

    for (int x = 0; x < 5; ++x) {
      const array_t<F, 64> c_bits = u64_to_bits_le<F>(C[x]);
      for (int z = 0; z < 64; ++z)
        row.c[x][z] = c_bits[z];
    }

    // Compute C'[x] = C[x] ⊕ C[x-1] ⊕ ROT(C[x+1], 1).
    uint64_t Cprime[5];
    for (int x = 0; x < 5; ++x)
      Cprime[x] = C[x] ^ C[(x + 4) % 5] ^ rotl64(C[(x + 1) % 5], 1);

    for (int x = 0; x < 5; ++x) {
      const array_t<F, 64> c_prime_bits = u64_to_bits_le<F>(Cprime[x]);
      for (int z = 0; z < 64; ++z)
        row.c_prime[x][z] = c_prime_bits[z];
    }

    // Apply θ to state: A'[x][y] = A[x][y] ⊕ C[x] ⊕ C'[x].
    for (int x = 0; x < 5; ++x)
      for (int y = 0; y < 5; ++y)
        state[x][y] ^= C[x] ^ Cprime[x];

    for (int x = 0; x < 5; ++x)
      for (int y = 0; y < 5; ++y) {
        const array_t<F, 64> a_prime_bits = u64_to_bits_le<F>(state[x][y]);
        for (int z = 0; z < 64; ++z)
          row.a_prime[y][x][z] = a_prime_bits[z];
      }

    // Steps ρ and π: Rotate and permute into B[x][y].
    uint64_t B[5][5];
    for (int x = 0; x < 5; ++x)
      for (int y = 0; y < 5; ++y) {
        const int nx = (x + 3 * y) % 5;
        const int ny = x;
        B[x][y] = rotl64(state[nx][ny], R[nx][ny]);
      }

    // Step χ: Apply nonlinear mixing to produce A''.
    for (int x = 0; x < 5; ++x)
      for (int y = 0; y < 5; ++y)
        state[x][y] = B[x][y] ^ ((~B[(x + 1) % 5][y]) & B[(x + 2) % 5][y]);

    for (int x = 0; x < 5; ++x)
      for (int y = 0; y < 5; ++y) {
        const array_t<F, 4> a_prime_prime_limbs = u64_to_16_bit_limbs<F>(state[x][y]);
        for (int z = 0; z < 4; z++)
          row.a_prime_prime[y][x][z] = a_prime_prime_limbs[z];
      }

    // Split A''[0][0] into bits for round constant application.
    const array_t<F, 64> a_prime_prime_0_0_bits = u64_to_bits_le<F>(state[0][0]);
    for (int z = 0; z < 64; z++) {
      row.a_prime_prime_0_0_bits[z] = a_prime_prime_0_0_bits[z];
    }

    // Step ι: XOR round constant into A''[0][0] to get A'''[0][0].
    state[0][0] ^= RC[round];
    const array_t<F, 4> a_prime_prime_prime_0_0_limbs = u64_to_16_bit_limbs<F>(state[0][0]);
    for (int z = 0; z < 4; z++) {
      row.a_prime_prime_prime_0_0_limbs[z] = a_prime_prime_prime_0_0_limbs[z];
    }
  }

  /// Access A''' from a Keccak column structure.
  template <typename T>
  __device__ inline T keccak_Appp(const KeccakCols<T>& cols, const size_t y, const size_t x, const size_t limb)
  {
    assert(y < 5 && x < 5 && limb < U64_LIMBS);
    return (y == 0 && x == 0) ? cols.a_prime_prime_prime_0_0_limbs[limb] : cols.a_prime_prime[y][x][limb];
  }

  /// Generate trace rows for a complete Keccak-f[1600] permutation.
  /// Fills up to NUM_ROUNDS (24) rows, each representing one round.
  template <class F>
  __device__ inline void
  generate_trace_rows_for_perm(KeccakCols<F>* row_ptrs[NUM_ROUNDS], const int row_count, const uint64_t input[25])
  {
    // Initialize state from input (5x5 layout).
    uint64_t state[5][5];
    for (int y = 0; y < 5; ++y)
      for (int x = 0; x < 5; ++x)
        state[x][y] = input[y * 5 + x];

    // Write preimage to all rows and initial A-state to round 0.
    for (int y = 0; y < 5; ++y)
      for (int x = 0; x < 5; ++x) {
        const uint64_t v = input[y * 5 + x];

        for (int limb = 0; limb < U64_LIMBS; ++limb) {
          const uint16_t limb_val = (v >> (16 * limb)) & 0xFFFFu;
          const F fe = F::from_canonical_u16(limb_val);

          // Preimage is identical for all rows.
          for (int r = 0; r < row_count; ++r)
            row_ptrs[r]->preimage[y][x][limb] = fe;

          // Initial A goes only into round-0 row.
          row_ptrs[0]->a[y][x][limb] = fe;
        }
      }

    // Generate round 0.
    generate_trace_row_for_round(*row_ptrs[0], 0, state);

    // Generate rounds 1 through row_count-1.
    // Each round's A is taken from previous round's A'''.
    for (int r = 1; r < row_count; ++r) {
      KeccakCols<F>& cur = *row_ptrs[r];
      KeccakCols<F>& prev = *row_ptrs[r - 1];

      // Copy A'''(prev) → A(cur).
      for (int y = 0; y < 5; ++y)
        for (int x = 0; x < 5; ++x)
          for (int limb = 0; limb < U64_LIMBS; ++limb)
            cur.a[y][x][limb] = keccak_Appp(prev, y, x, limb);

      generate_trace_row_for_round(cur, r, state);
    }
  }
} // namespace pico_gpu::keccak