// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0

#pragma once

#include <cstdint>

namespace pico_gpu {
  /// Extension degree
  constexpr static const size_t EXTENSION_DEGREE = 4;
  constexpr static const size_t BASE_ALU_DATAPAR = 2;
  constexpr static const size_t EXT_ALU_DATAPAR = 4;
  /// Bit width of the Ecall chip's BitRange cap on num_extra_clk; mirrors Rust
  /// riscv_ecall::NUM_EXTRA_CLK_BITS.
  constexpr static const uint8_t NUM_EXTRA_CLK_BITS = 6;
  constexpr static const size_t VAR_MEM_DATAPAR = 4;
  constexpr static const size_t CONST_MEM_DATAPAR = 1;
  constexpr static const size_t SELECT_DATAPAR = 2;
  constexpr static const size_t POSEIDON2_DATAPAR = 1;

  /// Bit width of the global-interaction y[6] range check — the split point for the
  /// (lo16, mid, top) limb decomposition in the hash-to-curve gadget. Mirrors the Rust side.
  constexpr static const uint32_t Y6_RANGE_BITS = 30;

  /// Low-limb width (bits) of the y6 range-check split — y6_limbs[0] holds the low 16 bits.
  constexpr static const uint32_t Y6_LO_LIMB_BITS = 16;

  /// Chunk-clk budget: the memory-access timestamp diff is range-checked to this width as a
  /// 16-bit low limb plus a BitRange(9) high limb (those sites hardcode the limb widths); the Cpu
  /// clk limbs use the same split as a representation only (the row chain bounds clk, see Rust
  /// riscv_cpu/chunk_clk).
  /// The consumers of this constant are the Global chip's 16 + 8 + 1 split of `message[0]`,
  /// whose boolean high limb carries bit `CHUNK_CLK_BITS - 1`, and `GLOBAL_KIND_SHIFT` below.
  /// Mirrors the Rust
  /// `ChunkSplitConfig::MAX_CHUNK_CLK_BITS`; a chunk may hold up to 2^(CHUNK_CLK_BITS-2)
  /// instructions (max chunk clk = 4 * CHUNK_SIZE).
  constexpr static const uint32_t CHUNK_CLK_BITS = 25;

  /// The global-interaction digest packs the interaction kind above the chunk-clk-sized message
  /// element: `values[0] + kind * 2^GLOBAL_KIND_SHIFT`. A kind below 32 keeps the sum under 2^30:
  /// (2^25 - 1) + 31 * 2^25 = 2^30 - 1 < p. Comfortably sufficient rather than tight -- the
  /// modulus alone would permit a larger kind (asserted at keygen on the Rust side).
  constexpr static const uint32_t GLOBAL_KIND_SHIFT = CHUNK_CLK_BITS;

  /// Ed25519
  constexpr static const size_t WORD_SIZE = 4;
  constexpr static const size_t ED25519_LIMBS = 32;
  constexpr static const size_t ED25519_WITNESS = 62;
  constexpr static const size_t WORDS_FIELD_ELEM = ED25519_LIMBS / 4;
  constexpr static const size_t WORDS_CURVE_POINT = ED25519_LIMBS / 2;
  constexpr static const size_t BYTES_FIELD_ELEM = WORDS_FIELD_ELEM * WORD_SIZE;
  constexpr static const size_t COMPRESSED_POINT_BYTES = 32;

  /// The number of bits in a byte.
  constexpr static const uintptr_t BYTE_SIZE = 8;

  /// The width of the permutation.
  constexpr static const uintptr_t WIDTH = 16;

  constexpr static const uintptr_t STATE_SIZE = 25;

  /// The number of bytes necessary to represent a 64-bit integer.
  constexpr static const uintptr_t LONG_WORD_SIZE = (2 * WORD_SIZE);

  /// The number of 32 bit words in the Pico proof's committed value digest.
  constexpr static const uintptr_t PV_DIGEST_NUM_WORDS = 8;

  constexpr static const uintptr_t DIGEST_SIZE = 8;

  constexpr static const uintptr_t PERMUTATION_WIDTH = 16;

  /// Committed permutation inputs per riscv Poseidon2 slot (the absorption rate; the capacity
  /// lanes are AIR constants). Mirrors Rust `RISCV_POSEIDON2_NUM_INPUTS`.
  constexpr static const uintptr_t RISCV_POSEIDON2_NUM_INPUTS = 8;
  /// Committed permutation outputs per riscv Poseidon2 slot: the lanes the Poseidon2 lookup
  /// reads. Mirrors Rust `RISCV_POSEIDON2_NUM_OUTPUTS`.
  constexpr static const uintptr_t RISCV_POSEIDON2_NUM_OUTPUTS = 7;

  /// RISC-V X0 register
  constexpr static const uint32_t REGISTER_X0 = 0;

  /// An opcode (short for "operation code") specifies the operation to be performed by the processor.
  ///
  /// In the context of the RISC-V ISA, an opcode specifies which operation (i.e., addition,
  /// subtraction, multiplication, etc.) to perform on up to three operands such as registers,
  /// immediates, or memory addresses.
  ///
  /// While the Pico zkVM targets the RISC-V ISA, it uses a custom instruction encoding that uses
  /// a different set of opcodes. The main difference is that the Pico zkVM encodes register
  /// operations and immediate operations as the same opcode. For example, the RISC-V opcodes ADD and
  /// ADDI both become ADD inside the Pico zkVM. We utilize flags inside the instruction itself to
  /// distinguish between the two.
  ///
  /// Refer to the "RV32I Reference Card" [here](https://github.com/johnwinans/rvalp/releases) for
  /// more details.
  enum class Opcode : uint8_t {
    /// rd ← rs1 + rs2, pc ← pc + 4
    ADD = 0,
    /// rd ← rs1 - rs2, pc ← pc + 4
    SUB = 1,
    /// rd ← rs1 ^ rs2, pc ← pc + 4
    XOR = 2,
    /// rd ← rs1 | rs2, pc ← pc + 4
    OR = 3,
    /// rd ← rs1 & rs2, pc ← pc + 4
    AND = 4,
    /// rd ← rs1 << rs2, pc ← pc + 4
    SLL = 5,
    /// rd ← rs1 >> rs2 (logical), pc ← pc + 4
    SRL = 6,
    /// rd ← rs1 >> rs2 (arithmetic), pc ← pc + 4
    SRA = 7,
    /// rd ← (rs1 < rs2) ? 1 : 0 (signed), pc ← pc + 4
    SLT = 8,
    /// rd ← (rs1 < rs2) ? 1 : 0 (unsigned), pc ← pc + 4
    SLTU = 9,
    /// rd ← sx(m8(rs1 + imm)), pc ← pc + 4
    LB = 10,
    /// rd ← sx(m16(rs1 + imm)), pc ← pc + 4
    LH = 11,
    /// rd ← sx(m32(rs1 + imm)), pc ← pc + 4
    LW = 12,
    /// rd ← zx(m8(rs1 + imm)), pc ← pc + 4
    LBU = 13,
    /// rd ← zx(m16(rs1 + imm)), pc ← pc + 4
    LHU = 14,
    /// m8(rs1 + imm) ← rs2[7:0], pc ← pc + 4
    SB = 15,
    /// m16(rs1 + imm) ← rs2[15:0], pc ← pc + 4
    SH = 16,
    /// m32(rs1 + imm) ← rs2[31:0], pc ← pc + 4
    SW = 17,
    /// pc ← pc + ((rs1 == rs2) ? imm : 4)
    BEQ = 18,
    /// pc ← pc + ((rs1 != rs2) ? imm : 4)
    BNE = 19,
    /// pc ← pc + ((rs1 < rs2) ? imm : 4) (signed)
    BLT = 20,
    /// pc ← pc + ((rs1 >= rs2) ? imm : 4) (signed)
    BGE = 21,
    /// pc ← pc + ((rs1 < rs2) ? imm : 4) (unsigned)
    BLTU = 22,
    /// pc ← pc + ((rs1 >= rs2) ? imm : 4) (unsigned)
    BGEU = 23,
    /// rd ← pc + 4, pc ← pc + imm
    JAL = 24,
    /// rd ← pc + 4, pc ← (rs1 + imm) & ∼1
    JALR = 25,
    /// rd ← pc + imm, pc ← pc + 4
    ///
    /// Kept to stay one-to-one with the Rust `Opcode` enum, but no compiled program carries it
    /// since X12: the transpiler re-encodes `AUIPC rd, imm` as the two-immediate
    /// `ADD rd, imm, pc`. Nothing in these headers branches on it.
    AUIPC = 27,
    /// Transfer control to the debugger.
    ECALL = 28,
    /// Transfer control to the operating system.
    EBREAK = 29,
    /// rd ← rs1 * rs2 (signed), pc ← pc + 4
    MUL = 30,
    /// rd ← rs1 * rs2 (half), pc ← pc + 4
    MULH = 31,
    /// rd ← rs1 * rs2 (half unsigned), pc ← pc + 4
    MULHU = 32,
    /// rd ← rs1 * rs2 (half signed unsigned), pc ← pc + 4
    MULHSU = 33,
    /// rd ← rs1 / rs2 (signed), pc ← pc + 4
    DIV = 34,
    /// rd ← rs1 / rs2 (unsigned), pc ← pc + 4
    DIVU = 35,
    /// rd ← rs1 % rs2 (signed), pc ← pc + 4
    REM = 36,
    /// rd ← rs1 % rs2 (unsigned), pc ← pc + 4
    REMU = 37,
    /// Unimplemented instruction.
    UNIMP = 39,
    /// RV64 W-variant opcodes
    ADDW = 40,
    SUBW = 41,
    SLLW = 42,
    SRLW = 43,
    SRAW = 44,
    LWU = 45,
    LD = 46,
    SD = 47,
    MULW = 48,
    DIVW = 49,
    DIVUW = 50,
    REMW = 51,
    REMUW = 52,
  };

  /// AluEvent matching the Rust u64 AluEvent layout (#[repr(C)]).
  /// Fields: clk (u64), opcode (u8 + 7-byte padding), a/b/c (u64), is_imm (u8 @40),
  /// is_cmp (u8 @41: merged branch comparison, `a = lt | gt << 16`), 6-byte padding.
  struct AluEventU64 {
    uint64_t clk;
    Opcode opcode;
    uint8_t _pad[7];
    uint64_t a;
    uint64_t b;
    uint64_t c;
    bool is_imm;
    bool is_cmp;
    uint8_t _pad2[6];
  };

  enum class ByteOpcode : uint8_t {
    AND = 0,
    OR = 1,
    XOR = 2,
    SLL = 3,
    ShrCarry = 4,
    LTU = 5,
    MSB = 6,
    U8Range = 7,
    U16Range = 8,
    BitRange = 9,
  };

  /// System Calls.
  ///
  /// A system call is invoked by the the `ecall` instruction with a specific value in register t0.
  /// The syscall number is a 32-bit integer with the following little-endian layout:
  ///
  /// | Byte 0 | Byte 1 | Byte 2 | Byte 3 |
  /// | ------ | ------ | ------ | ------ |
  /// |   ID   | Table  | Cycles | Unused |
  ///
  /// where:
  /// - Byte 0: The system call identifier.
  /// - Byte 1: Whether the handler of the system call has its own table. This is used in the CPU
  ///   table to determine whether to lookup the syscall using the syscall interaction.
  /// - Byte 2: The number of additional cycles the syscall uses. This is used to make sure the # of
  ///   memory accesses is bounded.
  /// - Byte 3: Currently unused.
  enum class SyscallCode : uint32_t {
    /// Halts the program.
    HALT = 0,
    /// Write to the output buffer.
    WRITE = 2,
    /// Enter unconstrained block.
    ENTER_UNCONSTRAINED = 3,
    /// Exit unconstrained block.
    EXIT_UNCONSTRAINED = 4,
    /// Executes the `SHA_EXTEND` precompile.
    SHA_EXTEND = 3145989,
    /// Executes the `SHA_COMPRESS` precompile.
    SHA_COMPRESS = 65798,
    /// Executes the `ED_ADD` precompile.
    ED_ADD = 65799,
    /// Executes the `ED_DECOMPRESS` precompile.
    ED_DECOMPRESS = 264,
    /// Executes the `KECCAK_PERMUTE` precompile.
    KECCAK_PERMUTE = 65801,
    /// Executes the `SECP256K1_ADD` precompile.
    SECP256K1_ADD = 65802,
    /// Executes the `SECP256K1_DOUBLE` precompile.
    SECP256K1_DOUBLE = 267,
    /// Executes the `SECP256K1_DECOMPRESS` precompile.
    SECP256K1_DECOMPRESS = 268,
    /// Executes the `BN254_ADD` precompile.
    BN254_ADD = 65806,
    /// Executes the `BN254_DOUBLE` precompile.
    BN254_DOUBLE = 271,
    /// Executes the `COMMIT` precompile.
    COMMIT = 16,
    /// Executes the `COMMIT_DEFERRED_PROOFS` precompile.
    COMMIT_DEFERRED_PROOFS = 26,
    /// Executes the `VERIFY_PICO_PROOF` precompile.
    VERIFY_PICO_PROOF = 27,
    /// Executes the `BLS12381_DECOMPRESS` precompile.
    BLS12381_DECOMPRESS = 284,
    /// Executes the `HINT_LEN` precompile.
    HINT_LEN = 240,
    /// Executes the `HINT_READ` precompile.
    HINT_READ = 241,
    /// Executes the `UINT256_MUL` precompile.
    UINT256_MUL = 65821,
    /// Executes the `U256XU2048_MUL` precompile.
    U256XU2048_MUL = 65839,
    /// Executes the `BLS12381_ADD` precompile.
    BLS12381_ADD = 65822,
    /// Executes the `BLS12381_DOUBLE` precompile.
    BLS12381_DOUBLE = 287,
    /// Executes the `BLS12381_FP_ADD` precompile.
    BLS12381_FP_ADD = 65824,
    /// Executes the `BLS12381_FP_SUB` precompile.
    BLS12381_FP_SUB = 65825,
    /// Executes the `BLS12381_FP_MULprecompile.
    BLS12381_FP_MUL = 65826,
    /// Executes the `BLS12381_FP2_ADD` precompile.
    BLS12381_FP2_ADD = 65827,
    /// Executes the `BLS12381_FP2_SUB` precompile.
    BLS12381_FP2_SUB = 65828,
    /// Executes the `BLS12381_FP2_MUL` precompile.
    BLS12381_FP2_MUL = 65829,
    /// Executes the `BN254_FP_ADD` precompile.
    BN254_FP_ADD = 65830,
    /// Executes the `BN254_FP_SUB` precompile.
    BN254_FP_SUB = 65831,
    /// Executes the `BN254_FP_MUL` precompile.
    BN254_FP_MUL = 65832,
    /// Executes the `BN254_FP2_ADD` precompile.
    BN254_FP2_ADD = 65833,
    /// Executes the `BN254_FP2_SUB` precompile.
    BN254_FP2_SUB = 65834,
    /// Executes the `BN254_FP2_MUL` precompile.
    BN254_FP2_MUL = 65835,
    /// Executes the `SECP256R1_ADD` precompile.
    SECP256R1_ADD = 65840,
    /// Executes the `SECP256R1_DOUBLE` precompile.
    SECP256R1_DOUBLE = 305,
    /// Executes the `SECP256R1_DECOMPRESS` precompile.
    SECP256R1_DECOMPRESS = 306,
  };

  enum class Register : uint8_t {
    /// %x0
    X0 = 0,
    /// %x1
    X1 = 1,
    /// %x2
    X2 = 2,
    /// %x3
    X3 = 3,
    /// %x4
    X4 = 4,
    /// %x5
    X5 = 5,
    /// %x6
    X6 = 6,
    /// %x7
    X7 = 7,
    /// %x8
    X8 = 8,
    /// %x9
    X9 = 9,
    /// %x10
    X10 = 10,
    /// %x11
    X11 = 11,
    /// %x12
    X12 = 12,
    /// %x13
    X13 = 13,
    /// %x14
    X14 = 14,
    /// %x15
    X15 = 15,
    /// %x16
    X16 = 16,
    /// %x17
    X17 = 17,
    /// %x18
    X18 = 18,
    /// %x19
    X19 = 19,
    /// %x20
    X20 = 20,
    /// %x21
    X21 = 21,
    /// %x22
    X22 = 22,
    /// %x23
    X23 = 23,
    /// %x24
    X24 = 24,
    /// %x25
    X25 = 25,
    /// %x26
    X26 = 26,
    /// %x27
    X27 = 27,
    /// %x28
    X28 = 28,
    /// %x29
    X29 = 29,
    /// %x30
    X30 = 30,
    /// %x31
    X31 = 31,
  };


  /// An array of four bytes to represent a 32-bit value.
  ///
  /// We use the generic type `T to represent the different representations of a byte, ranging from
  /// a `u8` to a `AB::Var` or `AB::Expr`.
  template <typename T>
  struct Word {
    T _0[WORD_SIZE];
  };











  // ─── U64 Memory types ────────────────────────────────────────────────

  struct MemoryRecordU64 {
    uint32_t chunk;
    uint32_t timestamp;
    uint64_t value;
  };

  /// Mirrors the Rust `FfiMemoryLocalEvent` (riscv_memory/event.rs): the emulator's entry plus
  /// two values the host decides and the kernel only consumes.
  struct MemoryLocalEventU64 {
    uint64_t addr;
    MemoryRecordU64 initial_mem_access;
    MemoryRecordU64 final_mem_access;
    /// Whether the entry receives its initial state from the Global bus (the `has_initial`
    /// column). 0 on a fresh entry -- the first touch of a zero-initial main-memory word outside
    /// the program image -- which declares its initial state instead and writes no initial
    /// Global event.
    uint32_t has_initial;
    /// Number of Global events the entries before this one in the record write (two each, one
    /// for a fresh entry): where this entry's events go, relative to the chip's start index.
    uint32_t global_slot;
  };

  template <typename T>
  struct SingleMemoryLocalU64 {
    T addr[3];
    T initial_chunk;
    T final_chunk;
    T initial_clk;
    T final_clk;
    Word<T> initial_value;
    Word<T> final_value;
    T initial_value_lower;
    T initial_value_upper;
    T final_value_lower;
    T final_value_upper;
    T is_real;
    /// Whether this entry receives its initial state from the Global bus; zero on a fresh entry,
    /// which declares a zero initial state instead (the AIR pins initial_value to zero on it).
    /// Zero on padding rows, so it doubles as the degree-1 multiplicity of the Global receive.
    T has_initial;
    /// Whether this entry's interval starts inside this chunk (initial_chunk == final_chunk).
    /// Only the chunk's first interval for an address inherits its start from an earlier chunk;
    /// a syscall splits an address's activity into further intervals that all start here.
    T initial_in_chunk;
    /// The clk this entry puts on the regional memory bus for its initial state: the real
    /// initial_clk when the interval starts here, the sentinel 0 when it starts earlier.
    /// The Global message keeps the real value either way.
    T initial_clk_regional;
    /// final_clk - initial_clk - 1 in a 16 + 9 decomposition, proving initial_clk < final_clk
    /// on entries that start in this chunk. Also rules out the entry whose two Global messages
    /// carry the same payload, which is otherwise free to add.
    T clk_diff_16bit_limb;
    T clk_diff_hi9;
  };

  struct MemoryInitializeFinalizeEventU64 {
    uint64_t addr;
    uint64_t value;
    uint32_t chunk;
    uint32_t timestamp;
    uint32_t used;
    uint32_t _pad;
  };

  template <typename T>
  struct LtWordU16Gadget {
    T u16_flags[WORD_SIZE];
    T not_eq_inv;
    T comparison_limbs[2];
    T bit;
  };

  // ─── End U64 Memory types (column structs below after IsZeroOperation) ───

  /// Operation columns for verifying that an element is within the range `[0, modulus)`.
  template <typename T, uintptr_t N>
  struct AssertLtColsBits {
    /// Boolean flags to indicate the first byte in which the element is smaller than the modulus.
    T bit_flags[N];
  };

  /// A set of columns needed to compute whether the given word is 0.
  template <typename T>
  struct IsZeroOperation {
    /// The inverse of the input.
    T inverse;
    /// Result indicating whether the input is 0. This equals `inverse * input == 0`.
    T result;
  };

  template <typename T>
  struct MemoryInitializeFinalizeColsU64 {
    T chunk;
    T timestamp;
    T addr[3];
    T prev_addr[3];
    LtWordU16Gadget<T> lt_cols;
    Word<T> value;
    T value_lower;
    T value_upper;
    T is_real;
    T is_comp;
    IsZeroOperation<T> is_prev_addr_zero;
  };

  template <typename T>
  struct IsZeroWordOperation {
    /// `IsZeroOperation` to check if each byte in the input word is zero.
    IsZeroOperation<T> is_zero_byte[WORD_SIZE];
    /// A boolean flag indicating whether the lower word (the bottom 16 bits of the input) is 0.
    /// This equals `is_zero_byte[0] * is_zero_byte[1]`.
    T is_lower_half_zero;
    /// A boolean flag indicating whether the upper word (the top 16 bits of the input) is 0. This
    /// equals `is_zero_byte[2] * is_zero_byte[3]`.
    T is_upper_half_zero;
    /// A boolean flag indicating whether the word is zero. This equals `is_zero_byte[0] * ... *
    /// is_zero_byte[WORD_SIZE - 1]`.
    T result;
  };

  template <typename T>
  struct IsEqualWordOperation {
    /// An operation to check whether the differences in limbs are all 0 (i.e., `a[0] - b[0]`,
    /// `a[1] - b[1]`, `a[2] - b[2]`, `a[3] - b[3]]`). The result of `IsEqualWordOperation` is
    /// `is_diff_zero.result`.
    IsZeroWordOperation<T> is_diff_zero;
  };

  template <typename T>
  struct DivRemCols {
    /// The output operand.
    Word<T> a;
    /// The first input operand.
    Word<T> b;
    /// The second input operand.
    Word<T> c;
    /// Results of dividing `b` by `c`.
    Word<T> quotient;
    /// Remainder when dividing `b` by `c`.
    Word<T> remainder;
    /// `abs(remainder)`, used to check `abs(remainder) < abs(c)`.
    Word<T> abs_remainder;
    /// `abs(c)`, used to check `abs(remainder) < abs(c)`.
    Word<T> abs_c;
    /// `max(abs(c), 1)`, used to check `abs(remainder) < abs(c)`.
    Word<T> max_abs_c_or_1;
    /// The result of `c * quotient`.
    T c_times_quotient[LONG_WORD_SIZE];
    /// Carry propagated when adding `remainder` by `c * quotient`.
    T carry[LONG_WORD_SIZE];
    /// Flag to indicate division by 0.
    IsZeroWordOperation<T> is_c_0;
    /// Flag to indicate whether the opcode is DIV.
    T is_div;
    /// Flag to indicate whether the opcode is DIVU.
    T is_divu;
    /// Flag to indicate whether the opcode is REM.
    T is_rem;
    /// Flag to indicate whether the opcode is REMU.
    T is_remu;
    /// Flag to indicate whether the division operation overflows.
    ///
    /// Overflow occurs in a specific case of signed 32-bit integer division: when `b` is the
    /// minimum representable value (`-2^31`, the smallest negative number) and `c` is `-1`. In
    /// this case, the division result exceeds the maximum positive value representable by a
    /// 32-bit signed integer.
    T is_overflow;
    /// Flag for whether the value of `b` matches the unique overflow case `b = -2^31` and `c =
    /// -1`.
    IsEqualWordOperation<T> is_overflow_b;
    /// Flag for whether the value of `c` matches the unique overflow case `b = -2^31` and `c =
    /// -1`.
    IsEqualWordOperation<T> is_overflow_c;
    /// The most significant bit of `b`.
    T b_msb;
    /// The most significant bit of remainder.
    T rem_msb;
    /// The most significant bit of `c`.
    T c_msb;
    /// Flag to indicate whether `b` is negative.
    T b_neg;
    /// Flag to indicate whether `rem_neg` is negative.
    T rem_neg;
    /// Flag to indicate whether `c` is negative.
    T c_neg;
    /// Selector to determine whether an ALU Event is sent for absolute value computation of `c`.
    T abs_c_alu_event;
    /// Selector to determine whether an ALU Event is sent for absolute value computation of `rem`.
    T abs_rem_alu_event;
    /// Selector to know whether this row is enabled.
    T is_real;
    /// Column to modify multiplicity for remainder range check event.
    T remainder_check_multiplicity;
  };

  template <typename T>
  struct FieldBitDecomposition {
    /// The bit decoposition of the`value`.
    T bits[32];
    /// The upper bits are all ones.
    IsZeroOperation<T> upper_all_one;
  };

  template <typename T>
  struct FieldWordRangeChecker {
    /// 1 iff value[1] < TOP_LIMB, where TOP_LIMB = (p - 1) >> 16.
    T ms_limb_lt_top;
    /// Mersenne31 only: the inverse witnessing value[0] != 0xFFFF on rows where the most
    /// significant limb is maxed. Pinned to zero on KoalaBear, where the tail
    /// case is a plain value[0] == 0 assertion that needs no witness.
    T low_limb_not_max_inv;
  };

  /// Memory Read Record.
  ///
  /// This object encapsulates the information needed to prove a memory read operation. This
  /// includes the value, chunk, timestamp, and previous chunk and timestamp.
  struct MemoryReadRecord {
    /// The value.
    uint32_t value;
    /// The chunk number.
    uint32_t chunk;
    /// The timestamp.
    uint32_t timestamp;
    /// The previous chunk number.
    uint32_t prev_chunk;
    /// The previous timestamp.
    uint32_t prev_timestamp;
  };

  /// Memory Write Record.
  ///
  /// This object encapsulates the information needed to prove a memory write operation. This
  /// includes the value, chunk, timestamp, previous value, previous chunk, and previous timestamp.
  struct MemoryWriteRecord {
    /// The value.
    uint32_t value;
    /// The chunk number.
    uint32_t chunk;
    /// The timestamp.
    uint32_t timestamp;
    /// The previous value.
    uint32_t prev_value;
    /// The previous chunk number.
    uint32_t prev_chunk;
    /// The previous timestamp.
    uint32_t prev_timestamp;
  };

  /// U16→U8 auxiliary gadget: stores the low bytes from each u16 limb of a Word<T>.
  template <typename T>
  struct U16ToU8Gadget {
    T low_bytes[WORD_SIZE];
  };

  /// U64 SyscallEvent matching Rust #[repr(C)] layout (40 bytes).
  struct SyscallEventU64 {
    uint32_t chunk;
    uint32_t _pad1;
    uint64_t clk;
    uint32_t syscall_id;
    uint32_t _pad2;
    uint64_t arg1;
    uint64_t arg2;
  };

  /// U64 SyscallCols: no chunk, arg1/arg2 are Addr<T> = [T; 3] (9 field elements).
  template <typename T>
  struct SyscallColsU64 {
    // Leads the syscall bus tuple. clk restarts in every chunk, so without this the tuple
    // cannot tell two chunks apart.
    T chunk;
    T clk;
    T syscall_id;
    T arg1[3];
    T arg2[3];
    T is_real;
  };

  /// The syscall-decode gadget block of the Ecall chip.
  /// Must match Rust riscv_ecall/columns.rs::EcallCols (#[repr(C)]).
  template <typename T>
  struct EcallColsU64 {
    /// u16 -> u8 decomposition of op_a prev_value (syscall code); low_bytes[WORD_SIZE].
    U16ToU8Gadget<T> syscall_code_bytes;
    /// Whether the current ecall is ENTER_UNCONSTRAINED.
    IsZeroOperation<T> is_enter_unconstrained;
    /// Whether the current ecall is HINT_LEN.
    IsZeroOperation<T> is_hint_len;
    /// Whether the current ecall is HALT.
    IsZeroOperation<T> is_halt;
    /// Whether the current ecall is a COMMIT.
    IsZeroOperation<T> is_commit;
    /// Whether the current ecall is a COMMIT_DEFERRED_PROOFS.
    IsZeroOperation<T> is_commit_deferred_proofs;
    /// index_bitmap[word index] set to 1 for the COMMIT ecall, everything else 0.
    T index_bitmap[PV_DIGEST_NUM_WORDS];
    /// Columns to range check the halt/commit_deferred_proofs operand.
    FieldWordRangeChecker<T> operand_range_check_cols;
    /// The operand value to range check.
    Word<T> operand_to_check;
  };

  /// The whole column layout for the Ecall chip.
  /// Must match Rust riscv_ecall/columns.rs::EcallChipCols (#[repr(C)]).
  template <typename T>
  struct EcallChipColsU64 {
    /// The current chunk (message element 0).
    T chunk;
    /// The clock cycle of the ecall (message element 1).
    T clk;
    /// op_a prev_value - the syscall code word.
    Word<T> op_a_prev_value;
    /// op_a value after the ecall.
    Word<T> op_a_value;
    /// op_b value.
    Word<T> op_b_value;
    /// op_c value.
    Word<T> op_c_value;
    /// The Cpu-committed extra-clock scalar (byte 2 of the syscall code).
    T num_extra_clk;
    /// The Cpu-committed halt flag.
    T is_halt;
    /// send_to_table byte x is_real: multiplicity of the syscall-table send.
    T send_to_table;
    /// (is_halt or is_commit_deferred_proofs) x is_real: gates the operand range check.
    T range_check_operand;
    /// is_hint_len x is_real: the multiplicity of the u16 range check on op_a_value (HINT_LEN
    /// rows only), materialized like range_check_operand.
    T hint_len_mul;
    /// The four bytes of the committed_value_digest public-value word selected by
    /// ecall.index_bitmap, mirrored into the main trace so the u8 range-check interaction
    /// message is affine in current-row main columns. Only meaningful on COMMIT rows.
    Word<T> commit_digest_bytes;
    /// is_real x is_commit: the multiplicity of the digest-byte range check, materialized
    /// because an interaction count must be affine in current-row columns too.
    T commit_mul;
    /// is_real x is_commit_deferred_proofs, and the running count and sum of 2^word_index of
    /// such rows up to this one. The last row's pair is the chunk's published tally.
    T deferred_commit_mul;
    T deferred_commits;
    T deferred_commit_mask;
    /// The syscall-decode machinery.
    EcallColsU64<T> ecall;
    /// Whether this row is an actual ecall (multiplicity of the looked message).
    T is_real;
  };
  // Pinned against the Rust `NUM_ECALL_CHIP_COLS == 60` assertion in riscv_ecall/columns.rs.
  static_assert(sizeof(EcallChipColsU64<char>) == 60,
                "EcallChipColsU64 must stay field-for-field with Rust EcallChipCols");

  /// A 64-bit instruction operand whose bits 32..64 are a sign fill: the two low u16 limbs plus
  /// one sign bit (1 when bits 32..64 are all one, 0 when all zero), so limbs 2 and 3 of the
  /// operand word are `0xFFFF * sign` each. The Rust `SignExtImm::from_u64` asserts the
  /// invariant; the GPU writers take bit 63 and do not. Mirrors Rust SignExtImm.
  template <typename T>
  struct SignExtImm {
    T limb[2];
    T sign;
  };

  /// The column layout for instructions.
  template <typename T>
  struct InstructionCols {
    /// The opcode for this cycle.
    T opcode;
    /// The first operand for this instruction (register index; single limb).
    T op_a;
    /// The second operand: a register index or a sign-extended immediate.
    SignExtImm<T> op_b;
    /// The third operand: a register index or a sign-extended immediate.
    SignExtImm<T> op_c;
    /// Flags to indicate if op_a is register 0.
    T op_a_0;
  };

  template <typename T>
  struct MemoryAccessCols {
    /// The value of the memory access.
    Word<T> value;
    /// The clk of the previous access to this address, within this chunk. Carries the sentinel
    /// 0 when the previous access was in an earlier chunk; neither the previous chunk nor a
    /// selector between the two comparisons is committed any more.
    T prev_clk;
    /// The least significant 16 bit limb of clk - prev_clk - 1.
    T diff_16bit_limb;
    /// Bits 16..25 of the timestamp difference, merged into a single 9-bit limb (
    /// was an 8-bit limb + a boolean bit-24 column), BitRange(9)-checked. 25-bit chunk-clk
    /// budget. Mirrors Rust MemoryAccessCols.
    T diff_hi9;
  };

  /// Memory read-write access.
  template <typename T>
  struct MemoryReadWriteCols {
    Word<T> prev_value;
    MemoryAccessCols<T> access;
    __host__ __device__ __forceinline__ Word<T>& value_mut() { return access.value; }
  };

  /// Memory read access.
  template <typename T>
  struct MemoryReadCols {
    MemoryAccessCols<T> access;
    __host__ __device__ __forceinline__ Word<T>& value_mut() { return access.value; }
  };

  template <typename T>
  struct MemoryWriteCols {
    Word<T> prev_value;
    MemoryAccessCols<T> access;

    __host__ __device__ __forceinline__ Word<T>& value_mut() { return access.value; }
  };

  // ─── Lite access layout (Cpu register accesses, MemoryReadWrite) ───────
  // MemoryAccessCols without the prev_clk column. The predecessor clk is not a witness: the AIR
  // defines it from the two difference limbs as clk - (diff_16bit_limb + diff_hi9 * 2^16) - 1.
  // The three structs above stay for the precompile chips. Mirrors Rust MemoryAccessLiteCols /
  // MemoryReadLiteCols / MemoryReadWriteLiteCols.

  template <typename T>
  struct MemoryAccessLiteCols {
    /// The value of the memory access.
    Word<T> value;
    /// The least significant 16 bit limb of clk - prev_clk - 1.
    T diff_16bit_limb;
    /// Bits 16..25 of that difference, a single 9-bit limb, BitRange(9)-checked.
    T diff_hi9;
  };

  /// Memory read access without a committed prev_clk.
  template <typename T>
  struct MemoryReadLiteCols {
    MemoryAccessLiteCols<T> access;
    __host__ __device__ __forceinline__ Word<T>& value_mut() { return access.value; }
  };

  /// Memory read-write access without a committed prev_clk.
  template <typename T>
  struct MemoryReadWriteLiteCols {
    Word<T> prev_value;
    MemoryAccessLiteCols<T> access;
    __host__ __device__ __forceinline__ Word<T>& value_mut() { return access.value; }
  };



  template <typename T>
  struct FfiOption {
    enum class Tag {
      None,
      Some,
    };

    struct Some_Body {
      T _0;
    };

    Tag tag;
    union {
      Some_Body some;
    };
  };



  // ─── U64 Memory Read/Write column types ────────────────────────────

  template <typename T>
  struct MemoryInstructionColsU64 {
    // `opcode` is de-materialized (the dispatch message element is
    // the degree-1 expression sum(flag_i * OPCODE_i) on the AIR side).
    T op_a_0;
    T is_lb;
    T is_lbu;
    T is_lh;
    T is_lhu;
    T is_lw;
    T is_lwu;
    T is_ld;
    T is_sb;
    T is_sh;
    T is_sw;
    T is_sd;
    // Register-operand values. Mirrors the Rust
    // MemoryInstructionCols: the full access structs were replaced by plain
    // value words; the register accesses are proven on the Cpu chip's rows.
    Word<T> op_a_value;
    Word<T> op_b_value;
    Word<T> op_c_value;
  };

  // Field-for-field mirror of the Rust MemoryChipValueCols (49 columns):
  // 2 + 2 + 3 + 1 + 10 (lite access) + 7 scalars + 24 (instruction).
  template <typename F>
  struct MemoryChipValueColsU64 {
    F chunk;
    F clk;
    // Only the low 2 u16 limbs of the address sum are committed; the addition is proven in-chip
    // by a carry chain whose 3rd and 4th result limbs are literal 0s (guest addresses lie below
    // 2^32, Rust read_write/columns.rs).
    F addr_word[2];
    // The one-hot limb selector is not committed: the AIR forms the four degree-2 products of
    // these bits directly.
    F offset_bit[3];
    // Inverse of addr_word[1] for the stack guard (addr_limb1_inv * addr_word[1] == is_mem).
    F addr_limb1_inv;
    // Lite layout: no committed prev_clk (see MemoryAccessLiteCols).
    MemoryReadWriteLiteCols<F> memory_access;
    // The u16 limb selected by (bit1, bit2) from the memory value; also the low half of an
    // LW/LWU result (word alignment forces bit1 = 0 there).
    F selected_limb;
    F selected_byte;
    F msb;
    // Shared U16MSB operand: selected_limb on LH rows, the high half of the selected word on
    // LW and LWU rows, 0 elsewhere.
    F lhw_val;
    // Low byte of op_a[0] on SB rows and of selected_limb on LB/LBU rows.
    F register_low_byte;
    // SB: low byte of the memory limb being modified.
    F mem_limb_low_byte;
    // The value added to the selected limb by a sub-limb store: is_sh * (op_a[0] - selected_limb)
    // + is_sb * increment. Zero on load, SW/SD and padding rows; may be field-negative on SH/SB.
    F shift;
    // `store_value` / `unsigned_mem_val` are de-materialized (asserted
    // directly on memory_access.value / op_a on the AIR side).
    MemoryInstructionColsU64<F> instruction;
  };

  // ─── End U64 Memory Read/Write column types ────────────────────────

  /// FFI CPU Event.
  ///
  /// This object is a mirror of CpuEvent but has a stable ABI for FFI operations
  // ─── U64 CPU Event types ──────────────────────────────────────────────

  struct InstructionU64 {
    Opcode opcode;
    uint8_t op_a;
    uint8_t _pad[6];
    uint64_t op_b;
    uint64_t op_c;
    bool imm_b;
    bool imm_c;
    uint8_t _pad2[6];
  };

  struct MemoryReadRecordU64 {
    uint64_t value;
    uint32_t chunk;
    uint32_t timestamp;
    uint32_t prev_chunk;
    uint32_t prev_timestamp;
  };

  struct MemoryWriteRecordU64 {
    uint64_t value;
    uint32_t chunk;
    uint32_t timestamp;
    uint64_t prev_value;
    uint32_t prev_chunk;
    uint32_t prev_timestamp;
  };

  class MemoryRecordEnumU64 {
  public:
    enum class Tag {
      Read,
      Write,
    };

    struct Read_Body {
      MemoryReadRecordU64 _0;
    };

    struct Write_Body {
      MemoryWriteRecordU64 _0;
    };

    Tag tag;
    union {
      Read_Body read;
      Write_Body write;
    };

    __host__ __device__ __forceinline__ uint64_t value() const {
      if (tag == Tag::Read) return read._0.value;
      else return write._0.value;
    }

    __host__ __device__ __forceinline__ uint32_t prev_chunk() const {
      if (tag == Tag::Read) return read._0.prev_chunk;
      else return write._0.prev_chunk;
    }

    __host__ __device__ __forceinline__ uint32_t prev_timestamp() const {
      if (tag == Tag::Read) return read._0.prev_timestamp;
      else return write._0.prev_timestamp;
    }

    __host__ __device__ __forceinline__ uint32_t chunk() const {
      if (tag == Tag::Read) return read._0.chunk;
      else return write._0.chunk;
    }

    __host__ __device__ __forceinline__ uint32_t timestamp() const {
      if (tag == Tag::Read) return read._0.timestamp;
      else return write._0.timestamp;
    }
  };

  struct FfiCpuEventU64 {
    uint32_t chunk;
    uint32_t _pad1;
    uint64_t clk;
    uint64_t pc;
    uint64_t next_pc;
    InstructionU64 instruction;
    uint64_t a;
    FfiOption<MemoryRecordEnumU64> a_record;
    uint64_t b;
    FfiOption<MemoryRecordEnumU64> b_record;
    uint64_t c;
    FfiOption<MemoryRecordEnumU64> c_record;
    FfiOption<uint64_t> memory;
    FfiOption<MemoryRecordEnumU64> memory_record;
    uint32_t exit_code;
    uint32_t _pad2;
  };

  // Slimmed version of FfiCpuEventU64 for the GPU trace kernel.
  // Excludes memory, memory_record, and exit_code which are unused by the
  // riscv_cpu kernel, reducing per-event size from 304 to 232 bytes (~24%).
  struct FfiCpuEventSlimU64 {
    uint32_t chunk;
    uint32_t _pad1;
    uint64_t clk;
    uint64_t pc;
    uint64_t next_pc;
    InstructionU64 instruction;
    uint64_t a;
    FfiOption<MemoryRecordEnumU64> a_record;
    uint64_t b;
    FfiOption<MemoryRecordEnumU64> b_record;
    uint64_t c;
    FfiOption<MemoryRecordEnumU64> c_record;
  };

  template <typename F>
  struct Addr {
    F _0[3];
  };

  /// A program counter as 2 u16 limbs. A guest address stays below 0x7800_0000, so the third limb
  /// an Addr carries is identically zero on every pc and is not materialized; Addr remains the
  /// type of a memory address. Mirrors Rust Pc.
  template <typename F>
  struct Pc {
    F _0[2];
  };



  // ─── U64 CPU Column types ────────────────────────────────────────────

  template <class F>
  struct OpcodeSelectorColsU64 {
    F imm_b;
    F imm_c;
    // `is_alu` is derived in the constraints (one-hot complement of the other class
    // selectors, backed by the Program-table lookup), so it is no longer a committed column.
    F is_ecall;
    // per-variant memory flags moved to MemoryReadWriteChip (self-proven there);
    // the Cpu chip keeps only is_memory (+ is_store for the register-write constraint).
    F is_memory;
    F is_store;
    F is_beq;
    F is_bne;
    F is_blt;
    F is_bge;
    F is_bltu;
    F is_bgeu;
    F is_jalr;
    F is_jal;
    // No trap selector: the Program table holds no row for a UNIMP / EBREAK pc (program.cuh), so
    // no Cpu row can carry one (Rust riscv_cpu/opcode_selector/columns.rs).
  };

  template <class F>
  struct CpuColsU64 {
    // No chunk column: the AIR's messages carry the `execution_chunk` public value (Rust
    // riscv_cpu/columns.rs).
    // One `clk` column, not range-checked: the row chain bounds clk (Rust riscv_cpu/chunk_clk).
    F clk;
    F num_extra_clk;
    // Two limbs each (Rust `Pc`): a guest address fits in 32 bits.
    Pc<F> pc;
    Pc<F> next_pc;
    InstructionCols<F> instruction;
    OpcodeSelectorColsU64<F> opcode_selector;
    // Lite access layout: no committed prev_clk, the AIR defines it from the two difference
    // limbs (see MemoryAccessLiteCols).
    MemoryReadWriteLiteCols<F> op_a_access;
    MemoryReadLiteCols<F> op_b_access;
    MemoryReadLiteCols<F> op_c_access;
    // Branch compare flags (formerly the BranchCols view of the opcode_specific union).
    // The former a_eq_b column is derived (1 - a_lt_b - a_gt_b).
    F a_gt_b;
    F a_lt_b;
    // committed "ecall is HALT" flag, certified by the Ecall-chip lookup.
    F is_halt;
    F is_real;
    /// The chain's sink (Rust `CpuCols::is_last`). Written by the HOST after the slice launches,
    /// never by a kernel: `cpu_events_to_trace_kernel` is launched once per slice, so a
    /// kernel-side "last index" test would mark the last row of every slice.
    F is_last;
    // Multiplicity of the ALU dispatch: is_alu_instruction * (1 - op_a_0). Precomputed
    // host-side because a lookup multiplicity stays degree 1. Since X12 this also covers the
    // rows that used to be AUIPC: the transpiler re-encodes `AUIPC rd, imm` as the
    // two-immediate `ADD rd, imm, pc`, so the separate is_auipc_not_x0 column is gone.
    F is_alu_not_x0;
    // Bit 0 of the unmasked JALR target rs1 + imm; next_pc holds the masked value. The AIR
    // proves b + c = (next_pc0 + lsb, next_pc1, 0, 0) in-chip (Rust jump/constraints.rs).
    F jalr_lsb;
    // `is_sequential_instr` is derived (1 - branch - jal - jalr - is_halt); no column.
    // [pc0 >= 0xFFFC] on every real row: the carry out of limb 0 in pc+4. The Program message
    // carries it, so it is pinned to the preprocessed table's value (ProgramPreprocessedColsU64
    // below); its one reader is a jump row's return address pc+4=op_a. Only limb 0 of that
    // addition can carry (a guest address stays below 0x7800_0000), so limb 1 is a plain
    // equality and op_a's limb 2 is pinned to zero (a pc has no limb 2). Mirrors Rust
    // CpuCols::pc_carry.
    F pc_carry;
  };

  // ────────────────────────────────────────────────────────────────────

  /// A block of columns for septic extension.
  template <typename T>
  struct SepticBlock {
    T _0[7];
  };

  /// A set of columns needed to compute the global interaction elliptic curve digest.
  /// Must match Rust GlobalInteractionOperation #[repr(C)].
  template <typename T>
  struct GlobalInteractionOperation {
    // Hash-to-curve offset, a single byte (range-checked via a U8Range byte lookup).
    T offset;
    SepticBlock<T> x_coordinate;
    SepticBlock<T> y_coordinate;
    // Compact limb decomposition [lo16, mid, top] of the y_6 range-check value, validated by byte
    // lookups (replaces the 30-bit decomposition + inverse witness). The Poseidon2 input (m_trial)
    // and the full output are re-derived / bound via the lookup rather than committed.
    T y6_limbs[3];
  };

  /// Global Interaction Event.
  ///
  /// This event is emitted for all interactions that are sent or received across different chunks.
  /// Mirrors Rust's GlobalInteractionEvent #[repr(C)] with message: [u32; 8].
  struct GlobalInteractionEvent {
    /// The message (8 elements in u64; was 7 in u32).
    uint32_t message[8];
    /// Whether the interaction is received or sent.
    bool is_receive;
    /// The kind of the interaction event.
    uint8_t kind;
  };

  /// A set of columns needed to compute the global interaction elliptic curve digest.
  /// Must match Rust GlobalAccumulationOperation #[repr(C)]: `lambda[i]` is the slope witness of
  /// the chord from `cum[i]` to point i, `cum[i]` the running sum BEFORE point i is added (the
  /// sum after the row is the next row's `cum[0]`; the last row is always padding).
  /// It is critical that this struct is at the end of the main trace, as the permutation constraints
  /// will be dependent on this fact. It is also critical that `cum` is at the end of this struct,
  /// for the same reason.
  template <typename T, uintptr_t N>
  struct GlobalAccumulationOperation {
    SepticBlock<T> lambda[N];
    SepticBlock<T> cum[N][2];
  };

  template <typename T>
  struct GlobalCols {
    T message[8];
    T kind;
    T message_0_16bit_limb;
    T message_0_8bit_limb;
    /// The most significant (boolean) bit of message[0] (bit 24) — 25-bit widening, mirrors Rust.
    T message_0_high_bit;
    GlobalInteractionOperation<T> interaction;
    T is_real;
    T is_receive;
    T is_send;
    GlobalAccumulationOperation<T, 1> accumulation;
  };
  // Column budget the riscv_global TPC kernels hardcode (33 fixed + 7 lambda + 14 cum). The field
  // type is a 4-byte word, so the u32 instantiation has the same layout.
  static_assert(sizeof(GlobalCols<uint32_t>) == 54 * sizeof(uint32_t), "GlobalCols must be 54 columns");

  template <typename T>
  struct BytePreprocessedCols {
    /// The first byte operand.
    T b;
    /// The second byte operand.
    T c;
    /// The result of the `AND` operation on `b` and `c`
    T and_result;
    /// The result of the `OR` operation on `b` and `c`
    T or_result;
    /// The result of the `XOR` operation on `b` and `c`
    T xor_result;
    /// The result of the `SLL` operation on `b` and `c`
    T sll;
    /// The result of the `ShrCarry` operation on `b` and `c`
    T shr;
    T shr_carry;
    /// The result of the `LTU` operation on `b` and `c`
    T ltu;
    /// The most significant bit of `b`.
    T msb;
    /// A u16 value used for `U16Range`.
    T value_u16;
    /// The number of bits for `BitRange`.
    T bits;
  };

  struct ByteLookupEvent {
    ByteOpcode opcode;
    uint16_t a1;
    uint8_t a2;
    uint8_t b;
    uint8_t c;
  };

  template <typename T>
  struct ProgramPreprocessedColsU64 {
    Pc<T> pc;                           // 2 field elements (u16 limbs)
    InstructionCols<T> instruction;
    OpcodeSelectorColsU64<T> selectors;
    // [pc0 >= 0xFFFC]: the carry out of limb 0 in pc+4, a function of the row's pc. The Program
    // message carries it, pinning every real Cpu row's pc_carry (Rust riscv_program/columns.rs).
    T pc_carry;
  };

  template <typename T>
  struct ProgramMultCols {
    T multiplicity;
  };

  struct Poseidon2Event {
    uint32_t input[WIDTH];
    uint32_t output[WIDTH];
  };

  template <typename F>
  using Address = F;

  template <typename T>
  struct Poseidon2Io {
    T input[WIDTH];
    T output[WIDTH];
  };

  template <typename F>
  using RecursionPoseidon2Event = Poseidon2Io<F>;

  /// An instruction invoking the Poseidon2 permutation.
  template <typename F>
  struct Poseidon2SkinnyInstr {
    Poseidon2Io<Address<F>> addrs;
    F mults[WIDTH];
  };

  template <typename T>
  struct FullRound {
    T post[WIDTH];
  };

  template <typename T>
  struct PartialRound {
    T post_sbox;
  };

  template <typename T, const int NumHalfFullRounds, const int NumInternalRounds>
  struct Poseidon2ValueCols {
    T is_real;
    T input[WIDTH];
    FullRound<T> beginning_full_rounds[NumHalfFullRounds];
    PartialRound<T> partial_rounds[NumInternalRounds];
    FullRound<T> ending_full_rounds[NumHalfFullRounds];
  };

  /// Last ending full round of the riscv Poseidon2 wrapper: only the lanes the Poseidon2 lookup
  /// reads are committed. Must match Rust `RiscvLastFullRound` #[repr(C)] (KoalaBear: no S-box
  /// registers).
  template <typename T>
  struct RiscvLastFullRound {
    T post[RISCV_POSEIDON2_NUM_OUTPUTS];
  };

  /// riscv Poseidon2 wrapper slot: only the rate-8 inputs are committed (the capacity lanes are
  /// AIR constants) and the last ending full round commits only the looked-up output lanes. Must
  /// match Rust `RiscvPoseidon2ValueCols` #[repr(C)]; the recursion chips and the riscv Poseidon2
  /// precompile keep the full-width `Poseidon2ValueCols`.
  template <typename T, const int NumHalfFullRounds, const int NumInternalRounds>
  struct RiscvPoseidon2ValueCols {
    T is_real;
    T input[RISCV_POSEIDON2_NUM_INPUTS];
    FullRound<T> beginning_full_rounds[NumHalfFullRounds];
    PartialRound<T> partial_rounds[NumInternalRounds];
    FullRound<T> ending_full_rounds[NumHalfFullRounds - 1];
    RiscvLastFullRound<T> last_full_round;
  };
  // KoalaBear (4 half full rounds, 20 partial rounds): 1 + 8 + 4*16 + 20 + 3*16 + 7 = 148.
  static_assert(
    sizeof(RiscvPoseidon2ValueCols<uint32_t, 4, 20>) == 148 * sizeof(uint32_t),
    "RiscvPoseidon2ValueCols<KoalaBear> must be 148 columns");

  struct Poseidon2PermuteEvent {
    uint32_t chunk;
    uint32_t clk;
    uint32_t state_values[WIDTH];
    uint32_t input_memory_ptr;
    uint32_t output_memory_ptr;
    MemoryReadRecord state_read_records[WIDTH];
    MemoryWriteRecord state_write_records[WIDTH];
    // local_mem_access is not needed in generate main
  };

  template <typename T, const int NumHalfFullRounds, const int NumInternalRounds>
  struct Poseidon2Cols {
    T chunk;
    T clk;
    T input_memory_ptr;
    MemoryReadCols<T> input_memory[WIDTH];

    T output_memory_ptr;
    MemoryWriteCols<T> output_memory[WIDTH];

    Poseidon2ValueCols<T, NumHalfFullRounds, NumInternalRounds> value_cols;
  };

  // Simplified ByteRecordBehavior for C++ (Rust equivalent would be more complex)
  template <typename T>
  struct ByteRecordBehavior {
    // In a full implementation, this would track byte-level lookup events
    // For now, it's a placeholder to match the Rust interface

    __host__ __device__ void add_byte_lookup_event(/* ByteLookupEvent event */)
    {
      // Placeholder - would record byte operations for constraint system
    }
  };

  enum class BaseAluOpcode : uint8_t {
    AddF = 0,
    SubF = 1,
    MulF = 2,
    DivF = 3,
  };

  /// The inputs and outputs to an operation of the base field ALU.
  template <typename V>
  struct BaseAluIo {
    V out;
    V in1;
    V in2;
  };

  template <typename F>
  using BaseAluEvent = BaseAluIo<F>;

  /// An instruction invoking the base field ALU.
  template <typename F>
  struct BaseAluInstr {
    BaseAluOpcode opcode;
    F mult;
    BaseAluIo<Address<F>> addrs;
  };

  template <typename F>
  struct BaseAluAccessCols {
    BaseAluIo<Address<F>> addrs;
    F is_add;
    F is_sub;
    F is_mul;
    F is_div;
    F mult;
  };

  template <typename F>
  struct BaseAluPreprocessedCols {
    BaseAluAccessCols<F> accesses[BASE_ALU_DATAPAR];
  };

  template <typename F>
  struct BaseAluValueCols {
    BaseAluIo<F> vals;
  };

  template <typename F>
  struct BaseAluCols {
    BaseAluValueCols<F> values[BASE_ALU_DATAPAR];
  };

  /// The smallest unit of memory that can be read and written to.
  template <typename T>
  struct Block {
    T _0[EXTENSION_DEGREE];
  };

  enum class ExtAluOpcode : uint8_t {
    AddE = 0,
    SubE = 1,
    MulE = 2,
    DivE = 3,
  };

  /// The inputs and outputs to an operation of the extension field ALU.
  template <typename V>
  struct ExtAluIo {
    V out;
    V in1;
    V in2;
  };

  template <typename F>
  using ExtAluEvent = ExtAluIo<Block<F>>;

  /// An instruction invoking the extension field ALU.
  template <typename F>
  struct ExtAluInstr {
    ExtAluOpcode opcode;
    F mult;
    ExtAluIo<Address<F>> addrs;
  };

  template <typename F>
  struct ExtAluAccessCols {
    ExtAluIo<Address<F>> addrs;
    F is_add;
    F is_sub;
    F is_mul;
    F is_div;
    F mult;
  };

  template <typename F>
  struct ExtAluPreprocessedCols {
    ExtAluAccessCols<F> accesses[EXT_ALU_DATAPAR];
  };

  template <typename F>
  struct ExtAluValueCols {
    ExtAluIo<Block<F>> vals;
  };

  template <typename F>
  struct ExtAluCols {
    ExtAluValueCols<F> values[EXT_ALU_DATAPAR];
  };

  // AssertZeroExt: asserts that an extension-field value equals zero.
  // Layouts are kept byte-compatible with the Rust `#[repr(C)]` columns so the trace buffers can be
  // copied host-to-device with a raw memcpy; defined here alongside every other chip's layouts.

  /// Event: the EF value that should be zero.
  template <typename F>
  struct AssertZeroExtEvent {
    Block<F> val;
  };

  /// Instruction: the address of the value to check.
  template <typename F>
  struct AssertZeroExtInstr {
    Address<F> addr;
  };

  /// Preprocessed trace columns.
  template <typename F>
  struct AssertZeroExtPreprocessedCols {
    F is_real;
    Address<F> addr;
  };

  /// Main trace columns: just the value.
  template <typename F>
  struct AssertZeroExtCols {
    Block<F> val;
  };

  // MleOpen: asserts a specific (air_id, col_id) MLE column opening equals the claimed value.
  // Layouts byte-compatible with the Rust `#[repr(C)]` columns for the raw host-to-device copy.

  /// Event: the claimed opening value for the (air_id, col_id) column.
  template <typename F>
  struct MleOpenEvent {
    uint32_t col_ref; // encoded: air_id << 16 | col_id
    Block<F> opening_value;
  };

  /// Instruction: the encoded column ref and the address the opening value is read from.
  template <typename F>
  struct MleOpenInstr {
    uint32_t col_ref;
    Address<F> opening_value_addr;
  };

  /// Preprocessed trace columns.
  template <typename F>
  struct MleOpenPreprocessedCols {
    F is_real;
    F col_ref;
    Address<F> opening_value_addr;
  };

  /// Main trace columns: the claimed MLE opening value.
  template <typename F>
  struct MleOpenCols {
    Block<F> opening_value;
  };

  // FracFold (GKR fractional fold): (p1,q1) + (p2,q2) = (p1*q2 + p2*q1, q1*q2).
  // Layouts are kept byte-compatible with the Rust `#[repr(C)]` columns for the raw host-to-device copy.

  /// Event: the six EF values (four inputs + the two folded outputs).
  template <typename F>
  struct FracFoldEvent {
    Block<F> p1;
    Block<F> q1;
    Block<F> p2;
    Block<F> q2;
    Block<F> out_p;
    Block<F> out_q;
  };

  /// Instruction: input/output addresses and the output write multiplicity.
  template <typename F>
  struct FracFoldInstr {
    Address<F> in_p1_addr;
    Address<F> in_q1_addr;
    Address<F> in_p2_addr;
    Address<F> in_q2_addr;
    Address<F> out_p_addr;
    Address<F> out_q_addr;
    F mult;
  };

  /// Preprocessed trace columns.
  template <typename F>
  struct FracFoldPreprocessedCols {
    F is_real;
    Address<F> in_p1_addr;
    Address<F> in_q1_addr;
    Address<F> in_p2_addr;
    Address<F> in_q2_addr;
    Address<F> out_p_addr;
    Address<F> out_q_addr;
    F mult;
  };

  /// Main trace columns: the six EF values.
  template <typename F>
  struct FracFoldCols {
    Block<F> p1;
    Block<F> q1;
    Block<F> p2;
    Block<F> q2;
    Block<F> out_p;
    Block<F> out_q;
  };

  // UnivariateFold (high-degree univariate sumcheck round): the trace stores the padded polynomial
  // coefficients plus the Horner accumulator that constrains next_sum = p(r). MAX_COEFFS must match
  // the Rust chip constant. Layouts are byte-compatible with the Rust `#[repr(C)]` structs.
  static constexpr size_t UNIVARIATE_FOLD_MAX_COEFFS = 31;

  /// Flattened event (the Rust side pads Vec<Block<F>> into the fixed `coeffs` array).
  template <typename F>
  struct UnivariateFoldFlatEvent {
    Block<F> coeffs[UNIVARIATE_FOLD_MAX_COEFFS];
    Block<F> claimed_sum;
    Block<F> r;
    Block<F> next_sum;
  };

  /// Flattened instruction (Rust pads Vec<Address<F>> into `coeffs_addrs` and passes `num_coeffs`).
  template <typename F>
  struct UnivariateFoldInstr {
    Address<F> coeffs_addrs[UNIVARIATE_FOLD_MAX_COEFFS];
    F num_coeffs;
    Address<F> claimed_sum_addr;
    Address<F> r_addr;
    Address<F> next_sum_addr;
    F mult;
  };

  /// Preprocessed trace columns.
  template <typename F>
  struct UnivariateFoldPreprocessedCols {
    F is_real;
    F num_coeffs;
    Address<F> coeffs_addrs[UNIVARIATE_FOLD_MAX_COEFFS];
    Address<F> claimed_sum_addr;
    Address<F> r_addr;
    Address<F> next_sum_addr;
    F mult;
  };

  /// Main trace columns: coefficients, the claim/challenge/output, poly(0)/poly(1), and the Horner
  /// accumulator (`acc`).
  template <typename F>
  struct UnivariateFoldCols {
    Block<F> coeffs[UNIVARIATE_FOLD_MAX_COEFFS];
    Block<F> claimed_sum;
    Block<F> r;
    Block<F> next_sum;
    Block<F> eval_at_0;
    Block<F> eval_at_1;
    Block<F> acc[UNIVARIATE_FOLD_MAX_COEFFS - 1];
  };

  /// The inputs and outputs to the manual memory management/memory initialization table.
  template <typename V>
  struct MemIo {
    V inner;
  };

  template <typename F>
  using MemEvent = MemIo<Block<F>>;

  /// Data describing in what manner to access a particular memory block.
  template <typename F>
  struct RecursionMemoryAccessCols {
    /// The address to access.
    Address<F> addr;
    /// The multiplicity which to read/write.
    /// "Positive" values indicate a write, and "negative" values indicate a read.
    F mult;
  };

  template <typename F>
  struct MemInstrFfi {
    MemIo<Address<F>> addrs;
    MemIo<Block<F>> vals;
    F mult;
  };

  // ── Recursion pure-copy chips (layouts byte-compatible with the Rust #[repr(C)] columns) ──

  // UnpackBlock: unpacks a Block into 4 base-field lanes written to memory.
  /// Event: the Block value being unpacked.
  template <typename F>
  struct UnpackBlockEvent {
    Block<F> val;
  };
  /// Instruction: input Block address + per-lane output addresses and write multiplicities.
  template <typename F>
  struct UnpackBlockInstr {
    Address<F> in_addr;
    Address<F> out_addrs[4];
    F mults[4];
  };
  /// Preprocessed trace columns.
  template <typename F>
  struct UnpackBlockPreprocessedCols {
    F is_real;
    Address<F> in_addr;
    RecursionMemoryAccessCols<F> out[4];
  };
  /// Main trace columns: the unpacked Block value.
  template <typename F>
  struct UnpackBlockCols {
    Block<F> val;
  };

  // LagrangeEvalD2: degree-2 Lagrange evaluation round.
  /// Event: the EF values.
  template <typename F>
  struct LagrangeEvalD2Event {
    Block<F> claim;
    Block<F> s1;
    Block<F> s2;
    Block<F> r;
    Block<F> next_claim;
  };
  /// Instruction: input/output addresses and the write multiplicity.
  template <typename F>
  struct LagrangeEvalD2Instr {
    Address<F> claim_addr;
    Address<F> s1_addr;
    Address<F> s2_addr;
    Address<F> r_addr;
    Address<F> next_claim_addr;
    F mult;
  };
  /// Preprocessed trace columns.
  template <typename F>
  struct LagrangeEvalD2PreprocessedCols {
    F is_real;
    Address<F> claim_addr;
    Address<F> s1_addr;
    Address<F> s2_addr;
    Address<F> r_addr;
    Address<F> next_claim_addr;
    F mult;
  };
  /// Main trace columns.
  template <typename F>
  struct LagrangeEvalD2Cols {
    Block<F> claim;
    Block<F> s1;
    Block<F> s2;
    Block<F> r;
    Block<F> next_claim;
  };

  // PackBlock: packs four base-field lanes into a Block. DATAPAR independent ops per row.
  constexpr static const size_t PACK_BLOCK_DATAPAR = 4;
  /// Event: the four packed base-field lanes.
  template <typename F>
  struct PackBlockEvent {
    F lanes[4];
  };
  /// Instruction: four input lane addresses, output Block address, write multiplicity.
  template <typename F>
  struct PackBlockInstr {
    Address<F> in_addrs[4];
    Address<F> out_addr;
    F mult;
  };
  /// One pack operation's preprocessed data.
  template <typename F>
  struct PackBlockPreprocessedValueCols {
    F is_real;
    Address<F> in_addrs[4];
    Address<F> out_addr;
    F mult;
  };
  /// Preprocessed columns: DATAPAR independent operations per row.
  template <typename F>
  struct PackBlockPreprocessedCols {
    PackBlockPreprocessedValueCols<F> values[PACK_BLOCK_DATAPAR];
  };
  /// Main columns: the packed lanes of each operation.
  template <typename F>
  struct PackBlockCols {
    F values[PACK_BLOCK_DATAPAR][4];
  };

  // DigestSelect: per-lane select of two 8-lane digests. DATAPAR independent ops per row.
  constexpr static const size_t DIGEST_SELECT_DATAPAR = 2;
  /// Event: one select operation's values.
  template <typename F>
  struct DigestSelectEvent {
    F bit;
    F in1[8];
    F in2[8];
    F out[8];
  };
  /// Instruction: bit + input/output digest addresses + write multiplicities.
  template <typename F>
  struct DigestSelectInstr {
    Address<F> bit_addr;
    Address<F> in1_addrs[2];
    Address<F> in2_addrs[2];
    Address<F> out_addrs[2];
    F mults[2];
  };
  /// One select's preprocessed data.
  template <typename F>
  struct DigestSelectPreprocessedValueCols {
    F is_real;
    Address<F> bit_addr;
    Address<F> in1_addrs[2];
    Address<F> in2_addrs[2];
    Address<F> out_addrs[2];
    F mults[2];
  };
  /// Preprocessed columns: DATAPAR independent selects per row.
  template <typename F>
  struct DigestSelectPreprocessedCols {
    DigestSelectPreprocessedValueCols<F> values[DIGEST_SELECT_DATAPAR];
  };
  /// One select's values.
  template <typename F>
  struct DigestSelectValueCols {
    F bit;
    F in1[8];
    F in2[8];
    F out[8];
  };
  /// Main columns: DATAPAR independent selects per row.
  template <typename F>
  struct DigestSelectCols {
    DigestSelectValueCols<F> values[DIGEST_SELECT_DATAPAR];
  };

  // ExtDotProduct: extension-field dot product with a running accumulator over
  // PACKED base-field values. Each row folds GROUPS packed value-blocks (DATAPAR
  // mu lanes). The main event is fixed-size and copies 1:1 to a trace row; the
  // preprocessed rows are materialized host-side (per-instruction segment length
  // is data-dependent and unbounded) and copied into the device trace verbatim.
  constexpr static const size_t EXT_DOT_PRODUCT_DATAPAR = 8;
  constexpr static const size_t EXT_DOT_PRODUCT_GROUPS = 2;
  /// Main event: mus[DATAPAR] blocks, vals[GROUPS] packed blocks, running acc.
  template <typename F>
  struct ExtDotProductEvent {
    Block<F> mus[EXT_DOT_PRODUCT_DATAPAR];
    Block<F> vals[EXT_DOT_PRODUCT_GROUPS];
    Block<F> acc;
  };
  /// Main columns: identical layout to the event.
  template <typename F>
  struct ExtDotProductCols {
    Block<F> mus[EXT_DOT_PRODUCT_DATAPAR];
    Block<F> vals[EXT_DOT_PRODUCT_GROUPS];
    Block<F> acc;
  };
  /// Preprocessed columns: per-row element addresses + segment bookkeeping.
  template <typename F>
  struct ExtDotProductPreprocessedCols {
    F is_real, is_end;
    Address<F> mu_addrs[EXT_DOT_PRODUCT_DATAPAR];
    Address<F> val_addrs[EXT_DOT_PRODUCT_GROUPS];
    Address<F> acc_addr;
    F mult;
  };
  /// Per-instruction preprocessed metadata for the two-buffer expansion. `n_rows`/`mu_offset`/
  /// `val_offset`/`row_offset` are integer bookkeeping; the mu/val addresses live in flat side
  /// buffers concatenated across instructions.
  template <typename F>
  struct ExtDotProductPreprocMeta {
    uint32_t n_rows;
    uint32_t mu_offset;
    uint32_t val_offset;
    uint32_t row_offset;
    Address<F> acc_addr;
    F mult;
  };

  // GkrSumcheckRound: one GKR sumcheck round. main = EF values + the r^2 helper
  // (computed on the device); preproc = one row of memory addresses per instruction.
  template <typename F>
  struct GkrSumcheckRoundEvent {
    Block<F> claim, s1, s2, s3, r, xi, eq_in, next_claim, eq_out;
  };
  template <typename F>
  struct GkrSumcheckRoundInstr {
    Address<F> claim_addr, s1_addr, s2_addr, s3_addr, r_addr, xi_addr, eq_in_addr,
        next_claim_addr, eq_out_addr;
    F mult_next, mult_eq;
  };
  template <typename F>
  struct GkrSumcheckRoundPreprocessedCols {
    F is_real;
    Address<F> claim_addr, s1_addr, s2_addr, s3_addr, r_addr, xi_addr, eq_in_addr,
        next_claim_addr, eq_out_addr;
    F mult_next, mult_eq;
  };
  template <typename F>
  struct GkrSumcheckRoundCols {
    // r_sq is inserted between eq_in and next_claim (the event carries no r_sq).
    Block<F> claim, s1, s2, s3, r, xi, eq_in, r_sq, next_claim, eq_out;
  };

  // HornerEval: Horner evaluation of a univariate polynomial. main = DATAPAR
  // coefficients + x power helpers (computed on the device); preproc segments each
  // instruction into len/DATAPAR rows and is materialized fully on the device via a
  // two-buffer layout (per-instruction metadata + a flat coefficient-address buffer).
  constexpr static const size_t HORNER_EVAL_DATAPAR = 4;
  template <typename F>
  struct HornerEvalEvent {
    Block<F> coeffs[HORNER_EVAL_DATAPAR];
    Block<F> x;
    Block<F> acc;
  };
  template <typename F>
  struct HornerEvalPreprocessedCols {
    F is_real, is_start, is_end;
    Address<F> coeff_addrs[HORNER_EVAL_DATAPAR];
    Address<F> x_addr, out_addr;
    F mult;
  };
  template <typename F>
  struct HornerEvalCols {
    Block<F> coeffs[HORNER_EVAL_DATAPAR];
    Block<F> x, x2, x3, x4, acc;
  };
  /// Per-instruction preprocessed metadata for the two-buffer expansion. Byte-compatible
  /// with the Rust mirror; `num_coeffs`/`coeff_offset`/`row_offset` are integer bookkeeping.
  template <typename F>
  struct HornerEvalPreprocMeta {
    uint32_t num_coeffs;
    uint32_t coeff_offset;
    uint32_t row_offset;
    Address<F> x_addr;
    Address<F> out_addr;
    F mult;
  };

  // BinaryFoldStep: one WHIR binary k-fold butterfly step. main = inputs + amt/tih
  // helpers (computed on the device); preproc = one row of memory addresses per instruction.
  template <typename F>
  struct BinaryFoldStepEvent {
    F tw, xi, inv_tw, xi_inv;
    Block<F> alpha, v_lo, v_hi, out;
  };
  template <typename F>
  struct BinaryFoldStepInstr {
    Address<F> tw_addr, xi_addr, inv_tw_addr, xi_inv_addr, alpha_addr, v_lo_addr, v_hi_addr,
        out_addr;
    F mult;
  };
  template <typename F>
  struct BinaryFoldStepPreprocessedCols {
    F is_real;
    Address<F> tw_addr, xi_addr, inv_tw_addr, xi_inv_addr, alpha_addr, v_lo_addr, v_hi_addr,
        out_addr;
    F mult;
  };
  template <typename F>
  struct BinaryFoldStepCols {
    F tw, xi, inv_tw, xi_inv;
    Block<F> alpha, v_lo, v_hi, amt;
    F tih;
    Block<F> out;
  };

  // Poseidon2Packed: a Poseidon2 permutation with packed (Block-valued) I/O. The main columns are
  // the shared gadget's Poseidon2ValueCols (one permutation per row), so main GPU trace-gen reuses
  // the recursion poseidon2 kernel; only the preprocessed layout (4 packed Block addresses) is new.
  template <typename F>
  struct Poseidon2PackedInstr {
    Address<F> input_addrs[4];
    Address<F> output_addrs[4];
    F mults[4];
  };
  template <typename F>
  struct Poseidon2PackedPreprocessedCols {
    F is_real;
    Address<F> input[4];
    RecursionMemoryAccessCols<F> output[4];
  };

  // ExpBitsLen: KoalaBear-only exponentiation of `base` by the low `num_bits` bits of `bit_src`,
  // one 32-row request block per event/instruction. main recomputes the squaring ladder, the
  // reverse-product result, and the canonicality flags per row on the device; preproc uses a
  // two-buffer layout (per-instruction metadata + a flat per-row bit-output buffer).
  constexpr static const size_t EXP_BITS_LEN_ROWS = 32;       // NUM_BITS_MAX_PLUS_ONE
  constexpr static const size_t EXP_BITS_LEN_LOW_BITS = 24;   // LOW_BITS_COUNT
  /// H2D main event: `bit_src` is the canonical u32 (the field integer to decompose).
  template <typename F>
  struct ExpBitsLenFlatEvent {
    F base;
    uint32_t bit_src;
    uint32_t num_bits;
  };
  template <typename F>
  struct ExpBitsLenCols {
    F bit_idx, base, bit_src, num_bits, apply_bit, low_bits_left, in_low_region, result,
        result_multiplier, bit_src_mod_2, low_bits_are_zero, high_bits_all_one;
  };
  template <typename F>
  struct ExpBitsLenPreprocessedCols {
    F is_first, is_last, is_real;
    RecursionMemoryAccessCols<F> bit_src_in;
    RecursionMemoryAccessCols<F> result_out;
    RecursionMemoryAccessCols<F> bit_out;
    F base_const, num_bits_const;
  };
  /// Per-instruction preprocessed metadata; the per-row bit outputs live in a flat side buffer of
  /// (addr, mult) pairs.
  template <typename F>
  struct ExpBitsLenPreprocMeta {
    F base;
    uint32_t num_bits;
    Address<F> bit_src_in_addr;
    Address<F> result_out_addr;
    F result_mult;
    uint32_t bit_outs_offset;
    uint32_t bit_outs_count;
  };

  template <typename F>
  struct MemoryValueAccess {
    Block<F> value;
    RecursionMemoryAccessCols<F> access;
  };

  template <typename F>
  struct RecursionMemoryConstPreprocessedCols {
    MemoryValueAccess<F> values_and_accesses[CONST_MEM_DATAPAR];
  };

  template <typename F>
  struct RecursionConstantMemoryCols {
    F _nothing;
  };

  template <typename F>
  struct RecursionMemoryVarPreprocessedCols {
    RecursionMemoryAccessCols<F> accesses[VAR_MEM_DATAPAR];
  };

  template <typename F>
  struct RecursionVariableMemoryCols {
    Block<F> values[VAR_MEM_DATAPAR];
  };

  enum class FieldOperation : uint8_t {
    Add = 0,
    Mul = 1,
    Sub = 2,
    Div = 3,
  };

  /// FpEvent: u64-compatible version.
  /// NumDwords = number of u64 dwords per field element (= NumLimbs/8).
  template <const int NumDwords>
  struct FpEvent {
    uint32_t chunk;
    uint32_t clk;
    uint32_t x_ptr;
    uint64_t x[NumDwords];
    uint32_t y_ptr;
    uint64_t y[NumDwords];
    FieldOperation op;
    MemoryWriteRecordU64 x_memory_records[NumDwords];
    MemoryReadRecordU64 y_memory_records[NumDwords];
  };

  template <typename T, const int NumLimbs, const int NumWitnesses>
  struct FieldOpCols {
    T result[NumLimbs];
    T carry[NumLimbs];
    T witness[NumWitnesses];
  };

  template <typename T, const int NumLimbs, const int NumWitnesses>
  struct FieldLtCols {
    T byte_flags[NumLimbs];
    T lhs_comparison_byte;
    T rhs_comparison_byte;
  };

  /// AddrAddGadget: matches Rust AddrAddGadget<T>.
  template <typename T>
  struct FpAddrAddGadget {
    T value[3];              // 3 u16 limbs (ADDR_SIZE)
  };

  /// SyscallAddrGadget for FpOp: matches Rust SyscallAddrGadget<T>.
  template <typename T>
  struct FpSyscallAddrGadget {
    T addr[3];               // 3 u16 limbs
    T top_two_limb_min;
    IsZeroOperation<T> top_two_limb_max;
  };



  /// Fp2AddSubEvent: u64-compatible version.
  /// NumDwords = number of u64 dwords per curve point (= 2 * WordsFieldElement).
  template <const int NumDwords>
  struct Fp2AddSubEvent {
    uint32_t chunk;
    uint32_t clk;
    uint32_t x_ptr;
    uint64_t x[NumDwords];
    uint32_t y_ptr;
    uint64_t y[NumDwords];
    FieldOperation op;
    MemoryWriteRecordU64 x_memory_records[NumDwords];
    MemoryReadRecordU64 y_memory_records[NumDwords];
  };


  /// Fp2MulEvent: u64-compatible version.
  /// NumCurveDwords = number of u64 dwords per curve point (= 2 * NumFieldDwords).
  template <const int NumCurveDwords>
  struct Fp2MulEvent {
    uint32_t chunk;
    uint32_t clk;
    uint32_t x_ptr;
    uint64_t x[NumCurveDwords];
    uint32_t y_ptr;
    uint64_t y[NumCurveDwords];
    MemoryWriteRecordU64 x_memory_records[NumCurveDwords];
    MemoryReadRecordU64 y_memory_records[NumCurveDwords];
  };


  /// The inputs and outputs to a select operation.
  template <typename V>
  struct SelectIo {
    V bit;
    V out1;
    V out2;
    V in1;
    V in2;
  };

  template <typename F>
  using SelectEvent = SelectIo<F>;

  /// An instruction invoking the select operation.
  template <typename F>
  struct SelectInstr {
    SelectIo<Address<F>> addrs;
    F mult1;
    F mult2;
  };

  template <typename F>
  struct SelectPreprocessedValueCols {
    F is_real;
    SelectIo<Address<F>> addrs;
    F mult1;
    F mult2;
  };

  template <typename F>
  struct SelectPreprocessedCols {
    SelectPreprocessedValueCols<F> values[SELECT_DATAPAR];
  };

  template <typename F>
  struct SelectValueCols {
    SelectIo<F> vals;
  };

  template <typename F>
  struct SelectCols {
    SelectValueCols<F> values[SELECT_DATAPAR];
  };

  template <typename T>
  struct Poseidon2PreprocessedValueCols {
    Address<T> input[PERMUTATION_WIDTH];
    RecursionMemoryAccessCols<T> output[PERMUTATION_WIDTH];
    T is_real_neg;
  };

  template <typename T>
  struct Poseidon2PreprocessedCols {
    Poseidon2PreprocessedValueCols<T> values[POSEIDON2_DATAPAR];
  };

  /// Memory write columns with U16→U8 auxiliary for precompile chips.
  template <typename T>
  struct MemoryWriteColsU8 {
    MemoryWriteCols<T> inner;
    U16ToU8Gadget<T> prev_value_u8;
  };

  /// Memory read columns with U16→U8 auxiliary for precompile chips.
  template <typename T>
  struct MemoryReadColsU8 {
    MemoryReadCols<T> inner;
    U16ToU8Gadget<T> prev_value_u8;
  };

  /// Fp2AddSubCols: u64-compatible version with new gadgets.
  /// NumCurveDwords = number of u64 dwords per curve point (= NumLimbs/4 = 2 * NumFieldDwords).
  /// Arrays use NumCurveDwords (matching Rust's WordsCurvePoint).
  template <typename T, const int NumCurveDwords, const int NumLimbs, const int NumWitnesses>
  struct Fp2AddSubCols {
    T is_real;
    T chunk;
    T clk;
    T is_add;
    FpSyscallAddrGadget<T> x_ptr;
    FpSyscallAddrGadget<T> y_ptr;
    FpAddrAddGadget<T> x_addrs[NumCurveDwords];
    FpAddrAddGadget<T> y_addrs[NumCurveDwords];
    MemoryWriteColsU8<T> x_access[NumCurveDwords];
    MemoryReadColsU8<T> y_access[NumCurveDwords];
    FieldOpCols<T, NumLimbs, NumWitnesses> c0;
    FieldOpCols<T, NumLimbs, NumWitnesses> c1;
    FieldLtCols<T, NumLimbs, 0> c0_range;
    FieldLtCols<T, NumLimbs, 0> c1_range;
  };

  /// Fp2MulCols: u64-compatible version with new gadgets.
  /// NumCurveDwords = number of u64 dwords per curve point (= NumLimbs/4 = 2 * NumFieldDwords).
  /// Arrays use NumCurveDwords (matching Rust's WordsCurvePoint).
  template <typename T, const int NumCurveDwords, const int NumLimbs, const int NumWitnesses>
  struct Fp2MulCols {
    T is_real;
    T chunk;
    T clk;
    FpSyscallAddrGadget<T> x_ptr;
    FpSyscallAddrGadget<T> y_ptr;
    FpAddrAddGadget<T> x_addrs[NumCurveDwords];
    FpAddrAddGadget<T> y_addrs[NumCurveDwords];
    MemoryWriteColsU8<T> x_access[NumCurveDwords];
    MemoryReadColsU8<T> y_access[NumCurveDwords];
    FieldOpCols<T, NumLimbs, NumWitnesses> a0_mul_b0;
    FieldOpCols<T, NumLimbs, NumWitnesses> a1_mul_b1;
    FieldOpCols<T, NumLimbs, NumWitnesses> a0_mul_b1;
    FieldOpCols<T, NumLimbs, NumWitnesses> a1_mul_b0;
    FieldOpCols<T, NumLimbs, NumWitnesses> c0;
    FieldOpCols<T, NumLimbs, NumWitnesses> c1;
    FieldLtCols<T, NumLimbs, 0> c0_range;
    FieldLtCols<T, NumLimbs, 0> c1_range;
  };

  /// FpOpCols: u64-compatible version with new gadgets.
  /// NumDwords = number of u64 dwords per field element (= NumLimbs/8).
  template <typename T, const int NumDwords, const int NumLimbs, const int NumWitnesses>
  struct FpOpCols {
    T is_real;
    T chunk;
    T clk;
    T is_add;
    T is_sub;
    T is_mul;
    FpSyscallAddrGadget<T> x_ptr;
    FpSyscallAddrGadget<T> y_ptr;
    FpAddrAddGadget<T> x_addrs[NumDwords];
    FpAddrAddGadget<T> y_addrs[NumDwords];
    MemoryWriteColsU8<T> x_access[NumDwords];
    MemoryReadColsU8<T> y_access[NumDwords];
    FieldOpCols<T, NumLimbs, NumWitnesses> output;
    FieldLtCols<T, NumLimbs, NumWitnesses> output_range;
  };

  /// Memory write record for EC FFI events (u64 values).
  /// Must match Rust FfiEcMemoryWriteRecord: value, prev_value, chunk, timestamp, prev_chunk, prev_timestamp.
  struct FfiEcMemoryWriteRecord {
    uint64_t value;
    uint64_t prev_value;
    uint32_t chunk;
    uint32_t timestamp;
    uint32_t prev_chunk;
    uint32_t prev_timestamp;
  };

  /// Memory read record for EC FFI events (u64 value).
  /// Must match Rust MemoryReadRecord field order: value, chunk, timestamp, prev_chunk, prev_timestamp.
  struct FfiEcMemoryReadRecord {
    uint64_t value;
    uint32_t chunk;
    uint32_t timestamp;
    uint32_t prev_chunk;
    uint32_t prev_timestamp;
  };

  /// EllipticCurveAddEvent — used by Edwards add (old u32-based memory records).
  template <const int NumWords>
  struct EllipticCurveAddEvent {
    uint32_t chunk;
    uint32_t clk;
    uint32_t p_ptr;
    uint32_t p[2 * NumWords];
    uint32_t q_ptr;
    uint32_t q[2 * NumWords];
    MemoryWriteRecord p_memory_records[2 * NumWords];
    MemoryReadRecord q_memory_records[2 * NumWords];
  };

  /// WeierstrassAddEvent — used by Weierstrass add (u64 dword data, FfiEcMemory*Record).
  /// NumWordsDword = WordsCurvePoint (number of u64 dwords per curve point).
  template <const int NumWordsDword>
  struct WeierstrassAddEvent {
    uint32_t chunk;
    uint32_t clk;
    uint32_t p_ptr;
    // Point data as u64 dwords (NumWordsDword elements).
    // GPU code reinterprets each u64 as two u32 words in little-endian.
    uint64_t p[NumWordsDword];
    uint32_t q_ptr;
    uint64_t q[NumWordsDword];
    // Memory records with u64 value/prev_value, one per dword.
    FfiEcMemoryWriteRecord p_memory_records[NumWordsDword];
    FfiEcMemoryReadRecord q_memory_records[NumWordsDword];
  };

  /// SyscallAddrGadget — layout matches Rust SyscallAddrGadget<T>.
  /// populate() implementation is in syscall_addr.cuh (avoids circular include with byte.cuh).
  template <typename T>
  struct SyscallAddrGadget {
    T addr[3];
    T top_two_limb_min;
    IsZeroOperation<T> top_two_limb_max;

    __device__ void populate(T* byte_trace, uint64_t addr_val, uint64_t len);
  };

  /// AddrAddGadget — layout matches Rust AddrAddGadget<T>.
  /// populate() implementation is in syscall_addr.cuh.
  template <typename T>
  struct AddrAddGadget {
    T value[3];

    __device__ uint64_t populate(T* byte_trace, uint64_t base, uint64_t offset);
  };

  /// WeierstrassAddAssignCols for u64 branch.
  /// NumWordsDword = WordsCurvePoint (number of u64 dwords per curve point).
  template <typename T, int NumWordsDword, int NumLimbs, int NumWitnesses>
  struct WeierstrassAddAssignCols {
    T is_real;
    T chunk;
    T clk;
    SyscallAddrGadget<T> p_ptr;
    SyscallAddrGadget<T> q_ptr;
    AddrAddGadget<T> p_addrs[NumWordsDword];
    AddrAddGadget<T> q_addrs[NumWordsDword];
    MemoryWriteColsU8<T> p_access[NumWordsDword];
    MemoryReadColsU8<T> q_access[NumWordsDword];
    FieldOpCols<T, NumLimbs, NumWitnesses> slope_denominator;
    FieldOpCols<T, NumLimbs, NumWitnesses> inverse_check;
    FieldOpCols<T, NumLimbs, NumWitnesses> slope_numerator;
    FieldOpCols<T, NumLimbs, NumWitnesses> slope;
    FieldOpCols<T, NumLimbs, NumWitnesses> slope_squared;
    FieldOpCols<T, NumLimbs, NumWitnesses> p_x_plus_q_x;
    FieldOpCols<T, NumLimbs, NumWitnesses> x3_ins;
    FieldOpCols<T, NumLimbs, NumWitnesses> p_x_minus_x;
    FieldOpCols<T, NumLimbs, NumWitnesses> y3_ins;
    FieldOpCols<T, NumLimbs, NumWitnesses> slope_times_p_x_minus_x;
    FieldLtCols<T, NumLimbs, NumWitnesses> x3_range;
    FieldLtCols<T, NumLimbs, NumWitnesses> y3_range;
  };

  template <const int NumWords>
  struct EllipticCurveDoubleEvent {
    // The chunk number.
    uint32_t chunk;
    // The clock cycle.
    uint32_t clk;
    // The pointer to the first point.
    uint32_t p_ptr;
    // The first point as a list of words.
    uint32_t p[2 * NumWords];
    // The memory records for the first point.
    MemoryWriteRecord p_memory_records[2 * NumWords];
  };

  /// WeierstrassDoubleEvent — used by Weierstrass double (u64 dword data, FfiEcMemoryWriteRecord).
  /// NumWordsDword = WordsCurvePoint (number of u64 dwords per curve point).
  template <const int NumWordsDword>
  struct WeierstrassDoubleEvent {
    uint32_t chunk;
    uint32_t clk;
    uint32_t p_ptr;
    // Point data as u64 dwords (NumWordsDword elements).
    uint64_t p[NumWordsDword];
    // Memory records with u64 value/prev_value, one per dword.
    FfiEcMemoryWriteRecord p_memory_records[NumWordsDword];
  };

  /// WeierstrassDoubleAssignCols for u64 branch.
  /// NumWordsDword = WordsCurvePoint (number of u64 dwords per curve point).
  template <typename T, int NumWordsDword, int NumLimbs, int NumWitnesses>
  struct WeierstrassDoubleAssignCols {
    T is_real;
    T chunk;
    T clk;
    SyscallAddrGadget<T> p_ptr;
    AddrAddGadget<T> p_addrs[NumWordsDword];
    MemoryWriteColsU8<T> p_access[NumWordsDword];
    FieldOpCols<T, NumLimbs, NumWitnesses> slope_denominator;
    FieldOpCols<T, NumLimbs, NumWitnesses> inverse_check;
    FieldOpCols<T, NumLimbs, NumWitnesses> slope_numerator;
    FieldOpCols<T, NumLimbs, NumWitnesses> slope;
    FieldOpCols<T, NumLimbs, NumWitnesses> p_x_squared;
    FieldOpCols<T, NumLimbs, NumWitnesses> p_x_squared_times_3;
    FieldOpCols<T, NumLimbs, NumWitnesses> slope_squared;
    FieldOpCols<T, NumLimbs, NumWitnesses> p_x_plus_p_x;
    FieldOpCols<T, NumLimbs, NumWitnesses> x3_ins;
    FieldOpCols<T, NumLimbs, NumWitnesses> p_x_minus_x;
    FieldOpCols<T, NumLimbs, NumWitnesses> y3_ins;
    FieldOpCols<T, NumLimbs, NumWitnesses> slope_times_p_x_minus_x;
    FieldLtCols<T, NumLimbs, NumWitnesses> x3_range;
    FieldLtCols<T, NumLimbs, NumWitnesses> y3_range;
  };

  /// Elliptic Curve Point Decompress Event FFI.
  ///
  /// This event is emitted when an elliptic curve point decompression operation is performed.
  template <int NumWords, int NumLimbs>
  struct EllipticCurveDecompressEventFFI {
    /// The chunk number.
    uint32_t chunk;
    /// The clock cycle.
    uint32_t clk;
    /// The pointer to the point.
    uint32_t ptr;
    /// The sign bit of the point.
    bool sign_bit;
    /// The x coordinate as a list of bytes.
    uint8_t x_bytes[NumLimbs];
    /// The decompressed y coordinate as a list of bytes.
    uint8_t decompressed_y_bytes[NumLimbs];
    /// The memory records for the x coordinate (u64 values).
    /// NumWords/2 entries because each entry holds a u64 dword.
    FfiEcMemoryReadRecord x_memory_records[NumWords / 2];
    /// The memory records for the y coordinate (u64 values).
    FfiEcMemoryWriteRecord y_memory_records[NumWords / 2];
  };

  /// A set of columns to compute the square root in emulated arithmetic.
  ///
  /// *Safety*: The `FieldSqrtCols` asserts that `multiplication.result` is a square root of the given
  /// input lying within the range `[0, modulus)` with the least significant bit `lsb`.
  template <typename T, int NumLimbs, int NumWitness>
  struct FieldSqrtCols {
    /// The multiplication operation to verify that the sqrt and the input match.
    ///
    /// In order to save space, we actually store the sqrt of the input in `multiplication.result`
    /// since we'll receive the input again in the `eval` function.
    FieldOpCols<T, NumLimbs, NumWitness> multiplication;
    FieldLtCols<T, NumLimbs, NumWitness> range;
    T lsb;
  };

  /// A set of columns to compute `WeierstrassDecompress` that decompresses a point on a Weierstrass
  /// curve.
  template <typename T, int NumWords, int NumLimbs, int NumWitnesses>
  struct LexicographicChoiceCols {
    FieldLtCols<T, NumLimbs, NumWitnesses> comparison_lt_cols;
    // neg_y_range_check is gone: the host dropped it, the sibling range check on the chosen y
    // already covers the same bound.
    T is_y_eq_sqrt_y_result;
    T when_sqrt_y_res_is_lt;
    T when_neg_y_res_is_lt;
  };

  /// An array representing N limbs of T.
  ///
  /// Array allows us to constrain the correct array lengths so we can have # of limbs and # of
  /// witness limbs associated in NumLimbs / FieldParameters.
  /// See: https://github.com/RustCrypto/traits/issues/1481
  template <typename T, int N>
  struct Limbs {
    T _0[N];
  };

  /// A set of columns to compute `InnerProduct([a], [b])` where a, b are emulated elements.
  ///
  /// *Safety*: The `FieldInnerProductCols` asserts that `result = sum_i a_i * b_i mod M` where
  /// `M` is the modulus `P::modulus()` under the assumption that the length of `a` and `b` is small
  /// enough so that the vanishing polynomial has limbs bounded by the witness shift. It is the
  /// responsibility of the caller to ensure that the length of `a` and `b` is small enough.
  template <typename T, int NumLimbs, int NumWitness>
  struct FieldInnerProductCols {
    /// The result of `a inner product b`, where a, b are field elements
    Limbs<T, NumLimbs> result;
    Limbs<T, NumLimbs> carry;
    Limbs<T, NumWitness> witness;
  };

  /// A set of columns to compute `FieldDen(a, b)` where `a`, `b` are field elements.
  ///
  /// `a / (1 + b)` if `sign`
  /// `a / (1 - b) ` if `!sign`
  ///
  /// *Safety*: the operation assumes that the denominators are never zero. It is the responsibility
  /// of the caller to ensure that condition.
  template <typename T, int NumLimbs, int NumWitness>
  struct FieldDenCols {
    /// The result of `a den b`, where a, b are field elements
    Limbs<T, NumLimbs> result;
    Limbs<T, NumLimbs> carry;
    Limbs<T, NumWitness> witness;
  };

  /// A set of columns to compute `EdAdd` where a, b are field elements.
  /// Right now the number of limbs is assumed to be a constant, although this could be macro-ed
  /// or made generic in the future.
  template <typename T, int NumLimbs, int NumWitness>
  struct EdAddAssignCols {
    T is_real;
    T chunk;
    T clk;
    T p_ptr;
    T q_ptr;
    MemoryWriteCols<T> p_access[WORDS_CURVE_POINT];
    MemoryReadCols<T> q_access[WORDS_CURVE_POINT];
    FieldInnerProductCols<T, NumLimbs, NumWitness> x3_numerator;
    FieldInnerProductCols<T, NumLimbs, NumWitness> y3_numerator;
    FieldOpCols<T, NumLimbs, NumWitness> x1_mul_y1;
    FieldOpCols<T, NumLimbs, NumWitness> x2_mul_y2;
    FieldOpCols<T, NumLimbs, NumWitness> f;
    FieldOpCols<T, NumLimbs, NumWitness> d_mul_f;
    FieldDenCols<T, NumLimbs, NumWitness> x3_ins;
    FieldDenCols<T, NumLimbs, NumWitness> y3_ins;
  };

  /// A set of columns to compute `WeierstrassDecompress` that decompresses a point on a Weierstrass
  /// curve.
  template <typename T, int NumWords, int NumLimbs, int NumWitnesses>
  struct WeierstrassDecompressCols {
    T is_real;
    T chunk;
    T clk;
    SyscallAddrGadget<T> ptr;
    AddrAddGadget<T> x_addrs[NumWords / 2];
    AddrAddGadget<T> y_addrs[NumWords / 2];
    T sign_bit;
    MemoryReadColsU8<T> x_access[NumWords / 2];   // NumWords/2 = u64 dword count
    MemoryWriteColsU8<T> y_access[NumWords / 2];
    FieldLtCols<T, NumLimbs, NumWitnesses> range_x;
    FieldOpCols<T, NumLimbs, NumWitnesses> x_2;
    FieldOpCols<T, NumLimbs, NumWitnesses> x_3;
    FieldInnerProductCols<T, NumLimbs, NumWitnesses> ax_plus_b;
    FieldOpCols<T, NumLimbs, NumWitnesses> x_3_plus_ax_plus_b;
    FieldSqrtCols<T, NumLimbs, NumWitnesses> y;
    FieldOpCols<T, NumLimbs, NumWitnesses> neg_y;
    FieldLtCols<T, NumLimbs, NumWitnesses> neg_y_range_check;
  };

  /// Edwards Decompress Event.
  ///
  /// This event is emitted when an edwards decompression operation is performed.
  struct EdDecompressEvent {
    /// The chunk number.
    uint32_t chunk;
    /// The clock cycle.
    uint32_t clk;
    /// The pointer to the point.
    uint32_t ptr;
    /// The sign bit of the point.
    bool sign;
    /// The comprssed y coordinate as a list of bytes.
    uint8_t y_bytes[COMPRESSED_POINT_BYTES];
    /// The decompressed x coordinate as a list of bytes.
    uint8_t decompressed_x_bytes[BYTES_FIELD_ELEM];
    /// The memory records for the x coordinate.
    MemoryWriteRecord x_memory_records[WORDS_FIELD_ELEM];
    /// The memory records for the y coordinate.
    MemoryReadRecord y_memory_records[WORDS_FIELD_ELEM];
  };

  /// A set of columns to compute `EdDecompress` given a pointer to a 16 word slice formatted as such:
  /// The 31st byte of the slice is the sign bit. The second half of the slice is the 255-bit
  /// compressed Y (without sign bit).
  ///
  /// After `EdDecompress`, the first 32 bytes of the slice are overwritten with the decompressed X.
  template <typename T, int NumLimbs, int NumWitness>
  struct EdDecompressCols {
    T is_real;
    T chunk;
    T clk;
    T ptr;
    T sign;
    MemoryWriteCols<T> x_access[WORDS_FIELD_ELEM];
    MemoryReadCols<T> y_access[WORDS_FIELD_ELEM];
    FieldLtCols<T, NumLimbs, NumWitness> y_range;
    FieldOpCols<T, NumLimbs, NumWitness> yy;
    FieldOpCols<T, NumLimbs, NumWitness> u;
    FieldOpCols<T, NumLimbs, NumWitness> dyy;
    FieldOpCols<T, NumLimbs, NumWitness> v;
    FieldOpCols<T, NumLimbs, NumWitness> u_div_v;
    FieldSqrtCols<T, NumLimbs, NumWitness> x;
    FieldOpCols<T, NumLimbs, NumWitness> neg_x;
    /// Gadget to check if x is zero. This is used to ensure that when x is zero, sign must be 0.
    IsZeroOperation<T> x_is_zero;
  };

  /// The preprocessed columns for the CommitPVHash instruction.
  template <typename T>
  struct PublicValuesPreprocessedCols {
    T pv_idx[DIGEST_SIZE];
    RecursionMemoryAccessCols<T> pv_mem;
  };

  /// The cols for a CommitPVHash invocation.
  template <typename T>
  struct PublicValuesCols {
    T pv_element;
  };

  template <typename T>
  struct ExpReverseBitsLenPreprocessedCols {
    RecursionMemoryAccessCols<T> x_mem;
    RecursionMemoryAccessCols<T> exponent_mem;
    RecursionMemoryAccessCols<T> result_mem;
    T iteration_num;
    T is_first;
    T is_last;
    T is_real;
  };

  template <typename T>
  struct ExpReverseBitsLenCols {
    /// The base of the exponentiation.
    T x;
    /// The current bit of the exponent. This is read from memory.
    T current_bit;
    /// The previous accumulator squared.
    T prev_accum_squared;
    /// Is set to the value local.prev_accum_squared * local.multiplier.
    T prev_accum_squared_times_multiplier;
    /// The accumulator of the current iteration.
    T accum;
    /// The accumulator squared.
    T accum_squared;
    /// A column which equals x if `current_bit` is on, and 1 otherwise.
    T multiplier;
  };

  /// The inputs and outputs to an exp-reverse-bits operation.
  template <typename V>
  struct ExpReverseBitsIoFfi {
    V base;
    V result;
    uintptr_t len;
    uintptr_t initial_row;
  };

  /// An FFI-safe instruction invoking the exp-reverse-bits operation.
  template <typename F>
  struct ExpReverseBitsInstrFfi {
    ExpReverseBitsIoFfi<Address<F>> addrs;
    F mult;
  };

  template <typename F>
  struct ExpReverseBitsFfiEvent {
    F base;
    F result;
    uintptr_t len;
    uintptr_t initial_row;
  };

  /// The base-field-valued vector inputs to the batch FRI operation.
  template <typename V>
  struct BatchFRIBaseVecIo {
    V p_at_x;
  };

  /// The extension-field-valued single inputs to the batch FRI operation.
  template <typename V>
  struct BatchFRIExtSingleIo {
    V acc;
  };

  /// The extension-field-valued vector inputs to the batch FRI operation.
  template <typename V>
  struct BatchFRIExtVecIo {
    V p_at_z;
    V alpha_pow;
  };

  template <typename F>
  struct BatchFRIInstrFfi {
    BatchFRIExtSingleIo<Address<F>> ext_single_addrs;
    F acc_mult;
    uintptr_t offset;
    uintptr_t len;
  };

  /// The event encoding the data of a single iteration within the batch FRI operation.
  /// For any given event, we are accessing a single element of the `Vec` inputs, so that the event
  /// is not a type alias for `BatchFRIIo` like many of the other events.
  template <typename F>
  struct BatchFRIEvent {
    BatchFRIBaseVecIo<F> base_vec;
    BatchFRIExtSingleIo<Block<F>> ext_single;
    BatchFRIExtVecIo<Block<F>> ext_vec;
  };

  /// The preprocessed columns for a batch FRI invocation.
  template <typename F>
  struct BatchFRIPreprocessedCols {
    F is_real;
    F is_end;
    Address<F> acc_addr;
    Address<F> alpha_pow_addr;
    Address<F> p_at_z_addr;
    Address<F> p_at_x_addr;
  };

  /// The main columns for a batch FRI invocation.
  template <typename F>
  struct BatchFRICols {
    Block<F> acc;
    Block<F> alpha_pow;
    Block<F> p_at_z;
    F p_at_x;
  };

  /// FFI-compatible representation of ShaExtendEvent for CUDA interop
  /// This matches the C++ ShaExtendEvent structure exactly
  struct ShaExtendFfiEvent {
    uint32_t chunk;
    uint32_t _pad1;
    uint64_t clk;
    uint64_t w_ptr;
    MemoryReadRecordU64 w_i_minus_15_reads[48];
    MemoryReadRecordU64 w_i_minus_2_reads[48];
    MemoryReadRecordU64 w_i_minus_16_reads[48];
    MemoryReadRecordU64 w_i_minus_7_reads[48];
    MemoryWriteRecordU64 w_i_writes[48];
  };

  struct Uint256MulEvent {
    /// The chunk number
    uint32_t chunk;
    uint32_t _pad;
    /// The clock cycle
    uint64_t clk;
    /// The pointer to the x value
    uint64_t x_ptr;
    /// The x value as a list of words
    uint32_t x[8];
    /// The pointer to the y value
    uint64_t y_ptr;
    /// The y value as a list of words
    uint32_t y[8];
    /// The modulus as a list of word.
    uint32_t modulus[8];
    /// The memory records for the x value
    MemoryWriteRecord x_memory_records[8];
    /// The memory records for the y value
    MemoryReadRecord y_memory_records[8];
    /// The memory records for the modulus
    MemoryReadRecord modulus_memory_records[8];
  };



  enum class LookupType : uint8_t {
    Memory = 1,
    Program = 2,
    Instruction = 3,
    Alu = 4,
    Byte = 5,
    Range = 6,
    Field = 7,
    Syscall = 8,
    Poseidon2 = 9,
    Global = 10,
  };

  enum class SyscallChunkKind : uint8_t {
    Riscv = 0,
    Precompile = 1,
  };

  struct CpuExtraEventIndices {
    size_t add;
    size_t lt;
  };

  struct DivremExtraEventIndices {
    size_t add;
    size_t lt;
  };
} // namespace pico_gpu
