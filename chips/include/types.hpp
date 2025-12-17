#pragma once

#include <cstdint>

namespace pico_gpu {
  /// Extension degree
  constexpr static const size_t EXTENSION_DEGREE = 4;
  constexpr static const size_t BASE_ALU_DATAPAR = 2;
  constexpr static const size_t EXT_ALU_DATAPAR = 4;
  constexpr static const size_t VAR_MEM_DATAPAR = 4;
  constexpr static const size_t CONST_MEM_DATAPAR = 1;
  constexpr static const size_t SELECT_DATAPAR = 2;
  constexpr static const size_t POSEIDON2_DATAPAR = 1;

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
  /// Alu Instruction Event.
  ///
  /// This object encapsulated the information needed to prove a RISC-V ALU operation.
  struct AluEvent {
    /// The program counter.
    uint32_t pc;
    /// The opcode.
    Opcode opcode;
    /// The first operand value.
    uint32_t a;
    /// The second operand value.
    uint32_t b;
    /// The third operand value.
    uint32_t c;
  };

  /// An array of four bytes to represent a 32-bit value.
  ///
  /// We use the generic type `T to represent the different representations of a byte, ranging from
  /// a `u8` to a `AB::Var` or `AB::Expr`.
  template <typename T>
  struct Word {
    T _0[WORD_SIZE];
  };

  /// A set of columns needed to compute the add of two words.
  template <typename T>
  struct AddOperation {
    /// The result of `a + b`.
    Word<T> value;
    /// Trace.
    T carry[3];
  };

  /// The column layout for the chip.
  template <typename T>
  struct AddSubCols {
    /// Instance of `AddOperation` to handle addition logic in `AddSubChip`'s ALU operations.
    /// It's result will be `a` for the add operation and `b` for the sub operation.
    AddOperation<T> add_operation;
    /// The first input operand.  This will be `b` for add operations and `a` for sub operations.
    Word<T> operand_1;
    /// The second input operand.  This will be `c` for both operations.
    Word<T> operand_2;
    /// Boolean to indicate whether the row is for an add operation.
    T is_add;
    /// Boolean to indicate whether the row is for a sub operation.
    T is_sub;
  };

  /// The column layout for the chip.
  template <typename T>
  struct MulCols {
    /// The output operand.
    Word<T> a;
    /// The first input operand.
    Word<T> b;
    /// The second input operand.
    Word<T> c;
    /// Trace.
    T carry[LONG_WORD_SIZE];
    /// An array storing the product of `b * c` after the carry propagation.
    T product[LONG_WORD_SIZE];
    /// The most significant bit of `b`.
    T b_msb;
    /// The most significant bit of `c`.
    T c_msb;
    /// The sign extension of `b`.
    T b_sign_extend;
    /// The sign extension of `c`.
    T c_sign_extend;
    /// Flag indicating whether the opcode is `MUL` (`u32 x u32`).
    T is_mul;
    /// Flag indicating whether the opcode is `MULH` (`i32 x i32`, upper half).
    T is_mulh;
    /// Flag indicating whether the opcode is `MULHU` (`u32 x u32`, upper half).
    T is_mulhu;
    /// Flag indicating whether the opcode is `MULHSU` (`i32 x u32`, upper half).
    T is_mulhsu;
    /// Selector to know whether this row is enabled.
    T is_real;
  };

  /// The column layout for the chip.
  template <typename T>
  struct BitwiseCols {
    /// The output operand.
    Word<T> a;
    /// The first input operand.
    Word<T> b;
    /// The second input operand.
    Word<T> c;
    /// If the opcode is XOR.
    T is_xor;
    T is_or;
    /// If the opcode is AND.
    T is_and;
  };

  /// The column layout for the chip.
  template <typename T>
  struct LtCols {
    /// If the opcode is SLT.
    T is_slt;
    /// If the opcode is SLTU.
    T is_sltu;
    /// The output operand.
    Word<T> a;
    /// The first input operand.
    Word<T> b;
    /// The second input operand.
    Word<T> c;
    /// Boolean flag to indicate which byte pair differs if the operands are not equal.
    T byte_flags[4];
    /// The masking b\[3\] & 0x7F.
    T b_masked;
    /// The masking c\[3\] & 0x7F.
    T c_masked;
    /// The multiplication msb_b * is_slt.
    T bit_b;
    /// The multiplication msb_c * is_slt.
    T bit_c;
    /// An inverse of differing byte if c_comp != b_comp.
    T not_eq_inv;
    /// The most significant bit of operand b.
    T msb_b;
    /// The most significant bit of operand c.
    T msb_c;
    /// The result of the intermediate SLTU operation `b_comp < c_comp`.
    T sltu;
    /// A bollean flag for an intermediate comparison.
    T is_comp_eq;
    /// A boolean flag for comparing the sign bits.
    T is_sign_eq;
    /// The comparison bytes to be looked up.
    T comparison_bytes[2];
  };

  /// The column layout for the chip.
  template <typename T>
  struct ShiftLeftCols {
    /// The output operand.
    Word<T> a;
    /// The first input operand.
    Word<T> b;
    /// The second input operand.
    Word<T> c;
    /// The least significant byte of `c`. Used to verify `shift_by_n_bits` and `shift_by_n_bytes`.
    T c_least_sig_byte[BYTE_SIZE];
    /// A boolean array whose `i`th element indicates whether `num_bits_to_shift = i`.
    T shift_by_n_bits[BYTE_SIZE];
    /// The number to multiply to shift `b` by `num_bits_to_shift`. (i.e., `2^num_bits_to_shift`)
    T bit_shift_multiplier;
    /// The result of multiplying `b` by `bit_shift_multiplier`.
    T bit_shift_result[WORD_SIZE];
    /// The carry propagated when multiplying `b` by `bit_shift_multiplier`.
    T bit_shift_result_carry[WORD_SIZE];
    /// A boolean array whose `i`th element indicates whether `num_bytes_to_shift = i`.
    T shift_by_n_bytes[WORD_SIZE];
    T is_real;
  };

  /// The column layout for the chip.
  template <typename T>
  struct ShiftRightCols {
    /// The output operand.
    Word<T> a;
    /// The first input operand.
    Word<T> b;
    /// The second input operand.
    Word<T> c;
    /// A boolean array whose `i`th element indicates whether `num_bits_to_shift = i`.
    T shift_by_n_bits[BYTE_SIZE];
    /// A boolean array whose `i`th element indicates whether `num_bytes_to_shift = i`.
    T shift_by_n_bytes[WORD_SIZE];
    /// The result of "byte-shifting" the input operand `b` by `num_bytes_to_shift`.
    T byte_shift_result[LONG_WORD_SIZE];
    /// The result of "bit-shifting" the byte-shifted input by `num_bits_to_shift`.
    T bit_shift_result[LONG_WORD_SIZE];
    /// The carry output of `shrcarry` on each byte of `byte_shift_result`.
    T shr_carry_output_carry[LONG_WORD_SIZE];
    /// The shift byte output of `shrcarry` on each byte of `byte_shift_result`.
    T shr_carry_output_shifted_byte[LONG_WORD_SIZE];
    /// The most significant bit of `b`.
    T b_msb;
    /// The least significant byte of `c`. Used to verify `shift_by_n_bits` and `shift_by_n_bytes`.
    T c_least_sig_byte[BYTE_SIZE];
    /// If the opcode is SRL.
    T is_srl;
    /// If the opcode is SRA.
    T is_sra;
    /// Selector to know whether this row is enabled.
    T is_real;
  };

  /// Memory Record.
  ///
  /// This object encapsulates the information needed to prove a memory access operation. This
  /// includes the chunk, timestamp, and value of the memory address.
  struct MemoryRecord {
    /// The chunk number.
    uint32_t chunk;
    /// The timestamp.
    uint32_t timestamp;
    /// The value.
    uint32_t value;
  };

  /// Memory Local Event.
  ///
  /// This object encapsulates the information needed to prove a memory access operation within a
  /// chunk. This includes the address, initial memory access, and final memory access within a
  /// chunk.
  struct MemoryLocalEvent {
    /// The address.
    uint32_t addr;
    /// The initial memory access.
    MemoryRecord initial_mem_access;
    /// The final memory access.
    MemoryRecord final_mem_access;
  };

  template <typename T>
  struct SingleMemoryLocal {
    /// The address of the memory access.
    T addr;
    /// The initial chunk of the memory access.
    T initial_chunk;
    /// The final chunk of the memory access.
    T final_chunk;
    /// The initial clk of the memory access.
    T initial_clk;
    /// The final clk of the memory access.
    T final_clk;
    /// The initial value of the memory access.
    Word<T> initial_value;
    /// The final value of the memory access.
    Word<T> final_value;
    /// Whether the memory access is a real access.
    T is_real;
  };

  /// Memory Initialize/Finalize Event.
  ///
  /// This object encapsulates the information needed to prove a memory initialize or finalize
  /// operation. This includes the address, value, chunk, timestamp, and whether the memory is
  /// initialized or finalized.
  struct MemoryInitializeFinalizeEvent {
    /// The address.
    uint32_t addr;
    /// The value.
    uint32_t value;
    /// The chunk number.
    uint32_t chunk;
    /// The timestamp.
    uint32_t timestamp;
    /// The used flag.
    uint32_t used;
  };

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
    /// Most sig byte LE bit decomposition.
    T most_sig_byte_decomp[8];
    /// Check the range of the last byte.
    IsZeroOperation<T> upper_all_one;
  };

  template <typename T>
  struct MemoryInitializeFinalizeCols {
    /// The chunk number of the memory access.
    T chunk;
    /// The timestamp of the memory access.
    T timestamp;
    /// The address of the memory access.
    T addr;
    /// Comparison assertions for address to be strictly increasing.
    AssertLtColsBits<T, 32> lt_cols;
    /// A bit decomposition of `addr`.
    FieldBitDecomposition<T> addr_bits;
    /// The value of the memory access.
    T value[32];
    /// Whether the memory access is a real access.
    T is_real;
    /// Whether or not we are making the assertion `addr < addr_next`.
    T is_next_comp;
    /// A witness to assert whether or not we the previous address is zero.
    IsZeroOperation<T> is_prev_addr_zero;
    /// Auxiliary column, equal to `(1 - is_prev_addr_zero.result) * is_first_row`.
    T is_first_comp;
    /// A flag to indicate the last non-padded address. An auxiliary column needed for degree 3.
    T is_last_addr;
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

  /// Syscall Event.
  ///
  /// This object encapsulated the information needed to prove a syscall invocation from the CPU
  /// table. This includes its chunk, clk, syscall id, arguments, other relevant information.
  struct SyscallEvent {
    uint32_t chunk;
    /// The clock cycle.
    uint32_t clk;
    /// The syscall id.
    uint32_t syscall_id;
    /// The first operand value (`op_b`).
    uint32_t arg1;
    /// The second operand value (`op_c`).
    uint32_t arg2;
  };

  /// The column layout for the chip.
  template <typename T>
  struct SyscallCols {
    /// The chunk number of the syscall.
    T chunk;
    /// The clk of the syscall.
    T clk;
    /// The syscall_id of the syscall.
    T syscall_id;
    /// The arg1.
    T arg1;
    /// The arg2.
    T arg2;
    T is_real;
  };

  /// RISC-V 32IM Instruction.
  ///
  /// The structure of the instruction differs from the RISC-V ISA. We do not encode the instructions
  /// as 32-bit words, but instead use a custom encoding that is more friendly to decode in the
  /// Pico zkVM.
  struct Instruction {
    /// The operation to emulate.
    Opcode opcode;
    /// The first operand.
    uint32_t op_a;
    /// The second operand.
    uint32_t op_b;
    /// The third operand.
    uint32_t op_c;
    /// Whether the second operand is an immediate value.
    bool imm_b;
    /// Whether the third operand is an immediate value.
    bool imm_c;
  };

  /// The column layout for instructions.
  template <typename T>
  struct InstructionCols {
    /// The opcode for this cycle.
    T opcode;
    /// The first operand for this instruction.
    Word<T> op_a;
    /// The second operand for this instruction.
    Word<T> op_b;
    /// The third operand for this instruction.
    Word<T> op_c;
    /// Flags to indicate if op_a is register 0.
    T op_a_0;
  };

  template <typename T>
  struct MemoryAccessCols {
    /// The value of the memory access.
    Word<T> value;
    /// The previous chunk and timestamp that this memory access is being read from.
    T prev_chunk;
    T prev_clk;
    /// This will be true if the current chunk == prev_access's chunk, else false.
    T compare_clk;
    /// The following columns are decomposed limbs for the difference between the current access's
    /// timestamp and the previous access's timestamp.  Note the actual value of the timestamp
    /// is either the accesses' chunk or clk depending on the value of compare_clk.
    ///
    /// This column is the least significant 16 bit limb of current access timestamp - prev access
    /// timestamp.
    T diff_16bit_limb;
    /// This column is the most significant 8 bit limb of current access timestamp - prev access
    /// timestamp.
    T diff_8bit_limb;
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

  /// Memory Record Enum.
  ///
  /// This enum represents the different types of memory records that can be stored in the memory
  /// event such as reads and writes.
  class MemoryRecordEnum
  {
  public:
    enum class Tag {
      /// Read.
      Read,
      /// Write.
      Write,
    };

    struct Read_Body {
      MemoryReadRecord _0;
    };

    struct Write_Body {
      MemoryWriteRecord _0;
    };

    Tag tag;
    union {
      Read_Body read;
      Write_Body write;
    };

    __host__ __device__ __forceinline__ uint32_t value() const
    {
      if (tag == MemoryRecordEnum::Tag::Read) {
        return read._0.value;
      } else {
        return write._0.value;
      }
    }
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

  template <typename T>
  struct MemoryInstructionCols {
    // The opcode for this cycle
    T opcode;

    // Flag to indicate if op_a is register 0
    T op_a_0;

    // Memory instruction flags
    T is_lb;
    T is_lbu;
    T is_lh;
    T is_lhu;
    T is_lw;
    T is_sb;
    T is_sh;
    T is_sw;

    // Operand memory access descriptions
    MemoryReadWriteCols<T> op_a_access;
    MemoryReadCols<T> op_b_access;
    MemoryReadCols<T> op_c_access;
  };

  template <typename F>
  struct MemoryChipValueCols {
    // The current chunk
    F chunk;

    // The clock cycle value for memory offset
    F clk;

    // addr_word = aligned + offset; must be checked not to overflow
    Word<F> addr_word;
    FieldWordRangeChecker<F> addr_word_range_checker;
    F addr_aligned;

    // Least significant byte decomposition (little-endian) of aligned address
    F aa_least_sig_byte_decomp[6];
    F addr_offset;

    // Memory read/write info
    MemoryReadWriteCols<F> memory_access;
    F offset_is_one;
    F offset_is_two;
    F offset_is_three;

    // Most significant byte decomposition of memory value (used for sign detection)
    F most_sig_byte_decomp[8];

    // Unsigned memory value after applying offset logic
    Word<F> unsigned_mem_val;

    // Flags for positive or negative memory value, not writing to x0
    F mem_value_is_pos_not_x0;
    F mem_value_is_neg_not_x0;

    // Memory instruction info
    MemoryInstructionCols<F> instruction;
  };

  /// FFI CPU Event.
  ///
  /// This object is a mirror of CpuEvent but has a stable ABI for FFI operations
  struct FfiCpuEvent {
    /// The chunk number.
    uint32_t chunk;
    /// The clock cycle.
    uint32_t clk;
    /// The program counter.
    uint32_t pc;
    /// The next program counter.
    uint32_t next_pc;
    /// The instruction.
    Instruction instruction;
    /// The first operand.
    uint32_t a;
    /// The first operand memory record.
    FfiOption<MemoryRecordEnum> a_record;
    /// The second operand.
    uint32_t b;
    /// The second operand memory record.
    FfiOption<MemoryRecordEnum> b_record;
    /// The third operand.
    uint32_t c;
    /// The third operand memory record.
    FfiOption<MemoryRecordEnum> c_record;
    /// The memory value.
    FfiOption<uint32_t> memory;
    /// The memory record.
    FfiOption<MemoryRecordEnum> memory_record;
    /// The exit code.
    uint32_t exit_code;
  };

  template <class F>
  struct BranchCols {
    /// The current program counter.
    Word<F> pc;
    FieldWordRangeChecker<F> pc_range_checker;

    /// The next program counter.
    Word<F> next_pc;
    FieldWordRangeChecker<F> next_pc_range_checker;

    /// Whether a equals b.
    F a_eq_b;

    /// Whether a is greater than b.
    F a_gt_b;

    /// Whether a is less than b.
    F a_lt_b;
  };

  template <class F>
  struct JumpCols {
    /// The current program counter.
    Word<F> pc;
    FieldWordRangeChecker<F> pc_range_checker;

    /// The next program counter.
    Word<F> next_pc;
    FieldWordRangeChecker<F> next_pc_range_checker;

    // A range checker for `op_a` which may contain `pc + 4`.
    FieldWordRangeChecker<F> op_a_range_checker;
  };

  template <class F>
  struct AuipcCols {
    /// The current program counter.
    Word<F> pc;
    FieldWordRangeChecker<F> pc_range_checker;
  };

  template <class F>
  struct EcallCols {
    /// Whether the current ecall is ENTER_UNCONSTRAINED.
    IsZeroOperation<F> is_enter_unconstrained;

    /// Whether the current ecall is HINT_LEN.
    IsZeroOperation<F> is_hint_len;

    /// Whether the current ecall is HALT.
    IsZeroOperation<F> is_halt;

    /// Whether the current ecall is a COMMIT.
    IsZeroOperation<F> is_commit;

    /// Whether the current ecall is a COMMIT_DEFERRED.
    IsZeroOperation<F> is_commit_deferred_proofs;

    /// Field to store the word index passed into the COMMIT ecall.  index_bitmap[word index]
    /// should be set to 1 and everything else set to 0.
    F index_bitmap[PV_DIGEST_NUM_WORDS];

    /// Columns to babybear range check the halt/commit_deferred_proofs operand.
    FieldWordRangeChecker<F> operand_range_check_cols;

    /// The operand value to babybear range check.
    Word<F> operand_to_check;
  };

  template <class F>
  union OpcodeSpecificCols {
    BranchCols<F> branch;
    JumpCols<F> jump;
    AuipcCols<F> auipc;
    EcallCols<F> ecall;
  };

  template <class F>
  struct OpcodeSelectorCols {
    /// Whether op_b F is an immediate value.
    F imm_b;

    /// Whether op_c F is an immediate value.
    F imm_c;

    /// Table selectors for opcodes.
    F is_alu;

    /// Table selectors for opcodes.
    F is_ecall;

    /// Memory Instructions.
    F is_lb;
    F is_lbu;
    F is_lh;
    F is_lhu;
    F is_lw;
    F is_sb;
    F is_sh;
    F is_sw;

    /// Branch Instructions.
    F is_beq;
    F is_bne;
    F is_blt;
    F is_bge;
    F is_bltu;
    F is_bgeu;

    /// Jump Instructions.
    F is_jalr;
    F is_jal;

    /// MF iscellaneous.
    F is_auipc;
    F is_unimpl;
  };

  template <class F>
  struct CpuCols {
    /// The current chunk.
    F chunk;

    /// The clock cycle value.  This should be within 24 bits.
    F clk;
    /// The least significant 16 bit limb of clk.
    F clk_16bit_limb;
    /// The most significant 8 bit limb of clk.
    F clk_8bit_limb;

    /// The program counter value.
    F pc;

    /// The expected next program counter value.
    F next_pc;

    /// Columns related to the instruction.
    InstructionCols<F> instruction;

    /// Selectors for the opcode.
    OpcodeSelectorCols<F> opcode_selector;

    /// Operand values, either from registers or immediate values.
    MemoryReadWriteCols<F> op_a_access;
    MemoryReadCols<F> op_b_access;
    MemoryReadCols<F> op_c_access;

    OpcodeSpecificCols<F> opcode_specific;

    /// Selector to label whether this row is a non padded row.
    F is_real;

    /// The branching column is equal to:
    ///
    /// > is_beq & a_eq_b ||
    /// > is_bne & (a_lt_b | a_gt_b) ||
    /// > (is_blt | is_bltu) & a_lt_b ||
    /// > (is_bge | is_bgeu) & (a_eq_b | a_gt_b)
    F branching;

    /// The not branching column is equal to:
    ///
    /// > is_beq & !a_eq_b ||
    /// > is_bne & !(a_lt_b | a_gt_b) ||
    /// > (is_blt | is_bltu) & !a_lt_b ||
    /// > (is_bge | is_bgeu) & !(a_eq_b | a_gt_b)
    F not_branching;

    /// The result of selectors.is_ecall * the send_to_table column for the ECALL opcode.
    F ecall_mul_send_to_table;

    /// The result of selectors.is_ecall * (is_halt)
    F ecall_range_check_operand;

    /// This is true for all instructions that are not jumps, branches, and halt.  Those
    /// instructions may move the program counter to a non sequential instruction.
    F is_sequential_instr;
  };

  /// A block of columns for septic extension.
  template <typename T>
  struct SepticBlock {
    T _0[7];
  };

  /// A set of columns needed to compute the global interaction elliptic curve digest.
  template <typename T>
  struct GlobalInteractionOperation {
    T offset_bits[8];
    SepticBlock<T> x_coordinate;
    SepticBlock<T> y_coordinate;
    T y6_bit_decomp[30];
    T range_check_witness;
    T poseidon2_input[PERMUTATION_WIDTH];
    T poseidon2_output[PERMUTATION_WIDTH];
  };

  /// Global Interaction Event.
  ///
  /// This event is emitted for all interactions that are sent or received across different chunks.
  struct GlobalInteractionEvent {
    /// The message.
    uint32_t message[7];
    /// Whether the interaction is received or sent.
    bool is_receive;
    /// The kind of the interaction event.
    uint8_t kind;
  };

  /// A set of columns needed to compute the global interaction elliptic curve digest.
  /// It is critical that this struct is at the end of the main trace, as the permutation constraints
  /// will be dependent on this fact. It is also critical the the cumulative sum is at the end of this
  /// struct, for the same reason.
  template <typename T, uintptr_t N>
  struct GlobalAccumulationOperation {
    SepticBlock<T> initial_digest[2];
    SepticBlock<T> sum_checker[N];
    SepticBlock<T> cumulative_sum[N][2];
  };

  template <typename T>
  struct GlobalCols {
    T message[7];
    T kind;
    GlobalInteractionOperation<T> interaction;
    T is_receive;
    T is_send;
    T is_real;
    GlobalAccumulationOperation<T, 1> accumulation;
  };

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
  };

  struct ByteLookupEvent {
    ByteOpcode opcode;
    uint16_t a1;
    uint8_t a2;
    uint8_t b;
    uint8_t c;
  };

  /// The column layout for the chip.
  template <typename T>
  struct ProgramPreprocessedCols {
    T pc;
    InstructionCols<T> instruction;
    OpcodeSelectorCols<T> selectors;
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
#ifdef FEATURE_BABY_BEAR
    T sbox[WIDTH];
#endif
    T post[WIDTH];
  };

  template <typename T>
  struct PartialRound {
#ifdef FEATURE_BABY_BEAR
    T sbox;
#endif
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

  template <const int NumWords>
  struct FpEvent {
    uint32_t chunk;
    uint32_t clk;
    uint32_t x_ptr;
    uint32_t x[NumWords];
    uint32_t y_ptr;
    uint32_t y[NumWords];
    FieldOperation op;
    MemoryWriteRecord x_memory_records[NumWords];
    MemoryReadRecord y_memory_records[NumWords];
  };

  template <typename T, const int NumLimbs, const int NumWitnesses>
  struct FieldOpCols {
    T result[NumLimbs];
    T carry[NumLimbs];
    T witness_low[NumWitnesses];
    T witness_high[NumWitnesses];
  };

  template <typename T, const int NumLimbs, const int NumWitnesses>
  struct FieldLtCols {
    T byte_flags[NumLimbs];
    T lhs_comparison_byte;
    T rhs_comparison_byte;
  };

  template <typename T, const int NumWords, const int NumLimbs, const int NumWitnesses>
  struct FpOpCols {
    T is_real;
    T chunk;
    T clk;
    T is_add;
    T is_sub;
    T is_mul;
    T x_ptr;
    T y_ptr;
    MemoryWriteCols<T> x_access[NumWords];
    MemoryReadCols<T> y_access[NumWords];
    FieldOpCols<T, NumLimbs, NumWitnesses> output;
  };

  template <const int NumWords>
  struct Fp2AddSubEvent {
    uint32_t chunk;
    uint32_t clk;
    uint32_t x_ptr;
    uint32_t x[2 * NumWords];
    uint32_t y_ptr;
    uint32_t y[2 * NumWords];
    FieldOperation op;
    MemoryWriteRecord x_memory_records[2 * NumWords];
    MemoryReadRecord y_memory_records[2 * NumWords];
  };

  template <typename T, const int NumWords, const int NumLimbs, const int NumWitnesses>
  struct Fp2AddSubCols {
    T is_real;
    T chunk;
    T clk;
    T is_add;
    T x_ptr;
    T y_ptr;
    MemoryWriteCols<T> x_access[2 * NumWords];
    MemoryReadCols<T> y_access[2 * NumWords];
    FieldOpCols<T, NumLimbs, NumWitnesses> c0;
    FieldOpCols<T, NumLimbs, NumWitnesses> c1;
  };

  template <const int NumWords>
  struct Fp2MulEvent {
    uint32_t chunk;
    uint32_t clk;
    uint32_t x_ptr;
    uint32_t x[2 * NumWords];
    uint32_t y_ptr;
    uint32_t y[2 * NumWords];
    MemoryWriteRecord x_memory_records[2 * NumWords];
    MemoryReadRecord y_memory_records[2 * NumWords];
  };

  template <typename T, const int NumWords, const int NumLimbs, const int NumWitnesses>
  struct Fp2MulCols {
    T is_real;
    T chunk;
    T clk;
    T x_ptr;
    T y_ptr;
    MemoryWriteCols<T> x_access[2 * NumWords];
    MemoryReadCols<T> y_access[2 * NumWords];
    FieldOpCols<T, NumLimbs, NumWitnesses> a0_mul_b0;
    FieldOpCols<T, NumLimbs, NumWitnesses> a1_mul_b1;
    FieldOpCols<T, NumLimbs, NumWitnesses> a0_mul_b1;
    FieldOpCols<T, NumLimbs, NumWitnesses> a1_mul_b0;
    FieldOpCols<T, NumLimbs, NumWitnesses> c0;
    FieldOpCols<T, NumLimbs, NumWitnesses> c1;
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

  template <const int NumWords>
  struct EllipticCurveAddEvent {
    // The chunk number.
    uint32_t chunk;
    // The clock cycle.
    uint32_t clk;
    // The pointer to the first point.
    uint32_t p_ptr;
    // The first point as a list of words.
    uint32_t p[2 * NumWords];
    // The pointer to the second point.
    uint32_t q_ptr;
    // The second point as a list of words.
    uint32_t q[2 * NumWords];
    // The memory records for the first point.
    MemoryWriteRecord p_memory_records[2 * NumWords];
    // The memory records for the second point.
    MemoryReadRecord q_memory_records[2 * NumWords];
    // // The local memory access records.
    // MemoryLocalEvent local_mem_access[2 * NumWords]; // TODO: It seems to be unused in the trace generation.
  };

  template <typename T, int NumWords, int NumLimbs, int NumWitnesses>
  struct WeierstrassAddAssignCols {
    T is_real;
    T chunk;
    T clk;
    T p_ptr;
    T q_ptr;
    MemoryWriteCols<T> p_access[2 * NumWords];
    MemoryReadCols<T> q_access[2 * NumWords];
    FieldOpCols<T, NumLimbs, NumWitnesses> slope_denominator;
    FieldOpCols<T, NumLimbs, NumWitnesses> slope_numerator;
    FieldOpCols<T, NumLimbs, NumWitnesses> slope;
    FieldOpCols<T, NumLimbs, NumWitnesses> slope_squared;
    FieldOpCols<T, NumLimbs, NumWitnesses> p_x_plus_q_x;
    FieldOpCols<T, NumLimbs, NumWitnesses> x3_ins;
    FieldOpCols<T, NumLimbs, NumWitnesses> p_x_minus_x;
    FieldOpCols<T, NumLimbs, NumWitnesses> y3_ins;
    FieldOpCols<T, NumLimbs, NumWitnesses> slope_times_p_x_minus_x;
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
    // // The local memory access records.
    // MemoryLocalEvent local_mem_access[2 * NumWords]; // TODO: It seems to be unused in the trace generation.
  };

  /// A set of columns to double a point on a Weierstrass curve.
  ///
  /// Right now the number of limbs is assumed to be a constant, although this could be macro-ed or
  /// made generic in the future.
  template <typename T, int NumWords, int NumLimbs, int NumWitnesses>
  struct WeierstrassDoubleAssignCols {
    T is_real;
    T chunk;
    T clk;
    T p_ptr;
    MemoryWriteCols<T> p_access[2 * NumWords];
    FieldOpCols<T, NumLimbs, NumWitnesses> slope_denominator;
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
    /// The memory records for the x coordinate.
    MemoryReadRecord x_memory_records[NumWords];
    /// The memory records for the y coordinate.
    MemoryWriteRecord y_memory_records[NumWords];
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
  struct WeierstrassDecompressCols {
    T is_real;
    T chunk;
    T clk;
    T ptr;
    T sign_bit;
    MemoryReadCols<T> x_access[NumWords];
    MemoryReadWriteCols<T> y_access[NumWords];
    FieldLtCols<T, NumLimbs, NumWitnesses> range_x;
    FieldOpCols<T, NumLimbs, NumWitnesses> x_2;
    FieldOpCols<T, NumLimbs, NumWitnesses> x_3;
    FieldOpCols<T, NumLimbs, NumWitnesses> x_3_plus_b;
    FieldSqrtCols<T, NumLimbs, NumWitnesses> y;
    FieldOpCols<T, NumLimbs, NumWitnesses> neg_y;
  };

  /// A set of columns to compute `WeierstrassDecompress` that decompresses a point on a Weierstrass
  /// curve.
  template <typename T, int NumWords, int NumLimbs, int NumWitnesses>
  struct LexicographicChoiceCols {
    FieldLtCols<T, NumLimbs, NumWitnesses> comparison_lt_cols;
    FieldLtCols<T, NumLimbs, NumWitnesses> neg_y_range_check;
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
    Limbs<T, NumWitness> witness_low;
    Limbs<T, NumWitness> witness_high;
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
    Limbs<T, NumWitness> witness_low;
    Limbs<T, NumWitness> witness_high;
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

  /// FFI-compatible representation of ShaCompressEvent for CUDA interop
  /// This matches the C++ ShaCompressEvent structure exactly
  struct ShaCompressFfiEvent {
    uint32_t chunk;
    uint32_t clk;
    uint32_t w_ptr;
    uint32_t h_ptr;
    uint32_t h[8];
    uint32_t w[64];
    MemoryReadRecord h_read_records[8];
    MemoryReadRecord w_i_read_records[64];
    MemoryWriteRecord h_write_records[8];
  };

  /// FFI-compatible representation of ShaExtendEvent for CUDA interop
  /// This matches the C++ ShaExtendEvent structure exactly
  struct ShaExtendFfiEvent {
    uint32_t chunk;
    uint32_t clk;
    uint32_t w_ptr;
    MemoryReadRecord w_i_minus_15_reads[48];
    MemoryReadRecord w_i_minus_2_reads[48];
    MemoryReadRecord w_i_minus_16_reads[48];
    MemoryReadRecord w_i_minus_7_reads[48];
    MemoryWriteRecord w_i_writes[48];
  };

  struct Uint256MulEvent {
    /// The chunk number
    uint32_t chunk;
    /// The clock cycle
    uint32_t clk;
    /// The pointer to the x value
    uint32_t x_ptr;
    /// The x value as a list of words
    uint32_t x[8];
    /// The pointer to the y value
    uint32_t y_ptr;
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

  template <class T>
  struct Uint256MulCols {
    /// The chunk number of the syscall.
    T chunk;

    /// The clock cycle of the syscall.
    T clk;

    /// The pointer to the first input.
    T x_ptr;

    /// The pointer to the second input, which contains the y value and the modulus.
    T y_ptr;

    // Memory columns.
    // x_memory is written to with the result, which is why it is of type MemoryWriteCols.
    MemoryWriteCols<T> x_memory[8];
    MemoryReadCols<T> y_memory[8];
    MemoryReadCols<T> modulus_memory[8];

    /// Columns for checking if modulus is zero. If it's zero,
    /// then use 2^256 as the effective modulus.
    IsZeroOperation<T> modulus_is_zero;

    /// Column that is equal to is_real * (1 - modulus_is_zero.result).
    T modulus_is_not_zero;

    // Output values. We compute (x * y) % modulus.
    FieldOpCols<T, 32, 63> output;
    FieldLtCols<T, 32, 63> output_range_check;

    T is_real;
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

  struct MemoryReadWriteExtraEventIndices {
    size_t add_sub;
  };
} // namespace pico_gpu
