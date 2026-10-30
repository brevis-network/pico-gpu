// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Binary format of the constraint DAG: every node is one 128-bit `Rule` holding an opcode, two
// source operands, a destination slot and two flags.
#pragma once

#include <cstdint>

typedef struct {
    uint64_t low;
    uint64_t high;
} Rule;

typedef enum { OP_ADD, OP_SUB, OP_MUL, OP_NEG, OP_VAR, OP_INV } OperationType;

// Kind of a source operand (the low 4 bits of its encoding).
typedef enum {
    ENTRY_PREPROCESSED, // 0: preprocessed (fixed) trace column
    ENTRY_MAIN,         // 1: main trace column of some part
    ENTRY_PERMUTATION,  // 2: reserved, no longer emitted; keeps the codes below in place
    ENTRY_PUBLIC,       // 3: public value
    ENTRY_CHALLENGE,    // 4: verifier challenge
    ENTRY_EXPOSED,      // 5: reserved, no longer emitted
    SRC_INTERMEDIATE,   // 6: result of an earlier rule, kept in the intermediate buffer
    SRC_CONSTANT,       // 7: field constant stored inline in the index bits
    SRC_IS_FIRST,       // 8: first-row selector
    SRC_IS_LAST,        // 9: last-row selector
    SRC_IS_TRANSITION   // 10: transition selector (1 - last-row selector)
} EntryType;

typedef struct {
    EntryType type;
    uint8_t part;
    uint8_t offset;
    uint32_t index;
} SourceInfo;

// A fully unpacked rule.
typedef struct {
    bool is_constraint;
    bool buffer_result;
    OperationType op;
    SourceInfo x;
    SourceInfo y;
    uint32_t z_index; // destination slot in the intermediate buffer
} DecodedRule;

// Just the fields every rule needs (opcode, flags and x). Interpreters decode y and z separately,
// only when the opcode calls for them, to save registers.
typedef struct {
    bool is_constraint;
    bool buffer_result;
    OperationType op;
    SourceInfo x;
} RuleHeader;

__host__ __device__ __forceinline__ SourceInfo decode_source(uint64_t word);
__host__ __device__ __forceinline__ DecodedRule decode_rule(Rule packed);

// Partial decoders for the lazy path.
__host__ __device__ __forceinline__ RuleHeader decode_rule_header(Rule packed);
__host__ __device__ __forceinline__ SourceInfo decode_y(Rule packed);
__host__ __device__ __forceinline__ uint32_t decode_z_index(Rule packed);

// Fields of a 48-bit source operand: kind, part, rotation and column index.
static const uint64_t ENTRY_SRC_MASK = 0xF;
static const uint64_t ENTRY_PART_SHIFT = 4;
static const uint64_t ENTRY_PART_MASK = 0xFF;
static const uint64_t ENTRY_OFFSET_SHIFT = 12;
static const uint64_t ENTRY_OFFSET_MASK = 0xF;
static const uint64_t ENTRY_INDEX_SHIFT = 16;
static const uint64_t ENTRY_INDEX_MASK = 0xFFFFFFFF;
// Intermediate slots and inline constants keep their value at their own position instead.
static const uint64_t SOURCE_INTERMEDIATE_SHIFT = 4;
static const uint64_t SOURCE_INTERMEDIATE_MASK = 0xFFFFF;
static const uint64_t SOURCE_CONSTANT_SHIFT = 16;
static const uint64_t SOURCE_CONSTANT_MASK = 0xFFFFFFFF;
// Where x, y and z sit in the 128-bit rule: x is the low 48 bits of `low`, y spans the top 16 bits
// of `low` plus the bottom 32 of `high`, and z is the next 24 bits of `high`.
static const uint64_t LOW_48_BITS_MASK = 0xFFFFFFFFFFFF;
static const int Y_HIGH_SHIFT = 16;
static const uint64_t Y_HIGH_MASK = 0xFFFFFFFF;
static const int Z_LOW_SHIFT = 32;
static const uint64_t Z_LOW_MASK = 0xFFFFFF;
// Opcode and flags, at the top of `high`.
static const uint64_t OP_MASK = 0x3F;
static const int OP_SHIFT = 56;
static const uint64_t BUFFER_RESULT_MASK = 0x4000000000000000;
static const uint64_t IS_CONSTRAINT_MASK = 0x8000000000000000;

// Compile-time cross-checks of the hand-written masks.
const uint64_t one = 1;
static_assert(LOW_48_BITS_MASK == (one << 48) - 1, "LOW_48_BITS_MASK must be (1 << 48) - 1");
static_assert(Y_HIGH_MASK == (one << 32) - 1, "Y_HIGH_MASK must be (1 << 32) - 1");
static_assert(Z_LOW_MASK == (one << 24) - 1, "Z_LOW_MASK must be (1 << 24) - 1");
static_assert(OP_MASK == (one << 6) - 1, "OP_MASK must be (1 << 6) - 1");
static_assert(BUFFER_RESULT_MASK == one << 62, "BUFFER_RESULT_MASK must be (1 << 62)");
static_assert(IS_CONSTRAINT_MASK == one << 63, "IS_CONSTRAINT_MASK must be (1 << 63)");

__host__ __device__ __forceinline__ SourceInfo decode_source(uint64_t word) {
    SourceInfo info;
    info.type = (EntryType)(word & ENTRY_SRC_MASK);
    info.part = (word >> ENTRY_PART_SHIFT) & ENTRY_PART_MASK;
    info.offset = (word >> ENTRY_OFFSET_SHIFT) & ENTRY_OFFSET_MASK;
    info.index = (word >> ENTRY_INDEX_SHIFT) & ENTRY_INDEX_MASK;

    if (info.type == SRC_INTERMEDIATE) {
        info.part = 0;
        info.offset = 0;
        info.index = (word >> SOURCE_INTERMEDIATE_SHIFT) & SOURCE_INTERMEDIATE_MASK;
    } else if (info.type == SRC_CONSTANT) {
        info.index = (word >> SOURCE_CONSTANT_SHIFT) & SOURCE_CONSTANT_MASK;
    }
    return info;
}

__host__ __device__ __forceinline__ DecodedRule decode_rule(Rule packed) {
    DecodedRule out;

    uint64_t x_word = (packed.low & LOW_48_BITS_MASK);
    out.x = decode_source(x_word);

    uint64_t y_word = ((packed.low >> 48) | ((packed.high & Y_HIGH_MASK) << Y_HIGH_SHIFT));
    out.y = decode_source(y_word);

    uint64_t z_word = (packed.high >> Z_LOW_SHIFT) & Z_LOW_MASK;
    out.z_index = (z_word >> SOURCE_INTERMEDIATE_SHIFT) & SOURCE_INTERMEDIATE_MASK;

    out.op = (OperationType)((packed.high >> OP_SHIFT) & OP_MASK);

    out.buffer_result = (packed.high & BUFFER_RESULT_MASK) != 0;
    out.is_constraint = (packed.high & IS_CONSTRAINT_MASK) != 0;

    return out;
}

// Opcode, flags and x only.
__host__ __device__ __forceinline__ RuleHeader decode_rule_header(Rule packed) {
    RuleHeader hdr;

    uint64_t x_word = (packed.low & LOW_48_BITS_MASK);
    hdr.x = decode_source(x_word);

    hdr.op = (OperationType)((packed.high >> OP_SHIFT) & OP_MASK);
    hdr.buffer_result = (packed.high & BUFFER_RESULT_MASK) != 0;
    hdr.is_constraint = (packed.high & IS_CONSTRAINT_MASK) != 0;

    return hdr;
}

// Second operand; only binary opcodes (ADD, SUB, MUL) read it.
__host__ __device__ __forceinline__ SourceInfo decode_y(Rule packed) {
    uint64_t y_word = ((packed.low >> 48) | ((packed.high & Y_HIGH_MASK) << Y_HIGH_SHIFT));
    return decode_source(y_word);
}

// Destination slot; only meaningful when buffer_result is set.
__host__ __device__ __forceinline__ uint32_t decode_z_index(Rule packed) {
    uint64_t z_word = (packed.high >> Z_LOW_SHIFT) & Z_LOW_MASK;
    return (z_word >> SOURCE_INTERMEDIATE_SHIFT) & SOURCE_INTERMEDIATE_MASK;
}
