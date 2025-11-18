#pragma once

#include <cstdint>
#include "types.hpp"

namespace pico_gpu::curves {

  // Enumeration of supported curve types
  enum class CurveType { Secp256k1, Secp256r1, Bn254, Ed25519, Bls12381, Unknown };

  // Convert CurveType to string representation
  __PICO_HOSTDEV__ inline const char* curve_type_to_string(CurveType type);

  // Base class for elliptic curve parameters
  template <const int NumWords>
  class EllipticCurveParameters
  {
  public:
    static constexpr CurveType CURVE_TYPE = CurveType::Unknown;
  };

  template <const int NumWords>
  class AffinePoint;

  // Interface for elliptic curve operations
  template <const int NumWords>
  class EllipticCurve : public EllipticCurveParameters<NumWords>
  {
  public:
    static constexpr size_t NB_WITNESS_LIMBS = NumWords;

    __PICO_HOSTDEV__ inline static AffinePoint<NumWords>
    ec_add(const AffinePoint<NumWords>& p, const AffinePoint<NumWords>& q);

    __PICO_HOSTDEV__ inline static AffinePoint<NumWords> ec_double(const AffinePoint<NumWords>& p);

    __PICO_HOSTDEV__ inline static AffinePoint<NumWords> ec_generator();

    __PICO_HOSTDEV__ inline static AffinePoint<NumWords> ec_neutral();

    __PICO_HOSTDEV__ inline static AffinePoint<NumWords> ec_neg(const AffinePoint<NumWords>& p);
  };

  // Affine point implementation
  template <const int NumWords>
  class AffinePoint
  {
  public:
    uint32_t x[NumWords];
    uint32_t y[NumWords];

    __PICO_HOSTDEV__ inline AffinePoint()
    {
      for (size_t i = 0; i < NumWords; i++) {
        this->x[i] = 0;
        this->y[i] = 0;
      }
    }

    __PICO_HOSTDEV__ inline AffinePoint(const uint32_t x[NumWords], const uint32_t y[NumWords])
    {
      for (size_t i = 0; i < NumWords; i++) {
        this->x[i] = x[i];
        this->y[i] = y[i];
      }
    }

    __PICO_HOSTDEV__ inline static AffinePoint<NumWords> zero() { return AffinePoint<NumWords>(); }

    // Convert from words in little-endian format
    __PICO_HOSTDEV__ inline static AffinePoint<NumWords> from_words_le(const uint32_t words[2 * NumWords])
    {
      return AffinePoint<NumWords>(words, words + NumWords);
    }

    // Convert to words in little-endian format
    __PICO_HOSTDEV__ inline void write_to_words_le(uint32_t words[2 * NumWords]) const
    {
      for (size_t i = 0; i < NumWords; i++) {
        words[i] = x[i];
        words[i + NumWords] = y[i];
      }
    }

    // Operator overloads
    __PICO_HOSTDEV__ inline AffinePoint<NumWords> operator+(const AffinePoint<NumWords>& other) const
    {
      return EllipticCurve<NumWords>::ec_add(*this, other);
    }

    __PICO_HOSTDEV__ inline AffinePoint<NumWords> operator-() const { return EllipticCurve<NumWords>::ec_neg(*this); }
  };

  // Implementation of curve_type_to_string
  __PICO_HOSTDEV__ inline const char* curve_type_to_string(CurveType type)
  {
    switch (type) {
    case CurveType::Secp256k1:
      return "Secp256k1";
    case CurveType::Secp256r1:
      return "Secp256r1";
    case CurveType::Bn254:
      return "Bn254";
    case CurveType::Ed25519:
      return "Ed25519";
    case CurveType::Bls12381:
      return "Bls12381";
    default:
      return "Unknown";
    }
  }
} // namespace pico_gpu::curves
