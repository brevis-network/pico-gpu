#pragma once

#include <cstdio>
#include <poseidon2/constants.cuh>
#include <poseidon2/kernels.cuh>

#include "kb31_t.hpp"

#define FUN __host__ __device__ __noinline__

class kb31_cipolla_t
{
public:
  kb31_t real;
  kb31_t imag;

  FUN kb31_cipolla_t(kb31_t real, kb31_t imag)
  {
    this->real = kb31_t(real);
    this->imag = kb31_t(imag);
  }

  FUN static kb31_cipolla_t one() { return kb31_cipolla_t(kb31_t::one(), kb31_t::zero()); }

  FUN kb31_cipolla_t mul_ext(kb31_cipolla_t other, kb31_t nonresidue)
  {
    kb31_t new_real = real * other.real + nonresidue * imag * other.imag;
    kb31_t new_imag = real * other.imag + imag * other.real;
    return kb31_cipolla_t(new_real, new_imag);
  }

  FUN kb31_cipolla_t pow(uint32_t exponent, kb31_t nonresidue)
  {
    kb31_cipolla_t result = kb31_cipolla_t::one();
    kb31_cipolla_t base = *this;

    while (exponent) {
      if (exponent & 1) { result = result.mul_ext(base, nonresidue); }
      exponent >>= 1;
      base = base.mul_ext(base, nonresidue);
    }

    return result;
  }
};

namespace constants {
#ifdef __CUDA_ARCH__
  __constant__ constexpr const kb31_t kb31_frobenius_const[49] = {
    kb31_t(int(1)),          kb31_t(int(1)),          kb31_t(int(1)),          kb31_t(int(1)),
    kb31_t(int(1)),          kb31_t(int(1)),          kb31_t(int(1)),          kb31_t(int(249932420)),
    kb31_t(int(1507635516)), kb31_t(int(418400407)),  kb31_t(int(1455931098)), kb31_t(int(1798331821)),
    kb31_t(int(219395913)),  kb31_t(int(967671407)),  kb31_t(int(420027540)),  kb31_t(int(1057411591)),
    kb31_t(int(328132812)),  kb31_t(int(217796077)),  kb31_t(int(1147686967)), kb31_t(int(1432067604)),
    kb31_t(int(2087546270)), kb31_t(int(155884019)),  kb31_t(int(654771568)),  kb31_t(int(1636205984)),
    kb31_t(int(356126182)),  kb31_t(int(599803706)),  kb31_t(int(1696590466)), kb31_t(int(544132138)),
    kb31_t(int(669140182)),  kb31_t(int(1883835916)), kb31_t(int(1430461478)), kb31_t(int(90835923)),
    kb31_t(int(92515503)),   kb31_t(int(1561748253)), kb31_t(int(731963140)),  kb31_t(int(231339423)),
    kb31_t(int(721642200)),  kb31_t(int(1349944512)), kb31_t(int(1333756424)), kb31_t(int(1546547860)),
    kb31_t(int(1347939566)), kb31_t(int(666560000)),  kb31_t(int(1131530000)), kb31_t(int(1837697502)),
    kb31_t(int(1326477062)), kb31_t(int(1364068631)), kb31_t(int(2109424084)), kb31_t(int(938605660)),
    kb31_t(int(629063286)),
  };

  __constant__ constexpr const kb31_t kb31_double_frobenius_const[49] = {
    kb31_t(int(1)),          kb31_t(int(1)),          kb31_t(int(1)),          kb31_t(int(1)),
    kb31_t(int(1)),          kb31_t(int(1)),          kb31_t(int(1)),          kb31_t(int(1803832602)),
    kb31_t(int(782887483)),  kb31_t(int(1586804983)), kb31_t(int(1965167389)), kb31_t(int(1856929877)),
    kb31_t(int(372963244)),  kb31_t(int(2026350969)), kb31_t(int(1591818588)), kb31_t(int(2076120805)),
    kb31_t(int(75603410)),   kb31_t(int(247345090)),  kb31_t(int(210033395)),  kb31_t(int(1121012471)),
    kb31_t(int(358379851)),  kb31_t(int(1697524332)), kb31_t(int(1219662271)), kb31_t(int(929458421)),
    kb31_t(int(401687366)),  kb31_t(int(1373501145)), kb31_t(int(2060913949)), kb31_t(int(1130910364)),
    kb31_t(int(2068736911)), kb31_t(int(1813198575)), kb31_t(int(2101550341)), kb31_t(int(1947105710)),
    kb31_t(int(375042771)),  kb31_t(int(698819630)),  kb31_t(int(697469116)),  kb31_t(int(39587578)),
    kb31_t(int(2024152476)), kb31_t(int(10976667)),   kb31_t(int(1358333694)), kb31_t(int(413844500)),
    kb31_t(int(1977602137)), kb31_t(int(1978969975)), kb31_t(int(934181796)),  kb31_t(int(1083823847)),
    kb31_t(int(1953070371)), kb31_t(int(1649242345)), kb31_t(int(903686288)),  kb31_t(int(518936657)),
    kb31_t(int(648589698)),
  };

  __constant__ constexpr const kb31_t KB31_A_EC_LOGUP[7] = {
    kb31_t(int(0x31415926)), kb31_t(int(0x53589793)), kb31_t(int(0x23846264)), kb31_t(int(0x33832795)),
    kb31_t(int(0x02884197)), kb31_t(int(0x16939937)), kb31_t(int(0x51058209))};

  __constant__ constexpr const kb31_t KB31_B_EC_LOGUP[7] = {
    kb31_t(int(0x74944592)), kb31_t(int(0x30781640)), kb31_t(int(0x62862089)), kb31_t(int(0x9862803)),
    kb31_t(int(0x48253421)), kb31_t(int(0x17067982)), kb31_t(int(0x14808651))};

  __constant__ constexpr const kb31_t kb31_dummy_x[7] = {
    kb31_t(int(546938927)),  kb31_t(int(1509379008)), kb31_t(int(230266369)), kb31_t(int(757535510)),
    kb31_t(int(1712632789)), kb31_t(int(595785706)),  kb31_t(int(1272488796))};
  __constant__ constexpr const kb31_t kb31_dummy_y[7] = {
    kb31_t(int(585969973)), kb31_t(int(1703627363)), kb31_t(int(1435009742)), kb31_t(int(276846985)),
    kb31_t(int(544259301)), kb31_t(int(968414589)),  kb31_t(int(67451462))};

  __constant__ constexpr kb31_t kb31_start_x[7] = {
    kb31_t(int(1282297783)), kb31_t(int(884251427)), kb31_t(int(1390186945)), kb31_t(int(132125341)),
    kb31_t(int(714101915)),  kb31_t(int(511950180)), kb31_t(int(1023825808))};
  __constant__ constexpr kb31_t kb31_start_y[7] = {
    kb31_t(int(1188614483)), kb31_t(int(1724750090)), kb31_t(int(1138584195)), kb31_t(int(1198897381)),
    kb31_t(int(1166527600)), kb31_t(int(679696589)),  kb31_t(int(864127960))};

#endif

#ifndef __CUDA_ARCH__
  static constexpr const kb31_t kb31_frobenius_const[49] = {
    kb31_t(int(1)),          kb31_t(int(1)),          kb31_t(int(1)),          kb31_t(int(1)),
    kb31_t(int(1)),          kb31_t(int(1)),          kb31_t(int(1)),          kb31_t(int(249932420)),
    kb31_t(int(1507635516)), kb31_t(int(418400407)),  kb31_t(int(1455931098)), kb31_t(int(1798331821)),
    kb31_t(int(219395913)),  kb31_t(int(967671407)),  kb31_t(int(420027540)),  kb31_t(int(1057411591)),
    kb31_t(int(328132812)),  kb31_t(int(217796077)),  kb31_t(int(1147686967)), kb31_t(int(1432067604)),
    kb31_t(int(2087546270)), kb31_t(int(155884019)),  kb31_t(int(654771568)),  kb31_t(int(1636205984)),
    kb31_t(int(356126182)),  kb31_t(int(599803706)),  kb31_t(int(1696590466)), kb31_t(int(544132138)),
    kb31_t(int(669140182)),  kb31_t(int(1883835916)), kb31_t(int(1430461478)), kb31_t(int(90835923)),
    kb31_t(int(92515503)),   kb31_t(int(1561748253)), kb31_t(int(731963140)),  kb31_t(int(231339423)),
    kb31_t(int(721642200)),  kb31_t(int(1349944512)), kb31_t(int(1333756424)), kb31_t(int(1546547860)),
    kb31_t(int(1347939566)), kb31_t(int(666560000)),  kb31_t(int(1131530000)), kb31_t(int(1837697502)),
    kb31_t(int(1326477062)), kb31_t(int(1364068631)), kb31_t(int(2109424084)), kb31_t(int(938605660)),
    kb31_t(int(629063286)),
  };

  static constexpr const kb31_t kb31_double_frobenius_const[49] = {
    kb31_t(int(1)),          kb31_t(int(1)),          kb31_t(int(1)),          kb31_t(int(1)),
    kb31_t(int(1)),          kb31_t(int(1)),          kb31_t(int(1)),          kb31_t(int(1803832602)),
    kb31_t(int(782887483)),  kb31_t(int(1586804983)), kb31_t(int(1965167389)), kb31_t(int(1856929877)),
    kb31_t(int(372963244)),  kb31_t(int(2026350969)), kb31_t(int(1591818588)), kb31_t(int(2076120805)),
    kb31_t(int(75603410)),   kb31_t(int(247345090)),  kb31_t(int(210033395)),  kb31_t(int(1121012471)),
    kb31_t(int(358379851)),  kb31_t(int(1697524332)), kb31_t(int(1219662271)), kb31_t(int(929458421)),
    kb31_t(int(401687366)),  kb31_t(int(1373501145)), kb31_t(int(2060913949)), kb31_t(int(1130910364)),
    kb31_t(int(2068736911)), kb31_t(int(1813198575)), kb31_t(int(2101550341)), kb31_t(int(1947105710)),
    kb31_t(int(375042771)),  kb31_t(int(698819630)),  kb31_t(int(697469116)),  kb31_t(int(39587578)),
    kb31_t(int(2024152476)), kb31_t(int(10976667)),   kb31_t(int(1358333694)), kb31_t(int(413844500)),
    kb31_t(int(1977602137)), kb31_t(int(1978969975)), kb31_t(int(934181796)),  kb31_t(int(1083823847)),
    kb31_t(int(1953070371)), kb31_t(int(1649242345)), kb31_t(int(903686288)),  kb31_t(int(518936657)),
    kb31_t(int(648589698)),
  };

  static constexpr const kb31_t KB31_A_EC_LOGUP[7] = {
    kb31_t(int(0x31415926)), kb31_t(int(0x53589793)), kb31_t(int(0x23846264)), kb31_t(int(0x33832795)),
    kb31_t(int(0x02884197)), kb31_t(int(0x16939937)), kb31_t(int(0x51058209))};
  static constexpr const kb31_t KB31_B_EC_LOGUP[7] = {
    kb31_t(int(0x74944592)), kb31_t(int(0x30781640)), kb31_t(int(0x62862089)), kb31_t(int(0x9862803)),
    kb31_t(int(0x48253421)), kb31_t(int(0x17067982)), kb31_t(int(0x14808651))};

  static constexpr kb31_t kb31_dummy_x[7] = {kb31_t(int(546938927)), kb31_t(int(1509379008)), kb31_t(int(230266369)),
                                             kb31_t(int(757535510)), kb31_t(int(1712632789)), kb31_t(int(595785706)),
                                             kb31_t(int(1272488796))};
  static constexpr kb31_t kb31_dummy_y[7] = {kb31_t(int(585969973)), kb31_t(int(1703627363)), kb31_t(int(1435009742)),
                                             kb31_t(int(276846985)), kb31_t(int(544259301)),  kb31_t(int(968414589)),
                                             kb31_t(int(67451462))};

  static constexpr kb31_t kb31_start_x[7] = {kb31_t(int(1282297783)), kb31_t(int(884251427)), kb31_t(int(1390186945)),
                                             kb31_t(int(132125341)),  kb31_t(int(714101915)), kb31_t(int(511950180)),
                                             kb31_t(int(1023825808))};
  static constexpr kb31_t kb31_start_y[7] = {kb31_t(int(1188614483)), kb31_t(int(1724750090)), kb31_t(int(1138584195)),
                                             kb31_t(int(1198897381)), kb31_t(int(1166527600)), kb31_t(int(679696589)),
                                             kb31_t(int(864127960))};

#endif
} // namespace constants

class kb31_septic_extension_t
{
  // The value of KoalaBear septic extension element.
public:
  kb31_t value[7];
  static constexpr const kb31_t* frobenius_const = constants::kb31_frobenius_const;
  static constexpr const kb31_t* double_frobenius_const = constants::kb31_double_frobenius_const;
  static constexpr const kb31_t* A_EC_LOGUP = constants::KB31_A_EC_LOGUP;
  static constexpr const kb31_t* B_EC_LOGUP = constants::KB31_B_EC_LOGUP;

  FUN kb31_septic_extension_t()
  {
    for (uintptr_t i = 0; i < 7; i++) {
      this->value[i] = kb31_t(0);
    }
  }

  FUN kb31_septic_extension_t(kb31_t value)
  {
    this->value[0] = value;
    for (uintptr_t i = 1; i < 7; i++) {
      this->value[i] = kb31_t(0);
    }
  }

  FUN kb31_septic_extension_t(kb31_t value[7])
  {
    for (uintptr_t i = 0; i < 7; i++) {
      this->value[i] = value[i];
    }
  }

  FUN kb31_septic_extension_t(const kb31_t value[7])
  {
    for (uintptr_t i = 0; i < 7; i++) {
      this->value[i] = value[i];
    }
  }

  FUN kb31_septic_extension_t(const kb31_septic_extension_t& other)
  {
    for (int i = 0; i < 7; ++i) {
      value[i] = other.value[i];
    }
  }

  static FUN kb31_septic_extension_t zero() { return kb31_septic_extension_t(); }

  static FUN kb31_septic_extension_t one() { return kb31_septic_extension_t(kb31_t::one()); }

  static FUN kb31_septic_extension_t two() { return kb31_septic_extension_t(kb31_t::two()); }

  static FUN kb31_septic_extension_t from_canonical_u32(uint32_t n)
  {
    return kb31_septic_extension_t(kb31_t::from_canonical_u32(n));
  }

  FUN kb31_septic_extension_t& operator=(const kb31_septic_extension_t& other)
  {
    if (this != &other) {
      for (int i = 0; i < 7; ++i) {
        value[i] = other.value[i];
      }
    }
    return *this;
  }

  FUN kb31_septic_extension_t& operator+=(const kb31_t b)
  {
    value[0] += b;
    return *this;
  }

  friend FUN kb31_septic_extension_t operator+(kb31_septic_extension_t a, const kb31_t b) { return a += b; }

  FUN kb31_septic_extension_t& operator+=(const kb31_septic_extension_t b)
  {
    for (uintptr_t i = 0; i < 7; i++) {
      value[i] += b.value[i];
    }
    return *this;
  }

  friend FUN kb31_septic_extension_t operator+(kb31_septic_extension_t a, const kb31_septic_extension_t b)
  {
    return a += b;
  }

  FUN kb31_septic_extension_t& operator-=(const kb31_t b)
  {
    value[0] -= b;
    return *this;
  }

  friend FUN kb31_septic_extension_t operator-(kb31_septic_extension_t a, const kb31_t b) { return a -= b; }

  FUN kb31_septic_extension_t& operator-=(const kb31_septic_extension_t b)
  {
    for (uintptr_t i = 0; i < 7; i++) {
      value[i] -= b.value[i];
    }
    return *this;
  }

  friend FUN kb31_septic_extension_t operator-(kb31_septic_extension_t a, const kb31_septic_extension_t b)
  {
    return a -= b;
  }

  FUN kb31_septic_extension_t operator-() const
  {
    auto res = *this;
    for (int i = 0; i < 7; ++i) {
      res.value[i] = kb31_t::zero() - value[i];
    }
    return res;
  }

  FUN kb31_septic_extension_t& operator*=(const kb31_t b)
  {
    for (uintptr_t i = 0; i < 7; i++) {
      value[i] *= b;
    }
    return *this;
  }

  friend FUN kb31_septic_extension_t operator*(kb31_septic_extension_t a, const kb31_t b) { return a *= b; }

  FUN kb31_septic_extension_t& operator*=(const kb31_septic_extension_t b)
  {
    {
      kb31_t res[13] = {};
      for (uintptr_t i = 0; i < 13; i++) {
        res[i] = kb31_t::zero();
      }
      for (uintptr_t i = 0; i < 7; i++) {
        for (uintptr_t j = 0; j < 7; j++) {
          res[i + j] += value[i] * b.value[j];
        }
      }
      for (uintptr_t i = 12; i >= 7; i--) {
        res[i - 7] += res[i] * kb31_t::from_canonical_u32(2);
        res[i - 1] -= res[i] * kb31_t::from_canonical_u32(2);
        res[i] = 0;
      }
      for (uintptr_t i = 0; i < 7; i++) {
        value[i] = res[i];
      }
    }
    return *this;
  }

  friend FUN kb31_septic_extension_t operator*(kb31_septic_extension_t a, const kb31_septic_extension_t b)
  {
    return a *= b;
  }

  FUN bool operator==(const kb31_septic_extension_t rhs) const
  {
    for (uintptr_t i = 0; i < 7; i++) {
      if (value[i].val != rhs.value[i].val) { return false; }
    }
    return true;
  }

  FUN kb31_septic_extension_t frobenius() const
  {
    kb31_t res[7] = {};
    res[0] = value[0];
    for (uintptr_t i = 1; i < 7; i++) {
      res[i] = kb31_t::zero();
    }
    for (uintptr_t i = 1; i < 7; i++) {
      for (uintptr_t j = 0; j < 7; j++) {
        res[j] += value[i] * frobenius_const[7 * i + j];
      }
    }
    return kb31_septic_extension_t(res);
  }

  FUN kb31_septic_extension_t double_frobenius() const
  {
    kb31_t res[7] = {};
    res[0] = value[0];
    for (uintptr_t i = 1; i < 7; i++) {
      res[i] = kb31_t::zero();
    }
    for (uintptr_t i = 1; i < 7; i++) {
      for (uintptr_t j = 0; j < 7; j++) {
        res[j] += value[i] * double_frobenius_const[7 * i + j];
      }
    }
    return kb31_septic_extension_t(res);
  }

  FUN kb31_septic_extension_t pow_r_1() const
  {
    kb31_septic_extension_t base = frobenius();
    base *= double_frobenius();
    kb31_septic_extension_t base_p2 = base.double_frobenius();
    kb31_septic_extension_t base_p4 = base_p2.double_frobenius();
    return base * base_p2 * base_p4;
  }

  FUN kb31_t pow_r() const
  {
    kb31_septic_extension_t pow_r1 = pow_r_1();
    kb31_septic_extension_t pow_r = pow_r1 * *this;
    return pow_r.value[0];
  }

  FUN kb31_septic_extension_t reciprocal() const
  {
    kb31_septic_extension_t pow_r1 = pow_r_1();
    kb31_septic_extension_t pow_r = pow_r1 * *this;
    return pow_r1 * pow_r.value[0].reciprocal();
  }

  friend FUN kb31_septic_extension_t operator/(kb31_septic_extension_t a, kb31_septic_extension_t b)
  {
    return a * b.reciprocal();
  }

  FUN kb31_septic_extension_t& operator/=(const kb31_septic_extension_t a) { return *this *= a.reciprocal(); }

  FUN kb31_septic_extension_t sqrt(kb31_t pow_r) const
  {
    if (*this == kb31_septic_extension_t::zero()) { return *this; }

    kb31_septic_extension_t n_iter = *this;
    kb31_septic_extension_t n_power = *this;
    for (uintptr_t i = 1; i < 30; i++) {
      n_iter *= n_iter;
      if (i >= 23) // 30 - kb31_t::TOP_BITS
      {
        n_power *= n_iter;
      }
    }

    kb31_septic_extension_t n_frobenius = n_power.frobenius();
    kb31_septic_extension_t denominator = n_frobenius;

    n_frobenius = n_frobenius.double_frobenius();
    denominator *= n_frobenius;
    n_frobenius = n_frobenius.double_frobenius();
    denominator *= n_frobenius;
    denominator *= *this;

    kb31_t base = pow_r.reciprocal();
    kb31_t g = kb31_t::from_canonical_u32(3);
    kb31_t a = kb31_t::one();
    kb31_t nonresidue = kb31_t::one() - base;

    while (true) {
      kb31_t is_square = nonresidue ^ 1065353216; // (kb31_t::MOD - 1 >> 1)
      if (is_square.val != kb31_t::one().val) { break; }
      a *= g;
      nonresidue = a.square() - base;
    }

    kb31_cipolla_t x = kb31_cipolla_t(a, kb31_t::one());
    x = x.pow(1065353217, nonresidue); // (kb31_t::MOD + 1 >> 1)

    return denominator * x.real;
  }

  FUN kb31_septic_extension_t universal_hash() const
  {
    return *this * kb31_septic_extension_t(A_EC_LOGUP) + kb31_septic_extension_t(B_EC_LOGUP);
  }

  FUN kb31_septic_extension_t curve_formula() const
  {
    kb31_septic_extension_t result = *this * *this * *this;
    result += *this;
    result += *this;
    result.value[5] += kb31_t::from_canonical_u32(611);
    return result;
  }

  FUN bool is_receive() const
  {
    uint32_t limb = value[6].as_canonical_u32();
    return 1 <= limb && limb <= (kb31_t::MOD - 1) / 2;
  }

  FUN bool is_send() const
  {
    uint32_t limb = value[6].as_canonical_u32();
    return (kb31_t::MOD + 1) / 2 <= limb && limb <= (kb31_t::MOD - 1);
  }

  FUN bool is_exception() const { return value[6] == kb31_t::zero(); }
};

class kb31_septic_curve_t
{
public:
  kb31_septic_extension_t x;
  kb31_septic_extension_t y;

  static constexpr const kb31_t* dummy_x = constants::kb31_dummy_x;
  static constexpr const kb31_t* dummy_y = constants::kb31_dummy_y;
  static constexpr const kb31_t* start_x = constants::kb31_start_x;
  static constexpr const kb31_t* start_y = constants::kb31_start_y;

  FUN kb31_septic_curve_t()
  {
    this->x = kb31_septic_extension_t::zero();
    this->y = kb31_septic_extension_t::zero();
  }

  FUN kb31_septic_curve_t(kb31_septic_extension_t x, kb31_septic_extension_t y)
  {
    this->x = x;
    this->y = y;
  }

  FUN kb31_septic_curve_t(kb31_t value[14])
  {
    for (uintptr_t i = 0; i < 7; i++) {
      this->x.value[i] = value[i];
    }
    for (uintptr_t i = 0; i < 7; i++) {
      this->y.value[i] = value[i + 7];
    }
  }

  FUN kb31_septic_curve_t(kb31_t value_x[7], kb31_t value_y[7])
  {
    for (uintptr_t i = 0; i < 7; i++) {
      this->x.value[i] = value_x[i];
      this->y.value[i] = value_y[i];
    }
  }

  static FUN kb31_septic_curve_t dummy_point()
  {
    kb31_septic_extension_t x;
    kb31_septic_extension_t y;
    for (uintptr_t i = 0; i < 7; i++) {
      x.value[i] = dummy_x[i];
      y.value[i] = dummy_y[i];
    }
    return kb31_septic_curve_t(x, y);
  }

  static FUN kb31_septic_curve_t start_point()
  {
    kb31_septic_extension_t x;
    kb31_septic_extension_t y;
    for (uintptr_t i = 0; i < 7; i++) {
      x.value[i] = start_x[i];
      y.value[i] = start_y[i];
    }
    return kb31_septic_curve_t(x, y);
  }

  static FUN kb31_septic_curve_t infinity()
  {
    return kb31_septic_curve_t(kb31_septic_extension_t::zero(), kb31_septic_extension_t::zero());
  }

  FUN bool is_infinity() const { return x == kb31_septic_extension_t::zero() && y == kb31_septic_extension_t::zero(); }

  FUN kb31_septic_curve_t neg() const
  {
    kb31_septic_curve_t point{this->x, -this->y};
    return point;
  }

  FUN kb31_septic_curve_t& operator+=(const kb31_septic_curve_t b)
  {
    if (b.is_infinity()) { return *this; }
    if (is_infinity()) {
      x = b.x;
      y = b.y;
      return *this;
    }

    kb31_septic_extension_t x_diff = b.x - x;
    if (x_diff == kb31_septic_extension_t::zero()) {
      if (y == b.y) {
        kb31_septic_extension_t y2 = y + y;
        kb31_septic_extension_t x2 = x * x;
        kb31_septic_extension_t slope = (x2 + x2 + x2 + kb31_t::two()) / y2;
        kb31_septic_extension_t result_x = slope * slope - x - x;
        kb31_septic_extension_t result_y = slope * (x - result_x) - y;
        x = result_x;
        y = result_y;
        return *this;
      } else {
        x = kb31_septic_extension_t::zero();
        y = kb31_septic_extension_t::zero();
        return *this;
      }
    } else {
      kb31_septic_extension_t slope = (b.y - y) / x_diff;
      kb31_septic_extension_t new_x = slope * slope - x - b.x;
      y = slope * (x - new_x) - y;
      x = new_x;
      return *this;
    }
  }

  friend FUN kb31_septic_curve_t operator+(kb31_septic_curve_t a, const kb31_septic_curve_t b) { return a += b; }

  static FUN kb31_septic_extension_t
  sum_checker_x(const kb31_septic_curve_t& p1, const kb31_septic_curve_t& p2, const kb31_septic_curve_t& p3)
  {
    kb31_septic_extension_t x_diff = p2.x - p1.x;
    kb31_septic_extension_t y_diff = p2.y - p1.y;
    return (p1.x + p2.x + p3.x) * x_diff * x_diff - y_diff * y_diff;
  }

  /// Lift an x coordinate into an elliptic curve.
  /// As an x-coordinate may not be a valid one, we allow an additional value in `[0, 256)` to the hash input.
  /// Also, we always return the curve point with y-coordinate within `[1, (p-1)/2]`, where p is the characteristic.
  /// The returned values are the curve point, the offset used, and the hash input and output.
  static FUN void lift_x(
    const kb31_septic_extension_t& m,
    kb31_septic_curve_t& point,
    uint8_t& offset,
    kb31_t m_trial[16],
    kb31_t m_hash[16],
    const poseidon2::Poseidon2Constants<kb31_t>& poseidon2_constants)
  {
    using namespace poseidon2;
    for (int i = 0; i < 7; ++i) {
      m_trial[i] = m.value[i];
    }
    for (int i = 8; i < 16; ++i) {
      m_trial[i] = kb31_t::zero();
    }
    for (offset = 0; offset < 256; ++offset) {
      m_trial[7] = kb31_t::from_canonical_u8(offset);

      for (int i = 0; i < 16; ++i) {
        m_hash[i] = m_trial[i];
      }
      permute_state<kb31_t, 16>(m_hash, poseidon2_constants);

      auto x_trial = kb31_septic_extension_t(m_hash);

      auto y_sq = x_trial.curve_formula();
      auto y = y_sq.sqrt(y_sq.pow_r());
      if (!(y == kb31_septic_extension_t::zero())) {
        if (y.is_exception()) { continue; }
        if (y.is_send()) {
          point.x = x_trial;
          point.y = -y;
          return;
        }
        point.x = x_trial;
        point.y = y;
        return;
      }
    }
    printf("curve point couldn't be found after 256 attempts\n");
    assert(false);
  }
};

class kb31_septic_digest_t
{
public:
  kb31_septic_curve_t point;

  FUN kb31_septic_digest_t() { this->point = kb31_septic_curve_t(); }

  FUN kb31_septic_digest_t(kb31_t value[14]) { this->point = kb31_septic_curve_t(value); }

  FUN kb31_septic_digest_t(kb31_septic_extension_t x, kb31_septic_extension_t y)
  {
    this->point = kb31_septic_curve_t(x, y);
  }

  FUN kb31_septic_digest_t(kb31_septic_curve_t point) { this->point = point; }
};