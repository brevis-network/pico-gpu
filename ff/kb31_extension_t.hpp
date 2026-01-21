#pragma once

#include <ff/kb31_t.hpp>

// Only use KoalaBear
using field_t = kb31_t;

class kb31_extension_t {
   public:
    static constexpr size_t D = 4;
    static constexpr field_t W = field_t {3};

    field_t value[D];

    __device__ __forceinline__ kb31_extension_t() {}

    __device__ __forceinline__ kb31_extension_t(field_t value[4]) {
        for (size_t i = 0; i < D; i++) {
            this->value[i] = value[i];
        }
    }

    __device__ __forceinline__ kb31_extension_t(field_t value) {
        this->value[0] = value;
        for (size_t i = 1; i < D; i++) {
            this->value[i] = field_t(0);
        }
    }

    static __device__ __forceinline__ const kb31_extension_t zero() {
        field_t values[D] = {field_t(0), field_t(0), field_t(0), field_t(0)};
        return kb31_extension_t(values);
    }

    static __device__ __forceinline__ const kb31_extension_t one() {
        field_t values[D] = {field_t::one(), field_t(0), field_t(0), field_t(0)};
        return kb31_extension_t(values);
    }

    __device__ __forceinline__ kb31_extension_t& operator+=(const kb31_extension_t b) {
        for (size_t i = 0; i < D; i++) {
            value[i] += b.value[i];
        }
        return *this;
    }

    friend __device__ __forceinline__ kb31_extension_t operator+(kb31_extension_t a,
                                             const kb31_extension_t b) {
        return a += b;
    }

    __device__ __forceinline__ kb31_extension_t& operator-=(const kb31_extension_t b) {
        for (size_t i = 0; i < D; i++) {
            value[i] -= b.value[i];
        }
        return *this;
    }

    friend __device__ __forceinline__ kb31_extension_t operator-(kb31_extension_t a,
                                             const kb31_extension_t b) {
        return a -= b;
    }

    __device__ __forceinline__ kb31_extension_t& operator*=(const kb31_extension_t b) {
        field_t product[4] = {field_t(0), field_t(0), field_t(0), field_t(0)};
        for (size_t i = 0; i < D; i++) {
            for (size_t j = 0; j < D; j++) {
                if (i + j >= D) {
                    product[i + j - D] += value[i] * b.value[j] * W;
                } else {
                    product[i + j] += value[i] * b.value[j];
                }
            }
        }

        for (size_t i = 0; i < D; i++) {
            value[i] = product[i];
        }

        return *this;
    }

    __device__ __forceinline__ kb31_extension_t& operator*=(const field_t b) {
        for (size_t i = 0; i < D; i++) {
            value[i] *= b;
        }
        return *this;
    }

    friend __device__ __forceinline__ kb31_extension_t operator*(kb31_extension_t a,
                                             const kb31_extension_t b) {
        return a *= b;
    }

    friend __device__ __forceinline__ kb31_extension_t operator*(kb31_extension_t a,
                                             const field_t b) {
        return a *= b;
    }

    __device__ __forceinline__ kb31_extension_t& operator/=(const kb31_extension_t b) {
        *this *= b.reciprocal();
        return *this;
    }

    friend __device__ __forceinline__ kb31_extension_t operator/(kb31_extension_t a,
                                             const kb31_extension_t b) {
        return a /= b;
    }

    __device__ __forceinline__ kb31_extension_t exp_power_of_two(size_t log_power) {
        kb31_extension_t ret = *this;
        for (size_t i = 0; i < log_power; i++) {
            ret *= ret;
        }
        return ret;
    }

    friend __device__ __forceinline__ bool operator!=(
        const kb31_extension_t& lhs, 
        const kb31_extension_t& rhs
    ) {
        for (int i = 0; i < D; ++i) {
            if (lhs.value[i].val != rhs.value[i].val)
                return true;
        }
        return false;
    }

    __device__ __forceinline__ kb31_extension_t frobenius() {
        field_t z0 = field_t(2113994754);
        field_t z = z0;
        kb31_extension_t result;
        for (size_t i = 0; i < D; i++) {
            result.value[i] = value[i] * z;
            z *= z0;
        }
        return result;
    }

    __device__ __forceinline__ kb31_extension_t frobenius_inverse() const {
        kb31_extension_t f = one();
        for (size_t i = 1; i < D; i++) {
            f = (f * *this).frobenius();
        }

        kb31_extension_t a = *this;
        kb31_extension_t b = f;
        field_t g = field_t(0);
        for (size_t i = 1; i < D; i++) {
            g += a.value[i] * b.value[4 - i];
        }
        g *= field_t(11);
        g += a.value[0] * b.value[0];
        return f * g.reciprocal();
    }

    __device__ __forceinline__ kb31_extension_t reciprocal() const {
        bool is_zero = true;
        for (size_t i = 0; i < D; i++) {
            if (value[i].val != 0) {
                is_zero = false;
                break;
            }
        }

        if (is_zero) {
            return zero();
        }

        return frobenius_inverse();
    }
};

__device__ __forceinline__ field_t atomicAdd(field_t* address, field_t value) {
    field_t old_val, new_val;
    do {
        old_val = *address;
        new_val = old_val + value;
    } while (atomicCAS((uint32_t*)&(address->val), old_val.val, new_val.val) != old_val.val);
    return new_val;
}

__device__ __forceinline__ kb31_extension_t atomicAdd(kb31_extension_t* address, kb31_extension_t value) {
    kb31_extension_t old = *address;
    for (int i = 0; i < kb31_extension_t::D; ++i) {
        old.value[i] = atomicAdd(&address->value[i], value.value[i]);
    }
    return old; 
}
