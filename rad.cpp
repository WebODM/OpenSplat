// Writer for Spark's .RAD LoD gaussian splat format.
//
// C++ port of the Rust build-lod tool (https://github.com/sparkjsdev/spark/tree/main/rust/build-lod)
//
// Notes:
//  - f16 conversions ported from half-2.6.0 (round-to-nearest-even).
//  - Heap ported from Rust 1.93.1 alloc::collections::BinaryHeap; its
//    internal array order is observable (bhatt_lod iterates it).
//  - Floats in JSON are formatted with a port of ryu-1.0.20 (the same
//    library serde_json uses).
//  - Compression is raw deflate level 6 via zlib. This is the one deliberate
//    deviation from byte-exactness: the reference uses miniz_oxide, whose
//    compressed bytes differ, but both decode to identical payloads.

#include "rad.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <zlib.h>

namespace rad {
namespace {

// ---------------------------------------------------------------------------
// f16 (port of half-2.6.0 src/binary16/arch.rs software fallback)
// ---------------------------------------------------------------------------

inline uint32_t f32Bits(float f){
    uint32_t x;
    std::memcpy(&x, &f, 4);
    return x;
}

inline float f32FromBits(uint32_t x){
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

// half-2.6.0 f32_to_f16_fallback
uint16_t f16FromF32(float value){
    uint32_t x = f32Bits(value);

    uint32_t sign = x & 0x80000000u;
    uint32_t exp = x & 0x7F800000u;
    uint32_t man = x & 0x007FFFFFu;

    if (exp == 0x7F800000u){
        uint32_t nanBit = (man == 0) ? 0 : 0x0200u;
        return static_cast<uint16_t>((sign >> 16) | 0x7C00u | nanBit | (man >> 13));
    }

    uint32_t halfSign = sign >> 16;
    int32_t unbiasedExp = static_cast<int32_t>(exp >> 23) - 127;
    int32_t halfExp = unbiasedExp + 15;

    if (halfExp >= 0x1F){
        return static_cast<uint16_t>(halfSign | 0x7C00u);
    }

    if (halfExp <= 0){
        if (14 - halfExp > 24){
            return static_cast<uint16_t>(halfSign);
        }
        uint32_t man2 = man | 0x00800000u;
        uint32_t halfMan = man2 >> (14 - halfExp);
        uint32_t roundBit = 1u << (13 - halfExp);
        if ((man2 & roundBit) != 0 && (man2 & (3 * roundBit - 1)) != 0){
            halfMan += 1;
        }
        return static_cast<uint16_t>(halfSign | halfMan);
    }

    uint32_t halfExpBits = static_cast<uint32_t>(halfExp) << 10;
    uint32_t halfMan = man >> 13;
    uint32_t roundBit = 0x00001000u;
    if ((man & roundBit) != 0 && (man & (3 * roundBit - 1)) != 0){
        return static_cast<uint16_t>((halfSign | halfExpBits | halfMan) + 1);
    }else{
        return static_cast<uint16_t>(halfSign | halfExpBits | halfMan);
    }
}

// half-2.6.0 f16_to_f32_fallback
float f16ToF32(uint16_t i){
    if ((i & 0x7FFFu) == 0){
        return f32FromBits(static_cast<uint32_t>(i) << 16);
    }

    uint32_t halfSign = i & 0x8000u;
    uint32_t halfExp = i & 0x7C00u;
    uint32_t halfMan = i & 0x03FFu;

    if (halfExp == 0x7C00u){
        if (halfMan == 0){
            return f32FromBits((halfSign << 16) | 0x7F800000u);
        }else{
            return f32FromBits((halfSign << 16) | 0x7FC00000u | (halfMan << 13));
        }
    }

    uint32_t sign = halfSign << 16;
    int32_t unbiasedExp = (static_cast<int32_t>(halfExp) >> 10) - 15;

    if (halfExp == 0){
        // leading_zeros_u16(halfMan) - 6; halfMan != 0 here
        int32_t lz = 0;
        uint16_t m = static_cast<uint16_t>(halfMan);
        while (!(m & 0x8000u)){ lz++; m <<= 1; }
        int32_t e = lz - 6;
        uint32_t exp = static_cast<uint32_t>(127 - 15 - e) << 23;
        uint32_t man = (halfMan << (14 + e)) & 0x7FFFFFu;
        return f32FromBits(sign | exp | man);
    }

    uint32_t exp = static_cast<uint32_t>(unbiasedExp + 127) << 23;
    uint32_t man = (halfMan & 0x03FFu) << 13;
    return f32FromBits(sign | exp | man);
}

// A stored f16 value (bit pattern). Mirrors half::f16.
struct F16 {
    uint16_t bits = 0;

    F16() = default;
    static F16 fromF32(float v){ F16 h; h.bits = f16FromF32(v); return h; }
    static F16 fromBits(uint16_t b){ F16 h; h.bits = b; return h; }
    float toF32() const { return f16ToF32(bits); }
    bool isNan() const { return (bits & 0x7C00u) == 0x7C00u && (bits & 0x03FFu) != 0; }
};

// half-2.6.0 f16::max: if other > self && !other.is_nan() { other } else { self }
// (non-NaN f16 ordering matches f32 ordering of the converted values, incl. -0 == 0)
inline F16 f16Max(F16 self, F16 other){
    if (!self.isNan() && !other.isNan() && other.toF32() > self.toF32()) return other;
    return self;
}

// ---------------------------------------------------------------------------
// Rust semantics helpers
// ---------------------------------------------------------------------------

// Rust `as` cast f32 -> integer: truncate toward zero, saturate, NaN -> 0.
inline int64_t rustCastI64(float v){
    if (std::isnan(v)) return 0;
    if (v <= -9223372036854775808.0f) return INT64_MIN;
    if (v >= 9223372036854775807.0f) return INT64_MAX;
    return static_cast<int64_t>(v);
}

inline int16_t rustCastI16(float v){
    if (std::isnan(v)) return 0;
    if (v <= -32768.0f) return INT16_MIN;
    if (v >= 32767.0f) return INT16_MAX;
    return static_cast<int16_t>(v);
}

inline uint8_t rustCastU8(float v){
    if (std::isnan(v)) return 0;
    if (v <= 0.0f) return 0;
    if (v >= 255.0f) return 255;
    return static_cast<uint8_t>(v);
}

inline int8_t rustCastI8(float v){
    if (std::isnan(v)) return 0;
    if (v <= -128.0f) return INT8_MIN;
    if (v >= 127.0f) return INT8_MAX;
    return static_cast<int8_t>(v);
}

inline uint16_t rustCastU16(uint64_t v){ return static_cast<uint16_t>(v); }
inline uint32_t rustCastU32(uint64_t v){ return static_cast<uint32_t>(v); }

// Rust f32::max / f32::min (IEEE maxNum/minNum: NaN-ignoring)
inline float rustMax(float a, float b){ return std::fmaxf(a, b); }
inline float rustMin(float a, float b){ return std::fminf(a, b); }

// Rust f32::clamp (propagates NaN self; assumes min <= max)
inline float rustClamp(float x, float mn, float mx){
    if (x < mn) return mn;
    if (x > mx) return mx;
    return x;
}

// Rust f32::total_cmp (IEEE 754 totalOrder)
inline int32_t totalOrderKey(float v){
    int32_t bits = static_cast<int32_t>(f32Bits(v));
    bits ^= static_cast<int32_t>((static_cast<uint32_t>(bits >> 31)) >> 1);
    return bits;
}

// ---------------------------------------------------------------------------
// Vector / quaternion / matrix math (scalar port of glam-0.30.8; the SIMD
// ops used by the reference are all element-wise IEEE identical)
// ---------------------------------------------------------------------------

struct Vec3 {
    float x = 0.0f, y = 0.0f, z = 0.0f;

    Vec3() = default;
    Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    static Vec3 splat(float v){ return Vec3(v, v, v); }

    float operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }
    float &operator[](int i){ return i == 0 ? x : (i == 1 ? y : z); }

    Vec3 operator+(const Vec3 &o) const { return Vec3(x + o.x, y + o.y, z + o.z); }
    Vec3 operator-(const Vec3 &o) const { return Vec3(x - o.x, y - o.y, z - o.z); }
    Vec3 operator*(const Vec3 &o) const { return Vec3(x * o.x, y * o.y, z * o.z); }
    Vec3 operator*(float s) const { return Vec3(x * s, y * s, z * s); }
    Vec3 operator/(float s) const { return Vec3(x / s, y / s, z / s); }

    // glam mul_add without the fma target feature calls f32::mul_add, which is
    // a correctly-rounded fused multiply-add (fmaf), NOT mul-then-add.
    Vec3 mulAdd(const Vec3 &a, const Vec3 &b) const {
        return Vec3(std::fmaf(x, a.x, b.x), std::fmaf(y, a.y, b.y), std::fmaf(z, a.z, b.z));
    }

    // glam sse2 max/min use _mm_max_ps/_mm_min_ps: a > b ? a : b per element
    Vec3 max(const Vec3 &o) const {
        return Vec3(x > o.x ? x : o.x, y > o.y ? y : o.y, z > o.z ? z : o.z);
    }

    Vec3 floorv() const { return Vec3(std::floor(x), std::floor(y), std::floor(z)); }

    float maxElement() const {
        // glam sse2 max_element: max(max(x, y), z) via _mm_max_ss (a > b ? a : b)
        float m = x > y ? x : y;
        return m > z ? m : z;
    }

    // glam sse2 dot3: (x*x' + y*y') + z*z'
    float dot(const Vec3 &o) const { return (x * o.x + y * o.y) + z * o.z; }
    float lengthSquared() const { return dot(*this); }
    float length() const { return std::sqrt(dot(*this)); }
    float distance(const Vec3 &o) const { return (*this - o).length(); }

    bool isFinite() const { return std::isfinite(x) && std::isfinite(y) && std::isfinite(z); }
};

struct I64Vec3 {
    int64_t x = 0, y = 0, z = 0;

    I64Vec3() = default;
    I64Vec3(int64_t x_, int64_t y_, int64_t z_) : x(x_), y(y_), z(z_) {}

    int64_t operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }
    bool operator==(const I64Vec3 &o) const { return x == o.x && y == o.y && z == o.z; }

    I64Vec3 minv(const I64Vec3 &o) const {
        return I64Vec3(std::min(x, o.x), std::min(y, o.y), std::min(z, o.z));
    }
    I64Vec3 maxv(const I64Vec3 &o) const {
        return I64Vec3(std::max(x, o.x), std::max(y, o.y), std::max(z, o.z));
    }
};

struct I64Vec3Hash {
    size_t operator()(const I64Vec3 &v) const {
        // Iteration order of the cell map is never observed by the reference
        // algorithm, so this hash need not match ahash.
        uint64_t h = 0x9E3779B97F4A7C15ull;
        for (uint64_t c : {static_cast<uint64_t>(v.x), static_cast<uint64_t>(v.y), static_cast<uint64_t>(v.z)}){
            h ^= c + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
        }
        return static_cast<size_t>(h);
    }
};

struct Quat {
    float x = 0.0f, y = 0.0f, z = 0.0f, w = 1.0f;

    Quat() = default;
    Quat(float x_, float y_, float z_, float w_) : x(x_), y(y_), z(z_), w(w_) {}

    // glam sse2 dot4: (x*x' + z*z') + (y*y' + w*w')
    float dot(const Quat &o) const { return (x * o.x + z * o.z) + (y * o.y + w * o.w); }
    float length() const { return std::sqrt(dot(*this)); }

    bool isFinite() const {
        return std::isfinite(x) && std::isfinite(y) && std::isfinite(z) && std::isfinite(w);
    }
};

// Column-major 3x3 matrix: m[c] is column c. col(c)[r] = m[c][r].
struct Mat3 {
    float m[3][3];

    static Mat3 identity(){
        Mat3 r{};
        r.m[0][0] = 1.0f; r.m[1][1] = 1.0f; r.m[2][2] = 1.0f;
        return r;
    }
    static Mat3 fromCols(const Vec3 &c0, const Vec3 &c1, const Vec3 &c2){
        Mat3 r;
        for (int i = 0; i < 3; i++){ r.m[0][i] = c0[i]; r.m[1][i] = c1[i]; r.m[2][i] = c2[i]; }
        return r;
    }
    Vec3 col(int c) const { return Vec3(m[c][0], m[c][1], m[c][2]); }

    // glam Mat3A::from_quat
    static Mat3 fromQuat(const Quat &q){
        float x2 = q.x + q.x;
        float y2 = q.y + q.y;
        float z2 = q.z + q.z;
        float xx = q.x * x2;
        float xy = q.x * y2;
        float xz = q.x * z2;
        float yy = q.y * y2;
        float yz = q.y * z2;
        float zz = q.z * z2;
        float wx = q.w * x2;
        float wy = q.w * y2;
        float wz = q.w * z2;
        return fromCols(
            Vec3(1.0f - (yy + zz), xy + wz, xz - wy),
            Vec3(xy - wz, 1.0f - (xx + zz), yz + wx),
            Vec3(xz + wy, yz - wx, 1.0f - (xx + yy)));
    }
};

// glam Quat::from_rotation_axes (Quat::from_mat3a passes the matrix columns)
Quat quatFromMat3(const Mat3 &mat){
    float m00 = mat.m[0][0], m01 = mat.m[0][1], m02 = mat.m[0][2];
    float m10 = mat.m[1][0], m11 = mat.m[1][1], m12 = mat.m[1][2];
    float m20 = mat.m[2][0], m21 = mat.m[2][1], m22 = mat.m[2][2];
    if (m22 <= 0.0f){
        float dif10 = m11 - m00;
        float omm22 = 1.0f - m22;
        if (dif10 <= 0.0f){
            float fourXsq = omm22 - dif10;
            float inv4x = 0.5f / std::sqrt(fourXsq);
            return Quat(fourXsq * inv4x, (m01 + m10) * inv4x, (m02 + m20) * inv4x, (m12 - m21) * inv4x);
        }else{
            float fourYsq = omm22 + dif10;
            float inv4y = 0.5f / std::sqrt(fourYsq);
            return Quat((m01 + m10) * inv4y, fourYsq * inv4y, (m12 + m21) * inv4y, (m20 - m02) * inv4y);
        }
    }else{
        float sum10 = m11 + m00;
        float opm22 = 1.0f + m22;
        if (sum10 <= 0.0f){
            float fourZsq = opm22 - sum10;
            float inv4z = 0.5f / std::sqrt(fourZsq);
            return Quat((m02 + m20) * inv4z, (m12 + m21) * inv4z, fourZsq * inv4z, (m01 - m10) * inv4z);
        }else{
            float fourWsq = opm22 + sum10;
            float inv4w = 0.5f / std::sqrt(fourWsq);
            return Quat((m12 - m21) * inv4w, (m20 - m02) * inv4w, (m01 - m10) * inv4w, fourWsq * inv4w);
        }
    }
}

// ---------------------------------------------------------------------------
// SymMat3 (port of spark-lib/src/symmat3.rs)
// Storage [xx, yy, zz, xy] + [xz, yz]
// ---------------------------------------------------------------------------

struct SymMat3 {
    float v0[4] = {0, 0, 0, 0}; // xx, yy, zz, xy
    float v1[2] = {0, 0};       // xz, yz

    static SymMat3 make(float xx, float yy, float zz, float xy, float xz, float yz){
        SymMat3 r;
        r.v0[0] = xx; r.v0[1] = yy; r.v0[2] = zz; r.v0[3] = xy;
        r.v1[0] = xz; r.v1[1] = yz;
        return r;
    }

    float xx() const { return v0[0]; }
    float yy() const { return v0[1]; }
    float zz() const { return v0[2]; }
    float xy() const { return v0[3]; }
    float xz() const { return v1[0]; }
    float yz() const { return v1[1]; }

    static SymMat3 newScaleQuaternion(const Vec3 &scale, const Quat &quat){
        Mat3 rot = Mat3::fromQuat(quat);
        Vec3 sx = rot.col(0) * scale.x;
        Vec3 sy = rot.col(1) * scale.y;
        Vec3 sz = rot.col(2) * scale.z;
        float xx = sx.x * sx.x + sy.x * sy.x + sz.x * sz.x;
        float yy = sx.y * sx.y + sy.y * sy.y + sz.y * sz.y;
        float zz = sx.z * sx.z + sy.z * sy.z + sz.z * sz.z;
        float xy = sx.x * sx.y + sy.x * sy.y + sz.x * sz.y;
        float xz = sx.x * sx.z + sy.x * sy.z + sz.x * sz.z;
        float yz = sx.y * sx.z + sy.y * sy.z + sz.y * sz.z;
        return make(xx, yy, zz, xy, xz, yz);
    }

    // Vec4/Vec2 mul_add = correctly-rounded fused multiply-add (see Vec3::mulAdd)
    void addWeighted(const SymMat3 &other, float weight){
        for (int i = 0; i < 4; i++) v0[i] = std::fmaf(other.v0[i], weight, v0[i]);
        for (int i = 0; i < 2; i++) v1[i] = std::fmaf(other.v1[i], weight, v1[i]);
    }

    static SymMat3 newAverage(const SymMat3 &a, const SymMat3 &b){
        SymMat3 r;
        for (int i = 0; i < 4; i++) r.v0[i] = std::fmaf(a.v0[i], 0.5f, b.v0[i] * 0.5f);
        for (int i = 0; i < 2; i++) r.v1[i] = std::fmaf(a.v1[i], 0.5f, b.v1[i] * 0.5f);
        return r;
    }

    float determinant() const {
        float m00 = v0[0], m11 = v0[1], m22 = v0[2];
        float m01 = v0[3], m02 = v1[0], m12 = v1[1];
        return m00 * (m11 * m22 - m12 * m12) -
               m01 * (m01 * m22 - m12 * m02) +
               m02 * (m01 * m12 - m11 * m02);
    }

    bool inverse(SymMat3 &out) const {
        float m00 = v0[0], m11 = v0[1], m22 = v0[2];
        float m01 = v0[3], m02 = v1[0], m12 = v1[1];

        float det = determinant();
        float diagMax = rustMax(rustMax(std::fabs(v0[0]), std::fabs(v0[1])), std::fabs(v0[2]));
        float relTol = 1e-9f * rustMax(diagMax * diagMax * diagMax, 1e-30f);
        if (std::fabs(det) < relTol){
            return false;
        }
        float invDet = 1.0f / det;

        out = make((m11 * m22 - m12 * m12) * invDet,
                   (m00 * m22 - m02 * m02) * invDet,
                   (m00 * m11 - m01 * m01) * invDet,
                   (m02 * m12 - m01 * m22) * invDet,
                   (m01 * m12 - m02 * m11) * invDet,
                   (m01 * m02 - m00 * m12) * invDet);
        return true;
    }

    void eigens(float vals[3], Vec3 vecs[3]) const {
        const int MAX_ITERS = 32;
        float eps;
        {
            float s = std::fabs(v0[0]) + std::fabs(v0[1]) + std::fabs(v0[2]);
            eps = 1e-6f * rustMax(s, 1.0f);
        }

        Mat3 current = Mat3::fromCols(
            Vec3(v0[0], v0[3], v1[0]),
            Vec3(v0[3], v0[1], v1[1]),
            Vec3(v1[0], v1[1], v0[2]));
        Mat3 eigs = Mat3::identity();

        auto offDiagNorm2 = [](const Mat3 &a) -> float {
            float a01 = a.m[0][1];
            float a02 = a.m[0][2];
            float a12 = a.m[1][2];
            return a01 * a01 + a02 * a02 + a12 * a12;
        };

        int k = 0;
        while (k < MAX_ITERS && offDiagNorm2(current) > (eps * eps)){
            int p = 0, q = 1;
            float maxVal = std::fabs(current.m[0][1]);
            const struct { int i, j; } cand[2] = {{0, 2}, {1, 2}};
            const float candVal[2] = {std::fabs(current.m[0][2]), std::fabs(current.m[1][2])};
            for (int c = 0; c < 2; c++){
                if (candVal[c] > maxVal){
                    maxVal = candVal[c];
                    p = cand[c].i;
                    q = cand[c].j;
                }
            }

            float apq = current.m[p][q];
            if (std::fabs(apq) > eps){
                float app = current.m[p][p];
                float aqq = current.m[q][q];
                float tau = aqq - app;
                float phi = 0.5f * std::atan2(2.0f * apq, tau);
                float c = std::cos(phi);
                float s = std::sin(phi);

                for (int r = 0; r < 3; r++){
                    float arp = current.m[r][p];
                    float arq = current.m[r][q];
                    current.m[r][p] = c * arp - s * arq;
                    current.m[r][q] = s * arp + c * arq;
                }
                for (int r = 0; r < 3; r++){
                    float apr = current.m[p][r];
                    float aqr = current.m[q][r];
                    current.m[p][r] = c * apr - s * aqr;
                    current.m[q][r] = s * apr + c * aqr;
                }
                current.m[p][q] = 0.0f;
                current.m[q][p] = 0.0f;

                for (int r = 0; r < 3; r++){
                    float vrp = eigs.m[r][p];
                    float vrq = eigs.m[r][q];
                    eigs.m[r][p] = c * vrp - s * vrq;
                    eigs.m[r][q] = s * vrp + c * vrq;
                }
            }
            k += 1;
        }

        float rawVals[3] = {current.m[0][0], current.m[1][1], current.m[2][2]};
        Vec3 rawVecs[3] = {
            Vec3(eigs.m[0][0], eigs.m[1][0], eigs.m[2][0]),
            Vec3(eigs.m[0][1], eigs.m[1][1], eigs.m[2][1]),
            Vec3(eigs.m[0][2], eigs.m[1][2], eigs.m[2][2]),
        };

        for (int j = 0; j < 3; j++){
            float n = std::sqrt(rawVecs[j].x * rawVecs[j].x + rawVecs[j].y * rawVecs[j].y + rawVecs[j].z * rawVecs[j].z);
            if (n > 0.0f){
                rawVecs[j].x /= n;
                rawVecs[j].y /= n;
                rawVecs[j].z /= n;
            }
        }

        // Stable sort of [0,1,2] by descending total_cmp of eigenvalue
        int idx[3] = {0, 1, 2};
        std::stable_sort(idx, idx + 3, [&](int a, int b){
            return totalOrderKey(rawVals[b]) < totalOrderKey(rawVals[a]);
        });
        for (int j = 0; j < 3; j++){
            vals[j] = rawVals[idx[j]];
            vecs[j] = rawVecs[idx[j]];
        }
    }

    void positiveEigens(float vals[3], Vec3 vecs[3]) const {
        eigens(vals, vecs);
        float det =
            vecs[0][0] * (vecs[1][1] * vecs[2][2] - vecs[1][2] * vecs[2][1]) -
            vecs[0][1] * (vecs[1][0] * vecs[2][2] - vecs[1][2] * vecs[2][0]) +
            vecs[0][2] * (vecs[1][0] * vecs[2][1] - vecs[1][1] * vecs[2][0]);
        if (det < 0.0f){
            vecs[2] = Vec3(-vecs[2][0], -vecs[2][1], -vecs[2][2]);
        }
    }
};

// ---------------------------------------------------------------------------
// RustBinaryHeap (port of Rust 1.93.1 alloc BinaryHeap; the internal array
// order is observable via iter() in bhatt_lod, so the exact sift algorithms
// and extend/rebuild_tail heuristic must match)
// ---------------------------------------------------------------------------

// Key: (OrderedFloat<f32>, usize) compared lexicographically. Non-NaN floats
// compare as usual (with -0 == 0); indices break ties.
struct HeapKey {
    float key;
    size_t index;

    bool operator<=(const HeapKey &o) const {
        if (key < o.key) return true;
        if (key > o.key) return false;
        return index <= o.index;
    }
    bool operator>=(const HeapKey &o) const { return o <= *this; }
    bool operator<(const HeapKey &o) const { return !(o <= *this); }
};

class RustBinaryHeap {
public:
    std::vector<HeapKey> data;

    size_t len() const { return data.size(); }
    bool isEmpty() const { return data.empty(); }

    void push(const HeapKey &item){
        size_t oldLen = data.size();
        data.push_back(item);
        siftUp(0, oldLen);
    }

    bool pop(HeapKey &out){
        if (data.empty()) return false;
        HeapKey item = data.back();
        data.pop_back();
        if (!data.empty()){
            std::swap(item, data[0]);
            siftDownToBottom(0);
        }
        out = item;
        return true;
    }

    void extend(const std::vector<HeapKey> &items){
        size_t rebuildFrom = data.size();
        data.insert(data.end(), items.begin(), items.end());
        rebuildTail(rebuildFrom);
    }

private:
    // Hole-based sift_up: returns new position
    size_t siftUp(size_t start, size_t pos){
        HeapKey element = data[pos];
        size_t hole = pos;
        while (hole > start){
            size_t parent = (hole - 1) / 2;
            if (element <= data[parent]) break;
            data[hole] = data[parent];
            hole = parent;
        }
        data[hole] = element;
        return hole;
    }

    size_t siftDownRange(size_t pos, size_t end){
        HeapKey element = data[pos];
        size_t hole = pos;
        size_t child = 2 * hole + 1;
        while (child + 2 <= end){ // Rust: child <= end.saturating_sub(2)
            child += (data[child] <= data[child + 1]) ? 1 : 0;
            if (element >= data[child]){
                data[hole] = element;
                return hole;
            }
            data[hole] = data[child];
            hole = child;
            child = 2 * hole + 1;
        }
        if (child == end - 1 && element < data[child]){
            data[hole] = data[child];
            hole = child;
        }
        data[hole] = element;
        return hole;
    }

    void siftDown(size_t pos){ siftDownRange(pos, data.size()); }

    void siftDownToBottom(size_t pos){
        size_t end = data.size();
        size_t start = pos;
        HeapKey element = data[pos];
        size_t hole = pos;
        size_t child = 2 * hole + 1;
        while (child + 2 <= end){ // Rust: child <= end.saturating_sub(2)
            child += (data[child] <= data[child + 1]) ? 1 : 0;
            data[hole] = data[child];
            hole = child;
            child = 2 * hole + 1;
        }
        if (child == end - 1){
            data[hole] = data[child];
            hole = child;
        }
        data[hole] = element;
        siftUp(start, hole);
    }

    void rebuildTail(size_t start){
        if (start == data.size()) return;

        size_t tailLen = data.size() - start;

        auto log2Fast = [](size_t x) -> size_t {
            size_t r = 0;
            while (x >>= 1) r++;
            return r;
        };

        bool betterToRebuild;
        if (start < tailLen){
            betterToRebuild = true;
        }else if (data.size() <= 2048){
            betterToRebuild = 2 * data.size() < tailLen * log2Fast(start);
        }else{
            betterToRebuild = 2 * data.size() < tailLen * 11;
        }

        if (betterToRebuild){
            rebuild();
        }else{
            for (size_t i = start; i < data.size(); i++){
                siftUp(0, i);
            }
        }
    }

    void rebuild(){
        size_t n = data.size() / 2;
        while (n > 0){
            n -= 1;
            siftDown(n);
        }
    }
};

// ---------------------------------------------------------------------------
// ryu (port of ryu-1.0.20, the float formatter used by serde_json).
// f32 goes through f2d + format32, f64 through d2d + format64.
// ---------------------------------------------------------------------------

namespace ryu {

const char DIGIT_TABLE[201] =
    "00010203040506070809101112131415161718192021222324252627282930313233343536373839"
    "40414243444546474849505152535455565758596061626364656667686970717273747576777879"
    "8081828384858687888990919293949596979899";

const uint64_t DOUBLE_POW5_INV_SPLIT[342][2] = {
    {1ull, 2305843009213693952ull}, {11068046444225730970ull, 1844674407370955161ull},
    {5165088340638674453ull, 1475739525896764129ull}, {7821419487252849886ull, 1180591620717411303ull},
    {8824922364862649494ull, 1888946593147858085ull}, {7059937891890119595ull, 1511157274518286468ull},
    {13026647942995916322ull, 1208925819614629174ull}, {9774590264567735146ull, 1934281311383406679ull},
    {11509021026396098440ull, 1547425049106725343ull}, {16585914450600699399ull, 1237940039285380274ull},
    {15469416676735388068ull, 1980704062856608439ull}, {16064882156130220778ull, 1584563250285286751ull},
    {9162556910162266299ull, 1267650600228229401ull}, {7281393426775805432ull, 2028240960365167042ull},
    {16893161185646375315ull, 1622592768292133633ull}, {2446482504291369283ull, 1298074214633706907ull},
    {7603720821608101175ull, 2076918743413931051ull}, {2393627842544570617ull, 1661534994731144841ull},
    {16672297533003297786ull, 1329227995784915872ull}, {11918280793837635165ull, 2126764793255865396ull},
    {5845275820328197809ull, 1701411834604692317ull}, {15744267100488289217ull, 1361129467683753853ull},
    {3054734472329800808ull, 2177807148294006166ull}, {17201182836831481939ull, 1742245718635204932ull},
    {6382248639981364905ull, 1393796574908163946ull}, {2832900194486363201ull, 2230074519853062314ull},
    {5955668970331000884ull, 1784059615882449851ull}, {1075186361522890384ull, 1427247692705959881ull},
    {12788344622662355584ull, 2283596308329535809ull}, {13920024512871794791ull, 1826877046663628647ull},
    {3757321980813615186ull, 1461501637330902918ull}, {10384555214134712795ull, 1169201309864722334ull},
    {5547241898389809503ull, 1870722095783555735ull}, {4437793518711847602ull, 1496577676626844588ull},
    {10928932444453298728ull, 1197262141301475670ull}, {17486291911125277965ull, 1915619426082361072ull},
    {6610335899416401726ull, 1532495540865888858ull}, {12666966349016942027ull, 1225996432692711086ull},
    {12888448528943286597ull, 1961594292308337738ull}, {17689456452638449924ull, 1569275433846670190ull},
    {14151565162110759939ull, 1255420347077336152ull}, {7885109000409574610ull, 2008672555323737844ull},
    {9997436015069570011ull, 1606938044258990275ull}, {7997948812055656009ull, 1285550435407192220ull},
    {12796718099289049614ull, 2056880696651507552ull}, {2858676849947419045ull, 1645504557321206042ull},
    {13354987924183666206ull, 1316403645856964833ull}, {17678631863951955605ull, 2106245833371143733ull},
    {3074859046935833515ull, 1684996666696914987ull}, {13527933681774397782ull, 1347997333357531989ull},
    {10576647446613305481ull, 2156795733372051183ull}, {15840015586774465031ull, 1725436586697640946ull},
    {8982663654677661702ull, 1380349269358112757ull}, {18061610662226169046ull, 2208558830972980411ull},
    {10759939715039024913ull, 1766847064778384329ull}, {12297300586773130254ull, 1413477651822707463ull},
    {15986332124095098083ull, 2261564242916331941ull}, {9099716884534168143ull, 1809251394333065553ull},
    {14658471137111155161ull, 1447401115466452442ull}, {4348079280205103483ull, 1157920892373161954ull},
    {14335624477811986218ull, 1852673427797059126ull}, {7779150767507678651ull, 1482138742237647301ull},
    {2533971799264232598ull, 1185710993790117841ull}, {15122401323048503126ull, 1897137590064188545ull},
    {12097921058438802501ull, 1517710072051350836ull}, {5988988032009131678ull, 1214168057641080669ull},
    {16961078480698431330ull, 1942668892225729070ull}, {13568862784558745064ull, 1554135113780583256ull},
    {7165741412905085728ull, 1243308091024466605ull}, {11465186260648137165ull, 1989292945639146568ull},
    {16550846638002330379ull, 1591434356511317254ull}, {16930026125143774626ull, 1273147485209053803ull},
    {4951948911778577463ull, 2037035976334486086ull}, {272210314680951647ull, 1629628781067588869ull},
    {3907117066486671641ull, 1303703024854071095ull}, {6251387306378674625ull, 2085924839766513752ull},
    {16069156289328670670ull, 1668739871813211001ull}, {9165976216721026213ull, 1334991897450568801ull},
    {7286864317269821294ull, 2135987035920910082ull}, {16897537898041588005ull, 1708789628736728065ull},
    {13518030318433270404ull, 1367031702989382452ull}, {6871453250525591353ull, 2187250724783011924ull},
    {9186511415162383406ull, 1749800579826409539ull}, {11038557946871817048ull, 1399840463861127631ull},
    {10282995085511086630ull, 2239744742177804210ull}, {8226396068408869304ull, 1791795793742243368ull},
    {13959814484210916090ull, 1433436634993794694ull}, {11267656730511734774ull, 2293498615990071511ull},
    {5324776569667477496ull, 1834798892792057209ull}, {7949170070475892320ull, 1467839114233645767ull},
    {17427382500606444826ull, 1174271291386916613ull}, {5747719112518849781ull, 1878834066219066582ull},
    {15666221734240810795ull, 1503067252975253265ull}, {12532977387392648636ull, 1202453802380202612ull},
    {5295368560860596524ull, 1923926083808324180ull}, {4236294848688477220ull, 1539140867046659344ull},
    {7078384693692692099ull, 1231312693637327475ull}, {11325415509908307358ull, 1970100309819723960ull},
    {9060332407926645887ull, 1576080247855779168ull}, {14626963555825137356ull, 1260864198284623334ull},
    {12335095245094488799ull, 2017382717255397335ull}, {9868076196075591040ull, 1613906173804317868ull},
    {15273158586344293478ull, 1291124939043454294ull}, {13369007293925138595ull, 2065799902469526871ull},
    {7005857020398200553ull, 1652639921975621497ull}, {16672732060544291412ull, 1322111937580497197ull},
    {11918976037903224966ull, 2115379100128795516ull}, {5845832015580669650ull, 1692303280103036413ull},
    {12055363241948356366ull, 1353842624082429130ull}, {841837113407818570ull, 2166148198531886609ull},
    {4362818505468165179ull, 1732918558825509287ull}, {14558301248600263113ull, 1386334847060407429ull},
    {12225235553534690011ull, 2218135755296651887ull}, {2401490813343931363ull, 1774508604237321510ull},
    {1921192650675145090ull, 1419606883389857208ull}, {17831303500047873437ull, 2271371013423771532ull},
    {6886345170554478103ull, 1817096810739017226ull}, {1819727321701672159ull, 1453677448591213781ull},
    {16213177116328979020ull, 1162941958872971024ull}, {14873036941900635463ull, 1860707134196753639ull},
    {15587778368262418694ull, 1488565707357402911ull}, {8780873879868024632ull, 1190852565885922329ull},
    {2981351763563108441ull, 1905364105417475727ull}, {13453127855076217722ull, 1524291284333980581ull},
    {7073153469319063855ull, 1219433027467184465ull}, {11317045550910502167ull, 1951092843947495144ull},
    {12742985255470312057ull, 1560874275157996115ull}, {10194388204376249646ull, 1248699420126396892ull},
    {1553625868034358140ull, 1997919072202235028ull}, {8621598323911307159ull, 1598335257761788022ull},
    {17965325103354776697ull, 1278668206209430417ull}, {13987124906400001422ull, 2045869129935088668ull},
    {121653480894270168ull, 1636695303948070935ull}, {97322784715416134ull, 1309356243158456748ull},
    {14913111714512307107ull, 2094969989053530796ull}, {8241140556867935363ull, 1675975991242824637ull},
    {17660958889720079260ull, 1340780792994259709ull}, {17189487779326395846ull, 2145249268790815535ull},
    {13751590223461116677ull, 1716199415032652428ull}, {18379969808252713988ull, 1372959532026121942ull},
    {14650556434236701088ull, 2196735251241795108ull}, {652398703163629901ull, 1757388200993436087ull},
    {11589965406756634890ull, 1405910560794748869ull}, {7475898206584884855ull, 2249456897271598191ull},
    {2291369750525997561ull, 1799565517817278553ull}, {9211793429904618695ull, 1439652414253822842ull},
    {18428218302589300235ull, 2303443862806116547ull}, {7363877012587619542ull, 1842755090244893238ull},
    {13269799239553916280ull, 1474204072195914590ull}, {10615839391643133024ull, 1179363257756731672ull},
    {2227947767661371545ull, 1886981212410770676ull}, {16539753473096738529ull, 1509584969928616540ull},
    {13231802778477390823ull, 1207667975942893232ull}, {6413489186596184024ull, 1932268761508629172ull},
    {16198837793502678189ull, 1545815009206903337ull}, {5580372605318321905ull, 1236652007365522670ull},
    {8928596168509315048ull, 1978643211784836272ull}, {18210923379033183008ull, 1582914569427869017ull},
    {7190041073742725760ull, 1266331655542295214ull}, {436019273762630246ull, 2026130648867672343ull},
    {7727513048493924843ull, 1620904519094137874ull}, {9871359253537050198ull, 1296723615275310299ull},
    {4726128361433549347ull, 2074757784440496479ull}, {7470251503888749801ull, 1659806227552397183ull},
    {13354898832594820487ull, 1327844982041917746ull}, {13989140502667892133ull, 2124551971267068394ull},
    {14880661216876224029ull, 1699641577013654715ull}, {11904528973500979224ull, 1359713261610923772ull},
    {4289851098633925465ull, 2175541218577478036ull}, {18189276137874781665ull, 1740432974861982428ull},
    {3483374466074094362ull, 1392346379889585943ull}, {1884050330976640656ull, 2227754207823337509ull},
    {5196589079523222848ull, 1782203366258670007ull}, {15225317707844309248ull, 1425762693006936005ull},
    {5913764258841343181ull, 2281220308811097609ull}, {8420360221814984868ull, 1824976247048878087ull},
    {17804334621677718864ull, 1459980997639102469ull}, {17932816512084085415ull, 1167984798111281975ull},
    {10245762345624985047ull, 1868775676978051161ull}, {4507261061758077715ull, 1495020541582440929ull},
    {7295157664148372495ull, 1196016433265952743ull}, {7982903447895485668ull, 1913626293225524389ull},
    {10075671573058298858ull, 1530901034580419511ull}, {4371188443704728763ull, 1224720827664335609ull},
    {14372599139411386667ull, 1959553324262936974ull}, {15187428126271019657ull, 1567642659410349579ull},
    {15839291315758726049ull, 1254114127528279663ull}, {3206773216762499739ull, 2006582604045247462ull},
    {13633465017635730761ull, 1605266083236197969ull}, {14596120828850494932ull, 1284212866588958375ull},
    {4907049252451240275ull, 2054740586542333401ull}, {236290587219081897ull, 1643792469233866721ull},
    {14946427728742906810ull, 1315033975387093376ull}, {16535586736504830250ull, 2104054360619349402ull},
    {5849771759720043554ull, 1683243488495479522ull}, {15747863852001765813ull, 1346594790796383617ull},
    {10439186904235184007ull, 2154551665274213788ull}, {15730047152871967852ull, 1723641332219371030ull},
    {12584037722297574282ull, 1378913065775496824ull}, {9066413911450387881ull, 2206260905240794919ull},
    {10942479943902220628ull, 1765008724192635935ull}, {8753983955121776503ull, 1412006979354108748ull},
    {10317025513452932081ull, 2259211166966573997ull}, {874922781278525018ull, 1807368933573259198ull},
    {8078635854506640661ull, 1445895146858607358ull}, {13841606313089133175ull, 1156716117486885886ull},
    {14767872471458792434ull, 1850745787979017418ull}, {746251532941302978ull, 1480596630383213935ull},
    {597001226353042382ull, 1184477304306571148ull}, {15712597221132509104ull, 1895163686890513836ull},
    {8880728962164096960ull, 1516130949512411069ull}, {10793931984473187891ull, 1212904759609928855ull},
    {17270291175157100626ull, 1940647615375886168ull}, {2748186495899949531ull, 1552518092300708935ull},
    {2198549196719959625ull, 1242014473840567148ull}, {18275073973719576693ull, 1987223158144907436ull},
    {10930710364233751031ull, 1589778526515925949ull}, {12433917106128911148ull, 1271822821212740759ull},
    {8826220925580526867ull, 2034916513940385215ull}, {7060976740464421494ull, 1627933211152308172ull},
    {16716827836597268165ull, 1302346568921846537ull}, {11989529279587987770ull, 2083754510274954460ull},
    {9591623423670390216ull, 1667003608219963568ull}, {15051996368420132820ull, 1333602886575970854ull},
    {13015147745246481542ull, 2133764618521553367ull}, {3033420566713364587ull, 1707011694817242694ull},
    {6116085268112601993ull, 1365609355853794155ull}, {9785736428980163188ull, 2184974969366070648ull},
    {15207286772667951197ull, 1747979975492856518ull}, {1097782973908629988ull, 1398383980394285215ull},
    {1756452758253807981ull, 2237414368630856344ull}, {5094511021344956708ull, 1789931494904685075ull},
    {4075608817075965366ull, 1431945195923748060ull}, {6520974107321544586ull, 2291112313477996896ull},
    {1527430471115325346ull, 1832889850782397517ull}, {12289990821117991246ull, 1466311880625918013ull},
    {17210690286378213644ull, 1173049504500734410ull}, {9090360384495590213ull, 1876879207201175057ull},
    {18340334751822203140ull, 1501503365760940045ull}, {14672267801457762512ull, 1201202692608752036ull},
    {16096930852848599373ull, 1921924308174003258ull}, {1809498238053148529ull, 1537539446539202607ull},
    {12515645034668249793ull, 1230031557231362085ull}, {1578287981759648052ull, 1968050491570179337ull},
    {12330676829633449412ull, 1574440393256143469ull}, {13553890278448669853ull, 1259552314604914775ull},
    {3239480371808320148ull, 2015283703367863641ull}, {17348979556414297411ull, 1612226962694290912ull},
    {6500486015647617283ull, 1289781570155432730ull}, {10400777625036187652ull, 2063650512248692368ull},
    {15699319729512770768ull, 1650920409798953894ull}, {16248804598352126938ull, 1320736327839163115ull},
    {7551343283653851484ull, 2113178124542660985ull}, {6041074626923081187ull, 1690542499634128788ull},
    {12211557331022285596ull, 1352433999707303030ull}, {1091747655926105338ull, 2163894399531684849ull},
    {4562746939482794594ull, 1731115519625347879ull}, {7339546366328145998ull, 1384892415700278303ull},
    {8053925371383123274ull, 2215827865120445285ull}, {6443140297106498619ull, 1772662292096356228ull},
    {12533209867169019542ull, 1418129833677084982ull}, {5295740528502789974ull, 2269007733883335972ull},
    {15304638867027962949ull, 1815206187106668777ull}, {4865013464138549713ull, 1452164949685335022ull},
    {14960057215536570740ull, 1161731959748268017ull}, {9178696285890871890ull, 1858771135597228828ull},
    {14721654658196518159ull, 1487016908477783062ull}, {4398626097073393881ull, 1189613526782226450ull},
    {7037801755317430209ull, 1903381642851562320ull}, {5630241404253944167ull, 1522705314281249856ull},
    {814844308661245011ull, 1218164251424999885ull}, {1303750893857992017ull, 1949062802279999816ull},
    {15800395974054034906ull, 1559250241823999852ull}, {5261619149759407279ull, 1247400193459199882ull},
    {12107939454356961969ull, 1995840309534719811ull}, {5997002748743659252ull, 1596672247627775849ull},
    {8486951013736837725ull, 1277337798102220679ull}, {2511075177753209390ull, 2043740476963553087ull},
    {13076906586428298482ull, 1634992381570842469ull}, {14150874083884549109ull, 1307993905256673975ull},
    {4194654460505726958ull, 2092790248410678361ull}, {18113118827372222859ull, 1674232198728542688ull},
    {3422448617672047318ull, 1339385758982834151ull}, {16543964232501006678ull, 2143017214372534641ull},
    {9545822571258895019ull, 1714413771498027713ull}, {15015355686490936662ull, 1371531017198422170ull},
    {5577825024675947042ull, 2194449627517475473ull}, {11840957649224578280ull, 1755559702013980378ull},
    {16851463748863483271ull, 1404447761611184302ull}, {12204946739213931940ull, 2247116418577894884ull},
    {13453306206113055875ull, 1797693134862315907ull}, {3383947335406624054ull, 1438154507889852726ull},
    {16482362180876329456ull, 2301047212623764361ull}, {9496540929959153242ull, 1840837770099011489ull},
    {11286581558709232917ull, 1472670216079209191ull}, {5339916432225476010ull, 1178136172863367353ull},
    {4854517476818851293ull, 1885017876581387765ull}, {3883613981455081034ull, 1508014301265110212ull},
    {14174937629389795797ull, 1206411441012088169ull}, {11611853762797942306ull, 1930258305619341071ull},
    {5600134195496443521ull, 1544206644495472857ull}, {15548153800622885787ull, 1235365315596378285ull},
    {6430302007287065643ull, 1976584504954205257ull}, {16212288050055383484ull, 1581267603963364205ull},
    {12969830440044306787ull, 1265014083170691364ull}, {9683682259845159889ull, 2024022533073106183ull},
    {15125643437359948558ull, 1619218026458484946ull}, {8411165935146048523ull, 1295374421166787957ull},
    {17147214310975587960ull, 2072599073866860731ull}, {10028422634038560045ull, 1658079259093488585ull},
    {8022738107230848036ull, 1326463407274790868ull}, {9147032156827446534ull, 2122341451639665389ull},
    {11006974540203867551ull, 1697873161311732311ull}, {5116230817421183718ull, 1358298529049385849ull},
    {15564666937357714594ull, 2173277646479017358ull}, {1383687105660440706ull, 1738622117183213887ull},
    {12174996128754083534ull, 1390897693746571109ull}, {8411947361780802685ull, 2225436309994513775ull},
    {6729557889424642148ull, 1780349047995611020ull}, {5383646311539713719ull, 1424279238396488816ull},
    {1235136468979721303ull, 2278846781434382106ull}, {15745504434151418335ull, 1823077425147505684ull},
    {16285752362063044992ull, 1458461940118004547ull}, {5649904260166615347ull, 1166769552094403638ull},
    {5350498001524674232ull, 1866831283351045821ull}, {591049586477829062ull, 1493465026680836657ull},
    {11540886113407994219ull, 1194772021344669325ull}, {18673707743239135ull, 1911635234151470921ull},
    {14772334225162232601ull, 1529308187321176736ull}, {8128518565387875758ull, 1223446549856941389ull},
    {1937583260394870242ull, 1957514479771106223ull}, {8928764237799716840ull, 1566011583816884978ull},
    {14521709019723594119ull, 1252809267053507982ull}, {8477339172590109297ull, 2004494827285612772ull},
    {17849917782297818407ull, 1603595861828490217ull}, {6901236596354434079ull, 1282876689462792174ull},
    {18420676183650915173ull, 2052602703140467478ull}, {3668494502695001169ull, 1642082162512373983ull},
    {10313493231639821582ull, 1313665730009899186ull}, {9122891541139893884ull, 2101865168015838698ull},
    {14677010862395735754ull, 1681492134412670958ull}, {673562245690857633ull, 1345193707530136767ull},
};
const uint64_t DOUBLE_POW5_SPLIT[326][2] = {
    {0ull, 1152921504606846976ull}, {0ull, 1441151880758558720ull},
    {0ull, 1801439850948198400ull}, {0ull, 2251799813685248000ull},
    {0ull, 1407374883553280000ull}, {0ull, 1759218604441600000ull},
    {0ull, 2199023255552000000ull}, {0ull, 1374389534720000000ull},
    {0ull, 1717986918400000000ull}, {0ull, 2147483648000000000ull},
    {0ull, 1342177280000000000ull}, {0ull, 1677721600000000000ull},
    {0ull, 2097152000000000000ull}, {0ull, 1310720000000000000ull},
    {0ull, 1638400000000000000ull}, {0ull, 2048000000000000000ull},
    {0ull, 1280000000000000000ull}, {0ull, 1600000000000000000ull},
    {0ull, 2000000000000000000ull}, {0ull, 1250000000000000000ull},
    {0ull, 1562500000000000000ull}, {0ull, 1953125000000000000ull},
    {0ull, 1220703125000000000ull}, {0ull, 1525878906250000000ull},
    {0ull, 1907348632812500000ull}, {0ull, 1192092895507812500ull},
    {0ull, 1490116119384765625ull}, {4611686018427387904ull, 1862645149230957031ull},
    {9799832789158199296ull, 1164153218269348144ull}, {12249790986447749120ull, 1455191522836685180ull},
    {15312238733059686400ull, 1818989403545856475ull}, {14528612397897220096ull, 2273736754432320594ull},
    {13692068767113150464ull, 1421085471520200371ull}, {12503399940464050176ull, 1776356839400250464ull},
    {15629249925580062720ull, 2220446049250313080ull}, {9768281203487539200ull, 1387778780781445675ull},
    {7598665485932036096ull, 1734723475976807094ull}, {274959820560269312ull, 2168404344971008868ull},
    {9395221924704944128ull, 1355252715606880542ull}, {2520655369026404352ull, 1694065894508600678ull},
    {12374191248137781248ull, 2117582368135750847ull}, {14651398557727195136ull, 1323488980084844279ull},
    {13702562178731606016ull, 1654361225106055349ull}, {3293144668132343808ull, 2067951531382569187ull},
    {18199116482078572544ull, 1292469707114105741ull}, {8913837547316051968ull, 1615587133892632177ull},
    {15753982952572452864ull, 2019483917365790221ull}, {12152082354571476992ull, 1262177448353618888ull},
    {15190102943214346240ull, 1577721810442023610ull}, {9764256642163156992ull, 1972152263052529513ull},
    {17631875447420442880ull, 1232595164407830945ull}, {8204786253993389888ull, 1540743955509788682ull},
    {1032610780636961552ull, 1925929944387235853ull}, {2951224747111794922ull, 1203706215242022408ull},
    {3689030933889743652ull, 1504632769052528010ull}, {13834660704216955373ull, 1880790961315660012ull},
    {17870034976990372916ull, 1175494350822287507ull}, {17725857702810578241ull, 1469367938527859384ull},
    {3710578054803671186ull, 1836709923159824231ull}, {26536550077201078ull, 2295887403949780289ull},
    {11545800389866720434ull, 1434929627468612680ull}, {14432250487333400542ull, 1793662034335765850ull},
    {8816941072311974870ull, 2242077542919707313ull}, {17039803216263454053ull, 1401298464324817070ull},
    {12076381983474541759ull, 1751623080406021338ull}, {5872105442488401391ull, 2189528850507526673ull},
    {15199280947623720629ull, 1368455531567204170ull}, {9775729147674874978ull, 1710569414459005213ull},
    {16831347453020981627ull, 2138211768073756516ull}, {1296220121283337709ull, 1336382355046097823ull},
    {15455333206886335848ull, 1670477943807622278ull}, {10095794471753144002ull, 2088097429759527848ull},
    {6309871544845715001ull, 1305060893599704905ull}, {12499025449484531656ull, 1631326116999631131ull},
    {11012095793428276666ull, 2039157646249538914ull}, {11494245889320060820ull, 1274473528905961821ull},
    {532749306367912313ull, 1593091911132452277ull}, {5277622651387278295ull, 1991364888915565346ull},
    {7910200175544436838ull, 1244603055572228341ull}, {14499436237857933952ull, 1555753819465285426ull},
    {8900923260467641632ull, 1944692274331606783ull}, {12480606065433357876ull, 1215432671457254239ull},
    {10989071563364309441ull, 1519290839321567799ull}, {9124653435777998898ull, 1899113549151959749ull},
    {8008751406574943263ull, 1186945968219974843ull}, {5399253239791291175ull, 1483682460274968554ull},
    {15972438586593889776ull, 1854603075343710692ull}, {759402079766405302ull, 1159126922089819183ull},
    {14784310654990170340ull, 1448908652612273978ull}, {9257016281882937117ull, 1811135815765342473ull},
    {16182956370781059300ull, 2263919769706678091ull}, {7808504722524468110ull, 1414949856066673807ull},
    {5148944884728197234ull, 1768687320083342259ull}, {1824495087482858639ull, 2210859150104177824ull},
    {1140309429676786649ull, 1381786968815111140ull}, {1425386787095983311ull, 1727233711018888925ull},
    {6393419502297367043ull, 2159042138773611156ull}, {13219259225790630210ull, 1349401336733506972ull},
    {16524074032238287762ull, 1686751670916883715ull}, {16043406521870471799ull, 2108439588646104644ull},
    {803757039314269066ull, 1317774742903815403ull}, {14839754354425000045ull, 1647218428629769253ull},
    {4714634887749086344ull, 2059023035787211567ull}, {9864175832484260821ull, 1286889397367007229ull},
    {16941905809032713930ull, 1608611746708759036ull}, {2730638187581340797ull, 2010764683385948796ull},
    {10930020904093113806ull, 1256727927116217997ull}, {18274212148543780162ull, 1570909908895272496ull},
    {4396021111970173586ull, 1963637386119090621ull}, {5053356204195052443ull, 1227273366324431638ull},
    {15540067292098591362ull, 1534091707905539547ull}, {14813398096695851299ull, 1917614634881924434ull},
    {13870059828862294966ull, 1198509146801202771ull}, {12725888767650480803ull, 1498136433501503464ull},
    {15907360959563101004ull, 1872670541876879330ull}, {14553786618154326031ull, 1170419088673049581ull},
    {4357175217410743827ull, 1463023860841311977ull}, {10058155040190817688ull, 1828779826051639971ull},
    {7961007781811134206ull, 2285974782564549964ull}, {14199001900486734687ull, 1428734239102843727ull},
    {13137066357181030455ull, 1785917798878554659ull}, {11809646928048900164ull, 2232397248598193324ull},
    {16604401366885338411ull, 1395248280373870827ull}, {16143815690179285109ull, 1744060350467338534ull},
    {10956397575869330579ull, 2180075438084173168ull}, {6847748484918331612ull, 1362547148802608230ull},
    {17783057643002690323ull, 1703183936003260287ull}, {17617136035325974999ull, 2128979920004075359ull},
    {17928239049719816230ull, 1330612450002547099ull}, {17798612793722382384ull, 1663265562503183874ull},
    {13024893955298202172ull, 2079081953128979843ull}, {5834715712847682405ull, 1299426220705612402ull},
    {16516766677914378815ull, 1624282775882015502ull}, {11422586310538197711ull, 2030353469852519378ull},
    {11750802462513761473ull, 1268970918657824611ull}, {10076817059714813937ull, 1586213648322280764ull},
    {12596021324643517422ull, 1982767060402850955ull}, {5566670318688504437ull, 1239229412751781847ull},
    {2346651879933242642ull, 1549036765939727309ull}, {7545000868343941206ull, 1936295957424659136ull},
    {4715625542714963254ull, 1210184973390411960ull}, {5894531928393704067ull, 1512731216738014950ull},
    {16591536947346905892ull, 1890914020922518687ull}, {17287239619732898039ull, 1181821263076574179ull},
    {16997363506238734644ull, 1477276578845717724ull}, {2799960309088866689ull, 1846595723557147156ull},
    {10973347230035317489ull, 1154122327223216972ull}, {13716684037544146861ull, 1442652909029021215ull},
    {12534169028502795672ull, 1803316136286276519ull}, {11056025267201106687ull, 2254145170357845649ull},
    {18439230838069161439ull, 1408840731473653530ull}, {13825666510731675991ull, 1761050914342066913ull},
    {3447025083132431277ull, 2201313642927583642ull}, {6766076695385157452ull, 1375821026829739776ull},
    {8457595869231446815ull, 1719776283537174720ull}, {10571994836539308519ull, 2149720354421468400ull},
    {6607496772837067824ull, 1343575221513417750ull}, {17482743002901110588ull, 1679469026891772187ull},
    {17241742735199000331ull, 2099336283614715234ull}, {15387775227926763111ull, 1312085177259197021ull},
    {5399660979626290177ull, 1640106471573996277ull}, {11361262242960250625ull, 2050133089467495346ull},
    {11712474920277544544ull, 1281333180917184591ull}, {10028907631919542777ull, 1601666476146480739ull},
    {7924448521472040567ull, 2002083095183100924ull}, {14176152362774801162ull, 1251301934489438077ull},
    {3885132398186337741ull, 1564127418111797597ull}, {9468101516160310080ull, 1955159272639746996ull},
    {15140935484454969608ull, 1221974545399841872ull}, {479425281859160394ull, 1527468181749802341ull},
    {5210967620751338397ull, 1909335227187252926ull}, {17091912818251750210ull, 1193334516992033078ull},
    {12141518985959911954ull, 1491668146240041348ull}, {15176898732449889943ull, 1864585182800051685ull},
    {11791404716994875166ull, 1165365739250032303ull}, {10127569877816206054ull, 1456707174062540379ull},
    {8047776328842869663ull, 1820883967578175474ull}, {836348374198811271ull, 2276104959472719343ull},
    {7440246761515338900ull, 1422565599670449589ull}, {13911994470321561530ull, 1778206999588061986ull},
    {8166621051047176104ull, 2222758749485077483ull}, {2798295147690791113ull, 1389224218428173427ull},
    {17332926989895652603ull, 1736530273035216783ull}, {17054472718942177850ull, 2170662841294020979ull},
    {8353202440125167204ull, 1356664275808763112ull}, {10441503050156459005ull, 1695830344760953890ull},
    {3828506775840797949ull, 2119787930951192363ull}, {86973725686804766ull, 1324867456844495227ull},
    {13943775212390669669ull, 1656084321055619033ull}, {3594660960206173375ull, 2070105401319523792ull},
    {2246663100128858359ull, 1293815875824702370ull}, {12031700912015848757ull, 1617269844780877962ull},
    {5816254103165035138ull, 2021587305976097453ull}, {5941001823691840913ull, 1263492066235060908ull},
    {7426252279614801142ull, 1579365082793826135ull}, {4671129331091113523ull, 1974206353492282669ull},
    {5225298841145639904ull, 1233878970932676668ull}, {6531623551432049880ull, 1542348713665845835ull},
    {3552843420862674446ull, 1927935892082307294ull}, {16055585193321335241ull, 1204959932551442058ull},
    {10846109454796893243ull, 1506199915689302573ull}, {18169322836923504458ull, 1882749894611628216ull},
    {11355826773077190286ull, 1176718684132267635ull}, {9583097447919099954ull, 1470898355165334544ull},
    {11978871809898874942ull, 1838622943956668180ull}, {14973589762373593678ull, 2298278679945835225ull},
    {2440964573842414192ull, 1436424174966147016ull}, {3051205717303017741ull, 1795530218707683770ull},
    {13037379183483547984ull, 2244412773384604712ull}, {8148361989677217490ull, 1402757983365377945ull},
    {14797138505523909766ull, 1753447479206722431ull}, {13884737113477499304ull, 2191809349008403039ull},
    {15595489723564518921ull, 1369880843130251899ull}, {14882676136028260747ull, 1712351053912814874ull},
    {9379973133180550126ull, 2140438817391018593ull}, {17391698254306313589ull, 1337774260869386620ull},
    {3292878744173340370ull, 1672217826086733276ull}, {4116098430216675462ull, 2090272282608416595ull},
    {266718509671728212ull, 1306420176630260372ull}, {333398137089660265ull, 1633025220787825465ull},
    {5028433689789463235ull, 2041281525984781831ull}, {10060300083759496378ull, 1275800953740488644ull},
    {12575375104699370472ull, 1594751192175610805ull}, {1884160825592049379ull, 1993438990219513507ull},
    {17318501580490888525ull, 1245899368887195941ull}, {7813068920331446945ull, 1557374211108994927ull},
    {5154650131986920777ull, 1946717763886243659ull}, {915813323278131534ull, 1216698602428902287ull},
    {14979824709379828129ull, 1520873253036127858ull}, {9501408849870009354ull, 1901091566295159823ull},
    {12855909558809837702ull, 1188182228934474889ull}, {2234828893230133415ull, 1485227786168093612ull},
    {2793536116537666769ull, 1856534732710117015ull}, {8663489100477123587ull, 1160334207943823134ull},
    {1605989338741628675ull, 1450417759929778918ull}, {11230858710281811652ull, 1813022199912223647ull},
    {9426887369424876662ull, 2266277749890279559ull}, {12809333633531629769ull, 1416423593681424724ull},
    {16011667041914537212ull, 1770529492101780905ull}, {6179525747111007803ull, 2213161865127226132ull},
    {13085575628799155685ull, 1383226165704516332ull}, {16356969535998944606ull, 1729032707130645415ull},
    {15834525901571292854ull, 2161290883913306769ull}, {2979049660840976177ull, 1350806802445816731ull},
    {17558870131333383934ull, 1688508503057270913ull}, {8113529608884566205ull, 2110635628821588642ull},
    {9682642023980241782ull, 1319147268013492901ull}, {16714988548402690132ull, 1648934085016866126ull},
    {11670363648648586857ull, 2061167606271082658ull}, {11905663298832754689ull, 1288229753919426661ull},
    {1047021068258779650ull, 1610287192399283327ull}, {15143834390605638274ull, 2012858990499104158ull},
    {4853210475701136017ull, 1258036869061940099ull}, {1454827076199032118ull, 1572546086327425124ull},
    {1818533845248790147ull, 1965682607909281405ull}, {3442426662494187794ull, 1228551629943300878ull},
    {13526405364972510550ull, 1535689537429126097ull}, {3072948650933474476ull, 1919611921786407622ull},
    {15755650962115585259ull, 1199757451116504763ull}, {15082877684217093670ull, 1499696813895630954ull},
    {9630225068416591280ull, 1874621017369538693ull}, {8324733676974063502ull, 1171638135855961683ull},
    {5794231077790191473ull, 1464547669819952104ull}, {7242788847237739342ull, 1830684587274940130ull},
    {18276858095901949986ull, 2288355734093675162ull}, {16034722328366106645ull, 1430222333808546976ull},
    {1596658836748081690ull, 1787777917260683721ull}, {6607509564362490017ull, 2234722396575854651ull},
    {1823850468512862308ull, 1396701497859909157ull}, {6891499104068465790ull, 1745876872324886446ull},
    {17837745916940358045ull, 2182346090406108057ull}, {4231062170446641922ull, 1363966306503817536ull},
    {5288827713058302403ull, 1704957883129771920ull}, {6611034641322878003ull, 2131197353912214900ull},
    {13355268687681574560ull, 1331998346195134312ull}, {16694085859601968200ull, 1664997932743917890ull},
    {11644235287647684442ull, 2081247415929897363ull}, {4971804045566108824ull, 1300779634956185852ull},
    {6214755056957636030ull, 1625974543695232315ull}, {3156757802769657134ull, 2032468179619040394ull},
    {6584659645158423613ull, 1270292612261900246ull}, {17454196593302805324ull, 1587865765327375307ull},
    {17206059723201118751ull, 1984832206659219134ull}, {6142101308573311315ull, 1240520129162011959ull},
    {3065940617289251240ull, 1550650161452514949ull}, {8444111790038951954ull, 1938312701815643686ull},
    {665883850346957067ull, 1211445438634777304ull}, {832354812933696334ull, 1514306798293471630ull},
    {10263815553021896226ull, 1892883497866839537ull}, {17944099766707154901ull, 1183052186166774710ull},
    {13206752671529167818ull, 1478815232708468388ull}, {16508440839411459773ull, 1848519040885585485ull},
    {12623618533845856310ull, 1155324400553490928ull}, {15779523167307320387ull, 1444155500691863660ull},
    {1277659885424598868ull, 1805194375864829576ull}, {1597074856780748586ull, 2256492969831036970ull},
    {5609857803915355770ull, 1410308106144398106ull}, {16235694291748970521ull, 1762885132680497632ull},
    {1847873790976661535ull, 2203606415850622041ull}, {12684136165428883219ull, 1377254009906638775ull},
    {11243484188358716120ull, 1721567512383298469ull}, {219297180166231438ull, 2151959390479123087ull},
    {7054589765244976505ull, 1344974619049451929ull}, {13429923224983608535ull, 1681218273811814911ull},
    {12175718012802122765ull, 2101522842264768639ull}, {14527352785642408584ull, 1313451776415480399ull},
    {13547504963625622826ull, 1641814720519350499ull}, {12322695186104640628ull, 2052268400649188124ull},
    {16925056528170176201ull, 1282667750405742577ull}, {7321262604930556539ull, 1603334688007178222ull},
    {18374950293017971482ull, 2004168360008972777ull}, {4566814905495150320ull, 1252605225005607986ull},
    {14931890668723713708ull, 1565756531257009982ull}, {9441491299049866327ull, 1957195664071262478ull},
    {1289246043478778550ull, 1223247290044539049ull}, {6223243572775861092ull, 1529059112555673811ull},
    {3167368447542438461ull, 1911323890694592264ull}, {1979605279714024038ull, 1194577431684120165ull},
    {7086192618069917952ull, 1493221789605150206ull}, {18081112809442173248ull, 1866527237006437757ull},
    {13606538515115052232ull, 1166579523129023598ull}, {7784801107039039482ull, 1458224403911279498ull},
    {507629346944023544ull, 1822780504889099373ull}, {5246222702107417334ull, 2278475631111374216ull},
    {3278889188817135834ull, 1424047269444608885ull}, {8710297504448807696ull, 1780059086805761106ull},
};

inline uint32_t decimalLength9(uint32_t v){
    if (v >= 100000000) return 9;
    if (v >= 10000000) return 8;
    if (v >= 1000000) return 7;
    if (v >= 100000) return 6;
    if (v >= 10000) return 5;
    if (v >= 1000) return 4;
    if (v >= 100) return 3;
    if (v >= 10) return 2;
    return 1;
}

inline uint32_t decimalLength17(uint64_t v){
    if (v >= 10000000000000000ull) return 17;
    if (v >= 1000000000000000ull) return 16;
    if (v >= 100000000000000ull) return 15;
    if (v >= 10000000000000ull) return 14;
    if (v >= 1000000000000ull) return 13;
    if (v >= 100000000000ull) return 12;
    if (v >= 10000000000ull) return 11;
    if (v >= 1000000000ull) return 10;
    if (v >= 100000000ull) return 9;
    if (v >= 10000000ull) return 8;
    if (v >= 1000000ull) return 7;
    if (v >= 100000ull) return 6;
    if (v >= 10000ull) return 5;
    if (v >= 1000ull) return 4;
    if (v >= 100ull) return 3;
    if (v >= 10ull) return 2;
    return 1;
}

inline int32_t pow5bits(int32_t e){ return static_cast<int32_t>(((static_cast<uint32_t>(e) * 1217359u) >> 19) + 1); }
inline uint32_t log10Pow2(int32_t e){ return (static_cast<uint32_t>(e) * 78913u) >> 18; }
inline uint32_t log10Pow5(int32_t e){ return (static_cast<uint32_t>(e) * 732923u) >> 20; }

const int32_t DOUBLE_POW5_INV_BITCOUNT = 125;
const int32_t DOUBLE_POW5_BITCOUNT = 125;
const int32_t FLOAT_POW5_INV_BITCOUNT = DOUBLE_POW5_INV_BITCOUNT - 64;
const int32_t FLOAT_POW5_BITCOUNT = DOUBLE_POW5_BITCOUNT - 64;

inline uint32_t pow5Factor32(uint32_t value){
    uint32_t count = 0;
    for (;;){
        uint32_t q = value / 5;
        uint32_t r = value % 5;
        if (r != 0) break;
        value = q;
        count += 1;
    }
    return count;
}

inline bool multipleOfPowerOf5_32(uint32_t value, uint32_t p){ return pow5Factor32(value) >= p; }
inline bool multipleOfPowerOf2_32(uint32_t value, uint32_t p){ return (value & ((1u << p) - 1)) == 0; }

inline uint32_t mulShift32(uint32_t m, uint64_t factor, int32_t shift){
    uint32_t factorLo = static_cast<uint32_t>(factor);
    uint32_t factorHi = static_cast<uint32_t>(factor >> 32);
    uint64_t bits0 = static_cast<uint64_t>(m) * factorLo;
    uint64_t bits1 = static_cast<uint64_t>(m) * factorHi;
    uint64_t sum = (bits0 >> 32) + bits1;
    return static_cast<uint32_t>(sum >> (shift - 32));
}

inline uint32_t mulPow5InvDivPow2(uint32_t m, uint32_t q, int32_t j){
    return mulShift32(m, DOUBLE_POW5_INV_SPLIT[q][1] + 1, j);
}

inline uint32_t mulPow5DivPow2(uint32_t m, uint32_t i, int32_t j){
    return mulShift32(m, DOUBLE_POW5_SPLIT[i][1], j);
}

inline uint32_t pow5Factor64(uint64_t value){
    const uint64_t M_INV_5 = 14757395258967641293ull;
    const uint64_t N_DIV_5 = 3689348814741910323ull;
    uint32_t count = 0;
    for (;;){
        value = value * M_INV_5; // wrapping
        if (value > N_DIV_5) break;
        count += 1;
    }
    return count;
}

inline bool multipleOfPowerOf5_64(uint64_t value, uint32_t p){ return pow5Factor64(value) >= p; }
inline bool multipleOfPowerOf2_64(uint64_t value, uint32_t p){ return (value & ((1ull << p) - 1)) == 0; }

inline uint64_t mulShift64(uint64_t m, const uint64_t mul[2], uint32_t j){
    unsigned __int128 b0 = static_cast<unsigned __int128>(m) * mul[0];
    unsigned __int128 b2 = static_cast<unsigned __int128>(m) * mul[1];
    return static_cast<uint64_t>(((b0 >> 64) + b2) >> (j - 64));
}

inline uint64_t mulShiftAll64(uint64_t m, const uint64_t mul[2], uint32_t j, uint64_t *vp,
                              uint64_t *vm, uint32_t mmShift){
    *vp = mulShift64(4 * m + 2, mul, j);
    *vm = mulShift64(4 * m - 1 - mmShift, mul, j);
    return mulShift64(4 * m, mul, j);
}

struct FloatingDecimal32 {
    uint32_t mantissa;
    int32_t exponent;
};

struct FloatingDecimal64 {
    uint64_t mantissa;
    int32_t exponent;
};

FloatingDecimal32 f2d(uint32_t ieeeMantissa, uint32_t ieeeExponent){
    const uint32_t FLOAT_MANTISSA_BITS = 23;
    const int32_t FLOAT_BIAS = 127;

    int32_t e2;
    uint32_t m2;
    if (ieeeExponent == 0){
        e2 = 1 - FLOAT_BIAS - static_cast<int32_t>(FLOAT_MANTISSA_BITS) - 2;
        m2 = ieeeMantissa;
    }else{
        e2 = static_cast<int32_t>(ieeeExponent) - FLOAT_BIAS - static_cast<int32_t>(FLOAT_MANTISSA_BITS) - 2;
        m2 = (1u << FLOAT_MANTISSA_BITS) | ieeeMantissa;
    }
    bool even = (m2 & 1) == 0;
    bool acceptBounds = even;

    uint32_t mv = 4 * m2;
    uint32_t mp = 4 * m2 + 2;
    uint32_t mmShift = (ieeeMantissa != 0 || ieeeExponent <= 1) ? 1 : 0;
    uint32_t mm = 4 * m2 - 1 - mmShift;

    uint32_t vr, vp, vm;
    int32_t e10;
    bool vmIsTrailingZeros = false;
    bool vrIsTrailingZeros = false;
    uint8_t lastRemovedDigit = 0;
    if (e2 >= 0){
        uint32_t q = log10Pow2(e2);
        e10 = static_cast<int32_t>(q);
        int32_t k = FLOAT_POW5_INV_BITCOUNT + pow5bits(static_cast<int32_t>(q)) - 1;
        int32_t i = -e2 + static_cast<int32_t>(q) + k;
        vr = mulPow5InvDivPow2(mv, q, i);
        vp = mulPow5InvDivPow2(mp, q, i);
        vm = mulPow5InvDivPow2(mm, q, i);
        if (q != 0 && (vp - 1) / 10 <= vm / 10){
            int32_t l = FLOAT_POW5_INV_BITCOUNT + pow5bits(static_cast<int32_t>(q) - 1) - 1;
            lastRemovedDigit = static_cast<uint8_t>(
                mulPow5InvDivPow2(mv, q - 1, -e2 + static_cast<int32_t>(q) - 1 + l) % 10);
        }
        if (q <= 9){
            if (mv % 5 == 0){
                vrIsTrailingZeros = multipleOfPowerOf5_32(mv, q);
            }else if (acceptBounds){
                vmIsTrailingZeros = multipleOfPowerOf5_32(mm, q);
            }else{
                vp -= multipleOfPowerOf5_32(mp, q) ? 1 : 0;
            }
        }
    }else{
        uint32_t q = log10Pow5(-e2);
        e10 = static_cast<int32_t>(q) + e2;
        int32_t i = -e2 - static_cast<int32_t>(q);
        int32_t k = pow5bits(i) - FLOAT_POW5_BITCOUNT;
        int32_t j = static_cast<int32_t>(q) - k;
        vr = mulPow5DivPow2(mv, static_cast<uint32_t>(i), j);
        vp = mulPow5DivPow2(mp, static_cast<uint32_t>(i), j);
        vm = mulPow5DivPow2(mm, static_cast<uint32_t>(i), j);
        if (q != 0 && (vp - 1) / 10 <= vm / 10){
            j = static_cast<int32_t>(q) - 1 - (pow5bits(i + 1) - FLOAT_POW5_BITCOUNT);
            lastRemovedDigit = static_cast<uint8_t>(
                mulPow5DivPow2(mv, static_cast<uint32_t>(i + 1), j) % 10);
        }
        if (q <= 1){
            vrIsTrailingZeros = true;
            if (acceptBounds){
                vmIsTrailingZeros = mmShift == 1;
            }else{
                vp -= 1;
            }
        }else if (q < 31){
            vrIsTrailingZeros = multipleOfPowerOf2_32(mv, q - 1);
        }
    }

    int32_t removed = 0;
    uint32_t output;
    if (vmIsTrailingZeros || vrIsTrailingZeros){
        while (vp / 10 > vm / 10){
            vmIsTrailingZeros &= vm - (vm / 10) * 10 == 0;
            vrIsTrailingZeros &= lastRemovedDigit == 0;
            lastRemovedDigit = static_cast<uint8_t>(vr % 10);
            vr /= 10;
            vp /= 10;
            vm /= 10;
            removed += 1;
        }
        if (vmIsTrailingZeros){
            while (vm % 10 == 0){
                vrIsTrailingZeros &= lastRemovedDigit == 0;
                lastRemovedDigit = static_cast<uint8_t>(vr % 10);
                vr /= 10;
                vp /= 10;
                vm /= 10;
                removed += 1;
            }
        }
        if (vrIsTrailingZeros && lastRemovedDigit == 5 && vr % 2 == 0){
            lastRemovedDigit = 4;
        }
        output = vr + (((vr == vm && (!acceptBounds || !vmIsTrailingZeros)) || lastRemovedDigit >= 5)
                           ? 1
                           : 0);
    }else{
        while (vp / 10 > vm / 10){
            lastRemovedDigit = static_cast<uint8_t>(vr % 10);
            vr /= 10;
            vp /= 10;
            vm /= 10;
            removed += 1;
        }
        output = vr + ((vr == vm || lastRemovedDigit >= 5) ? 1 : 0);
    }
    int32_t exp = e10 + removed;

    return FloatingDecimal32{output, exp};
}

FloatingDecimal64 d2d(uint64_t ieeeMantissa, uint32_t ieeeExponent){
    const uint32_t DOUBLE_MANTISSA_BITS = 52;
    const int32_t DOUBLE_BIAS = 1023;

    int32_t e2;
    uint64_t m2;
    if (ieeeExponent == 0){
        e2 = 1 - DOUBLE_BIAS - static_cast<int32_t>(DOUBLE_MANTISSA_BITS) - 2;
        m2 = ieeeMantissa;
    }else{
        e2 = static_cast<int32_t>(ieeeExponent) - DOUBLE_BIAS - static_cast<int32_t>(DOUBLE_MANTISSA_BITS) - 2;
        m2 = (1ull << DOUBLE_MANTISSA_BITS) | ieeeMantissa;
    }
    bool even = (m2 & 1) == 0;
    bool acceptBounds = even;

    uint64_t mv = 4 * m2;
    uint32_t mmShift = (ieeeMantissa != 0 || ieeeExponent <= 1) ? 1 : 0;

    uint64_t vr, vp, vm;
    int32_t e10;
    bool vmIsTrailingZeros = false;
    bool vrIsTrailingZeros = false;
    if (e2 >= 0){
        uint32_t q = log10Pow2(e2) - (e2 > 3 ? 1 : 0);
        e10 = static_cast<int32_t>(q);
        int32_t k = DOUBLE_POW5_INV_BITCOUNT + pow5bits(static_cast<int32_t>(q)) - 1;
        int32_t i = -e2 + static_cast<int32_t>(q) + k;
        vr = mulShiftAll64(m2, DOUBLE_POW5_INV_SPLIT[q], static_cast<uint32_t>(i), &vp, &vm, mmShift);
        if (q <= 21){
            uint32_t mvMod5 = static_cast<uint32_t>(mv) - 5u * static_cast<uint32_t>(mv / 5);
            if (mvMod5 == 0){
                vrIsTrailingZeros = multipleOfPowerOf5_64(mv, q);
            }else if (acceptBounds){
                vmIsTrailingZeros = multipleOfPowerOf5_64(mv - 1 - mmShift, q);
            }else{
                vp -= multipleOfPowerOf5_64(mv + 2, q) ? 1 : 0;
            }
        }
    }else{
        uint32_t q = log10Pow5(-e2) - (-e2 > 1 ? 1 : 0);
        e10 = static_cast<int32_t>(q) + e2;
        int32_t i = -e2 - static_cast<int32_t>(q);
        int32_t k = pow5bits(i) - DOUBLE_POW5_BITCOUNT;
        int32_t j = static_cast<int32_t>(q) - k;
        vr = mulShiftAll64(m2, DOUBLE_POW5_SPLIT[i], static_cast<uint32_t>(j), &vp, &vm, mmShift);
        if (q <= 1){
            vrIsTrailingZeros = true;
            if (acceptBounds){
                vmIsTrailingZeros = mmShift == 1;
            }else{
                vp -= 1;
            }
        }else if (q < 63){
            vrIsTrailingZeros = multipleOfPowerOf2_64(mv, q);
        }
    }

    int32_t removed = 0;
    uint8_t lastRemovedDigit = 0;
    uint64_t output;
    if (vmIsTrailingZeros || vrIsTrailingZeros){
        for (;;){
            uint64_t vpDiv10 = vp / 10;
            uint64_t vmDiv10 = vm / 10;
            if (vpDiv10 <= vmDiv10) break;
            uint32_t vmMod10 = static_cast<uint32_t>(vm) - 10u * static_cast<uint32_t>(vmDiv10);
            uint64_t vrDiv10 = vr / 10;
            uint32_t vrMod10 = static_cast<uint32_t>(vr) - 10u * static_cast<uint32_t>(vrDiv10);
            vmIsTrailingZeros &= vmMod10 == 0;
            vrIsTrailingZeros &= lastRemovedDigit == 0;
            lastRemovedDigit = static_cast<uint8_t>(vrMod10);
            vr = vrDiv10;
            vp = vpDiv10;
            vm = vmDiv10;
            removed += 1;
        }
        if (vmIsTrailingZeros){
            for (;;){
                uint64_t vmDiv10 = vm / 10;
                uint32_t vmMod10 = static_cast<uint32_t>(vm) - 10u * static_cast<uint32_t>(vmDiv10);
                if (vmMod10 != 0) break;
                uint64_t vpDiv10 = vp / 10;
                uint64_t vrDiv10 = vr / 10;
                uint32_t vrMod10 = static_cast<uint32_t>(vr) - 10u * static_cast<uint32_t>(vrDiv10);
                vrIsTrailingZeros &= lastRemovedDigit == 0;
                lastRemovedDigit = static_cast<uint8_t>(vrMod10);
                vr = vrDiv10;
                vp = vpDiv10;
                vm = vmDiv10;
                removed += 1;
            }
        }
        if (vrIsTrailingZeros && lastRemovedDigit == 5 && vr % 2 == 0){
            lastRemovedDigit = 4;
        }
        output = vr + (((vr == vm && (!acceptBounds || !vmIsTrailingZeros)) || lastRemovedDigit >= 5)
                           ? 1
                           : 0);
    }else{
        bool roundUp = false;
        uint64_t vpDiv100 = vp / 100;
        uint64_t vmDiv100 = vm / 100;
        if (vpDiv100 > vmDiv100){
            uint64_t vrDiv100 = vr / 100;
            uint32_t vrMod100 = static_cast<uint32_t>(vr) - 100u * static_cast<uint32_t>(vrDiv100);
            roundUp = vrMod100 >= 50;
            vr = vrDiv100;
            vp = vpDiv100;
            vm = vmDiv100;
            removed += 2;
        }
        for (;;){
            uint64_t vpDiv10 = vp / 10;
            uint64_t vmDiv10 = vm / 10;
            if (vpDiv10 <= vmDiv10) break;
            uint64_t vrDiv10 = vr / 10;
            uint32_t vrMod10 = static_cast<uint32_t>(vr) - 10u * static_cast<uint32_t>(vrDiv10);
            roundUp = vrMod10 >= 5;
            vr = vrDiv10;
            vp = vpDiv10;
            vm = vmDiv10;
            removed += 1;
        }
        output = vr + ((vr == vm || roundUp) ? 1 : 0);
    }
    int32_t exp = e10 + removed;

    return FloatingDecimal64{output, exp};
}

inline size_t writeExponent3(ptrdiff_t k, char *result){
    size_t sign = 0;
    if (k < 0){
        *result = '-';
        result += 1;
        k = -k;
        sign = 1;
    }
    if (k >= 100){
        *result = static_cast<char>('0' + (k / 100));
        k %= 100;
        std::memcpy(result + 1, DIGIT_TABLE + k * 2, 2);
        return sign + 3;
    }else if (k >= 10){
        std::memcpy(result, DIGIT_TABLE + k * 2, 2);
        return sign + 2;
    }else{
        *result = static_cast<char>('0' + k);
        return sign + 1;
    }
}

inline size_t writeExponent2(ptrdiff_t k, char *result){
    size_t sign = 0;
    if (k < 0){
        *result = '-';
        result += 1;
        k = -k;
        sign = 1;
    }
    if (k >= 10){
        std::memcpy(result, DIGIT_TABLE + k * 2, 2);
        return sign + 2;
    }else{
        *result = static_cast<char>('0' + k);
        return sign + 1;
    }
}

inline void writeMantissa(uint32_t output, char *result){
    while (output >= 10000){
        uint32_t c = output - 10000 * (output / 10000);
        output /= 10000;
        uint32_t c0 = (c % 100) << 1;
        uint32_t c1 = (c / 100) << 1;
        std::memcpy(result - 2, DIGIT_TABLE + c0, 2);
        std::memcpy(result - 4, DIGIT_TABLE + c1, 2);
        result -= 4;
    }
    if (output >= 100){
        uint32_t c = (output % 100) << 1;
        output /= 100;
        std::memcpy(result - 2, DIGIT_TABLE + c, 2);
        result -= 2;
    }
    if (output >= 10){
        uint32_t c = output << 1;
        std::memcpy(result - 2, DIGIT_TABLE + c, 2);
    }else{
        *(result - 1) = static_cast<char>('0' + output);
    }
}

inline void writeMantissaLong(uint64_t output, char *result){
    if ((output >> 32) != 0){
        uint32_t output2 = static_cast<uint32_t>(output - 100000000ull * (output / 100000000ull));
        output /= 100000000ull;

        uint32_t c = output2 % 10000;
        output2 /= 10000;
        uint32_t d = output2 % 10000;
        uint32_t c0 = (c % 100) << 1;
        uint32_t c1 = (c / 100) << 1;
        uint32_t d0 = (d % 100) << 1;
        uint32_t d1 = (d / 100) << 1;
        std::memcpy(result - 2, DIGIT_TABLE + c0, 2);
        std::memcpy(result - 4, DIGIT_TABLE + c1, 2);
        std::memcpy(result - 6, DIGIT_TABLE + d0, 2);
        std::memcpy(result - 8, DIGIT_TABLE + d1, 2);
        result -= 8;
    }
    writeMantissa(static_cast<uint32_t>(output), result);
}

// ryu pretty format32 (Buffer::format_finite for f32)
size_t format32(float f, char *result){
    uint32_t bits = f32Bits(f);
    bool sign = ((bits >> 31) & 1) != 0;
    uint32_t ieeeMantissa = bits & ((1u << 23) - 1);
    uint32_t ieeeExponent = (bits >> 23) & ((1u << 8) - 1);

    ptrdiff_t index = 0;
    if (sign){
        result[0] = '-';
        index = 1;
    }

    if (ieeeExponent == 0 && ieeeMantissa == 0){
        std::memcpy(result + index, "0.0", 3);
        return (sign ? 1 : 0) + 3;
    }

    FloatingDecimal32 v = f2d(ieeeMantissa, ieeeExponent);

    ptrdiff_t length = static_cast<ptrdiff_t>(decimalLength9(v.mantissa));
    ptrdiff_t k = v.exponent;
    ptrdiff_t kk = length + k;

    if (0 <= k && kk <= 13){
        writeMantissa(v.mantissa, result + index + length);
        for (ptrdiff_t i = length; i < kk; i++){
            result[index + i] = '0';
        }
        result[index + kk] = '.';
        result[index + kk + 1] = '0';
        return static_cast<size_t>(index + kk + 2);
    }else if (0 < kk && kk <= 13){
        writeMantissa(v.mantissa, result + index + length + 1);
        std::memmove(result + index, result + index + 1, static_cast<size_t>(kk));
        result[index + kk] = '.';
        return static_cast<size_t>(index + length + 1);
    }else if (-6 < kk && kk <= 0){
        result[index] = '0';
        result[index + 1] = '.';
        ptrdiff_t offset = 2 - kk;
        for (ptrdiff_t i = 2; i < offset; i++){
            result[index + i] = '0';
        }
        writeMantissa(v.mantissa, result + index + length + offset);
        return static_cast<size_t>(index + length + offset);
    }else if (length == 1){
        result[index] = static_cast<char>('0' + v.mantissa);
        result[index + 1] = 'e';
        return static_cast<size_t>(index + 2) + writeExponent2(kk - 1, result + index + 2);
    }else{
        writeMantissa(v.mantissa, result + index + length + 1);
        result[index] = result[index + 1];
        result[index + 1] = '.';
        result[index + length + 1] = 'e';
        return static_cast<size_t>(index + length + 2) +
               writeExponent2(kk - 1, result + index + length + 2);
    }
}

// ryu pretty format64 (Buffer::format_finite for f64)
size_t format64(double f, char *result){
    uint64_t bits;
    std::memcpy(&bits, &f, 8);
    bool sign = ((bits >> 63) & 1) != 0;
    uint64_t ieeeMantissa = bits & ((1ull << 52) - 1);
    uint32_t ieeeExponent = static_cast<uint32_t>(bits >> 52) & ((1u << 11) - 1);

    ptrdiff_t index = 0;
    if (sign){
        result[0] = '-';
        index = 1;
    }

    if (ieeeExponent == 0 && ieeeMantissa == 0){
        std::memcpy(result + index, "0.0", 3);
        return (sign ? 1 : 0) + 3;
    }

    FloatingDecimal64 v = d2d(ieeeMantissa, ieeeExponent);

    ptrdiff_t length = static_cast<ptrdiff_t>(decimalLength17(v.mantissa));
    ptrdiff_t k = v.exponent;
    ptrdiff_t kk = length + k;

    if (0 <= k && kk <= 16){
        writeMantissaLong(v.mantissa, result + index + length);
        for (ptrdiff_t i = length; i < kk; i++){
            result[index + i] = '0';
        }
        result[index + kk] = '.';
        result[index + kk + 1] = '0';
        return static_cast<size_t>(index + kk + 2);
    }else if (0 < kk && kk <= 16){
        writeMantissaLong(v.mantissa, result + index + length + 1);
        std::memmove(result + index, result + index + 1, static_cast<size_t>(kk));
        result[index + kk] = '.';
        return static_cast<size_t>(index + length + 1);
    }else if (-5 < kk && kk <= 0){
        result[index] = '0';
        result[index + 1] = '.';
        ptrdiff_t offset = 2 - kk;
        for (ptrdiff_t i = 2; i < offset; i++){
            result[index + i] = '0';
        }
        writeMantissaLong(v.mantissa, result + index + length + offset);
        return static_cast<size_t>(index + length + offset);
    }else if (length == 1){
        result[index] = static_cast<char>('0' + v.mantissa);
        result[index + 1] = 'e';
        return static_cast<size_t>(index + 2) + writeExponent3(kk - 1, result + index + 2);
    }else{
        writeMantissaLong(v.mantissa, result + index + length + 1);
        result[index] = result[index + 1];
        result[index + 1] = '.';
        result[index + length + 1] = 'e';
        return static_cast<size_t>(index + length + 2) +
               writeExponent3(kk - 1, result + index + length + 2);
    }
}

}

// serde_json formats f32 via ryu f32 and f64 via ryu f64
std::string formatF32(float v){
    char buf[16];
    size_t n = ryu::format32(v, buf);
    return std::string(buf, n);
}

std::string formatF64(double v){
    char buf[24];
    size_t n = ryu::format64(v, buf);
    return std::string(buf, n);
}

// ---------------------------------------------------------------------------
// JSON emitter matching serde_json output byte-for-byte.
// Compact: {"a":1,"b":[2,3]}   Pretty: 2-space indent, ": " separator.
// ---------------------------------------------------------------------------

struct JsonValue {
    enum Type { Null, Bool, UInt, F32AsF64, F64, F32, Str, RawStr, Arr, Obj } type = Null;
    bool b = false;
    uint64_t u = 0;
    float f32v = 0.0f;
    double f64v = 0.0;
    std::string s;
    std::vector<JsonValue> arr;
    std::vector<std::pair<std::string, JsonValue>> obj; // emitted in insertion order

    static JsonValue null(){ return JsonValue(); }
    static JsonValue boolean(bool v){ JsonValue j; j.type = Bool; j.b = v; return j; }
    static JsonValue uinteger(uint64_t v){ JsonValue j; j.type = UInt; j.u = v; return j; }
    // serde_json Value::Number stores f32 promoted to f64 (Number::from_f32)
    static JsonValue f32AsF64(float v){ JsonValue j; j.type = F32AsF64; j.f32v = v; return j; }
    static JsonValue f64(double v){ JsonValue j; j.type = F64; j.f64v = v; return j; }
    // direct f32 struct field serialization (write_f32 -> ryu f32)
    static JsonValue f32(float v){ JsonValue j; j.type = F32; j.f32v = v; return j; }
    static JsonValue str(const std::string &v){ JsonValue j; j.type = Str; j.s = v; return j; }
    static JsonValue array(){ JsonValue j; j.type = Arr; return j; }
    static JsonValue object(){ JsonValue j; j.type = Obj; return j; }

    void insert(const std::string &key, JsonValue v){ obj.push_back({key, std::move(v)}); }
};

// serde_json string escaping
void jsonEscapeTo(std::string &out, const std::string &s){
    static const char *HEX = "0123456789abcdef";
    out.push_back('"');
    for (unsigned char c : s){
        switch (c){
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20){
                    out += "\\u00";
                    out.push_back(HEX[(c >> 4) & 0xF]);
                    out.push_back(HEX[c & 0xF]);
                }else{
                    out.push_back(static_cast<char>(c));
                }
        }
    }
    out.push_back('"');
}

void jsonWrite(std::string &out, const JsonValue &v, bool pretty, int depth){
    switch (v.type){
        case JsonValue::Null: out += "null"; break;
        case JsonValue::Bool: out += v.b ? "true" : "false"; break;
        case JsonValue::UInt: out += std::to_string(v.u); break;
        case JsonValue::F32AsF64: out += formatF64(static_cast<double>(v.f32v)); break;
        case JsonValue::F64: out += formatF64(v.f64v); break;
        case JsonValue::F32: out += formatF32(v.f32v); break;
        case JsonValue::Str: jsonEscapeTo(out, v.s); break;
        case JsonValue::RawStr: out += v.s; break;
        case JsonValue::Arr: {
            if (v.arr.empty()){
                out += "[]";
                break;
            }
            out.push_back('[');
            for (size_t i = 0; i < v.arr.size(); i++){
                if (i) out.push_back(',');
                if (pretty){
                    out.push_back('\n');
                    out.append(static_cast<size_t>(depth + 1) * 2, ' ');
                }
                jsonWrite(out, v.arr[i], pretty, depth + 1);
            }
            if (pretty){
                out.push_back('\n');
                out.append(static_cast<size_t>(depth) * 2, ' ');
            }
            out.push_back(']');
            break;
        }
        case JsonValue::Obj: {
            if (v.obj.empty()){
                out += "{}";
                break;
            }
            out.push_back('{');
            for (size_t i = 0; i < v.obj.size(); i++){
                if (i) out.push_back(',');
                if (pretty){
                    out.push_back('\n');
                    out.append(static_cast<size_t>(depth + 1) * 2, ' ');
                }
                jsonEscapeTo(out, v.obj[i].first);
                out.push_back(':');
                if (pretty) out.push_back(' ');
                jsonWrite(out, v.obj[i].second, pretty, depth + 1);
            }
            if (pretty){
                out.push_back('\n');
                out.append(static_cast<size_t>(depth) * 2, ' ');
            }
            out.push_back('}');
            break;
        }
    }
}

std::string jsonToString(const JsonValue &v, bool pretty){
    std::string out;
    jsonWrite(out, v, pretty, 0);
    return out;
}

// ---------------------------------------------------------------------------
// Gsplat / GsplatArray (port of spark-lib/src/gsplat.rs + tsplat.rs)
// ---------------------------------------------------------------------------

// tsplat.rs ellipsoid_area (Knud Thomsen approximation)
float ellipsoidArea(const Vec3 &scales){
    const float P = 1.6075f;
    float numerator = std::pow(scales.x * scales.y, P) + std::pow(scales.x * scales.z, P) +
                      std::pow(scales.y * scales.z, P);
    return 4.0f * 3.14159265358979323846264338327950288f * std::pow(numerator / 3.0f, 1.0f / P);
}

struct Gsplat {
    Vec3 center;
    F16 opacity;
    F16 rgb[3];
    F16 lnScales[3];
    F16 quaternion[4]; // x, y, z, w

    static Gsplat make(const Vec3 &center, float opacity, const Vec3 &rgb, const Vec3 &scales,
                       const Quat &quaternion){
        Gsplat s;
        s.center = center;
        s.opacity = F16::fromF32(opacity);
        s.rgb[0] = F16::fromF32(rgb.x);
        s.rgb[1] = F16::fromF32(rgb.y);
        s.rgb[2] = F16::fromF32(rgb.z);
        s.lnScales[0] = F16::fromF32(std::log(scales.x));
        s.lnScales[1] = F16::fromF32(std::log(scales.y));
        s.lnScales[2] = F16::fromF32(std::log(scales.z));
        s.quaternion[0] = F16::fromF32(quaternion.x);
        s.quaternion[1] = F16::fromF32(quaternion.y);
        s.quaternion[2] = F16::fromF32(quaternion.z);
        s.quaternion[3] = F16::fromF32(quaternion.w);
        return s;
    }

    Vec3 getCenter() const { return center; }
    float getOpacity() const { return opacity.toF32(); }
    Vec3 getRgb() const { return Vec3(rgb[0].toF32(), rgb[1].toF32(), rgb[2].toF32()); }
    Vec3 getScales() const {
        return Vec3(std::exp(lnScales[0].toF32()), std::exp(lnScales[1].toF32()),
                    std::exp(lnScales[2].toF32()));
    }
    Quat getQuaternion() const {
        return Quat(quaternion[0].toF32(), quaternion[1].toF32(), quaternion[2].toF32(),
                    quaternion[3].toF32());
    }
    // gsplat.rs: max over f16 ln_scales, then exp
    float maxScale() const {
        return std::exp(f16Max(f16Max(lnScales[0], lnScales[1]), lnScales[2]).toF32());
    }

    void setCenter(const Vec3 &c){ center = c; }
    void setOpacity(float v){ opacity = F16::fromF32(v); }
    void setRgb(const Vec3 &v){
        rgb[0] = F16::fromF32(v.x);
        rgb[1] = F16::fromF32(v.y);
        rgb[2] = F16::fromF32(v.z);
    }
    void setScales(const Vec3 &scales){
        lnScales[0] = F16::fromF32(std::log(scales.x));
        lnScales[1] = F16::fromF32(std::log(scales.y));
        lnScales[2] = F16::fromF32(std::log(scales.z));
    }
    void setQuaternion(const Quat &q){
        quaternion[0] = F16::fromF32(q.x);
        quaternion[1] = F16::fromF32(q.y);
        quaternion[2] = F16::fromF32(q.z);
        quaternion[3] = F16::fromF32(q.w);
    }

    float area() const { return ellipsoidArea(getScales()); }

    // tsplat.rs lod_opacity
    float lodOpacity() const {
        float op = getOpacity();
        if (op > 1.0f){
            return std::sqrt(1.0f + 2.71828182845904523536028747135266250f * std::log(op));
        }
        return 1.0f;
    }

    float featureSize() const { return 2.0f * maxScale() * lodOpacity(); }

    // tsplat.rs grid: (center / step).floor().as_i64vec3()
    I64Vec3 grid(float stepSize) const {
        Vec3 g = (center / stepSize).floorv();
        return I64Vec3(rustCastI64(g.x), rustCastI64(g.y), rustCastI64(g.z));
    }
};

// tsplat.rs bhattacharyya_distance
float bhattacharyyaDistance(const Gsplat &a, const Gsplat &b){
    SymMat3 covA = SymMat3::newScaleQuaternion(a.getScales(), a.getQuaternion());
    SymMat3 covB = SymMat3::newScaleQuaternion(b.getScales(), b.getQuaternion());
    SymMat3 sigma = SymMat3::newAverage(covA, covB);
    SymMat3 inv;
    if (!sigma.inverse(inv)){
        return 0.0f;
    }

    Vec3 delta = b.getCenter() - a.getCenter();
    float quad = inv.xx() * delta.x * delta.x
        + inv.yy() * delta.y * delta.y
        + inv.zz() * delta.z * delta.z
        + 2.0f * inv.xy() * delta.x * delta.y
        + 2.0f * inv.xz() * delta.x * delta.z
        + 2.0f * inv.yz() * delta.y * delta.z;
    float term1 = 0.125f * quad;

    float detSigma = sigma.determinant();
    float detA = covA.determinant();
    float detB = covB.determinant();
    float term2 = 0.5f * std::log(detSigma / std::sqrt(detA * detB));

    return term1 + term2;
}

// tsplat.rs similarity_metric
float similarityMetric(const Gsplat &a, const Gsplat &b){
    float spatial = std::exp(-bhattacharyyaDistance(a, b));

    Vec3 colorDelta = a.getRgb() - b.getRgb();
    float colorDelta2 = colorDelta.lengthSquared();

    float metric = spatial * std::exp(-colorDelta2);
    if (std::isnan(metric)){
        return 0.0f;
    }
    return metric;
}

// tsplat.rs compute_swaps
std::vector<std::pair<size_t, size_t>> computeSwaps(const std::vector<size_t> &indexMap){
    size_t n = indexMap.size();
    std::vector<size_t> destOfSrc(n, 0);
    for (size_t newI = 0; newI < n; newI++){
        destOfSrc[indexMap[newI]] = newI;
    }

    std::vector<std::pair<size_t, size_t>> swaps;
    for (size_t i = 0; i < n; i++){
        while (destOfSrc[i] != i){
            size_t j = destOfSrc[i];
            swaps.push_back({i, j});
            std::swap(destOfSrc[i], destOfSrc[j]);
        }
    }
    return swaps;
}

template <typename T>
void applySwaps(std::vector<T> &data, const std::vector<std::pair<size_t, size_t>> &swaps){
    for (const auto &s : swaps){
        std::swap(data[s.first], data[s.second]);
    }
}

typedef std::array<F16, 9> GsplatSH1;  // 3 coeffs x rgb
typedef std::array<F16, 15> GsplatSH2; // 5 coeffs x rgb
typedef std::array<F16, 21> GsplatSH3; // 7 coeffs x rgb

struct GsplatArray {
    size_t maxShDegree = 0;
    std::vector<Gsplat> splats;
    std::vector<std::vector<size_t>> children;
    std::vector<GsplatSH1> sh1;
    std::vector<GsplatSH2> sh2;
    std::vector<GsplatSH3> sh3;

    size_t len() const { return splats.size(); }

    void prepareChildren(){ children.resize(len()); }
    bool hasChildren() const { return !children.empty(); }
    bool hasLodTree() const { return !children.empty(); }

    // gsplat.rs new_merged (step is always 0.0 from bhatt_lod)
    size_t newMerged(const size_t *indices, size_t numIndices, float step){
        size_t newIndex = splats.size();

        std::vector<float> weights(numIndices);
        for (size_t i = 0; i < numIndices; i++){
            const Gsplat &splat = splats[indices[i]];
            weights[i] = splat.area() * splat.getOpacity();
        }
        float sum = 0.0f;
        for (size_t i = 0; i < numIndices; i++) sum += weights[i];
        float totalWeight = rustMax(sum, 1.0e-30f);
        for (size_t i = 0; i < numIndices; i++) weights[i] /= totalWeight;

        Vec3 center = Vec3(0, 0, 0);
        Vec3 rgb = Vec3(0, 0, 0);

        for (size_t i = 0; i < numIndices; i++){
            const Gsplat &splat = splats[indices[i]];
            float weight = weights[i];
            center = splat.getCenter().mulAdd(Vec3::splat(weight), center);
            rgb = splat.getRgb().mulAdd(Vec3::splat(weight), rgb);
        }

        SymMat3 totalCov;
        float filter2 = (0.5f * step) * (0.5f * step); // powi(2)

        for (size_t i = 0; i < numIndices; i++){
            const Gsplat &splat = splats[indices[i]];
            float weight = weights[i];
            Vec3 delta = splat.getCenter() - center;
            SymMat3 cov = SymMat3::newScaleQuaternion(splat.getScales(), splat.getQuaternion());
            float xx = delta.x * delta.x + cov.xx() + filter2;
            float yy = delta.y * delta.y + cov.yy() + filter2;
            float zz = delta.z * delta.z + cov.zz() + filter2;
            float xy = delta.x * delta.y + cov.xy();
            float xz = delta.x * delta.z + cov.xz();
            float yz = delta.y * delta.z + cov.yz();
            totalCov.addWeighted(SymMat3::make(xx, yy, zz, xy, xz, yz), weight);
        }

        float vals[3];
        Vec3 vecs[3];
        totalCov.positiveEigens(vals, vecs);
        Vec3 scales = Vec3(std::sqrt(rustMax(vals[0], 0.0f)), std::sqrt(rustMax(vals[1], 0.0f)),
                           std::sqrt(rustMax(vals[2], 0.0f)));
        scales = scales.max(Vec3::splat(1.0e-30f));

        Mat3 basis = Mat3::fromCols(vecs[0], vecs[1], vecs[2]);
        Quat quaternion = quatFromMat3(basis);
        float opacity = totalWeight / ellipsoidArea(scales);
        opacity = rustClamp(opacity, 0.000001f, 1000.0f);
        // INFLATE_SCALE is false in the reference

        splats.push_back(Gsplat::make(center, opacity, rgb, scales, quaternion));
        children.push_back(std::vector<size_t>(indices, indices + numIndices));

        if (maxShDegree >= 1){
            Vec3 total[3] = {Vec3(0, 0, 0), Vec3(0, 0, 0), Vec3(0, 0, 0)};
            for (size_t i = 0; i < numIndices; i++){
                float weight = weights[i];
                const GsplatSH1 &s = sh1[indices[i]];
                for (int c = 0; c < 3; c++){
                    Vec3 v(s[c * 3 + 0].toF32(), s[c * 3 + 1].toF32(), s[c * 3 + 2].toF32());
                    total[c] = v.mulAdd(Vec3::splat(weight), total[c]);
                }
            }
            GsplatSH1 out;
            for (int c = 0; c < 3; c++){
                out[c * 3 + 0] = F16::fromF32(total[c].x);
                out[c * 3 + 1] = F16::fromF32(total[c].y);
                out[c * 3 + 2] = F16::fromF32(total[c].z);
            }
            sh1.push_back(out);
        }

        if (maxShDegree >= 2){
            Vec3 total[5] = {Vec3(0, 0, 0), Vec3(0, 0, 0), Vec3(0, 0, 0), Vec3(0, 0, 0), Vec3(0, 0, 0)};
            for (size_t i = 0; i < numIndices; i++){
                float weight = weights[i];
                const GsplatSH2 &s = sh2[indices[i]];
                for (int c = 0; c < 5; c++){
                    Vec3 v(s[c * 3 + 0].toF32(), s[c * 3 + 1].toF32(), s[c * 3 + 2].toF32());
                    total[c] = v.mulAdd(Vec3::splat(weight), total[c]);
                }
            }
            GsplatSH2 out;
            for (int c = 0; c < 5; c++){
                out[c * 3 + 0] = F16::fromF32(total[c].x);
                out[c * 3 + 1] = F16::fromF32(total[c].y);
                out[c * 3 + 2] = F16::fromF32(total[c].z);
            }
            sh2.push_back(out);
        }

        if (maxShDegree >= 3){
            Vec3 total[7];
            for (size_t i = 0; i < numIndices; i++){
                float weight = weights[i];
                const GsplatSH3 &s = sh3[indices[i]];
                for (int c = 0; c < 7; c++){
                    Vec3 v(s[c * 3 + 0].toF32(), s[c * 3 + 1].toF32(), s[c * 3 + 2].toF32());
                    total[c] = v.mulAdd(Vec3::splat(weight), total[c]);
                }
            }
            GsplatSH3 out;
            for (int c = 0; c < 7; c++){
                out[c * 3 + 0] = F16::fromF32(total[c].x);
                out[c * 3 + 1] = F16::fromF32(total[c].y);
                out[c * 3 + 2] = F16::fromF32(total[c].z);
            }
            sh3.push_back(out);
        }

        return newIndex;
    }

    void setChildren(size_t parent, const std::vector<size_t> &c){ children[parent] = c; }

    std::vector<size_t> getChildren(size_t parent) const { return children[parent]; }

    float similarity(size_t a, size_t b) const { return similarityMetric(splats[a], splats[b]); }

    template <typename F>
    void retain(F f){
        std::vector<bool> keep(splats.size());
        for (size_t i = 0; i < splats.size(); i++){
            keep[i] = f(splats[i]);
        }
        retainByMask(splats, keep);
        if (!children.empty()) retainByMask(children, keep);
        if (!sh1.empty()) retainByMask(sh1, keep);
        if (!sh2.empty()) retainByMask(sh2, keep);
        if (!sh3.empty()) retainByMask(sh3, keep);
    }

    void permute(const std::vector<size_t> &indexMap){
        assert(indexMap.size() == splats.size());
        auto swaps = computeSwaps(indexMap);
        applySwaps(splats, swaps);
        if (!children.empty()) applySwaps(children, swaps);
        if (!sh1.empty()) applySwaps(sh1, swaps);
        if (!sh2.empty()) applySwaps(sh2, swaps);
        if (!sh3.empty()) applySwaps(sh3, swaps);
    }

    void truncate(size_t count){
        if (splats.size() > count) splats.resize(count);
        if (!children.empty() && children.size() > count) children.resize(count);
        if (!sh1.empty() && sh1.size() > count) sh1.resize(count);
        if (!sh2.empty() && sh2.size() > count) sh2.resize(count);
        if (!sh3.empty() && sh3.size() > count) sh3.resize(count);
    }

    // tsplat.rs sort_by: stable sort of the index map by key, then permute
    void sortByFeatureSize(){
        std::vector<size_t> indexMap(len());
        for (size_t i = 0; i < indexMap.size(); i++) indexMap[i] = i;
        std::vector<float> keys(len());
        for (size_t i = 0; i < keys.size(); i++) keys[i] = splats[i].featureSize();
        // OrderedFloat<f32> ordering; keys are finite here so plain < works,
        // and stable_sort preserves equal-key order like Rust's stable sort.
        std::stable_sort(indexMap.begin(), indexMap.end(),
                         [&](size_t a, size_t b){ return keys[a] < keys[b]; });
        permute(indexMap);
    }

    // tsplat.rs encode_lod_opacity
    void encodeLodOpacity(){
        for (size_t i = 0; i < len(); i++){
            Gsplat &splat = splats[i];
            if (splat.getOpacity() > 1.0f){
                float d = splat.lodOpacity();
                splat.setOpacity(rustClamp(0.25f * (d - 1.0f) + 1.0f, 1.0f, 2.0f));
            }
        }
    }

private:
    template <typename T>
    static void retainByMask(std::vector<T> &v, const std::vector<bool> &keep){
        size_t out = 0;
        for (size_t i = 0; i < v.size(); i++){
            if (keep[i]){
                if (out != i) v[out] = std::move(v[i]);
                out++;
            }
        }
        v.resize(out);
    }
};

// ---------------------------------------------------------------------------
// bhatt_lod (port of spark-lib/src/bhatt_lod.rs)
// ---------------------------------------------------------------------------

const float MERGE_BASE = 2.0f;

void bhattRecurseToOutput(GsplatArray &splats, size_t index, std::vector<bool> &toOutput,
                          float lodBase, float &featureSizeOut, std::vector<size_t> &childrenOut){
    float featureSize;
    {
        const Gsplat &splat = splats.splats[index];
        featureSize = splat.area() * splat.getOpacity();
    }

    std::vector<size_t> children = splats.getChildren(index);
    if (children.empty()){
        featureSizeOut = featureSize;
        childrenOut.assign(1, index);
        return;
    }

    std::vector<size_t> newChildren;
    float maxChildFeatureSize = -std::numeric_limits<float>::infinity();

    for (size_t child : children){
        float childFeatureSize;
        std::vector<size_t> childChildren;
        bhattRecurseToOutput(splats, child, toOutput, lodBase, childFeatureSize, childChildren);
        maxChildFeatureSize = rustMax(maxChildFeatureSize, childFeatureSize);
        newChildren.insert(newChildren.end(), childChildren.begin(), childChildren.end());
    }

    if (featureSize >= (maxChildFeatureSize * lodBase)){
        toOutput[index] = true;
    }

    if (toOutput[index]){
        assert(newChildren.size() <= 65535);
        splats.setChildren(index, newChildren);
        featureSizeOut = featureSize;
        childrenOut.assign(1, index);
    }else{
        splats.setChildren(index, std::vector<size_t>());
        featureSizeOut = maxChildFeatureSize;
        childrenOut = std::move(newChildren);
    }
}

void bhattRecurseIndices(GsplatArray &splats, size_t index, std::vector<size_t> &indices,
                         float limitSize, std::vector<size_t> &frontier){
    if (splats.splats[index].featureSize() < limitSize){
        frontier.push_back(index);
        return;
    }

    std::vector<size_t> children = splats.getChildren(index);
    if (children.empty()){
        return;
    }

    std::vector<size_t> newChildren(children.size());
    for (size_t i = 0; i < children.size(); i++) newChildren[i] = indices.size() + i;
    splats.setChildren(index, newChildren);

    std::sort(children.begin(), children.end());
    for (size_t child : children){
        indices.push_back(child);
    }

    for (size_t child : children){
        bhattRecurseIndices(splats, child, indices, limitSize, frontier);
    }
}

void bhattComputeLodTree(GsplatArray &splats, float lodBase){
    // Differential-testing instrumentation, enabled via RAD_MERGE_LOG
    std::ofstream mergeLog;
    if (const char *mergeLogPath = std::getenv("RAD_MERGE_LOG")){
        mergeLog.open(mergeLogPath);
    }

    size_t initialLen = splats.len();
    if (initialLen == 0){
        return;
    }

    splats.sortByFeatureSize();
    splats.prepareChildren();

    std::vector<bool> isActive(splats.len(), true);

    float minFeatureSize = rustMax(splats.splats[0].featureSize(), 0.000001f);
    // Rust: min_feature_size.log(MERGE_BASE).ceil() as i16
    int16_t levelMin = rustCastI16(std::ceil(std::log(minFeatureSize) / std::log(MERGE_BASE)));

    int32_t level = levelMin;
    size_t frontier = 0;
    RustBinaryHeap active;
    std::unordered_map<I64Vec3, std::vector<size_t>, I64Vec3Hash> cells;

    for (;;){
        float step = std::pow(MERGE_BASE, static_cast<float>(level));

        size_t frontierStart = frontier;
        while (frontier < initialLen){
            if (splats.splats[frontier].featureSize() > step){
                break;
            }
            frontier += 1;
        }

        if (frontier > frontierStart){
            std::vector<HeapKey> newSplats;
            newSplats.reserve(frontier - frontierStart);
            for (size_t i = frontierStart; i < frontier; i++){
                newSplats.push_back(HeapKey{-splats.splats[i].featureSize(), i});
            }
            active.extend(newSplats);
        }

        cells.clear();

        // Iterate the heap's internal array order, like Rust's active.iter()
        for (const HeapKey &hk : active.data){
            size_t index = hk.index;
            I64Vec3 grid = splats.splats[index].grid(step);
            cells[grid].push_back(index);
        }

        std::vector<HeapKey> nextActive;

        HeapKey top;
        while (active.pop(top)){
            float negSize = top.key;
            size_t index = top.index;
            if (!isActive[index]){
                continue;
            }

            I64Vec3 grid = splats.splats[index].grid(step);

            size_t bestIndex = SIZE_MAX;
            float bestMetric = -std::numeric_limits<float>::infinity();
            I64Vec3 bestGrid(INT64_MAX, INT64_MAX, INT64_MAX);

            for (int64_t z = grid.z - 1; z <= grid.z + 1; z++){
                for (int64_t y = grid.y - 1; y <= grid.y + 1; y++){
                    for (int64_t x = grid.x - 1; x <= grid.x + 1; x++){
                        I64Vec3 g(x, y, z);
                        auto it = cells.find(g);
                        if (it != cells.end()){
                            for (size_t neighbor : it->second){
                                if (isActive[neighbor] && neighbor != index){
                                    float metric = splats.similarity(index, neighbor);
                                    if (metric > bestMetric){
                                        bestIndex = neighbor;
                                        bestMetric = metric;
                                        bestGrid = g;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if (bestIndex != SIZE_MAX){
                size_t bestNeighbor = bestIndex;
                size_t mergeIndices[2] = {index, bestNeighbor};
                size_t merged = splats.newMerged(mergeIndices, 2, 0.0f);
                if (mergeLog.is_open()){
                    const Gsplat &s = splats.splats[merged];
                    Vec3 c = s.getCenter();
                    Vec3 sc = s.getScales();
                    Quat q = s.getQuaternion();
                    char buf[256];
                    snprintf(buf, sizeof(buf),
                             "%zu %zu %zu %x %x %x %x %x %x %x %x %x %x %x\n",
                             index, bestNeighbor, merged,
                             f32Bits(c.x), f32Bits(c.y), f32Bits(c.z),
                             f32Bits(s.getOpacity()),
                             f32Bits(sc.x), f32Bits(sc.y), f32Bits(sc.z),
                             f32Bits(q.x), f32Bits(q.y), f32Bits(q.z), f32Bits(q.w));
                    mergeLog << buf;
                }

                isActive[index] = false;
                {
                    std::vector<size_t> &cellIndex = cells[grid];
                    cellIndex.erase(std::remove(cellIndex.begin(), cellIndex.end(), index),
                                    cellIndex.end());
                }

                isActive[bestNeighbor] = false;
                {
                    std::vector<size_t> &cellBest = cells[bestGrid];
                    cellBest.erase(std::remove(cellBest.begin(), cellBest.end(), bestNeighbor),
                                   cellBest.end());
                }

                isActive.push_back(true);

                float featureSize = splats.splats[merged].featureSize();
                if (featureSize > step){
                    nextActive.push_back(HeapKey{-featureSize, merged});
                }else{
                    I64Vec3 mergedGrid = splats.splats[merged].grid(step);
                    cells[mergedGrid].push_back(merged);

                    active.push(HeapKey{-featureSize, merged});
                }
            }else{
                // Can't find a neighbor to merge, so kick to next level
                nextActive.push_back(HeapKey{negSize, index});
            }
        }

        level += 1;
        active.extend(nextActive);

        if (frontier < initialLen){
            continue;
        }

        if (active.len() <= 1){
            break;
        }
    }

    size_t rootIndex = splats.len() - 1;

    std::vector<bool> toOutput;
    toOutput.resize(initialLen, true);
    toOutput.resize(splats.len(), false);
    toOutput[rootIndex] = true;

    {
        float rootFeatureSize;
        std::vector<size_t> rootChildren;
        bhattRecurseToOutput(splats, rootIndex, toOutput, lodBase, rootFeatureSize, rootChildren);
    }

    size_t outputCount = 0;
    for (bool b : toOutput){
        if (b) outputCount++;
    }

    std::vector<size_t> indices;
    indices.push_back(rootIndex);
    float limitSize = splats.splats[rootIndex].featureSize();
    std::vector<size_t> frontierIdx{rootIndex};

    for (;;){
        std::vector<size_t> nextFrontier;
        for (size_t index : frontierIdx){
            bhattRecurseIndices(splats, index, indices, limitSize, nextFrontier);
        }
        frontierIdx.clear();

        if (nextFrontier.empty()){
            break;
        }
        limitSize = limitSize / 4.0f;
        frontierIdx = std::move(nextFrontier);
    }

    assert(indices.size() == outputCount);

    for (size_t index = 0; index < toOutput.size(); index++){
        if (!toOutput[index]){
            indices.push_back(index);
        }
    }

    splats.permute(indices);
    splats.truncate(outputCount);
}

// ---------------------------------------------------------------------------
// chunk_tree (port of spark-lib/src/chunk_tree.rs chunk_tree_size)
// ---------------------------------------------------------------------------

const size_t BATCH_SIZE = 64 * 1024;
const size_t MIN_BATCH_SIZE = 8 * 1024;
const float STD_DEVS = 1.5f;

struct Aabb {
    Vec3 min;
    Vec3 max;

    static Aabb empty(){
        Aabb a;
        a.min = Vec3::splat(std::numeric_limits<float>::infinity());
        a.max = Vec3::splat(-std::numeric_limits<float>::infinity());
        return a;
    }

    // glam Vec3A::min/max are _mm_min_ps/_mm_max_ps: a < b ? a : b / a > b ? a : b
    Aabb extend(const Aabb &other) const {
        Aabb r;
        r.min = Vec3(min.x < other.min.x ? min.x : other.min.x,
                     min.y < other.min.y ? min.y : other.min.y,
                     min.z < other.min.z ? min.z : other.min.z);
        r.max = max.max(other.max);
        return r;
    }

    Vec3 center() const { return (min + max) * 0.5f; }
    Vec3 extent() const { return max - min; }

    float minElement() const {
        Vec3 e = extent();
        float m = e.x < e.y ? e.x : e.y;
        return m < e.z ? m : e.z;
    }

    // longest_axis: x if ex >= ey && ex >= ez, else y if ey >= ez, else z
    int longestAxis() const {
        Vec3 e = extent();
        if (e.x >= e.y && e.x >= e.z) return 0;
        if (e.y >= e.z) return 1;
        return 2;
    }

    static Aabb fromSplat(const Gsplat &s, float stdDevs){
        Vec3 clampedScales = s.getScales().max(Vec3::splat(1.0e-3f));
        Vec3 r = clampedScales * stdDevs;

        Mat3 rmat = Mat3::fromQuat(s.getQuaternion());
        // half = |R| * r: glam Mat3A * Vec3A = x_axis*v.x + y_axis*v.y + z_axis*v.z
        Vec3 ax(std::fabs(rmat.m[0][0]), std::fabs(rmat.m[0][1]), std::fabs(rmat.m[0][2]));
        Vec3 ay(std::fabs(rmat.m[1][0]), std::fabs(rmat.m[1][1]), std::fabs(rmat.m[1][2]));
        Vec3 az(std::fabs(rmat.m[2][0]), std::fabs(rmat.m[2][1]), std::fabs(rmat.m[2][2]));
        Vec3 half = ax * r.x + ay * r.y + az * r.z;

        Vec3 c = s.getCenter();
        Aabb a;
        a.min = c - half;
        a.max = c + half;
        return a;
    }
};

void chunkTreeSize(GsplatArray &splats, size_t root){
    std::vector<size_t> indices;
    indices.push_back(root);

    std::vector<std::vector<size_t>> batches;
    size_t batchesHead = 0; // VecDeque via head index
    batches.push_back({root});

    while (batchesHead < batches.size()){
        std::vector<size_t> batch = std::move(batches[batchesHead]);
        batchesHead++;

        RustBinaryHeap priority;
        for (size_t parent : batch){
            priority.push(HeapKey{splats.splats[parent].featureSize(), parent});
        }

        size_t startIndex = indices.size();
        size_t endIndex = ((startIndex + MIN_BATCH_SIZE + BATCH_SIZE - 1) / BATCH_SIZE) * BATCH_SIZE;

        HeapKey top;
        while (priority.pop(top)){
            std::vector<size_t> children = splats.getChildren(top.index);
            if ((indices.size() + children.size()) > endIndex){
                priority.push(top);
                break;
            }
            std::vector<size_t> newChildren(children.size());
            for (size_t i = 0; i < children.size(); i++) newChildren[i] = indices.size() + i;
            splats.setChildren(top.index, newChildren);

            for (size_t child : children){
                indices.push_back(child);
                priority.push(HeapKey{splats.splats[child].featureSize(), child});
            }
        }

        if (!priority.isEmpty()){
            Aabb aabb = Aabb::empty();
            for (const HeapKey &hk : priority.data){
                aabb = aabb.extend(Aabb::fromSplat(splats.splats[hk.index], STD_DEVS));
            }

            if (aabb.extent().maxElement() >= (3.0f * aabb.minElement())){
                int axis = aabb.longestAxis();
                float split = aabb.center()[axis];
                // partition keeps relative (internal array) order; pop order in the
                // next round is key-determined, so the batch order within a/b is
                // not observable — but keep it faithful anyway.
                std::vector<size_t> a, b;
                for (const HeapKey &hk : priority.data){
                    if (splats.splats[hk.index].getCenter()[axis] < split) a.push_back(hk.index);
                    else b.push_back(hk.index);
                }

                // sort_by_key(-(len)): stable descending by length, a first on tie
                if (b.size() > a.size()) std::swap(a, b);
                batches.push_back(std::move(a));
                batches.push_back(std::move(b));
                continue;
            }

            std::vector<size_t> octants[8];
            Vec3 split = aabb.center();

            for (const HeapKey &hk : priority.data){
                Vec3 center = splats.splats[hk.index].getCenter();
                int octant = (center.x < split.x ? 0 : 1) + (center.y < split.y ? 0 : 2) +
                             (center.z < split.z ? 0 : 4);
                octants[octant].push_back(hk.index);
            }

            // Hilbert order
            const int hilbert[8] = {0, 1, 3, 2, 6, 7, 5, 4};
            for (int i = 0; i < 8; i++){
                batches.push_back(std::move(octants[hilbert[i]]));
            }
        }
    }

    assert(indices.size() == splats.len());
    splats.permute(indices);
}

// ---------------------------------------------------------------------------
// Ingestion: SplatData -> GsplatArray, applying the exact transforms of
// spark-lib/src/ply.rs poll_data_standard for OpenSplat's savePly layout
// ---------------------------------------------------------------------------

const float SH_C0 = 0.28209479177387814f;

size_t maxShDegreeFromRestCoeffs(size_t numRestCoeffs){
    // f_rest count = numRestCoeffs * 3: 0 -> 0, 9 -> 1, 24 -> 2, 45 -> 3
    switch (numRestCoeffs){
        case 0: return 0;
        case 3: return 1;
        case 8: return 2;
        case 15: return 3;
        default: return SIZE_MAX;
    }
}

bool ingestSplats(const SplatData &data, GsplatArray &splats){
    size_t maxSh = maxShDegreeFromRestCoeffs(data.numRestCoeffs);
    if (maxSh == SIZE_MAX){
        std::cerr << "saveRad: invalid number of SH rest coefficients: " << data.numRestCoeffs
                  << std::endl;
        return false;
    }

    size_t n = data.numPoints;
    splats.maxShDegree = maxSh;
    splats.splats.resize(n);
    if (maxSh >= 1) splats.sh1.resize(n);
    if (maxSh >= 2) splats.sh2.resize(n);
    if (maxSh >= 3) splats.sh3.resize(n);

    size_t K = data.numRestCoeffs;

    for (size_t i = 0; i < n; i++){
        size_t i3 = i * 3;
        size_t i4 = i * 4;

        Vec3 center(data.means[i3], data.means[i3 + 1], data.means[i3 + 2]);

        float opLogistic = data.opacities[i];
        float opacity = 1.0f / (1.0f + std::exp(-opLogistic));

        Vec3 rgb(0.5f + data.featuresDc[i3] * SH_C0,
                 0.5f + data.featuresDc[i3 + 1] * SH_C0,
                 0.5f + data.featuresDc[i3 + 2] * SH_C0);

        Vec3 scale(std::exp(data.scales[i3]), std::exp(data.scales[i3 + 1]),
                   std::exp(data.scales[i3 + 2]));

        // stored order w,x,y,z (rot_0..rot_3); spark reads [rot_1, rot_2, rot_3, rot_0]
        float quatArr[4] = {data.quats[i4 + 1], data.quats[i4 + 2], data.quats[i4 + 3],
                            data.quats[i4]};
        float quatMagnitude = std::sqrt(((quatArr[0] * quatArr[0] + quatArr[1] * quatArr[1]) +
                                         quatArr[2] * quatArr[2]) + quatArr[3] * quatArr[3]);
        Quat quat(quatArr[0] / quatMagnitude, quatArr[1] / quatMagnitude,
                  quatArr[2] / quatMagnitude, quatArr[3] / quatMagnitude);

        // set_batch applies each setter (f16 quantization) individually.
        Gsplat &s = splats.splats[i];
        s.setCenter(center);
        s.setOpacity(opacity);
        s.setRgb(rgb);
        s.setScales(scale); // f16(ln(exp(stored)))
        s.setQuaternion(quat);

        // featuresRest [i][k][d] linear order equals spark's interleaved SH slots:
        // sh1 = coeffs 0..2, sh2 = coeffs 3..7, sh3 = coeffs 8..14
        const float *fr = data.featuresRest.data() + i * K * 3;
        if (maxSh >= 1){
            for (int d = 0; d < 9; d++) splats.sh1[i][d] = F16::fromF32(fr[d]);
        }
        if (maxSh >= 2){
            for (int d = 0; d < 15; d++) splats.sh2[i][d] = F16::fromF32(fr[9 + d]);
        }
        if (maxSh >= 3){
            for (int d = 0; d < 21; d++) splats.sh3[i][d] = F16::fromF32(fr[24 + d]);
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// Debug stage dumps (same binary format as the instrumented Rust oracle);
// enabled by setting RAD_DEBUG_DIR
// ---------------------------------------------------------------------------

void dumpStage(const GsplatArray &splats, const char *stage){
    const char *dir = std::getenv("RAD_DEBUG_DIR");
    if (!dir) return;

    std::string path = std::string(dir) + "/" + stage + ".bin";
    std::ofstream f(path, std::ios::binary);

    auto writeU64 = [&](uint64_t v){ f.write(reinterpret_cast<const char *>(&v), 8); };
    auto writeU32 = [&](uint32_t v){ f.write(reinterpret_cast<const char *>(&v), 4); };
    auto writeU16 = [&](uint16_t v){ f.write(reinterpret_cast<const char *>(&v), 2); };
    auto writeF32 = [&](float v){ f.write(reinterpret_cast<const char *>(&v), 4); };

    writeU64(splats.splats.size());
    writeU64(splats.maxShDegree);
    for (const Gsplat &s : splats.splats){
        writeF32(s.center.x);
        writeF32(s.center.y);
        writeF32(s.center.z);
        writeU16(s.opacity.bits);
        for (int d = 0; d < 3; d++) writeU16(s.rgb[d].bits);
        for (int d = 0; d < 3; d++) writeU16(s.lnScales[d].bits);
        for (int d = 0; d < 4; d++) writeU16(s.quaternion[d].bits);
    }
    writeU64(splats.children.size());
    for (const std::vector<size_t> &c : splats.children){
        writeU32(static_cast<uint32_t>(c.size()));
        for (size_t i : c) writeU32(static_cast<uint32_t>(i));
    }
    if (splats.maxShDegree >= 1){
        for (const GsplatSH1 &sh : splats.sh1) for (const F16 &v : sh) writeU16(v.bits);
    }
    if (splats.maxShDegree >= 2){
        for (const GsplatSH2 &sh : splats.sh2) for (const F16 &v : sh) writeU16(v.bits);
    }
    if (splats.maxShDegree >= 3){
        for (const GsplatSH3 &sh : splats.sh3) for (const F16 &v : sh) writeU16(v.bits);
    }
}

// ---------------------------------------------------------------------------
// Pipeline (port of build-lod/src/main.rs process_file_lod_tsplat, default
// options) minus the encoder, which is invoked by saveRad
// ---------------------------------------------------------------------------

struct PipelineResult {
    size_t inputSplatCount = 0;
    size_t inputShDegree = 0;
    bool removedAny = false;
    size_t emptySplatCount = 0;
    size_t initialSplatCount = 0;
    double lodDuration = 0.0;
    size_t finalSplatCount = 0;
    double chunkDuration = 0.0;
    size_t maxShDegree = 0;
};

bool runPipeline(const SplatData &data, GsplatArray &splats, PipelineResult &result){
    if (!ingestSplats(data, splats)){
        return false;
    }
    dumpStage(splats, "post_decode");

    result.inputSplatCount = splats.len();
    result.inputShDegree = splats.maxShDegree;

    // Validation (main.rs, no --skip-validate)
    {
        size_t invalidCount = 0;
        for (size_t i = 0; i < splats.len(); i++){
            const Gsplat &s = splats.splats[i];
            if (!s.getCenter().isFinite() || !s.getScales().isFinite() ||
                !s.getQuaternion().isFinite() || !std::isfinite(s.getOpacity()) ||
                !s.getRgb().isFinite()){
                invalidCount += 1;
            }
        }
        if (invalidCount > 0){
            std::cerr << "saveRad: found " << invalidCount << " invalid (non-finite) splats"
                      << std::endl;
            return false;
        }
    }

    splats.retain([](const Gsplat &s){
        return (s.getOpacity() > 0.0f) && (s.maxScale() > 0.0f) &&
               (s.getQuaternion().isFinite() && s.getQuaternion().length() > 0.0f);
    });
    dumpStage(splats, "post_retain");

    if (result.inputSplatCount != splats.len()){
        result.removedAny = true;
        result.emptySplatCount = result.inputSplatCount - splats.len();
        result.initialSplatCount = splats.len();
    }

    // LoD: default method Quality -> BhattLod { lod_base: 1.75 }
    auto lodStart = std::chrono::steady_clock::now();
    bhattComputeLodTree(splats, 1.75f);
    result.lodDuration = std::chrono::duration<double>(std::chrono::steady_clock::now() - lodStart).count();
    dumpStage(splats, "post_lod");

    result.finalSplatCount = splats.len();

    auto chunkStart = std::chrono::steady_clock::now();
    chunkTreeSize(splats, 0);
    result.chunkDuration = std::chrono::duration<double>(std::chrono::steady_clock::now() - chunkStart).count();
    dumpStage(splats, "post_chunk");

    result.maxShDegree = splats.maxShDegree;

    splats.encodeLodOpacity();
    dumpStage(splats, "post_elo");

    return true;
}

// ---------------------------------------------------------------------------
// Deflate via zlib: raw stream (no zlib header, windowBits -15), level 6 —
// the same parameters the Rust reference uses with miniz_oxide. zlib's
// compressed bytes differ from miniz_oxide's, but any valid raw deflate
// stream decodes to the identical payload, which is all .rad readers need.
// ---------------------------------------------------------------------------

std::vector<uint8_t> compressToVec(const std::vector<uint8_t> &data){
    z_stream strm;
    std::memset(&strm, 0, sizeof(strm));
    if (deflateInit2(&strm, 6, Z_DEFLATED, -15, 8, Z_DEFAULT_STRATEGY) != Z_OK){
        throw std::runtime_error("saveRad: deflateInit2 failed");
    }
    uLong bound = deflateBound(&strm, static_cast<uLong>(data.size()));
    std::vector<uint8_t> out(bound);
    strm.next_in = const_cast<Bytef *>(data.data());
    strm.avail_in = static_cast<uInt>(data.size());
    strm.next_out = out.data();
    strm.avail_out = static_cast<uInt>(bound);
    int ret = deflate(&strm, Z_FINISH);
    if (ret != Z_STREAM_END){
        deflateEnd(&strm);
        throw std::runtime_error("saveRad: deflate failed");
    }
    out.resize(strm.total_out);
    deflateEnd(&strm);
    return out;
}

// ---------------------------------------------------------------------------
// Property encoders (port of rad.rs encode_*). All planar (dimension-major)
// except oct88r8, which is 3 bytes per splat interleaved.
// ---------------------------------------------------------------------------

std::vector<uint8_t> encodeF32(const std::vector<float> &data, size_t dims, size_t count){
    std::vector<uint8_t> result;
    result.reserve(4 * dims * count);
    for (size_t d = 0; d < dims; d++){
        size_t index = d;
        for (size_t i = 0; i < count; i++){
            uint32_t bits = f32Bits(data[index]);
            result.push_back(static_cast<uint8_t>(bits));
            result.push_back(static_cast<uint8_t>(bits >> 8));
            result.push_back(static_cast<uint8_t>(bits >> 16));
            result.push_back(static_cast<uint8_t>(bits >> 24));
            index += dims;
        }
    }
    return result;
}

std::vector<uint8_t> encodeF16Prop(const std::vector<float> &data, size_t dims, size_t count){
    std::vector<uint8_t> result;
    result.reserve(2 * dims * count);
    for (size_t d = 0; d < dims; d++){
        size_t index = d;
        for (size_t i = 0; i < count; i++){
            uint16_t bits = f16FromF32(data[index]);
            result.push_back(static_cast<uint8_t>(bits));
            result.push_back(static_cast<uint8_t>(bits >> 8));
            index += dims;
        }
    }
    return result;
}

std::vector<uint8_t> encodeF32LeBytes(const std::vector<float> &data, size_t dims, size_t count){
    std::vector<uint8_t> result;
    result.reserve(4 * dims * count);
    for (int b = 0; b < 4; b++){
        for (size_t d = 0; d < dims; d++){
            size_t index = d;
            for (size_t i = 0; i < count; i++){
                uint32_t bits = f32Bits(data[index]);
                result.push_back(static_cast<uint8_t>(bits >> (8 * b)));
                index += dims;
            }
        }
    }
    return result;
}

std::vector<uint8_t> encodeR8(const std::vector<float> &data, size_t dims, size_t count, float min,
                              float max){
    std::vector<uint8_t> result;
    result.reserve(dims * count);
    for (size_t d = 0; d < dims; d++){
        size_t index = d;
        for (size_t i = 0; i < count; i++){
            float value = (data[index] - min) / (max - min) * 255.0f;
            result.push_back(rustCastU8(std::round(rustClamp(value, 0.0f, 255.0f))));
            index += dims;
        }
    }
    return result;
}

std::vector<uint8_t> encodeS8(const std::vector<float> &data, size_t dims, size_t count, float max){
    std::vector<uint8_t> result;
    result.reserve(dims * count);
    for (size_t d = 0; d < dims; d++){
        size_t index = d;
        for (size_t i = 0; i < count; i++){
            float value = data[index] / max * 127.0f;
            result.push_back(static_cast<uint8_t>(
                rustCastI8(std::round(rustClamp(value, -127.0f, 127.0f)))));
            index += dims;
        }
    }
    return result;
}

std::vector<uint8_t> encodeR8Delta(const std::vector<float> &data, size_t dims, size_t count,
                                   float min, float max){
    std::vector<uint8_t> result;
    result.reserve(dims * count);
    for (size_t d = 0; d < dims; d++){
        size_t index = d;
        uint8_t last = 0;
        for (size_t i = 0; i < count; i++){
            float v = (data[index] - min) / (max - min) * 255.0f;
            uint8_t value = rustCastU8(std::round(rustClamp(v, 0.0f, 255.0f)));
            result.push_back(static_cast<uint8_t>(value - last)); // wrapping_sub
            last = value;
            index += dims;
        }
    }
    return result;
}

// splat_encode.rs encode_scale8_zero
uint8_t encodeScale8Zero(float scale, float lnScaleZero, float lnScaleMin, float lnScaleMax){
    if (scale <= 0.0f) return 0;
    float lnScale = std::log(scale);
    if (lnScale <= lnScaleZero) return 0;
    float value = (lnScale - lnScaleMin) / (lnScaleMax - lnScaleMin) * 254.0f;
    return static_cast<uint8_t>(rustCastU8(std::round(rustClamp(value, 0.0f, 254.0f))) + 1);
}

std::vector<uint8_t> encodeLn0R8(const std::vector<float> &data, size_t dims, size_t count,
                                 float zero, float min, float max){
    std::vector<uint8_t> result;
    result.reserve(dims * count);
    for (size_t d = 0; d < dims; d++){
        size_t index = d;
        for (size_t i = 0; i < count; i++){
            result.push_back(encodeScale8Zero(data[index], zero, min, max));
            index += dims;
        }
    }
    return result;
}

std::vector<uint8_t> encodeLnF16(const std::vector<float> &data, size_t dims, size_t count){
    std::vector<uint8_t> result;
    result.reserve(2 * dims * count);
    for (size_t d = 0; d < dims; d++){
        size_t index = d;
        for (size_t i = 0; i < count; i++){
            uint16_t bits = f16FromF32(std::log(data[index]));
            result.push_back(static_cast<uint8_t>(bits));
            result.push_back(static_cast<uint8_t>(bits >> 8));
            index += dims;
        }
    }
    return result;
}

// splat_encode.rs float_to_u8
uint8_t floatToU8(float value, float min, float max){
    return rustCastU8(std::round(rustClamp((value - min) / (max - min) * 255.0f, 0.0f, 255.0f)));
}

// splat_encode.rs encode_quat_oct888 (quat in xyzw order)
void encodeQuatOct888(const float quatXyzw[4], uint8_t out[3]){
    float quat[4];
    if (quatXyzw[3] < 0.0f){
        for (int i = 0; i < 4; i++) quat[i] = -quatXyzw[i];
    }else{
        for (int i = 0; i < 4; i++) quat[i] = quatXyzw[i];
    }
    float theta = 2.0f * std::acos(rustClamp(quat[3], 0.0f, 1.0f));
    float s = std::sin(theta * 0.5f);

    float axis[3];
    if (std::fabs(s) < 1e-6f){
        axis[0] = 1.0f;
        axis[1] = 0.0f;
        axis[2] = 0.0f;
    }else{
        for (int i = 0; i < 3; i++) axis[i] = quat[i] / s;
    }
    float sum = std::fabs(axis[0]) + std::fabs(axis[1]) + std::fabs(axis[2]);
    float p[2] = {axis[0] / sum, axis[1] / sum};
    if (axis[2] < 0.0f){
        float p0 = (1.0f - std::fabs(p[1])) * (p[0] >= 0.0f ? 1.0f : -1.0f);
        float p1 = (1.0f - std::fabs(p[0])) * (p[1] >= 0.0f ? 1.0f : -1.0f);
        p[0] = p0;
        p[1] = p1;
    }
    out[0] = floatToU8(p[0], -1.0f, 1.0f);
    out[1] = floatToU8(p[1], -1.0f, 1.0f);
    out[2] = floatToU8(theta, 0.0f, 3.14159265358979323846264338327950288f);
}

std::vector<uint8_t> encodeQuatOct88R8(const std::vector<float> &data, size_t count){
    std::vector<uint8_t> result;
    result.reserve(3 * count);
    for (size_t i = 0; i < count; i++){
        uint8_t enc[3];
        encodeQuatOct888(&data[i * 4], enc);
        result.push_back(enc[0]);
        result.push_back(enc[1]);
        result.push_back(enc[2]);
    }
    return result;
}

std::vector<uint8_t> encodeU16Prop(const std::vector<uint16_t> &data, size_t count){
    std::vector<uint8_t> result;
    result.reserve(2 * count);
    for (size_t i = 0; i < count; i++){
        result.push_back(static_cast<uint8_t>(data[i]));
        result.push_back(static_cast<uint8_t>(data[i] >> 8));
    }
    return result;
}

std::vector<uint8_t> encodeUsizeAsU32(const std::vector<size_t> &data, size_t count){
    std::vector<uint8_t> result;
    result.reserve(4 * count);
    for (size_t i = 0; i < count; i++){
        uint32_t v = rustCastU32(data[i]);
        result.push_back(static_cast<uint8_t>(v));
        result.push_back(static_cast<uint8_t>(v >> 8));
        result.push_back(static_cast<uint8_t>(v >> 16));
        result.push_back(static_cast<uint8_t>(v >> 24));
    }
    return result;
}

// ---------------------------------------------------------------------------
// RadEncoder (port of rad.rs RadEncoder for the default configuration:
// no SH clusters, resolve everything from Auto)
// ---------------------------------------------------------------------------

const uint32_t RAD_MAGIC = 0x30444152;       // 'RAD0'
const uint32_t RAD_CHUNK_MAGIC = 0x43444152; // 'RADC'

inline size_t roundup8(size_t size){ return (size + 7) & ~static_cast<size_t>(7); }
inline size_t pad8(size_t size){ return (8 - (size & 7)) & 7; }

struct SplatEncodingParams {
    float rgbMin = 0.0f;
    float rgbMax = 1.0f;
    float lnScaleMin = -12.0f;
    float lnScaleMax = 9.0f;
    float sh1Max = 1.0f;
    float sh2Max = 1.0f;
    float sh3Max = 1.0f;
    bool lodOpacity = false;
};

enum class PropEncoding { F32LeBytes, F16, R8, R8Delta, Ln0R8, LnF16, Oct88R8, S8, U16, U32 };

const char *propEncodingName(PropEncoding e){
    switch (e){
        case PropEncoding::F32LeBytes: return "f32_lebytes";
        case PropEncoding::F16: return "f16";
        case PropEncoding::R8: return "r8";
        case PropEncoding::R8Delta: return "r8_delta";
        case PropEncoding::Ln0R8: return "ln_0r8";
        case PropEncoding::LnF16: return "ln_f16";
        case PropEncoding::Oct88R8: return "oct88r8";
        case PropEncoding::S8: return "s8";
        case PropEncoding::U16: return "u16";
        case PropEncoding::U32: return "u32";
    }
    return "";
}

struct EncodedProp {
    const char *name;
    PropEncoding encoding;
    bool hasMinMax = false;
    float min = 0.0f;
    float max = 0.0f;
    uint64_t offset = 0;
    uint64_t bytes = 0;
    std::vector<uint8_t> data;
};

struct RadEncoder {
    GsplatArray &splats;
    size_t maxSh; // min(getter max_sh_degree, 3)
    bool hasEncoding = false;
    SplatEncodingParams encoding;

    // Rust-enum-variant names of the current encoding state (for the comment)
    std::string centerEncoding = "Auto";
    std::string alphaEncoding = "Auto";
    std::string rgbEncoding = "Auto";
    std::string scalesEncoding = "Auto";
    std::string orientationEncoding = "Auto";
    std::string shEncoding = "Auto";
    std::string shLabelEncoding = "Auto";

    std::string comment;

    explicit RadEncoder(GsplatArray &s) : splats(s), maxSh(std::min(s.maxShDegree, size_t(3))) {}

    // Percentile helper: n-th order statistic with OrderedFloat comparator
    static float selectNth(std::vector<float> &v, size_t n){
        std::nth_element(v.begin(), v.begin() + n, v.end());
        return v[n];
    }

    void resolveEncoding(){
        // resolve_center_encoding
        centerEncoding = "F32LeBytes";

        // resolve_alpha_encoding
        {
            float maxAlpha = -std::numeric_limits<float>::infinity();
            for (const Gsplat &s : splats.splats){
                maxAlpha = rustMax(maxAlpha, s.getOpacity());
            }
            alphaEncoding = (maxAlpha > 1.0f) ? "F16" : "R8";
        }

        // resolve_rgb_encoding
        {
            size_t n = splats.len();
            std::vector<float> allRgb(n * 3);
            for (size_t i = 0; i < n; i++){
                Vec3 rgb = splats.splats[i].getRgb();
                allRgb[i * 3] = rgb.x;
                allRgb[i * 3 + 1] = rgb.y;
                allRgb[i * 3 + 2] = rgb.z;
            }
            size_t len = allRgb.size();
            size_t n1 = std::min(static_cast<size_t>(std::round(static_cast<float>(len) * 0.01f)), len - 1);
            size_t n99 = std::min(static_cast<size_t>(std::round(static_cast<float>(len) * 0.99f)), len - 1);
            float rgb1 = selectNth(allRgb, n1);
            float rgb99 = selectNth(allRgb, n99);
            float rgbMin = rustMin(rgb1, 0.0f);
            float rgbMax = rustMax(rgb99, 1.0f);

            if (rgbMin < -1.0f || rgbMax > 2.0f){
                rgbEncoding = "F16";
            }else{
                rgbEncoding = "R8Delta";
                hasEncoding = true;
                encoding.rgbMin = rgbMin;
                encoding.rgbMax = rgbMax;
            }
        }

        // resolve_scales_encoding
        {
            std::vector<float> scales;
            scales.reserve(splats.len() * 2);
            for (const Gsplat &s : splats.splats){
                Vec3 sc = s.getScales();
                float splatScales[3] = {sc.x, sc.y, sc.z};
                // 3-element stable ascending sort (Rust sort_by_key)
                std::stable_sort(splatScales, splatScales + 3);
                scales.push_back(splatScales[1]);
                scales.push_back(splatScales[2]);
            }
            size_t len = scales.size();
            size_t n1 = std::min(static_cast<size_t>(std::round(static_cast<float>(len) * 0.01f)), len - 1);
            size_t n99 = std::min(static_cast<size_t>(std::round(static_cast<float>(len) * 0.99f)), len - 1);
            float scale1 = selectNth(scales, n1);
            float scale99 = selectNth(scales, n99);
            float lnScaleMin = rustMin(std::log(rustMax(scale1, 1.0e-30f)), -12.0f);
            float lnScaleMax = rustMax(std::log(rustMax(scale99, 1.0e-30f)), 9.0f);

            if ((lnScaleMax - lnScaleMin) > 25.0f){
                scalesEncoding = "LnF16";
            }else{
                scalesEncoding = "Ln0R8";
                hasEncoding = true;
                encoding.lnScaleMin = lnScaleMin;
                encoding.lnScaleMax = lnScaleMax;
            }
        }

        // resolve_orientation_encoding
        orientationEncoding = "Oct88R8";

        // resolve_sh_encoding
        {
            size_t numSh = std::min(maxSh, splats.maxShDegree);
            if (numSh == 0){
                shEncoding = "S8";
            }else{
                hasEncoding = true;

                size_t n = splats.len();
                std::vector<float> allRgb(n * 9);
                for (size_t i = 0; i < n; i++){
                    for (int d = 0; d < 9; d++) allRgb[i * 9 + d] = splats.sh1[i][d].toF32();
                }
                {
                    size_t len = allRgb.size();
                    size_t n5 = std::min(static_cast<size_t>(std::round(static_cast<float>(len) * 0.05f)), len - 1);
                    size_t n95 = std::min(static_cast<size_t>(std::round(static_cast<float>(len) * 0.95f)), len - 1);
                    float sh5 = selectNth(allRgb, n5);
                    float sh95 = selectNth(allRgb, n95);
                    encoding.sh1Max = rustMax(rustMax(std::fabs(sh5), std::fabs(sh95)), 1.0f);
                }

                if (numSh >= 2){
                    allRgb.assign(n * 15, 0.0f);
                    for (size_t i = 0; i < n; i++){
                        for (int d = 0; d < 15; d++) allRgb[i * 15 + d] = splats.sh2[i][d].toF32();
                    }
                    size_t len = allRgb.size();
                    size_t n5 = std::min(static_cast<size_t>(std::round(static_cast<float>(len) * 0.05f)), len - 1);
                    size_t n95 = std::min(static_cast<size_t>(std::round(static_cast<float>(len) * 0.95f)), len - 1);
                    float sh5 = selectNth(allRgb, n5);
                    float sh95 = selectNth(allRgb, n95);
                    encoding.sh2Max = rustMax(rustMax(std::fabs(sh5), std::fabs(sh95)), 1.0f);
                }

                if (numSh >= 3){
                    allRgb.assign(n * 21, 0.0f);
                    for (size_t i = 0; i < n; i++){
                        for (int d = 0; d < 21; d++) allRgb[i * 21 + d] = splats.sh3[i][d].toF32();
                    }
                    size_t len = allRgb.size();
                    size_t n5 = std::min(static_cast<size_t>(std::round(static_cast<float>(len) * 0.05f)), len - 1);
                    size_t n95 = std::min(static_cast<size_t>(std::round(static_cast<float>(len) * 0.95f)), len - 1);
                    float sh5 = selectNth(allRgb, n5);
                    float sh95 = selectNth(allRgb, n95);
                    encoding.sh3Max = rustMax(rustMax(std::fabs(sh5), std::fabs(sh95)), 1.0f);
                }

                shEncoding = "S8";
            }
        }

        // resolve_sh_label_encoding: no clusters -> stays Auto
    }

    // json! serialization of the current encoding state (BTreeMap: keys sorted)
    JsonValue encodingStateJson() const {
        JsonValue obj = JsonValue::object();
        obj.insert("alpha", JsonValue::str(alphaEncoding));
        obj.insert("center", JsonValue::str(centerEncoding));
        if (!hasEncoding){
            obj.insert("encoding", JsonValue::null());
        }else{
            JsonValue enc = JsonValue::object();
            // json! -> Value object -> BTreeMap: alphabetical keys, f32 -> f64
            enc.insert("lnScaleMax", JsonValue::f32AsF64(encoding.lnScaleMax));
            enc.insert("lnScaleMin", JsonValue::f32AsF64(encoding.lnScaleMin));
            enc.insert("lodOpacity", JsonValue::boolean(encoding.lodOpacity));
            enc.insert("rgbMax", JsonValue::f32AsF64(encoding.rgbMax));
            enc.insert("rgbMin", JsonValue::f32AsF64(encoding.rgbMin));
            enc.insert("sh1Max", JsonValue::f32AsF64(encoding.sh1Max));
            enc.insert("sh2Max", JsonValue::f32AsF64(encoding.sh2Max));
            enc.insert("sh3Max", JsonValue::f32AsF64(encoding.sh3Max));
            obj.insert("encoding", std::move(enc));
        }
        obj.insert("orientation", JsonValue::str(orientationEncoding));
        obj.insert("rgb", JsonValue::str(rgbEncoding));
        obj.insert("scales", JsonValue::str(scalesEncoding));
        obj.insert("sh", JsonValue::str(shEncoding));
        obj.insert("sh_label", JsonValue::str(shLabelEncoding));
        return obj;
    }

    // SetSplatEncoding serialized from the struct: declaration order, f32 via ryu f32
    JsonValue setSplatEncodingJson(bool lodOpacity) const {
        JsonValue enc = JsonValue::object();
        enc.insert("rgbMin", JsonValue::f32(encoding.rgbMin));
        enc.insert("rgbMax", JsonValue::f32(encoding.rgbMax));
        enc.insert("lnScaleMin", JsonValue::f32(encoding.lnScaleMin));
        enc.insert("lnScaleMax", JsonValue::f32(encoding.lnScaleMax));
        enc.insert("sh1Max", JsonValue::f32(encoding.sh1Max));
        enc.insert("sh2Max", JsonValue::f32(encoding.sh2Max));
        enc.insert("sh3Max", JsonValue::f32(encoding.sh3Max));
        enc.insert("lodOpacity", JsonValue::boolean(lodOpacity));
        return enc;
    }

    EncodedProp encodeChunkCenter(size_t base, size_t count, std::vector<float> &buffer){
        if (buffer.size() < count * 3) buffer.resize(count * 3, 0.0f);
        for (size_t i = 0; i < count; i++){
            const Vec3 &c = splats.splats[base + i].center;
            buffer[i * 3] = c.x;
            buffer[i * 3 + 1] = c.y;
            buffer[i * 3 + 2] = c.z;
        }
        EncodedProp p;
        p.name = "center";
        p.encoding = PropEncoding::F32LeBytes;
        p.data = compressToVec(encodeF32LeBytes(buffer, 3, count));
        return p;
    }

    EncodedProp encodeChunkAlpha(size_t base, size_t count, std::vector<float> &buffer){
        if (buffer.size() < count) buffer.resize(count, 0.0f);
        for (size_t i = 0; i < count; i++){
            buffer[i] = splats.splats[base + i].getOpacity();
        }
        float maxAlpha = splats.hasLodTree() ? 2.0f : 1.0f;
        EncodedProp p;
        p.name = "alpha";
        if (alphaEncoding == "R8"){
            p.encoding = PropEncoding::R8;
            p.hasMinMax = true;
            p.min = 0.0f;
            p.max = maxAlpha;
            p.data = compressToVec(encodeR8(buffer, 1, count, 0.0f, maxAlpha));
        }else{
            p.encoding = PropEncoding::F16;
            p.data = compressToVec(encodeF16Prop(buffer, 1, count));
        }
        return p;
    }

    EncodedProp encodeChunkRgb(size_t base, size_t count, std::vector<float> &buffer){
        if (buffer.size() < count * 3) buffer.resize(count * 3, 0.0f);
        for (size_t i = 0; i < count; i++){
            Vec3 rgb = splats.splats[base + i].getRgb();
            buffer[i * 3] = rgb.x;
            buffer[i * 3 + 1] = rgb.y;
            buffer[i * 3 + 2] = rgb.z;
        }
        EncodedProp p;
        p.name = "rgb";
        if (rgbEncoding == "F16"){
            p.encoding = PropEncoding::F16;
            p.data = compressToVec(encodeF16Prop(buffer, 3, count));
        }else{
            p.encoding = PropEncoding::R8Delta;
            p.hasMinMax = true;
            p.min = encoding.rgbMin;
            p.max = encoding.rgbMax;
            p.data = compressToVec(encodeR8Delta(buffer, 3, count, encoding.rgbMin, encoding.rgbMax));
        }
        return p;
    }

    EncodedProp encodeChunkScales(size_t base, size_t count, std::vector<float> &buffer){
        if (buffer.size() < count * 3) buffer.resize(count * 3, 0.0f);
        for (size_t i = 0; i < count; i++){
            Vec3 sc = splats.splats[base + i].getScales();
            buffer[i * 3] = sc.x;
            buffer[i * 3 + 1] = sc.y;
            buffer[i * 3 + 2] = sc.z;
        }
        EncodedProp p;
        p.name = "scales";
        if (scalesEncoding == "LnF16"){
            p.encoding = PropEncoding::LnF16;
            p.data = compressToVec(encodeLnF16(buffer, 3, count));
        }else{
            p.encoding = PropEncoding::Ln0R8;
            p.hasMinMax = true;
            p.min = encoding.lnScaleMin;
            p.max = encoding.lnScaleMax;
            p.data = compressToVec(
                encodeLn0R8(buffer, 3, count, -30.0f, encoding.lnScaleMin, encoding.lnScaleMax));
        }
        return p;
    }

    EncodedProp encodeChunkOrientation(size_t base, size_t count, std::vector<float> &buffer){
        if (buffer.size() < count * 4) buffer.resize(count * 4, 0.0f);
        for (size_t i = 0; i < count; i++){
            Quat q = splats.splats[base + i].getQuaternion();
            buffer[i * 4] = q.x;
            buffer[i * 4 + 1] = q.y;
            buffer[i * 4 + 2] = q.z;
            buffer[i * 4 + 3] = q.w;
        }
        EncodedProp p;
        p.name = "orientation";
        p.encoding = PropEncoding::Oct88R8;
        p.data = compressToVec(encodeQuatOct88R8(buffer, count));
        return p;
    }

    EncodedProp encodeChunkSh(size_t base, size_t count, std::vector<float> &buffer, int degree){
        size_t elements = degree == 1 ? 9 : (degree == 2 ? 15 : 21);
        float shMax = degree == 1 ? encoding.sh1Max : (degree == 2 ? encoding.sh2Max : encoding.sh3Max);
        if (buffer.size() < count * elements) buffer.resize(count * elements, 0.0f);
        for (size_t i = 0; i < count; i++){
            if (degree == 1){
                for (size_t d = 0; d < 9; d++) buffer[i * 9 + d] = splats.sh1[base + i][d].toF32();
            }else if (degree == 2){
                for (size_t d = 0; d < 15; d++) buffer[i * 15 + d] = splats.sh2[base + i][d].toF32();
            }else{
                for (size_t d = 0; d < 21; d++) buffer[i * 21 + d] = splats.sh3[base + i][d].toF32();
            }
        }
        EncodedProp p;
        p.name = degree == 1 ? "sh1" : (degree == 2 ? "sh2" : "sh3");
        p.encoding = PropEncoding::S8;
        p.hasMinMax = true;
        p.min = -shMax;
        p.max = shMax;
        p.data = compressToVec(encodeS8(buffer, elements, count, shMax));
        return p;
    }

    EncodedProp encodeChunkChildCount(size_t base, size_t count, std::vector<uint16_t> &buffer){
        if (buffer.size() < count) buffer.resize(count, 0);
        for (size_t i = 0; i < count; i++){
            buffer[i] = static_cast<uint16_t>(splats.children[base + i].size());
        }
        EncodedProp p;
        p.name = "child_count";
        p.encoding = PropEncoding::U16;
        p.data = compressToVec(encodeU16Prop(buffer, count));
        return p;
    }

    EncodedProp encodeChunkChildStart(size_t base, size_t count, std::vector<size_t> &buffer){
        if (buffer.size() < count) buffer.resize(count, 0);
        for (size_t i = 0; i < count; i++){
            const std::vector<size_t> &c = splats.children[base + i];
            buffer[i] = c.empty() ? 0 : c[0];
        }
        EncodedProp p;
        p.name = "child_start";
        p.encoding = PropEncoding::U32;
        p.data = compressToVec(encodeUsizeAsU32(buffer, count));
        return p;
    }

    std::vector<uint8_t> encodeChunk(size_t base, size_t count, std::vector<float> &buffer,
                                     std::vector<uint16_t> &bufferU16,
                                     std::vector<size_t> &bufferUsize){
        size_t chunkMaxSh = std::min(splats.maxShDegree, maxSh);

        std::vector<EncodedProp> props;
        props.push_back(encodeChunkCenter(base, count, buffer));
        props.push_back(encodeChunkAlpha(base, count, buffer));
        props.push_back(encodeChunkRgb(base, count, buffer));
        props.push_back(encodeChunkScales(base, count, buffer));
        props.push_back(encodeChunkOrientation(base, count, buffer));

        if (chunkMaxSh >= 1) props.push_back(encodeChunkSh(base, count, buffer, 1));
        if (chunkMaxSh >= 2) props.push_back(encodeChunkSh(base, count, buffer, 2));
        if (chunkMaxSh >= 3) props.push_back(encodeChunkSh(base, count, buffer, 3));

        if (splats.hasLodTree()){
            props.push_back(encodeChunkChildCount(base, count, bufferU16));
            props.push_back(encodeChunkChildStart(base, count, bufferUsize));
        }

        uint64_t offset = 0;
        for (EncodedProp &p : props){
            p.offset = offset;
            p.bytes = p.data.size();
            offset += roundup8(p.data.size());
        }
        uint64_t payloadBytes = offset;

        // RadChunkMeta: compact JSON, struct declaration order
        JsonValue meta = JsonValue::object();
        meta.insert("version", JsonValue::uinteger(1));
        meta.insert("base", JsonValue::uinteger(base));
        meta.insert("count", JsonValue::uinteger(count));
        meta.insert("payloadBytes", JsonValue::uinteger(payloadBytes));
        meta.insert("maxSh", JsonValue::uinteger(chunkMaxSh));
        if (splats.hasLodTree()) meta.insert("lodTree", JsonValue::boolean(true));
        if (hasEncoding){
            meta.insert("splatEncoding", setSplatEncodingJson(splats.hasLodTree()));
        }
        JsonValue propsJson = JsonValue::array();
        for (const EncodedProp &p : props){
            JsonValue pj = JsonValue::object();
            pj.insert("offset", JsonValue::uinteger(p.offset));
            pj.insert("bytes", JsonValue::uinteger(p.bytes));
            pj.insert("property", JsonValue::str(p.name));
            pj.insert("encoding", JsonValue::str(propEncodingName(p.encoding)));
            pj.insert("compression", JsonValue::str("gz"));
            if (p.hasMinMax){
                pj.insert("min", JsonValue::f32(p.min));
                pj.insert("max", JsonValue::f32(p.max));
            }
            propsJson.arr.push_back(std::move(pj));
        }
        meta.insert("properties", std::move(propsJson));

        std::string metaStr = jsonToString(meta, false);

        std::vector<uint8_t> encoded;
        encoded.reserve(8 + roundup8(metaStr.size()) + 8 + static_cast<size_t>(payloadBytes));
        auto pushU32 = [&](uint32_t v){
            for (int b = 0; b < 4; b++) encoded.push_back(static_cast<uint8_t>(v >> (8 * b)));
        };
        auto pushU64 = [&](uint64_t v){
            for (int b = 0; b < 8; b++) encoded.push_back(static_cast<uint8_t>(v >> (8 * b)));
        };
        pushU32(RAD_CHUNK_MAGIC);
        pushU32(static_cast<uint32_t>(metaStr.size()));
        encoded.insert(encoded.end(), metaStr.begin(), metaStr.end());
        encoded.insert(encoded.end(), pad8(metaStr.size()), 0);
        pushU64(payloadBytes);
        for (const EncodedProp &p : props){
            encoded.insert(encoded.end(), p.data.begin(), p.data.end());
            encoded.insert(encoded.end(), pad8(p.data.size()), 0);
        }
        return encoded;
    }

    bool encode(std::ofstream &out){
        const size_t CHUNK_SIZE = 65536;

        size_t numSplats = splats.len();
        size_t fileMaxSh = std::min(splats.maxShDegree, maxSh);

        std::vector<float> buffer;
        size_t bufferDim = fileMaxSh == 0 ? 4 : (fileMaxSh == 1 ? 9 : (fileMaxSh == 2 ? 15 : 21));
        buffer.resize(CHUNK_SIZE * bufferDim, 0.0f);

        std::vector<uint16_t> bufferU16;
        std::vector<size_t> bufferUsize;
        if (splats.hasLodTree()){
            bufferU16.resize(CHUNK_SIZE, 0);
            bufferUsize.resize(CHUNK_SIZE, 0);
        }

        size_t numChunks = (numSplats + CHUNK_SIZE - 1) / CHUNK_SIZE;
        std::vector<std::vector<uint8_t>> chunks;
        chunks.reserve(numChunks);
        std::vector<std::pair<uint64_t, uint64_t>> chunkRanges; // offset, bytes
        chunkRanges.reserve(numChunks);
        uint64_t offset = 0;

        for (size_t chunkIndex = 0; chunkIndex < numChunks; chunkIndex++){
            size_t base = chunkIndex * CHUNK_SIZE;
            size_t count = std::min(numSplats - base, CHUNK_SIZE);
            std::vector<uint8_t> chunk = encodeChunk(base, count, buffer, bufferU16, bufferUsize);
            chunkRanges.push_back({offset, chunk.size()});
            offset += chunk.size();
            chunks.push_back(std::move(chunk));
        }
        uint64_t allChunkBytes = offset;

        // RadMeta: pretty JSON + trailing newline
        JsonValue meta = JsonValue::object();
        meta.insert("version", JsonValue::uinteger(1));
        meta.insert("type", JsonValue::str("gsplat"));
        meta.insert("count", JsonValue::uinteger(numSplats));
        meta.insert("maxSh", JsonValue::uinteger(fileMaxSh));
        if (splats.hasLodTree()) meta.insert("lodTree", JsonValue::boolean(true));
        meta.insert("chunkSize", JsonValue::uinteger(65536));
        meta.insert("allChunkBytes", JsonValue::uinteger(allChunkBytes));
        JsonValue chunksJson = JsonValue::array();
        for (const auto &cr : chunkRanges){
            JsonValue cj = JsonValue::object();
            cj.insert("offset", JsonValue::uinteger(cr.first));
            cj.insert("bytes", JsonValue::uinteger(cr.second));
            chunksJson.arr.push_back(std::move(cj));
        }
        meta.insert("chunks", std::move(chunksJson));
        if (hasEncoding){
            meta.insert("splatEncoding", setSplatEncodingJson(splats.hasLodTree()));
        }
        if (!comment.empty()){
            meta.insert("comment", JsonValue::str(comment));
        }

        std::string metaStr = jsonToString(meta, true);
        metaStr.push_back('\n');
        size_t metaBytesSize = metaStr.size();

        auto writeU32 = [&](uint32_t v){
            char b[4];
            for (int i = 0; i < 4; i++) b[i] = static_cast<char>(v >> (8 * i));
            out.write(b, 4);
        };
        writeU32(RAD_MAGIC);
        writeU32(static_cast<uint32_t>(metaBytesSize));
        out.write(metaStr.data(), static_cast<std::streamsize>(metaStr.size()));
        {
            size_t pad = pad8(metaBytesSize);
            if (pad != 0){
                char zeros[8] = {0};
                out.write(zeros, static_cast<std::streamsize>(pad));
            }
        }
        for (const std::vector<uint8_t> &chunk : chunks){
            assert((chunk.size() & 7) == 0);
            out.write(reinterpret_cast<const char *>(chunk.data()),
                      static_cast<std::streamsize>(chunk.size()));
        }
        return out.good();
    }
};

}

bool saveRad(const std::string &filename, const SplatData &data){
    GsplatArray splats;
    PipelineResult result;

    bool fixedDurations = std::getenv("OPENSPLAT_RAD_FIXED_DURATIONS") != nullptr;

    if (!runPipeline(data, splats, result)){
        return false;
    }

    RadEncoder encoder(splats);

    // Description map, mirroring build-lod main.rs (BTreeMap -> alphabetical keys)
    std::vector<std::pair<std::string, JsonValue>> description;
    description.push_back({"input_splat_count", JsonValue::uinteger(result.inputSplatCount)});
    description.push_back({"input_sh_degree", JsonValue::uinteger(result.inputShDegree)});
    if (result.removedAny){
        description.push_back({"empty_splat_count", JsonValue::uinteger(result.emptySplatCount)});
        description.push_back({"initial_splat_count", JsonValue::uinteger(result.initialSplatCount)});
    }
    description.push_back({"method", JsonValue::str("BhattLod { lod_base: 1.75 }")});
    description.push_back({"lod_duration", JsonValue::f64(fixedDurations ? 0.0 : result.lodDuration)});
    description.push_back({"final_splat_count", JsonValue::uinteger(result.finalSplatCount)});
    description.push_back({"chunk_duration", JsonValue::f64(fixedDurations ? 0.0 : result.chunkDuration)});
    description.push_back({"max_sh_degree", JsonValue::uinteger(result.maxShDegree)});

    description.push_back({"input_encoding", encoder.encodingStateJson()});

    encoder.resolveEncoding();
    description.push_back({"resolved_encoding", encoder.encodingStateJson()});

    // BTreeMap ordering
    std::stable_sort(description.begin(), description.end(),
                     [](const auto &a, const auto &b){ return a.first < b.first; });
    JsonValue descriptionJson = JsonValue::object();
    descriptionJson.obj = std::move(description);
    encoder.comment = jsonToString(descriptionJson, true);

    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()){
        std::cerr << "saveRad: cannot open " << filename << " for writing" << std::endl;
        return false;
    }
    try {
        return encoder.encode(out);
    } catch (const std::exception &e){
        std::cerr << "saveRad: " << e.what() << std::endl;
        return false;
    }
}

}

