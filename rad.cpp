// Writer for Spark's .RAD LoD gaussian splat format.
//
// C++ port of the Rust build-lod tool (https://github.com/sparkjsdev/spark/tree/main/rust/build-lod)
//
// Splat attributes are stored as IEEE 754 half-precision values during
// processing; this quantization is part of the format design and shapes the
// LoD construction.

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

#include <miniz.h>
#include <nlohmann/json.hpp>

namespace rad {
namespace {

// insertion-ordered to control metadata key order
using json = nlohmann::ordered_json;

// IEEE 754 half-precision conversions (round-to-nearest-even)

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
    float toF32() const { return f16ToF32(bits); }
    bool isNan() const { return (bits & 0x7C00u) == 0x7C00u && (bits & 0x03FFu) != 0; }
};

// half-2.6.0 f16::max: if other > self && !other.is_nan() { other } else { self }
// (non-NaN f16 ordering matches f32 ordering of the converted values, incl. -0 == 0)
inline F16 f16Max(F16 self, F16 other){
    if (!self.isNan() && !other.isNan() && other.toF32() > self.toF32()) return other;
    return self;
}

// Rust semantics helpers

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

inline uint32_t rustCastU32(uint64_t v){ return static_cast<uint32_t>(v); }

// Vector / quaternion / matrix math (scalar port of glam)

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

    bool isFinite() const { return std::isfinite(x) && std::isfinite(y) && std::isfinite(z); }
};

struct I64Vec3 {
    int64_t x = 0, y = 0, z = 0;

    I64Vec3() = default;
    I64Vec3(int64_t x_, int64_t y_, int64_t z_) : x(x_), y(y_), z(z_) {}

    int64_t operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }
    bool operator==(const I64Vec3 &o) const { return x == o.x && y == o.y && z == o.z; }
};

struct I64Vec3Hash {
    size_t operator()(const I64Vec3 &v) const {
        // Iteration order of the cell map is never observed, so any hash works.
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

// SymMat3 (port of spark-lib/src/symmat3.rs)
// Storage [xx, yy, zz, xy] + [xz, yz]

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
        float diagMax = std::fmaxf(std::fmaxf(std::fabs(v0[0]), std::fabs(v0[1])), std::fabs(v0[2]));
        float relTol = 1e-9f * std::fmaxf(diagMax * diagMax * diagMax, 1e-30f);
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
            eps = 1e-6f * std::fmaxf(s, 1.0f);
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

        // Stable sort of [0,1,2] by descending eigenvalue
        int idx[3] = {0, 1, 2};
        std::stable_sort(idx, idx + 3, [&](int a, int b){
            return rawVals[b] < rawVals[a];
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

// Max-heap keyed by (float, index), compared lexicographically. Keys are
// unique, so pop order is fully deterministic; the internal array order only
// affects tie-breaking in neighbor scans.

struct HeapKey {
    float key;
    size_t index;

    bool operator<(const HeapKey &o) const {
        if (key != o.key) return key < o.key;
        return index < o.index;
    }
};

struct MaxHeap {
    std::vector<HeapKey> data;

    size_t len() const { return data.size(); }
    bool isEmpty() const { return data.empty(); }

    void push(const HeapKey &item){
        data.push_back(item);
        std::push_heap(data.begin(), data.end());
    }

    bool pop(HeapKey &out){
        if (data.empty()) return false;
        std::pop_heap(data.begin(), data.end());
        out = data.back();
        data.pop_back();
        return true;
    }

    void extend(const std::vector<HeapKey> &items){
        data.insert(data.end(), items.begin(), items.end());
        std::make_heap(data.begin(), data.end());
    }
};

// Gsplat / GsplatArray (port of spark-lib/src/gsplat.rs + tsplat.rs)

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
        float totalWeight = std::fmaxf(sum, 1.0e-30f);
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
        Vec3 scales = Vec3(std::sqrt(std::fmaxf(vals[0], 0.0f)), std::sqrt(std::fmaxf(vals[1], 0.0f)),
                           std::sqrt(std::fmaxf(vals[2], 0.0f)));
        scales = scales.max(Vec3::splat(1.0e-30f));

        Mat3 basis = Mat3::fromCols(vecs[0], vecs[1], vecs[2]);
        Quat quaternion = quatFromMat3(basis);
        float opacity = totalWeight / ellipsoidArea(scales);
        opacity = std::clamp(opacity, 0.000001f, 1000.0f);

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
                splat.setOpacity(std::clamp(0.25f * (d - 1.0f) + 1.0f, 1.0f, 2.0f));
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

// bhatt_lod (port of spark-lib/src/bhatt_lod.rs)

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
        maxChildFeatureSize = std::fmaxf(maxChildFeatureSize, childFeatureSize);
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
    size_t initialLen = splats.len();
    if (initialLen == 0){
        return;
    }

    splats.sortByFeatureSize();
    splats.prepareChildren();

    std::vector<bool> isActive(splats.len(), true);

    float minFeatureSize = std::fmaxf(splats.splats[0].featureSize(), 0.000001f);
    // Rust: min_feature_size.log(MERGE_BASE).ceil() as i16
    int16_t levelMin = rustCastI16(std::ceil(std::log(minFeatureSize) / std::log(MERGE_BASE)));

    int32_t level = levelMin;
    size_t frontier = 0;
    MaxHeap active;
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

// chunk_tree (port of spark-lib/src/chunk_tree.rs chunk_tree_size)

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

        MaxHeap priority;
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

// Ingestion: SplatData -> GsplatArray, applying the same transforms
// spark-lib/src/ply.rs applies when reading a 3DGS PLY

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

// Pipeline (port of build-lod/src/main.rs process_file_lod_tsplat, default
// options) minus the encoder, which is invoked by saveRad

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

    if (result.inputSplatCount != splats.len()){
        result.removedAny = true;
        result.emptySplatCount = result.inputSplatCount - splats.len();
        result.initialSplatCount = splats.len();
    }

    // LoD: default method Quality -> BhattLod { lod_base: 1.75 }
    auto lodStart = std::chrono::steady_clock::now();
    bhattComputeLodTree(splats, 1.75f);
    result.lodDuration = std::chrono::duration<double>(std::chrono::steady_clock::now() - lodStart).count();

    result.finalSplatCount = splats.len();

    auto chunkStart = std::chrono::steady_clock::now();
    chunkTreeSize(splats, 0);
    result.chunkDuration = std::chrono::duration<double>(std::chrono::steady_clock::now() - chunkStart).count();

    result.maxShDegree = splats.maxShDegree;

    splats.encodeLodOpacity();

    return true;
}

// Raw deflate stream (no zlib header, windowBits -15), level 6

std::vector<uint8_t> compressToVec(const std::vector<uint8_t> &data){
    mz_stream strm;
    std::memset(&strm, 0, sizeof(strm));
    if (mz_deflateInit2(&strm, 6, MZ_DEFLATED, -MZ_DEFAULT_WINDOW_BITS, 8, MZ_DEFAULT_STRATEGY) != MZ_OK){
        throw std::runtime_error("saveRad: deflateInit2 failed");
    }
    mz_ulong bound = mz_deflateBound(&strm, static_cast<mz_ulong>(data.size()));
    std::vector<uint8_t> out(bound);
    strm.next_in = data.data();
    strm.avail_in = static_cast<unsigned int>(data.size());
    strm.next_out = out.data();
    strm.avail_out = static_cast<unsigned int>(bound);
    int ret = mz_deflate(&strm, MZ_FINISH);
    if (ret != MZ_STREAM_END){
        mz_deflateEnd(&strm);
        throw std::runtime_error("saveRad: deflate failed");
    }
    out.resize(strm.total_out);
    mz_deflateEnd(&strm);
    return out;
}

// Property encoders (port of rad.rs encode_*). All planar (dimension-major)
// except oct88r8, which is 3 bytes per splat interleaved.

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
            result.push_back(rustCastU8(std::round(std::clamp(value, 0.0f, 255.0f))));
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
                rustCastI8(std::round(std::clamp(value, -127.0f, 127.0f)))));
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
            uint8_t value = rustCastU8(std::round(std::clamp(v, 0.0f, 255.0f)));
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
    return static_cast<uint8_t>(rustCastU8(std::round(std::clamp(value, 0.0f, 254.0f))) + 1);
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
    return rustCastU8(std::round(std::clamp((value - min) / (max - min) * 255.0f, 0.0f, 255.0f)));
}

// splat_encode.rs encode_quat_oct888 (quat in xyzw order)
void encodeQuatOct888(const float quatXyzw[4], uint8_t out[3]){
    float quat[4];
    if (quatXyzw[3] < 0.0f){
        for (int i = 0; i < 4; i++) quat[i] = -quatXyzw[i];
    }else{
        for (int i = 0; i < 4; i++) quat[i] = quatXyzw[i];
    }
    float theta = 2.0f * std::acos(std::clamp(quat[3], 0.0f, 1.0f));
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

// RadEncoder (port of rad.rs RadEncoder for the default configuration:
// no SH clusters, resolve everything from Auto)

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
                maxAlpha = std::fmaxf(maxAlpha, s.getOpacity());
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
            float rgbMin = std::fminf(rgb1, 0.0f);
            float rgbMax = std::fmaxf(rgb99, 1.0f);

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
            float lnScaleMin = std::fminf(std::log(std::fmaxf(scale1, 1.0e-30f)), -12.0f);
            float lnScaleMax = std::fmaxf(std::log(std::fmaxf(scale99, 1.0e-30f)), 9.0f);

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
                    encoding.sh1Max = std::fmaxf(std::fmaxf(std::fabs(sh5), std::fabs(sh95)), 1.0f);
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
                    encoding.sh2Max = std::fmaxf(std::fmaxf(std::fabs(sh5), std::fabs(sh95)), 1.0f);
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
                    encoding.sh3Max = std::fmaxf(std::fmaxf(std::fabs(sh5), std::fabs(sh95)), 1.0f);
                }

                shEncoding = "S8";
            }
        }

        // resolve_sh_label_encoding: no clusters -> stays Auto
    }

    json encodingStateJson() const {
        json obj = json::object();
        obj["alpha"] = alphaEncoding;
        obj["center"] = centerEncoding;
        if (!hasEncoding){
            obj["encoding"] = nullptr;
        }else{
            json enc = json::object();
            enc["lnScaleMax"] = encoding.lnScaleMax;
            enc["lnScaleMin"] = encoding.lnScaleMin;
            enc["lodOpacity"] = encoding.lodOpacity;
            enc["rgbMax"] = encoding.rgbMax;
            enc["rgbMin"] = encoding.rgbMin;
            enc["sh1Max"] = encoding.sh1Max;
            enc["sh2Max"] = encoding.sh2Max;
            enc["sh3Max"] = encoding.sh3Max;
            obj["encoding"] = std::move(enc);
        }
        obj["orientation"] = orientationEncoding;
        obj["rgb"] = rgbEncoding;
        obj["scales"] = scalesEncoding;
        obj["sh"] = shEncoding;
        obj["sh_label"] = shLabelEncoding;
        return obj;
    }

    json setSplatEncodingJson(bool lodOpacity) const {
        json enc = json::object();
        enc["rgbMin"] = encoding.rgbMin;
        enc["rgbMax"] = encoding.rgbMax;
        enc["lnScaleMin"] = encoding.lnScaleMin;
        enc["lnScaleMax"] = encoding.lnScaleMax;
        enc["sh1Max"] = encoding.sh1Max;
        enc["sh2Max"] = encoding.sh2Max;
        enc["sh3Max"] = encoding.sh3Max;
        enc["lodOpacity"] = lodOpacity;
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

        json meta = json::object();
        meta["version"] = 1;
        meta["base"] = base;
        meta["count"] = count;
        meta["payloadBytes"] = payloadBytes;
        meta["maxSh"] = chunkMaxSh;
        if (splats.hasLodTree()) meta["lodTree"] = true;
        if (hasEncoding){
            meta["splatEncoding"] = setSplatEncodingJson(splats.hasLodTree());
        }
        json propsJson = json::array();
        for (const EncodedProp &p : props){
            json pj = json::object();
            pj["offset"] = p.offset;
            pj["bytes"] = p.bytes;
            pj["property"] = p.name;
            pj["encoding"] = propEncodingName(p.encoding);
            pj["compression"] = "gz";
            if (p.hasMinMax){
                pj["min"] = p.min;
                pj["max"] = p.max;
            }
            propsJson.push_back(std::move(pj));
        }
        meta["properties"] = std::move(propsJson);

        std::string metaStr = meta.dump();

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

        json meta = json::object();
        meta["version"] = 1;
        meta["type"] = "gsplat";
        meta["count"] = numSplats;
        meta["maxSh"] = fileMaxSh;
        if (splats.hasLodTree()) meta["lodTree"] = true;
        meta["chunkSize"] = 65536;
        meta["allChunkBytes"] = allChunkBytes;
        json chunksJson = json::array();
        for (const auto &cr : chunkRanges){
            json cj = json::object();
            cj["offset"] = cr.first;
            cj["bytes"] = cr.second;
            chunksJson.push_back(std::move(cj));
        }
        meta["chunks"] = std::move(chunksJson);
        if (hasEncoding){
            meta["splatEncoding"] = setSplatEncodingJson(splats.hasLodTree());
        }
        if (!comment.empty()){
            meta["comment"] = comment;
        }

        std::string metaStr = meta.dump(2);
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

    if (!runPipeline(data, splats, result)){
        return false;
    }

    RadEncoder encoder(splats);

    // nlohmann::json objects keep keys sorted alphabetically
    nlohmann::json description;
    description["input_splat_count"] = result.inputSplatCount;
    description["input_sh_degree"] = result.inputShDegree;
    if (result.removedAny){
        description["empty_splat_count"] = result.emptySplatCount;
        description["initial_splat_count"] = result.initialSplatCount;
    }
    description["method"] = "BhattLod { lod_base: 1.75 }";
    description["lod_duration"] = result.lodDuration;
    description["final_splat_count"] = result.finalSplatCount;
    description["chunk_duration"] = result.chunkDuration;
    description["max_sh_degree"] = result.maxShDegree;

    description["input_encoding"] = encoder.encodingStateJson();

    encoder.resolveEncoding();
    description["resolved_encoding"] = encoder.encodingStateJson();

    encoder.comment = description.dump(2);

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

