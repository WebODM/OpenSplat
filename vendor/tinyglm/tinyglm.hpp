// tinyglm: a minimal subset of GLM (https://github.com/g-truc/glm)
// providing only the types and functions used by OpenSplat.
// Column-major, same semantics as GLM. Same license as GLM (see LICENSE).
#ifndef TINYGLM_HPP
#define TINYGLM_HPP

#if defined(__CUDACC__) || defined(__HIPCC__)
#define TINYGLM_HD __host__ __device__
#else
#define TINYGLM_HD
#endif

namespace glm{

struct vec2{
    float x, y;
    TINYGLM_HD vec2() : x(0.0f), y(0.0f) {}
    TINYGLM_HD vec2(float x, float y) : x(x), y(y) {}
    TINYGLM_HD float &operator[](int i){ return (&x)[i]; }
    TINYGLM_HD const float &operator[](int i) const { return (&x)[i]; }
};

struct vec3{
    float x, y, z;
    TINYGLM_HD vec3() : x(0.0f), y(0.0f), z(0.0f) {}
    TINYGLM_HD vec3(float x, float y, float z) : x(x), y(y), z(z) {}
    TINYGLM_HD float &operator[](int i){ return (&x)[i]; }
    TINYGLM_HD const float &operator[](int i) const { return (&x)[i]; }
};

TINYGLM_HD inline vec2 operator+(const vec2 &a, const vec2 &b){ return vec2(a.x + b.x, a.y + b.y); }
TINYGLM_HD inline vec2 operator-(const vec2 &a){ return vec2(-a.x, -a.y); }
TINYGLM_HD inline vec2 operator*(float s, const vec2 &v){ return vec2(s * v.x, s * v.y); }
TINYGLM_HD inline vec2 operator*(const vec2 &v, float s){ return s * v; }

TINYGLM_HD inline vec3 operator+(const vec3 &a, const vec3 &b){ return vec3(a.x + b.x, a.y + b.y, a.z + b.z); }
TINYGLM_HD inline vec3 operator-(const vec3 &a, const vec3 &b){ return vec3(a.x - b.x, a.y - b.y, a.z - b.z); }
TINYGLM_HD inline vec3 operator-(const vec3 &a){ return vec3(-a.x, -a.y, -a.z); }
TINYGLM_HD inline vec3 operator*(float s, const vec3 &v){ return vec3(s * v.x, s * v.y, s * v.z); }
TINYGLM_HD inline vec3 operator*(const vec3 &v, float s){ return s * v; }

TINYGLM_HD inline float dot(const vec2 &a, const vec2 &b){ return a.x * b.x + a.y * b.y; }
TINYGLM_HD inline float dot(const vec3 &a, const vec3 &b){ return a.x * b.x + a.y * b.y + a.z * b.z; }

// Column-major 2x2 matrix; mat2(a, b, c, d) fills columns (a,b) and (c,d)
struct mat2{
    vec2 value[2];
    TINYGLM_HD mat2(){ value[0] = vec2(1.0f, 0.0f); value[1] = vec2(0.0f, 1.0f); }
    TINYGLM_HD mat2(float x0, float y0, float x1, float y1){
        value[0] = vec2(x0, y0);
        value[1] = vec2(x1, y1);
    }
    TINYGLM_HD vec2 &operator[](int i){ return value[i]; }
    TINYGLM_HD const vec2 &operator[](int i) const { return value[i]; }
};

TINYGLM_HD inline mat2 operator-(const mat2 &m){
    return mat2(-m[0].x, -m[0].y, -m[1].x, -m[1].y);
}
TINYGLM_HD inline vec2 operator*(const mat2 &m, const vec2 &v){
    return m[0] * v.x + m[1] * v.y;
}
TINYGLM_HD inline mat2 operator*(const mat2 &a, const mat2 &b){
    vec2 c0 = a * b[0];
    vec2 c1 = a * b[1];
    return mat2(c0.x, c0.y, c1.x, c1.y);
}

// Column-major 3x3 matrix; scalar constructor fills the diagonal
struct mat3{
    vec3 value[3];
    TINYGLM_HD mat3(){
        value[0] = vec3(1.0f, 0.0f, 0.0f);
        value[1] = vec3(0.0f, 1.0f, 0.0f);
        value[2] = vec3(0.0f, 0.0f, 1.0f);
    }
    TINYGLM_HD explicit mat3(float s){
        value[0] = vec3(s, 0.0f, 0.0f);
        value[1] = vec3(0.0f, s, 0.0f);
        value[2] = vec3(0.0f, 0.0f, s);
    }
    TINYGLM_HD mat3(float x0, float y0, float z0,
                    float x1, float y1, float z1,
                    float x2, float y2, float z2){
        value[0] = vec3(x0, y0, z0);
        value[1] = vec3(x1, y1, z1);
        value[2] = vec3(x2, y2, z2);
    }
    TINYGLM_HD vec3 &operator[](int i){ return value[i]; }
    TINYGLM_HD const vec3 &operator[](int i) const { return value[i]; }
};

TINYGLM_HD inline vec3 operator*(const mat3 &m, const vec3 &v){
    return m[0] * v.x + m[1] * v.y + m[2] * v.z;
}
TINYGLM_HD inline mat3 operator*(const mat3 &a, const mat3 &b){
    vec3 c0 = a * b[0];
    vec3 c1 = a * b[1];
    vec3 c2 = a * b[2];
    return mat3(c0.x, c0.y, c0.z, c1.x, c1.y, c1.z, c2.x, c2.y, c2.z);
}
TINYGLM_HD inline mat3 operator*(float s, const mat3 &m){
    return mat3(s * m[0].x, s * m[0].y, s * m[0].z,
                s * m[1].x, s * m[1].y, s * m[1].z,
                s * m[2].x, s * m[2].y, s * m[2].z);
}
TINYGLM_HD inline mat3 operator*(const mat3 &m, float s){ return s * m; }
TINYGLM_HD inline mat3 operator+(const mat3 &a, const mat3 &b){
    return mat3(a[0].x + b[0].x, a[0].y + b[0].y, a[0].z + b[0].z,
                a[1].x + b[1].x, a[1].y + b[1].y, a[1].z + b[1].z,
                a[2].x + b[2].x, a[2].y + b[2].y, a[2].z + b[2].z);
}

TINYGLM_HD inline mat2 transpose(const mat2 &m){
    return mat2(m[0].x, m[1].x, m[0].y, m[1].y);
}
TINYGLM_HD inline mat3 transpose(const mat3 &m){
    return mat3(m[0].x, m[1].x, m[2].x,
                m[0].y, m[1].y, m[2].y,
                m[0].z, m[1].z, m[2].z);
}

}

#endif
