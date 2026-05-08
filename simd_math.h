#pragma once
#include <immintrin.h>
#include <cstdint>

#include "core_types.h"

// 8-way Parallel Random Number Generator
struct SIMDRand {
    __m256i state; // 8 independent 32-bit seeds

    // Xorshift32: Very fast, perfectly suited for SIMD
    inline __m256 next_float() {
        state = _mm256_xor_si256(state, _mm256_slli_epi32(state, 13));
        state = _mm256_xor_si256(state, _mm256_srli_epi32(state, 17));
        state = _mm256_xor_si256(state, _mm256_slli_epi32(state, 5));
        
        // Convert to float between 0.0 and 1.0
        __m256i mask = _mm256_set1_epi32(0x7FFFFFFF);
        __m256i masked = _mm256_and_si256(state, mask);
        return _mm256_mul_ps(_mm256_cvtepi32_ps(masked), _mm256_set1_ps(1.0f / 2147483647.0f));
    }
};

// Memory Operations (Load/Store)

// Loads exactly 32 bytes (8 floats) from memory into a CPU register.
// WARNING: The memory address MUST be aligned to 32 bytes (alignas(32)).
inline __m256 simd_load(const float* aligned_mem) {
    return _mm256_load_ps(aligned_mem);
}

inline __m256i simd_load_int(const uint32_t* p) {
    return _mm256_load_si256(reinterpret_cast<const __m256i*>(p));
}

// Stores the contents of a 256-bit register back into 32 bytes of RAM.
inline void simd_store(float* aligned_mem, __m256 reg) {
    _mm256_store_ps(aligned_mem, reg);
}

inline void simd_store_int(uint32_t* p, __m256i v) {
    _mm256_store_si256(reinterpret_cast<__m256i*>(p), v);
}

// Broadcasts a single float into all 8 slots of a register (e.g., [1.0, 1.0...])
inline __m256 simd_set1(float val) {
    return _mm256_set1_ps(val);
}

inline __m256i simd_set1_int(uint32_t x) {
    return _mm256_set1_epi32(static_cast<int>(x));
}

// Sets all 8 slots to 0.0f (Hardware optimization: usually executes via an XOR)
inline __m256 simd_zero() {
    return _mm256_setzero_ps();
}

inline __m256 simd_abs(__m256 x) {
    __m256 sign_mask = _mm256_set1_ps(-0.0f);
    return _mm256_andnot_ps(sign_mask, x);
}

// Core Math
inline __m256 simd_add(__m256 a, __m256 b) {
    return _mm256_add_ps(a, b);
}

inline __m256 simd_sub(__m256 a, __m256 b) {
    return _mm256_sub_ps(a, b);
}

inline __m256 simd_mul(__m256 a, __m256 b) {
    return _mm256_mul_ps(a, b);
}

inline __m256 simd_div(__m256 a, __m256 b) {
    return _mm256_div_ps(a, b);
}

// fused multiply and add: (a * b) + c in one clock cycle
// most used operation in raytracing so this is worth using
inline __m256 simd_fmadd(__m256 a, __m256 b, __m256 c) {
    return _mm256_fmadd_ps(a, b, c);
}


// Vector Math
//dot product
inline __m256 simd_dot3(__m256 x1, __m256 y1, __m256 z1, 
                        __m256 x2, __m256 y2, __m256 z2) 
{
    __m256 dot = simd_mul(x1, x2);
    dot = simd_fmadd(y1, y2, dot);
    dot = simd_fmadd(z1, z2, dot);
    return dot;
}

//cross product
inline void simd_cross3(__m256 ax, __m256 ay, __m256 az,
                        __m256 bx, __m256 by, __m256 bz,
                        __m256& out_x, __m256& out_y, __m256& out_z) 
{
    out_x = simd_sub(simd_mul(ay, bz), simd_mul(az, by));
    out_y = simd_sub(simd_mul(az, bx), simd_mul(ax, bz));
    out_z = simd_sub(simd_mul(ax, by), simd_mul(ay, bx));
}


// Branchless Logic (removing some if statements from the hot path)
// mask if a < b = 1 if new hit dist is closer than old current dist
inline __m256 simd_cmp_less(__m256 a, __m256 b) {
    return _mm256_cmp_ps(a, b, _CMP_LT_OQ);
}

inline __m256 simd_cmp_less_equal(__m256 a, __m256 b) {
    return _mm256_cmp_ps(a, b, _CMP_LE_OQ);
}

// a > b = 1
inline __m256 simd_cmp_greater(__m256 a, __m256 b) {
    return _mm256_cmp_ps(a, b, _CMP_GT_OQ);
}

inline __m256 simd_cmp_greater_equal(__m256 a, __m256 b) {
    return _mm256_cmp_ps(a, b, _CMP_GE_OQ);
}

// bitwise and to combine masks
inline __m256 simd_and(__m256 mask1, __m256 mask2) {
    return _mm256_and_ps(mask1, mask2);
}

inline __m256 simd_or(__m256 a, __m256 b) {
    return _mm256_or_ps(a, b);
}

inline __m256 simd_min(__m256 a, __m256 b) {
    return _mm256_min_ps(a, b);
}

inline __m256 simd_max(__m256 a, __m256 b) {
    return _mm256_max_ps(a, b);
}

// hardware blend
// apply mask to old and new, if 1 take new if 0 take old
inline __m256 simd_blend(__m256 old_val, __m256 new_val, __m256 mask) {
    return _mm256_blendv_ps(old_val, new_val, mask);
}

// integer blend for updating material ID arrays using int in register
inline __m256i simd_blend_int(__m256i old_val, __m256i new_val, __m256 mask) {
    // Note: blendv_epi8 blends per byte. As long as the mask is all 1s or all 0s 
    // per 32-bit float slot (which _mm256_cmp_ps guarantees), this works perfectly for uint32_t.
    return _mm256_blendv_epi8(old_val, new_val, _mm256_castps_si256(mask));
}
// takes the top bit from each float in a mask and turns those into an int
// basically mask to int for cpp usage.
inline int simd_movemask(__m256 mask) {
    return _mm256_movemask_ps(mask);
}
