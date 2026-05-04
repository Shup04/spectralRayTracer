#pragma once
#include <immintrin.h> // Intel AVX/AVX2 intrinsics
#include <cstdint>

// Material: 64 byte
struct alignas(32) Material {
  float reflectance[8]; // 32 byte
  float emission[8]; // 32 byte
};

// BVH: 32 byte
struct alignas(32) BVHNode {
  float min_x, min_y, min_z; // 12 byte
  uint32_t left_first; // index of left child, or first triangle 4 byte
  
  float max_x, max_y, max_z; // 12 byte
  uint32_t triangle_count; // if > 0, this is a leaf
};

// Processes 8 rays simultaneously in AVX2 registers.
// Allows common data to be bundled together when going to cache
struct alignas(32) RayPacket {
    // Origins (3 * 32 bytes)
    float origin_x[8]; 
    float origin_y[8];
    float origin_z[8];

    // Directions (3 * 32 bytes)
    float dir_x[8];
    float dir_y[8];
    float dir_z[8];

    float inv_dir_x[8];
    float inv_dir_y[8];
    float inv_dir_z[8];

    // wavelengths: 8 bands spanning the spectrum (8 * 32 bytes)
    float spectrum_b0[8]; 
    float spectrum_b1[8];
    float spectrum_b2[8];
    float spectrum_b3[8];
    float spectrum_b4[8];
    float spectrum_b5[8];
    float spectrum_b6[8];
    float spectrum_b7[8];

    // The Hit Record: Used for the branchless mask blend (5 * 32 bytes)
    float closest_t[8];          // Distance to closest hit
    float normal_x[8];           // Surface normal X for bouncing
    float normal_y[8];           // Surface normal Y
    float normal_z[8];           // Surface normal Z
    uint32_t hit_material_id[8]; // Material ID of the closest hit
};
