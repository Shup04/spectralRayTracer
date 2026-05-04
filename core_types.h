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

// 64 byte
struct alignas(64) Triangle {
    // 16 byte: vertex
    float v0_x, v0_y, v0_z;
    uint32_t material_id;  

    // 16 byte: edge1
    float e1_x, e1_y, e1_z;
    float padding1;        // empty space 

    // 16 byte: edge2
    float e2_x, e2_y, e2_z;
    float padding2;        // empty space

    // 16 byte
    float norm_x, norm_y, norm_z; // Pre-calculated during .stl loading
    float padding3;        // empty stace
};

// the only dynamically allocated object at startup
// contains raw pointers to our alignes memory blocks
struct EngineState {
    BVHNode* bvh_nodes;
    Material* materials;
    Triangle* triangles;

    uint32_t total_bvh_nodes;
    uint32_t total_materials;
    uint32_t total_triangles;
};
