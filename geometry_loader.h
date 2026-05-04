#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <immintrin.h>
#include "core_types.h"

inline void build_stub_bvh(EngineState& engine) {
    // We only have one node: the root
    BVHNode& root = engine.bvh_nodes[0];
    
    float min_x = 1e30f, min_y = 1e30f, min_z = 1e30f;
    float max_x = -1e30f, max_y = -1e30f, max_z = -1e30f;

    // Scan every triangle to find the outer limits of the model
    for (uint32_t i = 0; i < engine.total_triangles; ++i) {
        const Triangle& tri = engine.triangles[i];
        
        // Check V0
        min_x = std::min(min_x, tri.v0_x); max_x = std::max(max_x, tri.v0_x);
        min_y = std::min(min_y, tri.v0_y); max_y = std::max(max_y, tri.v0_y);
        min_z = std::min(min_z, tri.v0_z); max_z = std::max(max_z, tri.v0_z);

        // Check V1 (V0 + E1)
        float v1x = tri.v0_x + tri.e1_x;
        float v1y = tri.v0_y + tri.e1_y;
        float v1z = tri.v0_z + tri.e1_z;
        min_x = std::min(min_x, v1x); max_x = std::max(max_x, v1x);
        min_y = std::min(min_y, v1y); max_y = std::max(max_y, v1y);
        min_z = std::min(min_z, v1z); max_z = std::max(max_z, v1z);

        // Check V2 (V0 + E2)
        float v2x = tri.v0_x + tri.e2_x;
        float v2y = tri.v0_y + tri.e2_y;
        float v2z = tri.v0_z + tri.e2_z;
        min_x = std::min(min_x, v2x); max_x = std::max(max_x, v2x);
        min_y = std::min(min_y, v2y); max_y = std::max(max_y, v2y);
        min_z = std::min(min_z, v2z); max_z = std::max(max_z, v2z);
    }

    // Set the root node to perfectly fit the model
    root.min_x = min_x - 0.01f; root.min_y = min_y - 0.01f; root.min_z = min_z - 0.01f;
    root.max_x = max_x + 0.01f; root.max_y = max_y + 0.01f; root.max_z = max_z + 0.01f;
    
    root.left_first = 0; // Point to triangle index 0
    root.triangle_count = engine.total_triangles; // Put EVERY triangle in this one box
}

inline bool load_binary_stl(const std::string& filepath, EngineState& engine) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        std::cerr << "FAILED: Could not open " << filepath << "\n";
        return false;
    }

    // skip the header, its 80 bytes and useless
    file.seekg(80);

    // Read the total triangle count
    uint32_t num_tris = 0;
    file.read(reinterpret_cast<char*>(&num_tris), sizeof(uint32_t));
    std::cout << "Loading STL with " << num_tris << " triangles...\n";

    // Allocate the Global Arenas
    engine.total_triangles = num_tris;
    engine.triangles = (Triangle*)_mm_malloc(sizeof(Triangle) * num_tris, 64);
    
    // For a binary tree, the maximum number of nodes is roughly 2N - 1
    engine.total_bvh_nodes = num_tris * 2;
    engine.bvh_nodes = (BVHNode*)_mm_malloc(sizeof(BVHNode) * engine.total_bvh_nodes, 64);
    
    // (You'll still need to malloc your materials separately later)

    // each tri is 50 bytes in an stl
    // This just organizes those 50 bytes into our padded format.
    for (uint32_t i = 0; i < num_tris; ++i) {
        float normal[3], v0[3], v1[3], v2[3];
        uint16_t attribute_byte_count;

        // Read the raw bytes
        file.read(reinterpret_cast<char*>(normal), 12);
        file.read(reinterpret_cast<char*>(v0), 12);
        file.read(reinterpret_cast<char*>(v1), 12);
        file.read(reinterpret_cast<char*>(v2), 12);
        file.read(reinterpret_cast<char*>(&attribute_byte_count), 2);

        // Map to our DOD Triangle struct
        Triangle& tri = engine.triangles[i];
        
        tri.v0_x = v0[0]; tri.v0_y = v0[1]; tri.v0_z = v0[2];
        
        // Pre-calculate edges to save hot-loop CPU cycles
        tri.e1_x = v1[0] - v0[0]; tri.e1_y = v1[1] - v0[1]; tri.e1_z = v1[2] - v0[2];
        tri.e2_x = v2[0] - v0[0]; tri.e2_y = v2[1] - v0[1]; tri.e2_z = v2[2] - v0[2];

        // Store the pre-calculated normal
        tri.norm_x = normal[0]; tri.norm_y = normal[1]; tri.norm_z = normal[2];
        tri.material_id = 0; 
    }

    std::cout << "Successfully loaded geometry into L1-aligned arena.\n";
    return true;
}
