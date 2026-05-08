#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>

#include <immintrin.h>
#include "core_types.h"

//Helper for tracking which node we are filling
uint32_t nodes_used = 1;

inline void translate_model(EngineState& engine, float tx, float ty, float tz) {
    for (uint32_t i = 0; i < engine.total_triangles; ++i) {
        Triangle& tri = engine.triangles[i];
        
        // Move the base vertex in 3D space
        tri.v0_x += tx;
        tri.v0_y += ty;
        tri.v0_z += tz;

        // Note: Edges (e1, e2) and Normals do NOT change during translation
        // because they represent the shape's orientation, not its position.
    }
}

inline void rotate_model_x(EngineState& engine, float angle_degrees) {
    float rad = angle_degrees * (3.14159265f / 180.0f);
    float cos_a = std::cos(rad);
    float sin_a = std::sin(rad);

    for (uint32_t i = 0; i < engine.total_triangles; ++i) {
        Triangle& tri = engine.triangles[i];

        // Helper lambda to rotate Y and Z around the X-axis
        auto rot = [&](float& y, float& z) {
            float old_y = y;
            y = old_y * cos_a - z * sin_a;
            z = old_y * sin_a + z * cos_a;
        };

        // 1. Rotate the base vertex
        rot(tri.v0_y, tri.v0_z);

        // 2. Rotate the edges
        rot(tri.e1_y, tri.e1_z);
        rot(tri.e2_y, tri.e2_z);

        // 3. Rotate the pre-calculated normal
        rot(tri.norm_y, tri.norm_z);
    }
}

inline void rotate_model_y(EngineState& engine, float angle_degrees) {
    float rad = angle_degrees * (3.14159265f / 180.0f);
    float cos_a = std::cos(rad);
    float sin_a = std::sin(rad);

    for (uint32_t i = 0; i < engine.total_triangles; ++i) {
        Triangle& tri = engine.triangles[i];

        // Helper lambda to rotate X and Z around the Y-axis
        auto rot = [&](float& x, float& z) {
            float old_x = x;
            x = old_x * cos_a + z * sin_a;
            z = -old_x * sin_a + z * cos_a;
        };

        // 1. Rotate the base vertex
        rot(tri.v0_x, tri.v0_z);

        // 2. Rotate the edges
        rot(tri.e1_x, tri.e1_z);
        rot(tri.e2_x, tri.e2_z);

        // 3. Rotate the pre-calculated normal
        rot(tri.norm_x, tri.norm_z);
    }
}

inline void rotate_model_z(EngineState& engine, float angle_degrees) {
    float rad = angle_degrees * (3.14159265f / 180.0f);
    float cos_a = std::cos(rad);
    float sin_a = std::sin(rad);

    for (uint32_t i = 0; i < engine.total_triangles; ++i) {
        Triangle& tri = engine.triangles[i];

        // Helper lambda to rotate X and Y around the Z-axis
        auto rot = [&](float& x, float& y) {
            float old_x = x;
            x = old_x * cos_a - y * sin_a;
            y = old_x * sin_a + y * cos_a;
        };

        // 1. Rotate the base vertex
        rot(tri.v0_x, tri.v0_y);

        // 2. Rotate the edges
        rot(tri.e1_x, tri.e1_y);
        rot(tri.e2_x, tri.e2_y);

        // 3. Rotate the pre-calculated normal
        rot(tri.norm_x, tri.norm_y);
    }
}

void update_node_bounds(uint32_t node_idx, EngineState& engine, uint32_t first, uint32_t count) {
    BVHNode& node = engine.bvh_nodes[node_idx];
    
    // Reset to "infinite" inverse bounds
    node.min_x = node.min_y = node.min_z = 1e30f;
    node.max_x = node.max_y = node.max_z = -1e30f;

    for (uint32_t i = 0; i < count; i++) {
        const Triangle& tri = engine.triangles[first + i];
        
        // Extract all three vertices
        float v0x = tri.v0_x, v0y = tri.v0_y, v0z = tri.v0_z;
        float v1x = tri.v0_x + tri.e1_x, v1y = tri.v0_y + tri.e1_y, v1z = tri.v0_z + tri.e1_z;
        float v2x = tri.v0_x + tri.e2_x, v2y = tri.v0_y + tri.e2_y, v2z = tri.v0_z + tri.e2_z;

        // Tighten the box around all vertices of this triangle
        node.min_x = std::min({node.min_x, v0x, v1x, v2x});
        node.max_x = std::max({node.max_x, v0x, v1x, v2x});
        
        node.min_y = std::min({node.min_y, v0y, v1y, v2y});
        node.max_y = std::max({node.max_y, v0y, v1y, v2y});
        
        node.min_z = std::min({node.min_z, v0z, v1z, v2z});
        node.max_z = std::max({node.max_z, v0z, v1z, v2z});
    }
}

void subdivide(uint32_t node_idx, EngineState& engine, uint32_t first, uint32_t count) {
    BVHNode& node = engine.bvh_nodes[node_idx];

    // 1. Calculate the bounding box for this node
    update_node_bounds(node_idx, engine, first, count);

    // 2. Base Case: If triangle count is small, this is a leaf node
    if (count <= 4) {
        node.left_first = first;
        node.triangle_count = count;
        return;
    }

    // 3. Internal Node: We need to split
    // Determine which axis to split on (choose the longest axis of the box)
    float extent_x = node.max_x - node.min_x;
    float extent_y = node.max_y - node.min_y;
    float extent_z = node.max_z - node.min_z;
    int axis = 0; // 0=x, 1=y, 2=z
    if (extent_y > extent_x) axis = 1;
    if (extent_z > (axis == 0 ? extent_x : extent_y)) axis = 2;

    float split_pos = (axis == 0) ? (node.min_x + extent_x * 0.5f) :
                      (axis == 1) ? (node.min_y + extent_y * 0.5f) :
                                    (node.min_z + extent_z * 0.5f);

    // 4. Partition triangles around the split plane using their centroids
    uint32_t i = first;
    uint32_t j = i + count - 1;
    while (i <= j) {
        const Triangle& tri = engine.triangles[i];
        float centroid = (axis == 0) ? (tri.v0_x + (tri.e1_x + tri.e2_x) * 0.333f) :
                         (axis == 1) ? (tri.v0_y + (tri.e1_y + tri.e2_y) * 0.333f) :
                                       (tri.v0_z + (tri.e1_z + tri.e2_z) * 0.333f);
        if (centroid < split_pos) {
            i++;
        } else {
            std::swap(engine.triangles[i], engine.triangles[j]);
            j--;
        }
    }

    // 5. Create Child Nodes
    uint32_t left_count = i - first;
    if (left_count == 0 || left_count == count) {
        // Fallback: If partitioning fails to split the group, just force a leaf
        node.left_first = first;
        node.triangle_count = count;
        return;
    }

    uint32_t left_child_idx = nodes_used++;
    nodes_used++; // Right child is always left_child_idx + 1
    
    node.left_first = left_child_idx;
    node.triangle_count = 0; // 0 means this is a branch, not a leaf

    subdivide(left_child_idx, engine, first, left_count);
    subdivide(left_child_idx + 1, engine, i, count - left_count);
}

// The new main entry point for BVH generation
inline void build_bvh(EngineState& engine) {
    nodes_used = 1; // Reset node counter (0 is root)
    subdivide(0, engine, 0, engine.total_triangles);
    std::cout << "BVH Built. Nodes used: " << nodes_used << "\n";
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
