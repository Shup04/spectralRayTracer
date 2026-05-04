#include <iostream>
#include <vector>
#include <cmath>
#include <immintrin.h>

#include "core_types.h"
#include "simd_math.h"
#include "wavefront_pipeline.h"
#include "display_sensor.h"

#include <cmath> // Needed for std::sqrt

void add_triangle(EngineState& engine, int index, 
                  float v0x, float v0y, float v0z,
                  float v1x, float v1y, float v1z,
                  float v2x, float v2y, float v2z) 
{
    Triangle& tri = engine.triangles[index];
    
    // 1. Set V0
    tri.v0_x = v0x; tri.v0_y = v0y; tri.v0_z = v0z;
    
    // 2. Calculate Edges
    tri.e1_x = v1x - v0x; tri.e1_y = v1y - v0y; tri.e1_z = v1z - v0z;
    tri.e2_x = v2x - v0x; tri.e2_y = v2y - v0y; tri.e2_z = v2z - v0z;

    // 3. Calculate True Surface Normal (Cross Product of e1 and e2)
    float nx = (tri.e1_y * tri.e2_z) - (tri.e1_z * tri.e2_y);
    float ny = (tri.e1_z * tri.e2_x) - (tri.e1_x * tri.e2_z);
    float nz = (tri.e1_x * tri.e2_y) - (tri.e1_y * tri.e2_x);

    // Normalize it
    float len = std::sqrt(nx*nx + ny*ny + nz*nz);
    tri.norm_x = nx / len; 
    tri.norm_y = ny / len; 
    tri.norm_z = nz / len;
    
    tri.material_id = 0;
}

int main() {
    std::cout << "Booting Wube-Tier Spectral Raytracer...\n";

    // 1. ALLOCATE THE GLOBAL ARENA 
    EngineState engine;
    engine.total_materials = 1;
    engine.total_bvh_nodes = 1;
    engine.total_triangles = 4; // We are building a 4-sided pyramid

    engine.materials = (Material*)_mm_malloc(sizeof(Material) * engine.total_materials, 64);
    engine.bvh_nodes = (BVHNode*)_mm_malloc(sizeof(BVHNode) * engine.total_bvh_nodes, 64);
    engine.triangles = (Triangle*)_mm_malloc(sizeof(Triangle) * engine.total_triangles, 64);

    // 2. HARDCODE A ROTATED PYRAMID
    // We are pulling the apex close to the camera, and pushing the base back.
    float apex_x =  0.0f, apex_y =  0.0f, apex_z = -3.0f; // Pointing right at us
    float p1_x   = -1.5f, p1_y   =  1.5f, p1_z   = -5.0f; // Base: Top-Left
    float p2_x   =  1.5f, p2_y   =  1.5f, p2_z   = -5.0f; // Base: Top-Right
    float p3_x   =  0.0f, p3_y   = -2.0f, p3_z   = -5.0f; // Base: Bottom-Center

    // The 3 faces meeting at the apex
    add_triangle(engine, 0, p1_x, p1_y, p1_z, apex_x, apex_y, apex_z, p2_x, p2_y, p2_z); // Top Face
    add_triangle(engine, 1, p3_x, p3_y, p3_z, apex_x, apex_y, apex_z, p1_x, p1_y, p1_z); // Left Face
    add_triangle(engine, 2, p2_x, p2_y, p2_z, apex_x, apex_y, apex_z, p3_x, p3_y, p3_z); // Right Face
    
    // The flat base covering the back
    add_triangle(engine, 3, p1_x, p1_y, p1_z, p2_x, p2_y, p2_z, p3_x, p3_y, p3_z); 

    // A single BVH Leaf Node that encapsulates the new coordinates
    BVHNode& root = engine.bvh_nodes[0];
    root.min_x = -1.6f; root.min_y = -2.1f; root.min_z = -5.1f;
    root.max_x =  1.6f; root.max_y =  1.6f; root.max_z = -2.9f;
    root.left_first = 0; 
    root.triangle_count = 4;

    // 3. SET UP THE DISPLAY SENSOR
    const int width = 800;
    const int height = 600;
    const float aspect_ratio = (float)width / (float)height;
    std::vector<Pixel> framebuffer(width * height);

    std::cout << "Firing " << width * height << " Primary Rays...\n";

    // 4. THE WAVEFRONT EXECUTION LOOP
    RayPacket packet;
    int ray_idx = 0;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            
            // Calculate scalar ray direction (Standard Pinhole Camera Math)
            float ndc_x = (2.0f * (x + 0.5f) / width - 1.0f) * aspect_ratio;
            float ndc_y = 1.0f - 2.0f * (y + 0.5f) / height;
            
            // Normalize the direction vector
            float len = std::sqrt(ndc_x*ndc_x + ndc_y*ndc_y + 1.0f);
            float dir_x = ndc_x / len;
            float dir_y = ndc_y / len;
            float dir_z = -1.0f / len; // Looking down the -Z axis

            // Pack this ray into the SoA RayPacket
            packet.origin_x[ray_idx] = 0.0f;
            packet.origin_y[ray_idx] = 0.0f;
            packet.origin_z[ray_idx] = 0.0f;

            packet.dir_x[ray_idx] = dir_x;
            packet.dir_y[ray_idx] = dir_y;
            packet.dir_z[ray_idx] = dir_z;

            // Pre-calculate inverse directions for the fast BVH slab test
            // Add a tiny epsilon to avoid Divide-By-Zero crashes
            packet.inv_dir_x[ray_idx] = 1.0f / (dir_x == 0.0f ? 1e-8f : dir_x);
            packet.inv_dir_y[ray_idx] = 1.0f / (dir_y == 0.0f ? 1e-8f : dir_y);
            packet.inv_dir_z[ray_idx] = 1.0f / (dir_z == 0.0f ? 1e-8f : dir_z);

            // Initialize hit tracking
            packet.closest_t[ray_idx] = 1e30f; // Infinity
            
            ray_idx++;

            // ONCE THE PACKET IS FULL (8 Rays), FIRE IT INTO THE ENGINE
            if (ray_idx == 8) {
                // Phase 3: Hardware-accelerated traversal
                traverse_bvh(packet, engine);

                // Phase 4: Unpack the hardware registers to the display buffer
                int pixel_base = (y * width + x) - 7;
                for (int i = 0; i < 8; ++i) {
                    if (packet.closest_t[i] > 1e29f) {
                        // Ray missed everything: Render Sky Color (Dark Grey)
                        framebuffer[pixel_base + i] = {30, 30, 30};
                    } else {
                        // Ray hit: Render the normal as a neon color
                        framebuffer[pixel_base + i] = {
                            normal_to_color(packet.normal_x[i]),
                            normal_to_color(packet.normal_y[i]),
                            normal_to_color(packet.normal_z[i])
                        };
                    }
                }
                
                // Reset the packet index for the next 8 pixels
                ray_idx = 0;
            }
        }
    }

    // 5. FLUSH TO DISK
    std::cout << "Simulation complete. Writing image.ppm...\n";
    write_ppm_image("image.ppm", width, height, framebuffer);

    // 6. FREE HARDWARE ARENAS
    _mm_free(engine.materials);
    _mm_free(engine.bvh_nodes);
    _mm_free(engine.triangles);

    std::cout << "Engine shut down cleanly.\n";
    return 0;
}
