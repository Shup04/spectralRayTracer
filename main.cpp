#include <iostream>
#include <vector>
#include <cmath>
#include <immintrin.h>

#include "core_types.h"
#include "simd_math.h"
#include "wavefront_pipeline.h"
#include "display_sensor.h"

int main() {
    std::cout << "Booting Wube-Tier Spectral Raytracer...\n";

    // 1. ALLOCATE THE GLOBAL ARENA (Strictly Aligned)
    EngineState engine;
    engine.total_materials = 1;
    engine.total_bvh_nodes = 1;
    engine.total_triangles = 1; // Assuming you added this to EngineState in core_types!

    engine.materials = (Material*)_mm_malloc(sizeof(Material) * engine.total_materials, 64);
    engine.bvh_nodes = (BVHNode*)_mm_malloc(sizeof(BVHNode) * engine.total_bvh_nodes, 64);
    
    // We need an array for the triangles. 
    // Make sure 'Triangle* triangles;' is in your EngineState struct!
    engine.triangles = (Triangle*)_mm_malloc(sizeof(Triangle) * engine.total_triangles, 64);

    // 2. HARDCODE A TEST SCENE
    // A single giant triangle floating directly in front of the camera at Z = -5.0
    Triangle& tri = engine.triangles[0];
    tri.v0_x = -2.0f; tri.v0_y = -2.0f; tri.v0_z = -5.0f;
    
    // Vertex 1 is at (2, -2, -5). Edge 1 = V1 - V0
    tri.e1_x =  4.0f; tri.e1_y =  0.0f; tri.e1_z =  0.0f;
    
    // Vertex 2 is at (0, 2, -5). Edge 2 = V2 - V0
    tri.e2_x =  2.0f; tri.e2_y =  4.0f; tri.e2_z =  0.0f;
    
    // The normal faces directly back at the camera (+Z)
    tri.norm_x = 0.0f; tri.norm_y = 0.0f; tri.norm_z = 1.0f;
    tri.material_id = 0;

    // A single BVH Leaf Node that perfectly encapsulates the triangle
    BVHNode& root = engine.bvh_nodes[0];
    root.min_x = -2.1f; root.min_y = -2.1f; root.min_z = -5.1f;
    root.max_x =  2.1f; root.max_y =  2.1f; root.max_z = -4.9f;
    // Top bit set to 1 to indicate Leaf Node, pure data is the triangle index (0)
    root.left_first = 0; 
    root.triangle_count = 1;

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
