#include <iostream>
#include <vector>
#include <cmath>
#include <immintrin.h>

#include "core_types.h"
#include "simd_math.h"
#include "wavefront_pipeline.h"
#include "display_sensor.h"
#include "geometry_loader.h"

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

    EngineState engine;
    
    // 1. LOAD AND BUILD (These handle their own mallocs)
    if (!load_binary_stl("orc.stl", engine)) return -1;
    rotate_model_x(engine, -90.0f);

    rotate_model_y(engine, 180.0f);
    build_bvh(engine);

    // 2. MANUALLY ALLOCATE ONLY THE MATERIALS
    engine.total_materials = 1;
    engine.materials = (Material*)_mm_malloc(sizeof(Material) * engine.total_materials, 64);

    // 3. SET UP DISPLAY
    const int width = 800;
    const int height = 600;
    const float aspect_ratio = (float)width / (float)height;
    std::vector<Pixel> framebuffer(width * height);

    std::cout << "Firing rays at " << engine.total_triangles << " triangles...\n";

    RayPacket packet;
    int ray_idx = 0;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float ndc_x = (2.0f * (x + 0.5f) / width - 1.0f) * aspect_ratio;
            float ndc_y = 1.0f - 2.0f * (y + 0.5f) / height;
            
            float len = std::sqrt(ndc_x*ndc_x + ndc_y*ndc_y + 1.0f);
            
            // CAMERA TWEAK: 
            // We set origin_z to 5.0f to move the camera BACK so we can see the model at the origin.
            packet.origin_x[ray_idx] = 0.0f;
            packet.origin_y[ray_idx] = 0.0f;
            packet.origin_z[ray_idx] = 150.0f; 

            packet.dir_x[ray_idx] = ndc_x / len;
            packet.dir_y[ray_idx] = ndc_y / len;
            packet.dir_z[ray_idx] = -1.0f / len; // Still looking towards -Z

            packet.inv_dir_x[ray_idx] = 1.0f / (packet.dir_x[ray_idx] == 0.0f ? 1e-8f : packet.dir_x[ray_idx]);
            packet.inv_dir_y[ray_idx] = 1.0f / (packet.dir_y[ray_idx] == 0.0f ? 1e-8f : packet.dir_y[ray_idx]);
            packet.inv_dir_z[ray_idx] = 1.0f / (packet.dir_z[ray_idx] == 0.0f ? 1e-8f : packet.dir_z[ray_idx]);

            packet.closest_t[ray_idx] = 1e30f;
            ray_idx++;

            // If packet has 8 rays, fully populated
            if (ray_idx == 8) {
                traverse_bvh(packet, engine);
                int pixel_base = (y * width + x) - 7;
                for (int i = 0; i < 8; ++i) {
                    if (packet.closest_t[i] > 1e29f) {
                        framebuffer[pixel_base + i] = {30, 30, 30};
                    } else {
                        framebuffer[pixel_base + i] = {
                            normal_to_color(packet.normal_x[i]),
                            normal_to_color(packet.normal_y[i]),
                            normal_to_color(packet.normal_z[i])
                        };
                    }
                }
                ray_idx = 0;
            }
        }
    }

    write_ppm_image("image.ppm", width, height, framebuffer);

    // CLEANUP
    _mm_free(engine.materials);
    _mm_free(engine.bvh_nodes);
    _mm_free(engine.triangles);

    return 0;
}
