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

    // Setup sky
    const float sky_spectrum[8] = { 1.0f, 0.9f, 0.8f, 0.7f, 0.6f, 0.5f, 0.5f, 0.5f };

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
            packet.origin_z[ray_idx] = 80.0f; 

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
                // 1. INITIALIZE THE RAYS
                bool active[8] = {true, true, true, true, true, true, true, true};
                
                // Final accumulated colors for these 8 pixels
                float pixel_r[8] = {0}, pixel_g[8] = {0}, pixel_b[8] = {0};

                // Initialize Throughput to 1.0 (Full Energy)
                for (int i = 0; i < 8; ++i) {
                    packet.spectrum_b0[i] = 1.0f; packet.spectrum_b1[i] = 1.0f;
                    packet.spectrum_b2[i] = 1.0f; packet.spectrum_b3[i] = 1.0f;
                    packet.spectrum_b4[i] = 1.0f; packet.spectrum_b5[i] = 1.0f;
                    packet.spectrum_b6[i] = 1.0f; packet.spectrum_b7[i] = 1.0f;
                }

                // 2. THE BOUNCE LOOP (Max 3 bounces for now)
                for (int bounce = 0; bounce < 3; ++bounce) {
                    
                    traverse_bvh(packet, engine);

                    for (int i = 0; i < 8; ++i) {
                        if (!active[i]) continue; // Skip dead rays

                        if (packet.closest_t[i] > 1e29f) {
                            // --- HIT THE SKY ---
                            // Multiply remaining throughput by the sky emission
                            // For visualization, let's map the 8 bands directly to RGB
                            // (Assuming b0-b2 = Blue, b3-b4 = Green, b5-b7 = Red)
                            pixel_b[i] += (packet.spectrum_b0[i] * sky_spectrum[0] + packet.spectrum_b1[i] * sky_spectrum[1]) * 0.5f;
                            pixel_g[i] += (packet.spectrum_b3[i] * sky_spectrum[3] + packet.spectrum_b4[i] * sky_spectrum[4]) * 0.5f;
                            pixel_r[i] += (packet.spectrum_b6[i] * sky_spectrum[6] + packet.spectrum_b7[i] * sky_spectrum[7]) * 0.5f;
                            
                            active[i] = false; // Ray is done.
                        } else {
                            // --- HIT THE MODEL ---
                            // 1. Material Absorption: COPPER (Absorb Blue, Reflect Red)
                            packet.spectrum_b0[i] *= 0.2f; // Kill UV/Blue
                            packet.spectrum_b1[i] *= 0.3f;
                            packet.spectrum_b2[i] *= 0.4f;
                            packet.spectrum_b3[i] *= 0.6f; // Moderate Green
                            packet.spectrum_b4[i] *= 0.7f;
                            packet.spectrum_b5[i] *= 0.8f;
                            packet.spectrum_b6[i] *= 0.9f; // High Red
                            packet.spectrum_b7[i] *= 0.95f; // High IR

                            // 2. Scatter (Basic hemisphere offset)
                            // We use a cheap pseudo-random offset based on pixel coords to scatter
                            float rx = ((x * 13 + y * 71 + bounce * 17) % 100) / 100.0f - 0.5f;
                            float ry = ((x * 37 + y * 19 + bounce * 23) % 100) / 100.0f - 0.5f;
                            float rz = ((x * 59 + y * 43 + bounce * 31) % 100) / 100.0f - 0.5f;

                            float nx = packet.normal_x[i] + rx;
                            float ny = packet.normal_y[i] + ry;
                            float nz = packet.normal_z[i] + rz;
                            float len = std::sqrt(nx*nx + ny*ny + nz*nz);

                            packet.dir_x[i] = nx / len;
                            packet.dir_y[i] = ny / len;
                            packet.dir_z[i] = nz / len;

                            // 3. Update Origin and Reset limits for next traverse
                            packet.origin_x[i] += packet.normal_x[i] * 0.001f;
                            packet.origin_y[i] += packet.normal_y[i] * 0.001f;
                            packet.origin_z[i] += packet.normal_z[i] * 0.001f;

                            packet.inv_dir_x[i] = 1.0f / (packet.dir_x[i] == 0.0f ? 1e-8f : packet.dir_x[i]);
                            packet.inv_dir_y[i] = 1.0f / (packet.dir_y[i] == 0.0f ? 1e-8f : packet.dir_y[i]);
                            packet.inv_dir_z[i] = 1.0f / (packet.dir_z[i] == 0.0f ? 1e-8f : packet.dir_z[i]);
                            packet.closest_t[i] = 1e30f;
                        }
                    }
                }

                // 3. WRITE TO FRAMEBUFFER
                int pixel_base = (y * width + x) - 7;
                for (int i = 0; i < 8; ++i) {
                    // Convert the 0.0 - 1.0 float to a 0-255 RGB byte
                    int r = std::min(255, std::max(0, (int)(pixel_r[i] * 255.0f)));
                    int g = std::min(255, std::max(0, (int)(pixel_g[i] * 255.0f)));
                    int b = std::min(255, std::max(0, (int)(pixel_b[i] * 255.0f)));
                    framebuffer[pixel_base + i] = {(uint8_t)r, (uint8_t)g, (uint8_t)b};
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
