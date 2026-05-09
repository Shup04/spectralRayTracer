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

SIMDRand rng;

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

    rotate_model_y(engine, 0.0f);
    build_bvh(engine);

    // 2. MANUALLY ALLOCATE ONLY THE MATERIALS
    engine.total_materials = 1;
    engine.materials = (Material*)_mm_malloc(sizeof(Material) * engine.total_materials, 64);

    // Setup sky
    const float sky_spectrum[8] = { 1.0f, 0.9f, 0.8f, 0.7f, 0.6f, 0.5f, 0.5f, 0.5f };
    float tmp_sun_dx = 0.45f;
    float tmp_sun_dy = 0.75f;
    float tmp_sun_dz = 0.50f;

    float tmp_sun_len = std::sqrt(
        tmp_sun_dx * tmp_sun_dx +
        tmp_sun_dy * tmp_sun_dy +
        tmp_sun_dz * tmp_sun_dz
    );

    const float SUN_DX = tmp_sun_dx / tmp_sun_len;
    const float SUN_DY = tmp_sun_dy / tmp_sun_len;
    const float SUN_DZ = tmp_sun_dz / tmp_sun_len;

    const float COPPER_DIFF_R = 0.75f;
    const float COPPER_DIFF_G = 0.38f;
    const float COPPER_DIFF_B = 0.14f;

    const float COPPER_SPEC_R = 1.50f;
    const float COPPER_SPEC_G = 0.85f;
    const float COPPER_SPEC_B = 0.35f;

    const float METAL_ROUGHNESS = 0.18f;
    const float RAY_EPSILON = 0.01f;

    // 3. SET UP DISPLAY
    const int width = 800;
    const int height = 600;
    const float aspect_ratio = (float)width / (float)height;
    std::vector<Pixel> framebuffer(width * height);

    // accumulator buffers per frame
    std::vector<float> accum_r(width * height, 0.0f);
    std::vector<float> accum_g(width * height, 0.0f);
    std::vector<float> accum_b(width * height, 0.0f);
    const int MAX_SAMPLES = 128;

    std::cout << "Firing rays at " << engine.total_triangles << " triangles...\n";



    RayPacket packet;
    int ray_idx = 0;
    for (int sample = 0; sample < MAX_SAMPLES; ++sample) {
      std::cout << "Rendering Sample " << sample + 1 << "/" << MAX_SAMPLES << "\r" << std::flush;
      for (int y = 0; y < height; ++y) {
          for (int x = 0; x < width; ++x) {

              float jitter_x = rng.next_float_scalar() - 0.5f;
              float jitter_y = rng.next_float_scalar() - 0.5f;

              float ndc_x = (2.0f * (x + 0.5f + jitter_x) / width - 1.0f) * aspect_ratio;
              float ndc_y = 1.0f - 2.0f * (y + 0.5f + jitter_y) / height;
              
              float len = std::sqrt(ndc_x*ndc_x + ndc_y*ndc_y + 1.0f);
              
              // CAMERA TWEAK: 
              // We set origin_z to 5.0f to move the camera BACK so we can see the model at the origin.
              packet.origin_x[ray_idx] = 0.0f;
              packet.origin_y[ray_idx] = 30.0f;
              packet.origin_z[ray_idx] = 50.0f; 

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
                  rng.init((uint32_t)(y * width + x + 1) * 914237u);
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
                      
                      // BVH Traversal
                      uint8_t alive_bits = 0;

                      for (int lane = 0; lane < 8; ++lane) {
                          if (active[lane]) {
                              alive_bits |= (1u << lane);
                          }
                      }

                      if (alive_bits == 0) {
                          break;
                      }

                      __m256 alive_mask = simd_mask_from_bits(alive_bits);

                      traverse_bvh(packet, engine, alive_mask);
                      // end BVH Traversal

                      for (int i = 0; i < 8; ++i) {
                          if (!active[i]) continue; // Skip dead rays

                          if (packet.closest_t[i] > 1e29f) {
                              // --- HIT THE SKY ---
                              
                              // 1. Base Sky Light (Ambient)
                              float r_light = sky_spectrum[6] * 0.5f;
                              float g_light = sky_spectrum[3] * 0.5f;
                              float b_light = sky_spectrum[0] * 0.5f;

                              // 2. The sun
                              float dot =
                                  packet.dir_x[i] * SUN_DX +
                                  packet.dir_y[i] * SUN_DY +
                                  packet.dir_z[i] * SUN_DZ;

                              // If the ray is within the "disc" of the sun, blast it with energy
                              if (dot > 0.98f) { 
                                  r_light += 15.0f; // High intensity white/yellow light
                                  g_light += 14.0f;
                                  b_light += 12.0f;
                              }

                              // Multiply remaining ray throughput by the light energy
                              pixel_b[i] += packet.spectrum_b0[i] * b_light;
                              pixel_g[i] += packet.spectrum_b3[i] * g_light;
                              pixel_r[i] += packet.spectrum_b6[i] * r_light;
                              
                              active[i] = false; // Ray is done.
                          } else {
                            // --- HIT THE MODEL ---

                            // ----------------------------------------------------
                            // 1. SAVE THE INCOMING RAY AND FIND THE HIT POINT
                            // ----------------------------------------------------
                            // We need the old ray direction before we change it.
                            // closest_t tells us how far along the ray the triangle hit was.

                            float old_dx = packet.dir_x[i];
                            float old_dy = packet.dir_y[i];
                            float old_dz = packet.dir_z[i];

                            float t = packet.closest_t[i];

                            float hit_x = packet.origin_x[i] + old_dx * t;
                            float hit_y = packet.origin_y[i] + old_dy * t;
                            float hit_z = packet.origin_z[i] + old_dz * t;

                            // ----------------------------------------------------
                            // 2. GET THE SURFACE NORMAL AND FACE IT TOWARD THE RAY
                            // ----------------------------------------------------
                            // Some STL triangles may have normals facing inward.
                            // This makes the normal usable for lighting and reflection.

                            float surf_nx = packet.normal_x[i];
                            float surf_ny = packet.normal_y[i];
                            float surf_nz = packet.normal_z[i];

                            float facing =
                                surf_nx * old_dx +
                                surf_ny * old_dy +
                                surf_nz * old_dz;

                            if (facing > 0.0f) {
                                surf_nx = -surf_nx;
                                surf_ny = -surf_ny;
                                surf_nz = -surf_nz;
                            }

                            // ----------------------------------------------------
                            // 3. SET UP THE SUN DIRECTION
                            // ----------------------------------------------------
                            // This is direct lighting. It gives the model readable shape.
                            // Positive Z means the sun is partly coming from the camera side.

                            float sun_dx = 0.45f;
                            float sun_dy = 0.75f;
                            float sun_dz = 0.50f;

                            float sun_len = std::sqrt(
                                sun_dx * sun_dx +
                                sun_dy * sun_dy +
                                sun_dz * sun_dz
                            );

                            sun_dx /= sun_len;
                            sun_dy /= sun_len;
                            sun_dz /= sun_len;

                            // ----------------------------------------------------
                            // 4. WEAK DIFFUSE LIGHTING
                            // ----------------------------------------------------
                            // This is intentionally weak.
                            // Strong diffuse orange is what made it look like plastic.

                            float ndotl = std::max(
                                0.0f,
                                surf_nx * SUN_DX +
                                surf_ny * SUN_DY +
                                surf_nz * SUN_DZ
                            );

                            float diffuse = 0.03f + 0.18f * ndotl;

                            // ----------------------------------------------------
                            // 5. COPPER-TINTED SPECULAR HIGHLIGHT
                            // ----------------------------------------------------
                            // This gives it the shiny metal read.
                            // View direction points back toward the camera.

                            // Ray directions should already be normalized, so view = -old_dir.
                            // No need to normalize view every hit.
                            float view_dx = -old_dx;
                            float view_dy = -old_dy;
                            float view_dz = -old_dz;

                            // Blinn-Phong half vector.
                            float half_x = SUN_DX + view_dx;
                            float half_y = SUN_DY + view_dy;
                            float half_z = SUN_DZ + view_dz;

                            float half_len = std::sqrt(
                                half_x * half_x +
                                half_y * half_y +
                                half_z * half_z
                            );

                            half_x /= half_len;
                            half_y /= half_len;
                            half_z /= half_len;

                            float ndoth = std::max(
                                0.0f,
                                surf_nx * half_x +
                                surf_ny * half_y +
                                surf_nz * half_z
                            );

                            // Faster than std::pow(ndoth, 80.0f).
                            // This gives roughly a sharp metallic highlight.
                            float s2  = ndoth * ndoth;
                            float s4  = s2 * s2;
                            float s8  = s4 * s4;
                            float s16 = s8 * s8;
                            float s32 = s16 * s16;
                            float s64 = s32 * s32;

                            float spec = s64;

                            // ----------------------------------------------------
                            // 6. ADD DIRECT COPPER LIGHT TO THIS PIXEL
                            // ----------------------------------------------------
                            // Copper reflects red/orange more than blue.
                            // The diffuse part is small.
                            // The specular part is stronger and copper-tinted.

                            float direct_r = diffuse * COPPER_DIFF_R + spec * COPPER_SPEC_R;
                            float direct_g = diffuse * COPPER_DIFF_G + spec * COPPER_SPEC_G;
                            float direct_b = diffuse * COPPER_DIFF_B + spec * COPPER_SPEC_B;

                            pixel_r[i] += packet.spectrum_b6[i] * direct_r;
                            pixel_g[i] += packet.spectrum_b3[i] * direct_g;
                            pixel_b[i] += packet.spectrum_b0[i] * direct_b;

                            // ----------------------------------------------------
                            // 7. APPLY COPPER ABSORPTION TO THE BOUNCED RAY
                            // ----------------------------------------------------
                            // This affects the next bounce, not just the first visible hit.
                            // Blue dies quickly, red survives longer.

                            packet.spectrum_b0[i] *= 0.10f;
                            packet.spectrum_b1[i] *= 0.18f;
                            packet.spectrum_b2[i] *= 0.28f;
                            packet.spectrum_b3[i] *= 0.48f;
                            packet.spectrum_b4[i] *= 0.62f;
                            packet.spectrum_b5[i] *= 0.78f;
                            packet.spectrum_b6[i] *= 0.92f;
                            packet.spectrum_b7[i] *= 0.96f;

                            // If the ray has basically no energy left, stop bouncing it.
                            float energy =
                                packet.spectrum_b0[i] +
                                packet.spectrum_b3[i] +
                                packet.spectrum_b6[i];

                            if (energy < 0.02f) {
                                active[i] = false;
                                continue;
                            }

                            // ----------------------------------------------------
                            // 8. MAKE A REFLECTION DIRECTION
                            // ----------------------------------------------------
                            // This is the main "metal" part.
                            // Instead of bouncing randomly from the normal like matte plastic,
                            // we reflect the incoming ray around the surface normal.

                            float dot_in_n =
                                old_dx * surf_nx +
                                old_dy * surf_ny +
                                old_dz * surf_nz;

                            float refl_x = old_dx - 2.0f * dot_in_n * surf_nx;
                            float refl_y = old_dy - 2.0f * dot_in_n * surf_ny;
                            float refl_z = old_dz - 2.0f * dot_in_n * surf_nz;

                            // ----------------------------------------------------
                            // 9. ADD ROUGHNESS TO THE REFLECTION
                            // ----------------------------------------------------
                            float rx = 2.0f * rng.next_float_scalar() - 1.0f;
                            float ry = 2.0f * rng.next_float_scalar() - 1.0f;
                            float rz = 2.0f * rng.next_float_scalar() - 1.0f;

                            float new_dx = refl_x + METAL_ROUGHNESS * rx;
                            float new_dy = refl_y + METAL_ROUGHNESS * ry;
                            float new_dz = refl_z + METAL_ROUGHNESS * rz;

                            float new_len = std::sqrt(
                                new_dx * new_dx +
                                new_dy * new_dy +
                                new_dz * new_dz
                            );

                            packet.dir_x[i] = new_dx / new_len;
                            packet.dir_y[i] = new_dy / new_len;
                            packet.dir_z[i] = new_dz / new_len;

                            // ----------------------------------------------------
                            // 10. MOVE THE RAY TO THE HIT POINT
                            // ----------------------------------------------------
                            // This is important.
                            // The next bounce must start at the surface, not back at the camera.

                            packet.origin_x[i] = hit_x + surf_nx * RAY_EPSILON;
                            packet.origin_y[i] = hit_y + surf_ny * RAY_EPSILON;
                            packet.origin_z[i] = hit_z + surf_nz * RAY_EPSILON;

                            // Update inverse direction for the BVH traversal.
                            packet.inv_dir_x[i] = 1.0f / (packet.dir_x[i] == 0.0f ? 1e-8f : packet.dir_x[i]);
                            packet.inv_dir_y[i] = 1.0f / (packet.dir_y[i] == 0.0f ? 1e-8f : packet.dir_y[i]);
                            packet.inv_dir_z[i] = 1.0f / (packet.dir_z[i] == 0.0f ? 1e-8f : packet.dir_z[i]);

                            // Reset closest hit distance for the next bounce.
                            packet.closest_t[i] = 1e30f;

                          }
                      }
                  }

                  // 3. WRITE TO FRAMEBUFFER
                  int pixel_base = (y * width + x) - 7;
                  for (int i = 0; i < 8; ++i) {
                      int p_idx = pixel_base + i;
                      
                      // Accumulate raw energy
                      accum_r[p_idx] += pixel_r[i];
                      accum_g[p_idx] += pixel_g[i];
                      accum_b[p_idx] += pixel_b[i];

                      // Average it based on how many samples we've done so far
                      float avg_r = accum_r[p_idx] / (sample + 1);
                      float avg_g = accum_g[p_idx] / (sample + 1);
                      float avg_b = accum_b[p_idx] / (sample + 1);

                      // Gamma correction (optional, but makes lighting look more realistic)
                      avg_r = std::sqrt(avg_r);
                      avg_g = std::sqrt(avg_g);
                      avg_b = std::sqrt(avg_b);

                      // Clamp and cast to byte
                      int r = std::min(255, std::max(0, (int)(avg_r * 255.0f)));
                      int g = std::min(255, std::max(0, (int)(avg_g * 255.0f)));
                      int b = std::min(255, std::max(0, (int)(avg_b * 255.0f)));
                      framebuffer[p_idx] = {(uint8_t)r, (uint8_t)g, (uint8_t)b};
                  }
                  ray_idx = 0;
              }
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
