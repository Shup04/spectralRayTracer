#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <immintrin.h>
#include <iostream>
#include <thread>
#include <vector>

#include "core_types.h"
#include "display_sensor.h"
#include "geometry_loader.h"
#include "simd_math.h"
#include "wavefront_pipeline.h"

void add_triangle(EngineState &engine, int index, float v0x, float v0y,
                  float v0z, float v1x, float v1y, float v1z, float v2x,
                  float v2y, float v2z) {
    Triangle &tri = engine.triangles[index];

    // 1. Set V0
    tri.v0_x = v0x;
    tri.v0_y = v0y;
    tri.v0_z = v0z;

    // 2. Calculate Edges
    tri.e1_x = v1x - v0x;
    tri.e1_y = v1y - v0y;
    tri.e1_z = v1z - v0z;
    tri.e2_x = v2x - v0x;
    tri.e2_y = v2y - v0y;
    tri.e2_z = v2z - v0z;

    // 3. Calculate True Surface Normal (Cross Product of e1 and e2)
    float nx = (tri.e1_y * tri.e2_z) - (tri.e1_z * tri.e2_y);
    float ny = (tri.e1_z * tri.e2_x) - (tri.e1_x * tri.e2_z);
    float nz = (tri.e1_x * tri.e2_y) - (tri.e1_y * tri.e2_x);

    // Normalize it
    float len = std::sqrt(nx * nx + ny * ny + nz * nz);
    tri.norm_x = nx / len;
    tri.norm_y = ny / len;
    tri.norm_z = nz / len;

    tri.material_id = 0;
}

struct RenderSettings {
    int width;
    int height;
    int max_bounces;
    float aspect_ratio;

    std::array<float, 8> sky_spectrum;

    float sun_dx;
    float sun_dy;
    float sun_dz;

    float copper_diff_r;
    float copper_diff_g;
    float copper_diff_b;

    float copper_spec_r;
    float copper_spec_g;
    float copper_spec_b;

    float metal_roughness;
    float ray_epsilon;
};

inline uint32_t make_packet_seed(int sample, int y, int x) {
    uint32_t seed = 2166136261u;
    seed ^= static_cast<uint32_t>(sample + 1) * 16777619u;
    seed ^= static_cast<uint32_t>(y + 1) * 2246822519u;
    seed ^= static_cast<uint32_t>(x + 1) * 3266489917u;
    return seed;
}

void render_sample_tiles(const EngineState &engine,
                         const RenderSettings &settings, int sample,
                         int tile_rows, std::atomic<int> &next_row,
                         std::vector<float> &accum_r,
                         std::vector<float> &accum_g,
                         std::vector<float> &accum_b) {
    RayPacket packet;
    SIMDRand rng;

    while (true) {
        int y_begin = next_row.fetch_add(tile_rows, std::memory_order_relaxed);
        if (y_begin >= settings.height) {
            break;
        }

        int y_end = std::min(settings.height, y_begin + tile_rows);

        for (int y = y_begin; y < y_end; ++y) {
            for (int x = 0; x < settings.width; x += 8) {
                rng.init(make_packet_seed(sample, y, x));

                bool active[8] = {false, false, false, false,
                                  false, false, false, false};
                float pixel_r[8] = {0.0f};
                float pixel_g[8] = {0.0f};
                float pixel_b[8] = {0.0f};

                for (int lane = 0; lane < 8; ++lane) {
                    int pixel_x = x + lane;
                    bool valid_pixel = pixel_x < settings.width;
                    active[lane] = valid_pixel;

                    packet.spectrum_b0[lane] = 1.0f;
                    packet.spectrum_b1[lane] = 1.0f;
                    packet.spectrum_b2[lane] = 1.0f;
                    packet.spectrum_b3[lane] = 1.0f;
                    packet.spectrum_b4[lane] = 1.0f;
                    packet.spectrum_b5[lane] = 1.0f;
                    packet.spectrum_b6[lane] = 1.0f;
                    packet.spectrum_b7[lane] = 1.0f;

                    if (!valid_pixel) {
                        packet.origin_x[lane] = 0.0f;
                        packet.origin_y[lane] = 0.0f;
                        packet.origin_z[lane] = 0.0f;
                        packet.dir_x[lane] = 0.0f;
                        packet.dir_y[lane] = 0.0f;
                        packet.dir_z[lane] = -1.0f;
                        packet.inv_dir_x[lane] = 1e8f;
                        packet.inv_dir_y[lane] = 1e8f;
                        packet.inv_dir_z[lane] = -1.0f;
                        packet.closest_t[lane] = 1e30f;
                        continue;
                    }

                    float jitter_x = rng.next_float_scalar() - 0.5f;
                    float jitter_y = rng.next_float_scalar() - 0.5f;

                    float ndc_x =
                        (2.0f * (pixel_x + 0.5f + jitter_x) / settings.width - 1.0f) *
                        settings.aspect_ratio;
                    float ndc_y = 1.0f - 2.0f * (y + 0.5f + jitter_y) / settings.height;

                    float len = std::sqrt(ndc_x * ndc_x + ndc_y * ndc_y + 1.0f);

                    packet.origin_x[lane] = 0.0f;
                    packet.origin_y[lane] = 30.0f;
                    packet.origin_z[lane] = 50.0f;

                    packet.dir_x[lane] = ndc_x / len;
                    packet.dir_y[lane] = ndc_y / len;
                    packet.dir_z[lane] = -1.0f / len;

                    packet.inv_dir_x[lane] =
                        1.0f / (packet.dir_x[lane] == 0.0f ? 1e-8f : packet.dir_x[lane]);
                    packet.inv_dir_y[lane] =
                        1.0f / (packet.dir_y[lane] == 0.0f ? 1e-8f : packet.dir_y[lane]);
                    packet.inv_dir_z[lane] =
                        1.0f / (packet.dir_z[lane] == 0.0f ? 1e-8f : packet.dir_z[lane]);

                    packet.closest_t[lane] = 1e30f;
                }

                for (int bounce = 0; bounce < settings.max_bounces; ++bounce) {
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

                    for (int i = 0; i < 8; ++i) {
                        if (!active[i])
                            continue;

                        if (packet.closest_t[i] > 1e29f) {
                            float r_light = settings.sky_spectrum[6] * 0.5f;
                            float g_light = settings.sky_spectrum[3] * 0.5f;
                            float b_light = settings.sky_spectrum[0] * 0.5f;

                            float dot = packet.dir_x[i] * settings.sun_dx +
                                        packet.dir_y[i] * settings.sun_dy +
                                        packet.dir_z[i] * settings.sun_dz;

                            if (dot > 0.98f) {
                                r_light += 15.0f;
                                g_light += 14.0f;
                                b_light += 12.0f;
                            }

                            pixel_b[i] += packet.spectrum_b0[i] * b_light;
                            pixel_g[i] += packet.spectrum_b3[i] * g_light;
                            pixel_r[i] += packet.spectrum_b6[i] * r_light;

                            active[i] = false;
                        } else {
                            float old_dx = packet.dir_x[i];
                            float old_dy = packet.dir_y[i];
                            float old_dz = packet.dir_z[i];

                            float t = packet.closest_t[i];

                            float hit_x = packet.origin_x[i] + old_dx * t;
                            float hit_y = packet.origin_y[i] + old_dy * t;
                            float hit_z = packet.origin_z[i] + old_dz * t;

                            float surf_nx = packet.normal_x[i];
                            float surf_ny = packet.normal_y[i];
                            float surf_nz = packet.normal_z[i];

                            float facing =
                                surf_nx * old_dx + surf_ny * old_dy + surf_nz * old_dz;

                            if (facing > 0.0f) {
                                surf_nx = -surf_nx;
                                surf_ny = -surf_ny;
                                surf_nz = -surf_nz;
                            }

                            float ndotl = std::max(0.0f, surf_nx * settings.sun_dx +
                                                             surf_ny * settings.sun_dy +
                                                             surf_nz * settings.sun_dz);

                            float diffuse = 0.03f + 0.18f * ndotl;

                            float view_dx = -old_dx;
                            float view_dy = -old_dy;
                            float view_dz = -old_dz;

                            float half_x = settings.sun_dx + view_dx;
                            float half_y = settings.sun_dy + view_dy;
                            float half_z = settings.sun_dz + view_dz;

                            float half_len = std::sqrt(half_x * half_x + half_y * half_y +
                                                       half_z * half_z);

                            half_x /= half_len;
                            half_y /= half_len;
                            half_z /= half_len;

                            float ndoth = std::max(0.0f, surf_nx * half_x + surf_ny * half_y +
                                                             surf_nz * half_z);

                            float s2 = ndoth * ndoth;
                            float s4 = s2 * s2;
                            float s8 = s4 * s4;
                            float s16 = s8 * s8;
                            float s32 = s16 * s16;
                            float s64 = s32 * s32;

                            float spec = s64;

                            float direct_r = diffuse * settings.copper_diff_r +
                                             spec * settings.copper_spec_r;
                            float direct_g = diffuse * settings.copper_diff_g +
                                             spec * settings.copper_spec_g;
                            float direct_b = diffuse * settings.copper_diff_b +
                                             spec * settings.copper_spec_b;

                            pixel_r[i] += packet.spectrum_b6[i] * direct_r;
                            pixel_g[i] += packet.spectrum_b3[i] * direct_g;
                            pixel_b[i] += packet.spectrum_b0[i] * direct_b;

                            packet.spectrum_b0[i] *= 0.10f;
                            packet.spectrum_b1[i] *= 0.18f;
                            packet.spectrum_b2[i] *= 0.28f;
                            packet.spectrum_b3[i] *= 0.48f;
                            packet.spectrum_b4[i] *= 0.62f;
                            packet.spectrum_b5[i] *= 0.78f;
                            packet.spectrum_b6[i] *= 0.92f;
                            packet.spectrum_b7[i] *= 0.96f;

                            float energy = packet.spectrum_b0[i] + packet.spectrum_b3[i] +
                                           packet.spectrum_b6[i];

                            if (energy < 0.02f) {
                                active[i] = false;
                                continue;
                            }

                            float dot_in_n =
                                old_dx * surf_nx + old_dy * surf_ny + old_dz * surf_nz;

                            float refl_x = old_dx - 2.0f * dot_in_n * surf_nx;
                            float refl_y = old_dy - 2.0f * dot_in_n * surf_ny;
                            float refl_z = old_dz - 2.0f * dot_in_n * surf_nz;

                            float rx = 2.0f * rng.next_float_scalar() - 1.0f;
                            float ry = 2.0f * rng.next_float_scalar() - 1.0f;
                            float rz = 2.0f * rng.next_float_scalar() - 1.0f;

                            float new_dx = refl_x + settings.metal_roughness * rx;
                            float new_dy = refl_y + settings.metal_roughness * ry;
                            float new_dz = refl_z + settings.metal_roughness * rz;

                            float new_len = std::sqrt(new_dx * new_dx + new_dy * new_dy +
                                                      new_dz * new_dz);

                            packet.dir_x[i] = new_dx / new_len;
                            packet.dir_y[i] = new_dy / new_len;
                            packet.dir_z[i] = new_dz / new_len;

                            packet.origin_x[i] = hit_x + surf_nx * settings.ray_epsilon;
                            packet.origin_y[i] = hit_y + surf_ny * settings.ray_epsilon;
                            packet.origin_z[i] = hit_z + surf_nz * settings.ray_epsilon;

                            packet.inv_dir_x[i] =
                                1.0f / (packet.dir_x[i] == 0.0f ? 1e-8f : packet.dir_x[i]);
                            packet.inv_dir_y[i] =
                                1.0f / (packet.dir_y[i] == 0.0f ? 1e-8f : packet.dir_y[i]);
                            packet.inv_dir_z[i] =
                                1.0f / (packet.dir_z[i] == 0.0f ? 1e-8f : packet.dir_z[i]);

                            packet.closest_t[i] = 1e30f;
                        }
                    }
                }

                int pixel_base = y * settings.width + x;
                for (int i = 0; i < 8 && x + i < settings.width; ++i) {
                    int p_idx = pixel_base + i;

                    accum_r[p_idx] += pixel_r[i];
                    accum_g[p_idx] += pixel_g[i];
                    accum_b[p_idx] += pixel_b[i];
                }
            }
        }
    }
}

int main() {
    std::cout << "Booting Wube-Tier Spectral Raytracer...\n";

    EngineState engine;

    // 1. LOAD AND BUILD (These handle their own mallocs)
    if (!load_binary_stl("orc.stl", engine))
        return -1;
    rotate_model_x(engine, -90.0f);

    rotate_model_y(engine, 0.0f);
    build_bvh(engine);

    // 2. MANUALLY ALLOCATE ONLY THE MATERIALS
    engine.total_materials = 1;
    engine.materials =
        (Material *)_mm_malloc(sizeof(Material) * engine.total_materials, 64);

    // Setup sky
    const float sky_spectrum[8] = {1.0f, 0.9f, 0.8f, 0.7f,
                                   0.6f, 0.5f, 0.5f, 0.5f};
    float tmp_sun_dx = 0.45f;
    float tmp_sun_dy = 0.75f;
    float tmp_sun_dz = 0.50f;

    float tmp_sun_len =
        std::sqrt(tmp_sun_dx * tmp_sun_dx + tmp_sun_dy * tmp_sun_dy +
                  tmp_sun_dz * tmp_sun_dz);

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

    RenderSettings settings = {width,
                               height,
                               3,
                               aspect_ratio,
                               {sky_spectrum[0], sky_spectrum[1], sky_spectrum[2],
                                sky_spectrum[3], sky_spectrum[4], sky_spectrum[5],
                                sky_spectrum[6], sky_spectrum[7]},
                               SUN_DX,
                               SUN_DY,
                               SUN_DZ,
                               COPPER_DIFF_R,
                               COPPER_DIFF_G,
                               COPPER_DIFF_B,
                               COPPER_SPEC_R,
                               COPPER_SPEC_G,
                               COPPER_SPEC_B,
                               METAL_ROUGHNESS,
                               RAY_EPSILON};

    unsigned int thread_count = std::thread::hardware_concurrency();
    if (thread_count == 0) {
        thread_count = 1;
    }
    thread_count = std::min(thread_count, static_cast<unsigned int>(height));

    std::cout << "Rendering with " << thread_count << " worker threads.\n";

    for (int sample = 0; sample < MAX_SAMPLES; ++sample) {
        std::cout << "Rendering Sample " << sample + 1 << "/" << MAX_SAMPLES << "\r"
                  << std::flush;

        std::vector<std::thread> workers;
        workers.reserve(thread_count);

        const int tile_rows = 8;
        std::atomic<int> next_row{0};

        for (unsigned int thread_idx = 0; thread_idx < thread_count; ++thread_idx) {
            workers.emplace_back(render_sample_tiles, std::cref(engine),
                                 std::cref(settings), sample, tile_rows,
                                 std::ref(next_row), std::ref(accum_r),
                                 std::ref(accum_g), std::ref(accum_b));
        }

        for (std::thread &worker : workers) {
            worker.join();
        }
    }

    for (int p_idx = 0; p_idx < width * height; ++p_idx) {
        float avg_r = accum_r[p_idx] / MAX_SAMPLES;
        float avg_g = accum_g[p_idx] / MAX_SAMPLES;
        float avg_b = accum_b[p_idx] / MAX_SAMPLES;

        avg_r = std::sqrt(avg_r);
        avg_g = std::sqrt(avg_g);
        avg_b = std::sqrt(avg_b);

        int r = std::min(255, std::max(0, static_cast<int>(avg_r * 255.0f)));
        int g = std::min(255, std::max(0, static_cast<int>(avg_g * 255.0f)));
        int b = std::min(255, std::max(0, static_cast<int>(avg_b * 255.0f)));
        framebuffer[p_idx] = {static_cast<uint8_t>(r), static_cast<uint8_t>(g),
                              static_cast<uint8_t>(b)};
    }

    write_ppm_image("image.ppm", width, height, framebuffer);

    // CLEANUP
    _mm_free(engine.materials);
    _mm_free(engine.bvh_nodes);
    _mm_free(engine.triangles);

    return 0;
}
