#define SPECTRAL_TEST
#include "../main.cpp"
#include <limits>

void check(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

RayPacket rays() {
    RayPacket packet{};
    for (int i = 0; i < 8; ++i) {
        packet.origin_x[i] = 0.25f;
        packet.origin_y[i] = 0.25f;
        packet.origin_z[i] = 2.0f;
        packet.dir_z[i] = -1.0f;
        packet.inv_dir_x[i] = packet.inv_dir_y[i] = 1e8f;
        packet.inv_dir_z[i] = -1.0f;
        packet.closest_t[i] = 1e30f;
        packet.hit_material_id[i] = 99;
        packet.normal_z[i] = -7;
    }
    packet.origin_x[1] = 2; // outside triangle
    packet.dir_z[2] = 1; packet.inv_dir_z[2] = 1; // behind ray
    packet.dir_z[3] = 0; packet.inv_dir_z[3] = 1e8f; // parallel
    packet.dir_x[3] = 1; packet.inv_dir_x[3] = 1;
    packet.closest_t[4] = 0.5f; // already has closer intersection
    // lane 5 is inactive despite aiming at triangle
    packet.origin_x[6] = 0; packet.origin_y[6] = 0; // vertex
    packet.origin_z[7] = -2; packet.dir_z[7] = 1; packet.inv_dir_z[7] = 1; // back face
    return packet;
}

int main() {
    Scene scene;
    auto &engine = scene.engine;
    engine.total_triangles = 8;
    engine.triangles = static_cast<Triangle *>(_mm_malloc(sizeof(Triangle) * 8, 64));
    engine.bvh_nodes = static_cast<BVHNode *>(_mm_malloc(sizeof(BVHNode) * 16, 64));
    for (int i = 0; i < 8; ++i) {
        float x = i * 4.0f;
        add_triangle(engine, i, x, 0, 0, x + 1, 0, 0, x, 1, 0);
        engine.triangles[i].material_id = i;
    }
    RayPacket direct = rays();
    intersect_triangle_packet(direct, engine.triangles[0], simd_mask_from_bits(0xdf));
    for (int i = 0; i < 8; ++i) {
        bool hit = i == 0 || i == 6 || i == 7;
        check(direct.closest_t[i] == (hit ? 2.0f : (i == 4 ? 0.5f : 1e30f)), "triangle mixed-lane distance");
        check(direct.hit_material_id[i] == (hit ? 0u : 99u), "triangle mixed-lane material preservation");
        check(direct.normal_z[i] == (hit ? 1.0f : -7.0f), "triangle mixed-lane normal preservation");
    }
    build_bvh(engine);
    check(nodes_used > 1, "synthetic geometry must exercise internal nodes");
    RayPacket traversed = rays();
    traverse_bvh(traversed, engine, simd_mask_from_bits(0xdf));
    for (int i = 0; i < 8; ++i) {
        check(traversed.closest_t[i] == direct.closest_t[i], "planar BVH must agree with direct triangle hits");
        check(traversed.hit_material_id[i] == direct.hit_material_id[i], "BVH material");
    }
    // Deterministic rays spread across leaves; compare BVH with brute force.
    for (int batch = 0; batch < 32; ++batch) {
        RayPacket brute = rays();
        for (int i = 0; i < 8; ++i) {
            brute.origin_x[i] = ((batch * 7 + i * 3) % 32) + 0.2f;
            brute.origin_y[i] = (i % 2) ? 2.0f : 0.2f;
            brute.origin_z[i] = 2;
            brute.dir_x[i] = brute.dir_y[i] = 0; brute.dir_z[i] = -1;
            brute.inv_dir_x[i] = brute.inv_dir_y[i] = 1e8f; brute.inv_dir_z[i] = -1;
            brute.closest_t[i] = 1e30f;
        }
        RayPacket bvh = brute;
        for (uint32_t j = 0; j < engine.total_triangles; ++j)
            intersect_triangle_packet(brute, engine.triangles[j], simd_mask_from_bits(0xb7));
        traverse_bvh(bvh, engine, simd_mask_from_bits(0xb7));
        for (int i = 0; i < 8; ++i) {
            check(bvh.closest_t[i] == brute.closest_t[i], "BVH versus brute force");
            check(bvh.hit_material_id[i] == brute.hit_material_id[i], "BVH versus brute force material");
        }
    }
    check(display_channel(-1) == 0 && display_channel(0) == 0, "lower display clamp");
    check(display_channel(1) == 255 && display_channel(10) == 255, "upper display clamp");
    check(display_channel(0.25f) == 127, "gamma conversion");
    bool rejected = false;
    try { display_channel(std::numeric_limits<float>::quiet_NaN()); }
    catch (const std::runtime_error &) { rejected = true; }
    check(rejected, "non-finite display input rejection");
    std::cout << "PASS: triangle mixed hit/miss/inactive/parallel/edge/backface lanes, planar multi-leaf BVH, 256 brute-force comparisons, display conversion\n";
}
