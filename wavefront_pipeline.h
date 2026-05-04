#pragma once
#include "core_types.h"
#include "simd_math.h"

// If ray[i] hits box, its 32bits in the mask are all 1, if missed, all 0
inline __m256 intersect_bvh_node(const RayPacket& packet, const BVHNode& node) {
    // turn floats into simd arrays for bitwise math.
    __m256 box_min_x = simd_set1(node.min_x);
    __m256 box_min_y = simd_set1(node.min_y);
    __m256 box_min_z = simd_set1(node.min_z);

    __m256 box_max_x = simd_set1(node.max_x);
    __m256 box_max_y = simd_set1(node.max_y);
    __m256 box_max_z = simd_set1(node.max_z);


    // load raypacket stuff for simd math. This stuff is aligned so its ok
    __m256 origin_x = simd_load(packet.origin_x);
    __m256 inv_dir_x = simd_load(packet.inv_dir_x);

    __m256 origin_y = simd_load(packet.origin_y);
    __m256 inv_dir_y = simd_load(packet.inv_dir_y);

    __m256 origin_z = simd_load(packet.origin_z);
    __m256 inv_dir_z = simd_load(packet.inv_dir_z);


    // Calculate plane intersections
    __m256 t1_x = simd_mul(simd_sub(box_min_x, origin_x), inv_dir_x);
    __m256 t2_x = simd_mul(simd_sub(box_max_x, origin_x), inv_dir_x);

    __m256 t1_y = simd_mul(simd_sub(box_min_y, origin_y), inv_dir_y);
    __m256 t2_y = simd_mul(simd_sub(box_max_y, origin_y), inv_dir_y);

    __m256 t1_z = simd_mul(simd_sub(box_min_z, origin_z), inv_dir_z);
    __m256 t2_z = simd_mul(simd_sub(box_max_z, origin_z), inv_dir_z);


    // calculate min or max for bvh
    __m256 t_near_x = simd_min(t1_x, t2_x);
    __m256 t_far_x  = simd_max(t1_x, t2_x);

    __m256 t_near_y = simd_min(t1_y, t2_y);
    __m256 t_far_y  = simd_max(t1_y, t2_y);

    __m256 t_near_z = simd_min(t1_z, t2_z);
    __m256 t_far_z  = simd_max(t1_z, t2_z);

    __m256 t_near = simd_max(t_near_x, simd_max(t_near_y, t_near_z));
    __m256 t_far = simd_min(t_far_x, simd_min(t_far_y, t_far_z));

    // branchless check, make a mask of rays that hit, AND are front faces.
    __m256 hit_mask = simd_cmp_greater(t_far, t_near);
    __m256 front_mask = simd_cmp_greater(t_far, simd_zero());
    
    return simd_and(hit_mask, front_mask);
}
