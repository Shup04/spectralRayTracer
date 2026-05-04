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

inline void intersect_triangle_packet(RayPacket& packet, const Triangle& tri, __m256 active_mask) {
  const __m256 zero = simd_zero();
  const __m256 one  = simd_set1(1.0f);
  const __m256 eps  = simd_set1(1e-8f);
  const __m256 ray_t_min = simd_set1(0.001f);

  // Load packet rays.
  __m256 ox = simd_load(packet.origin_x);
  __m256 oy = simd_load(packet.origin_y);
  __m256 oz = simd_load(packet.origin_z);

  __m256 dx = simd_load(packet.dir_x);
  __m256 dy = simd_load(packet.dir_y);
  __m256 dz = simd_load(packet.dir_z);

  __m256 old_closest_t = simd_load(packet.closest_t);

  // Broadcast triangle data.
  __m256 v0_x = simd_set1(tri.v0_x);
  __m256 v0_y = simd_set1(tri.v0_y);
  __m256 v0_z = simd_set1(tri.v0_z);

  __m256 e1_x = simd_set1(tri.e1_x);
  __m256 e1_y = simd_set1(tri.e1_y);
  __m256 e1_z = simd_set1(tri.e1_z);

  __m256 e2_x = simd_set1(tri.e2_x);
  __m256 e2_y = simd_set1(tri.e2_y);
  __m256 e2_z = simd_set1(tri.e2_z);

  // pvec = cross(ray_dir, edge2)
  __m256 pvec_x, pvec_y, pvec_z;
  simd_cross3(dx, dy, dz, e2_x, e2_y, e2_z, pvec_x, pvec_y, pvec_z);

  // det = dot(edge1, pvec)
  __m256 det = simd_dot3(e1_x, e1_y, e1_z, pvec_x, pvec_y, pvec_z);

  // Two-sided triangle test: reject nearly parallel rays.
  __m256 det_abs = simd_abs(det);
  __m256 det_mask = simd_cmp_greater(det_abs, eps);

  __m256 inv_det = simd_div(one, det);

  // tvec = origin - v0
  __m256 tvec_x = simd_sub(ox, v0_x);
  __m256 tvec_y = simd_sub(oy, v0_y);
  __m256 tvec_z = simd_sub(oz, v0_z);

  // u = dot(tvec, pvec) * inv_det
  __m256 u = simd_mul(
      simd_dot3(tvec_x, tvec_y, tvec_z, pvec_x, pvec_y, pvec_z),
      inv_det
  );

  __m256 u_mask = simd_and(
      simd_cmp_greater_equal(u, zero),
      simd_cmp_less_equal(u, one)
  );

  // qvec = cross(tvec, edge1)
  __m256 qvec_x, qvec_y, qvec_z;
  simd_cross3(tvec_x, tvec_y, tvec_z, e1_x, e1_y, e1_z, qvec_x, qvec_y, qvec_z);

  // v = dot(ray_dir, qvec) * inv_det
  __m256 v = simd_mul(
      simd_dot3(dx, dy, dz, qvec_x, qvec_y, qvec_z),
      inv_det
  );

  __m256 uv = simd_add(u, v);

  __m256 v_mask = simd_and(
      simd_cmp_greater_equal(v, zero),
      simd_cmp_less_equal(uv, one)
  );

  // t = dot(edge2, qvec) * inv_det
  __m256 t = simd_mul(
      simd_dot3(e2_x, e2_y, e2_z, qvec_x, qvec_y, qvec_z),
      inv_det
  );

  __m256 t_mask = simd_and(
      simd_cmp_greater(t, ray_t_min),
      simd_cmp_less(t, old_closest_t)
  );

  // Final lane mask.
  __m256 hit_mask = active_mask;
  hit_mask = simd_and(hit_mask, det_mask);
  hit_mask = simd_and(hit_mask, u_mask);
  hit_mask = simd_and(hit_mask, v_mask);
  hit_mask = simd_and(hit_mask, t_mask);

  // Update closest_t.
  __m256 new_closest_t = simd_blend(old_closest_t, t, hit_mask);
  simd_store(packet.closest_t, new_closest_t);

  // Update normal.
  __m256 old_nx = simd_load(packet.normal_x);
  __m256 old_ny = simd_load(packet.normal_y);
  __m256 old_nz = simd_load(packet.normal_z);

  __m256 tri_nx = simd_set1(tri.norm_x);
  __m256 tri_ny = simd_set1(tri.norm_y);
  __m256 tri_nz = simd_set1(tri.norm_z);

  simd_store(packet.normal_x, simd_blend(old_nx, tri_nx, hit_mask));
  simd_store(packet.normal_y, simd_blend(old_ny, tri_ny, hit_mask));
  simd_store(packet.normal_z, simd_blend(old_nz, tri_nz, hit_mask));

  // Update material ID.
  __m256i old_mat = simd_load_int(packet.hit_material_id);
  __m256i new_mat = simd_set1_int(tri.material_id);
  __m256i blended_mat = simd_blend_int(old_mat, new_mat, hit_mask);

  simd_store_int(packet.hit_material_id, blended_mat);
}

