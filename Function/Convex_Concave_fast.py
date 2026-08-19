"""
Convex_Concave_fast.py  —  Drop-in replacement cho Convex_Concave.py
Tối ưu hoá, giữ nguyên accuracy hoàn toàn by Le 2/2026

Các thay đổi:
  1. points_equal       → dùng norm thay vì np.allclose (bỏ overhead)
  2. find_closest_points→ dùng int index set thay vì tuple-key set
  3. _compute_angles_batch → vectorised numpy thay vì Python loop
  4. any_intersection_optimized → vectorised bbox filter khi n >= 8
  5. convex_concave_hull_visualization_v1 → B_X.extend thay vì B_X.append loop
"""

import numpy as np
import math
from scipy.spatial import ConvexHull
from sklearn.neighbors import KDTree

EPSILON = 1e-10


# ─── 1. Helpers ───────────────────────────────────────────────────────────────

def compute_angle(v1, v2):
    v1 = np.asarray(v1, dtype=np.float64)
    v2 = np.asarray(v2, dtype=np.float64)
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 < EPSILON or n2 < EPSILON:
        return 0.0
    return math.acos(float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)))


def _compute_angles_batch(candidates_rel, direction):
    """
    Vectorised: tính góc giữa mỗi hàng của candidates_rel và direction.
    candidates_rel đã được trừ vi (relative vectors).
    """
    norms = np.linalg.norm(candidates_rel, axis=1)
    nd = np.linalg.norm(direction)
    if nd < EPSILON:
        return np.zeros(len(candidates_rel))
    valid = norms > EPSILON
    cos_vals = np.ones(len(candidates_rel))
    if np.any(valid):
        dots = candidates_rel[valid] @ direction
        cos_vals[valid] = np.clip(dots / (norms[valid] * nd), -1.0, 1.0)
    return np.arccos(cos_vals)


def points_equal(p1, p2, tol=EPSILON):
    """Thay np.allclose bằng norm trực tiếp — nhanh hơn ~3x với vector nhỏ."""
    diff = np.asarray(p1, dtype=np.float64) - np.asarray(p2, dtype=np.float64)
    return float(diff @ diff) < tol * tol * max(len(diff), 1)


def point_to_key(point, precision=9):
    """Giữ nguyên để tương thích ngược với convex_concave_hull_indices."""
    return tuple(np.round(point, decimals=precision))


# ─── 2. Geometry helpers ──────────────────────────────────────────────────────

def get_bounding_box(p1, p2):
    return (min(p1[0],p2[0]), max(p1[0],p2[0]),
            min(p1[1],p2[1]), max(p1[1],p2[1]))


def bbox_intersect(b1, b2):
    if b1[1]+EPSILON < b2[0] or b1[0] > b2[1]+EPSILON: return False
    if b1[3]+EPSILON < b2[2] or b1[2] > b2[3]+EPSILON: return False
    return True


def line_segments_intersect(p1, q1, p2, q2):
    def orientation(p, q, r):
        val = (q[1]-p[1])*(r[0]-q[0]) - (q[0]-p[0])*(r[1]-q[1])
        if abs(val) < EPSILON: return 0
        return 1 if val > 0 else 2

    def on_segment(p, a, b):
        if abs((p[0]-a[0])*(b[1]-a[1]) - (p[1]-a[1])*(b[0]-a[0])) > EPSILON:
            return False
        return (min(a[0],b[0])-EPSILON <= p[0] <= max(a[0],b[0])+EPSILON and
                min(a[1],b[1])-EPSILON <= p[1] <= max(a[1],b[1])+EPSILON)

    o1=orientation(p1,q1,p2); o2=orientation(p1,q1,q2)
    o3=orientation(p2,q2,p1); o4=orientation(p2,q2,q1)
    if o1!=o2 and o3!=o4: return True
    if (o1==0 and on_segment(p2,p1,q1)) or (o2==0 and on_segment(q2,p1,q1)) or \
       (o3==0 and on_segment(p1,p2,q2)) or (o4==0 and on_segment(q1,p2,q2)):
        return True
    return False


def any_intersection_optimized(hull_points, segment):
    n = len(hull_points)
    if n < 2: return False
    p2 = np.asarray(segment[0], dtype=np.float64)
    q2 = np.asarray(segment[1], dtype=np.float64)
    seg_bbox = get_bounding_box(p2, q2)

    if n < 8:
        # Path gốc cho list nhỏ
        for i in range(n):
            a, b = hull_points[i], hull_points[(i+1) % n]
            if bbox_intersect(seg_bbox, get_bounding_box(a, b)):
                if line_segments_intersect(a, b, p2, q2):
                    return True
        return False

    # Vectorised bbox filter cho list lớn
    pts   = np.asarray(hull_points, dtype=np.float64)
    a_arr = pts
    b_arr = np.roll(pts, -1, axis=0)
    mnx = np.minimum(a_arr[:,0], b_arr[:,0]); mxx = np.maximum(a_arr[:,0], b_arr[:,0])
    mny = np.minimum(a_arr[:,1], b_arr[:,1]); mxy = np.maximum(a_arr[:,1], b_arr[:,1])
    sx0,sx1,sy0,sy1 = seg_bbox
    mask = (mxx+EPSILON>=sx0)&(mnx<=sx1+EPSILON)&(mxy+EPSILON>=sy0)&(mny<=sy1+EPSILON)
    for i in np.where(mask)[0]:
        if line_segments_intersect(a_arr[i], b_arr[i], p2, q2):
            return True
    return False


# ─── 3. Core algorithm ────────────────────────────────────────────────────────

def find_closest_points(X, S, K, theta, points_set, tree):
    """
    Thay đổi so với gốc:
      - local_used dùng integer index → O(1) int hash, không tạo tuple
      - Vectorised angle batch thay Python loop
      - K*3 query để có đủ candidates sau khi lọc
    Logic thuật toán 100% giống gốc.
    """
    vi, vj = S
    current_point = vi.copy()
    close_ch = []

    # Tìm index của vi
    _, vi_nn = tree.query([vi], k=1)
    local_used_idx = {int(vi_nn[0][0])}

    direction_to_target = vj - vi

    for _ in range(len(X)):
        if points_equal(current_point, vj):
            break

        k_query = min(K * 3, len(X))
        _, indices = tree.query([current_point], k=k_query)

        candidate_pts  = []
        candidate_idxs = []
        for idx in indices[0]:
            idx = int(idx)
            if idx in local_used_idx:
                continue
            pt = X[idx]
            if points_equal(pt, vj):
                continue
            candidate_pts.append(pt)
            candidate_idxs.append(idx)
            if len(candidate_pts) >= K:
                break

        if not candidate_pts:
            if not points_equal(current_point, vj):
                close_ch.append(vj)
            break

        cands_rel  = np.array(candidate_pts) - vi
        angles_arr = _compute_angles_batch(cands_rel, direction_to_target)

        valid_mask = angles_arr <= theta
        if not np.any(valid_mask):
            if not points_equal(current_point, vj):
                close_ch.append(vj)
            break

        valid_idxs   = np.where(valid_mask)[0]
        sorted_order = valid_idxs[np.argsort(angles_arr[valid_idxs])]

        added = False
        for vi_local in sorted_order:
            pi     = candidate_pts[vi_local]
            pi_idx = candidate_idxs[vi_local]
            angle  = float(angles_arr[vi_local])
            segment = [current_point, pi]
            if (not any_intersection_optimized(close_ch, segment) and
                    not any_intersection_optimized(points_set, segment)):
                close_ch.append(pi)
                local_used_idx.add(pi_idx)
                theta         = angle
                current_point = pi.copy()
                added = True
                break

        if not added:
            if not points_equal(current_point, vj):
                close_ch.append(vj)
            break

    return close_ch


# ─── 4. Public API ────────────────────────────────────────────────────────────

def convex_concave_hull_indices(X, K=3):
    X = np.asarray(X, dtype=np.float64)
    try:
        if len(X) < 3: return np.arange(len(X))
        if np.linalg.matrix_rank(X - X.mean(axis=0)) < 2: return np.arange(len(X))
        hull = ConvexHull(X)
    except (ValueError, np.linalg.LinAlgError):
        return np.arange(len(X))

    vertices     = X[hull.vertices]
    point_to_idx = {point_to_key(X[i]): i for i in range(len(X))}
    boundary_indices = []
    tree = KDTree(X)

    for i in range(len(vertices)):
        v1 = vertices[i]; v2 = vertices[(i+1) % len(vertices)]
        if len(boundary_indices) < 2:
            theta = math.pi
        else:
            theta = compute_angle(X[boundary_indices[-2]] - v1, v2 - v1)
        next_k_vertices = [vertices[(i+1+j) % len(vertices)]
                           for j in range(min(K, len(vertices)))]
        close_ch = find_closest_points(X, (v1, v2), K, theta, next_k_vertices, tree)
        for point in close_ch:
            pk = point_to_key(point)
            if pk in point_to_idx:
                boundary_indices.append(point_to_idx[pk])

    return (np.array(boundary_indices, dtype=int)
            if boundary_indices else np.arange(len(X)))


def convex_concave_hull_visualization(X, K=3):
    indices = convex_concave_hull_indices(X, K=K)
    return np.asarray(X)[indices]


def convex_concave_hull_visualization_v1(X, K=3):
    X = np.asarray(X, dtype=np.float64)
    try:
        if len(X) < 3: return X
        if np.linalg.matrix_rank(X - X.mean(axis=0)) < 2: return X
        hull = ConvexHull(X)
    except (ValueError, np.linalg.LinAlgError):
        return X

    vertices = X[hull.vertices]
    B_X  = []
    tree = KDTree(X)

    for i in range(len(vertices)):
        v1 = vertices[i]; v2 = vertices[(i+1) % len(vertices)]
        if len(B_X) < 2:
            theta = math.pi
        else:
            theta = compute_angle(np.asarray(B_X[-2]) - v1, v2 - v1)
        next_k_vertices = [vertices[(i+1+j) % len(vertices)]
                           for j in range(min(K, len(vertices)))]
        close_ch = find_closest_points(X, (v1, v2), K, theta, next_k_vertices, tree)
        B_X.extend(close_ch)   # extend thay vì loop append

    return np.array(B_X) if B_X else X
