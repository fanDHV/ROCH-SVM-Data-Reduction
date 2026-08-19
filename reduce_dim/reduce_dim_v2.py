import numpy as np
from scipy.spatial import ConvexHull
from sklearn.neighbors import KDTree
import time
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Function.function import *
def create_all_grids(X, ha, hb, hg, n_dims=None):
    
    n_samples, n_features = X.shape
    
    if n_dims is None:
        n_dims = min(ha, n_features)
    
    bounds = []
    for i in range(n_dims):
        min_val = X[:, i].min()
        max_val = X[:, i].max()
        bounds.append((min_val, max_val))
    cells = [{'bounds': bounds, 'samples': list(range(n_samples))}]
    
    grid_hg = None
    grid_ha = None
    grid_hb = None
    
    max_level = max(hg, ha, hb)
    
    for level in range(max_level):
        new_cells = []
        dim_to_split = level % n_dims
        
        for cell in cells:
            cell_bounds = cell['bounds']
            min_val, max_val = cell_bounds[dim_to_split]
            mid_val = (min_val + max_val) / 2
            
            bounds1 = cell_bounds.copy()
            bounds1[dim_to_split] = (min_val, mid_val)
            
            bounds2 = cell_bounds.copy()
            bounds2[dim_to_split] = (mid_val, max_val)
            
            samples1 = []
            samples2 = []
            
            for idx in cell['samples']:
                if X[idx, dim_to_split] <= mid_val:
                    samples1.append(idx)
                else:
                    samples2.append(idx)
            
            if len(samples1) > 0:
                new_cells.append({'bounds': bounds1, 'samples': samples1})
            if len(samples2) > 0:
                new_cells.append({'bounds': bounds2, 'samples': samples2})
        
        cells = new_cells
        
        current_depth = level + 1
        
        if current_depth == hg and grid_hg is None:
            grid_hg = [c.copy() for c in cells]
        if current_depth == ha and grid_ha is None:
            grid_ha = [c.copy() for c in cells]
        if current_depth == hb and grid_hb is None:
            grid_hb = [c.copy() for c in cells]
    
    if grid_hg is None:
        grid_hg = cells
    if grid_ha is None:
        grid_ha = cells
    if grid_hb is None:
        grid_hb = cells
    
    return grid_hg, grid_ha, grid_hb

def smooth_data(X, grid_hg, min_points=2):
    smoothed = set()
    isolated_count = 0
    
    for cell in grid_hg:
        samples = cell['samples']
        
        if len(samples) >= min_points:
            smoothed.update(samples)
        else:
            isolated_count += len(samples)
    
    smoothed_indices = np.array(sorted(smoothed))
    return smoothed_indices


def cluster_one_dimension(values, L_factor=0.10):
    n = len(values)
    if n == 0:
        return [], {}
    
    x_min, x_max = values.min(), values.max()
    L = L_factor * (x_max - x_min) if x_max > x_min else 0.01
    
    clusters = [{'samples': [0], 'center': values[0]}]
    
    for i in range(1, n):
        x_i = values[i]
        min_dist = float('inf')
        closest_idx = -1
        
        for j, cluster in enumerate(clusters):
            dist = abs(x_i - cluster['center'])
            if dist < min_dist:
                min_dist = dist
                closest_idx = j
        
        if min_dist <= L:
            clusters[closest_idx]['samples'].append(i)
            k = len(clusters[closest_idx]['samples'])
            clusters[closest_idx]['center'] = ((k - 1) / k) * clusters[closest_idx]['center'] + (1 / k) * x_i
        else:
            clusters.append({'samples': [i], 'center': x_i})
    
    merged = True
    while merged:
        merged = False
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                if i < len(clusters) and j < len(clusters):
                    if abs(clusters[i]['center'] - clusters[j]['center']) <= L:
                        clusters[i]['samples'].extend(clusters[j]['samples'])
                        clusters[i]['center'] = (clusters[i]['center'] + clusters[j]['center']) / 2
                        clusters.pop(j)
                        merged = True
                        break
            if merged:
                break
    
    value_to_center = {}
    for cluster in clusters:
        center = cluster['center']
        for sample_idx in cluster['samples']:
            value_to_center[values[sample_idx]] = center
    
    return clusters, value_to_center


def map_coords_to_indices(points2d, hull_coords):
    if hull_coords.shape[0] == 0:
        return np.array([], dtype=int)
    
    tree = KDTree(points2d)
    distances, local_indices = tree.query(hull_coords, k=1)
    
    flat = local_indices.flatten()
    return np.unique(flat).astype(int)
    
def select_2_dims_uncorrelated(data):
    n_dims = data.shape[1]
    if n_dims < 2:
        return 0, min(1, n_dims - 1)
    
    variances = np.var(data, axis=0)
    var_normalized = variances / (variances.max() + 1e-10)
    
    # Tính ma trận correlation (robust): bỏ qua các chiều có std = 0 để tránh chia cho 0
    stds = np.std(data, axis=0)
    non_constant = stds > 1e-12
    if np.sum(non_constant) < 2:
        # Nếu không đủ 2 chiều có biến thiên, fallback: chọn 2 chiều có variance lớn nhất
        top2 = np.argsort(variances)[-2:]
        return int(top2[0]), int(top2[1])

    corr_matrix = np.eye(n_dims, dtype=float)
    try:
        corr_sub = np.corrcoef(data[:, non_constant].T)
        corr_sub = np.nan_to_num(corr_sub, nan=0.0, posinf=0.0, neginf=0.0)
        idx = np.flatnonzero(non_constant)
        corr_matrix[np.ix_(idx, idx)] = corr_sub
    except Exception:
        # Nếu corrcoef lỗi vì lý do nào đó, coi như các chiều độc lập
        corr_matrix = np.eye(n_dims, dtype=float)
    
    best_score = -np.inf
    best_pair = (0, 1)
    
    for i in range(n_dims):
        for j in range(i + 1, n_dims):
            # Score = (variance_i + variance_j) - alpha * |correlation_ij|
            # alpha điều chỉnh mức độ quan trọng của correlation
            alpha = 0.5
            score = (var_normalized[i] + var_normalized[j]) - alpha * abs(corr_matrix[i, j])
            
            if score > best_score:
                best_score = score
                best_pair = (i, j)
    
    return best_pair[0], best_pair[1]

def reduce_points_cell(data, L=0.05, K_input=None, hull_func=None):
    n_dims = data.shape[1]
    if n_dims < 2:
        raise ValueError("Dữ liệu phải có ít nhất 2 chiều")
    
    dim_i, dim_j = select_2_dims_uncorrelated(data)
    remaining_dims = [d for d in range(n_dims) if d not in (dim_i, dim_j)]
    
    projected_points = data[:, [dim_i, dim_j]]
    hull_indices = []
    hull_coords = None 
    
    try:
        if hull_func is not None:
            if K_input is None:
                hull_coords = hull_func(projected_points)
                hull_indices = map_coords_to_indices(projected_points, hull_coords)
            else:
                # cch_vertices_indies trả về indices trực tiếp
                hull_indices = hull_func(projected_points, K=K_input)
                hull_coords = None
        else:
            hull = ConvexHull(projected_points)
            hull_indices = hull.vertices
            hull_coords = None
    except Exception as e:
        hull_indices = range(len(data))
    
    hull_indices = np.array(list(hull_indices)) if isinstance(hull_indices, set) else np.array(hull_indices)
    
    if len(hull_indices) == 0:
        return np.array([])
    
    if len(remaining_dims) == 0:
        hull_points_full = data[hull_indices]
        return hull_points_full
    
    #Cách 1: Clustering 1D cho các chiều còn lại
    # all_reduced_points = []
    # dimension_mappings = {}
    # for dim in remaining_dims:
    #     clusters, value_to_center = cluster_one_dimension(data[:, dim], L_factor=L)
    #     dimension_mappings[dim] = value_to_center
    
    
    
    # for hull_idx in hull_indices:
    #     original_point = data[hull_idx]
    #     hull_2d = projected_points[hull_idx]
        
    #     new_point = np.zeros(n_dims)
    #     new_point[dim_i] = hull_2d[0]
    #     new_point[dim_j] = hull_2d[1]
        
    #     for dim in remaining_dims:
    #         original_value = original_point[dim]
    #         if original_value in dimension_mappings[dim]:
    #             new_point[dim] = dimension_mappings[dim][original_value]
    #         else:
    #             new_point[dim] = original_value
        
    #     all_reduced_points.append(new_point)
    
    # if len(all_reduced_points) > 0:
    #     unique_points = np.unique(np.array(all_reduced_points), axis=0)
    # else:
    #     unique_points = np.array([])

    #Cách 2: Giữ nguyên giá trị gốc cho các chiều còn lại
    all_reduced_points = data[hull_indices]
    unique_points = np.unique(np.array(all_reduced_points), axis=0)
    return unique_points


def reduce(X, cells_hb, L=0.005, K_input=None, hull_func=None):
    all_border_points = []
    for cell in cells_hb:
        samples = cell['samples']
        if len(samples) == 0:
            continue
        cell_data = X[samples]
        try:
            reduced_points = reduce_points_cell(cell_data, L=L, K_input=K_input, hull_func=hull_func)
            if reduced_points.size > 0:
                all_border_points.append(reduced_points)
        except Exception:
            all_border_points.append(cell_data)
    
    if len(all_border_points) > 0:
        return np.vstack(all_border_points)
    else:
        return np.array([])


def reduce_points_nd_ha_hb_hg(X, ha, hb, hg, min_points=2, K_input = None, L = 0.005, hull_func=None):
        
    grid_hg, grid_ha, grid_hb = create_all_grids(X, ha = ha, hb = hb, hg = hg, n_dims=min(ha, X.shape[1]))
    
    smoothed_indices = smooth_data(X, grid_hg, min_points=min_points)
    smoothed_set = set(smoothed_indices)
    
    grid_hb_smoothed = []
    for cell in grid_hb:
        filtered_samples = [idx for idx in cell['samples'] if idx in smoothed_set]
        if len(filtered_samples) > 0:
            grid_hb_smoothed.append({
                'bounds': cell['bounds'],
                'samples': filtered_samples
            })
    
    X_reduced = reduce(X, grid_hb_smoothed, L= L, K_input=K_input, hull_func=hull_func)
    
    if X_reduced.size > 0:
        X_reduced = np.unique(X_reduced, axis=0)

    return X_reduced
