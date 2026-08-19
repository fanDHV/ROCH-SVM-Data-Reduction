
import numpy as np
from collections import defaultdict
from Function.function import * 
import numpy as np
from collections import defaultdict

def cluster_1d_feature(data, dim, L=0.1):
    if len(data) == 0:
        return {}, []
    
    data = np.array(data)
    feature_values = data[:, dim]
    
    if np.ptp(feature_values) == 0:
        center_val = float(feature_values[0])
        center_map = {tuple(point): center_val for point in data}
        return center_map, [center_val]
    feature_min = np.min(feature_values)
    feature_range = np.max(feature_values) - feature_min
    normalized_values = (feature_values - feature_min) / feature_range
    
    sorted_indices = np.argsort(normalized_values)
    sorted_values = normalized_values[sorted_indices]
    
    clusters = []
    
    for i, norm_val in enumerate(sorted_values):
        if not clusters:
            clusters.append((i, i, norm_val))
            continue
            
        last_start, last_end, last_center = clusters[-1]
        
        current_cluster_size = last_end - last_start + 1
        weighted_center = (last_center * current_cluster_size + norm_val) / (current_cluster_size + 1)
        
        if abs(norm_val - weighted_center) <= L:
            clusters[-1] = (last_start, i, weighted_center)
        else:
            clusters.append((i, i, norm_val))
    
    merged_clusters = []
    for start, end, center in clusters:
        if merged_clusters and abs(center - merged_clusters[-1][2]) <= L:
            # Gộp với cụm trước đó
            prev_start, prev_end, prev_center = merged_clusters[-1]
            prev_size = prev_end - prev_start + 1
            current_size = end - start + 1
            new_center = (prev_center * prev_size + center * current_size) / (prev_size + current_size)
            merged_clusters[-1] = (prev_start, end, new_center)
        else:
            merged_clusters.append((start, end, center))
    center_map = {}
    final_centers = []
    
    for start, end, norm_center in merged_clusters:
        
        original_center = norm_center * feature_range + feature_min
        final_centers.append(original_center)
        
       
        for idx in sorted_indices[start:end+1]:
            center_map[tuple(data[idx])] = original_center
    
    return center_map, final_centers

def reduce_points_nd(data, L=0.1, K_input = None, hull_func=None):
    n_dims = data.shape[1]
    if n_dims < 2:
        raise ValueError("Dữ liệu phải có ít nhất 2 chiều")
    
    all_edge_points = []
    for i in range(n_dims):
        for j in range(i+1, n_dims):
            projected_points = data[:, [i, j]]
            try:
                if hull_func and K_input is None:
                    hull_vertices = hull_func(projected_points)
                elif hull_func and K_input is not None:
                    # print("Using K_input for hull function", K_input)
                    hull_vertices = hull_func(projected_points, K=K_input)
                else:
                    hull = ConvexHull(projected_points)
                    hull_vertices = projected_points[hull.vertices]
                
                
                for vertex in hull_vertices:
                    distances = np.linalg.norm(projected_points - vertex, axis=1)
                    closest_idx = np.argmin(distances)
                    closest_point = data[closest_idx]
                    # new_point = np.copy(closest_point)
                    # new_point[i], new_point[j] = vertex[0], vertex[1]
                    all_edge_points.append(closest_point)
                    
            except Exception as e:
                # print(f"Error in dimensions ({i},{j}): {e}")

                all_edge_points.extend(data)

    # Loại bỏ điểm trùng lặp
    if all_edge_points:
        unique_points = np.unique(np.vstack(all_edge_points), axis=0)
    else:
        unique_points = np.array([])
    
    return unique_points

def n_dimensional_grid(data, hg=1):
    n_dims = data.shape[1]
    temp = 2 ** hg
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)
    
    bin_edges = [np.linspace(data_min[dim], data_max[dim], temp + 1) 
                for dim in range(n_dims)]
    digitized = np.array([np.clip(np.digitize(data[:, dim], edges) - 1, 0, temp - 1)
                         for dim, edges in enumerate(bin_edges)])
    multipliers = temp ** np.arange(n_dims - 1, -1, -1)
    cell_indices = digitized.T @ multipliers
    
    unique_cells, inverse_indices = np.unique(cell_indices, return_inverse=True)
    grid_dict = {}
    for i, cell_id in enumerate(unique_cells):
        grid_dict[cell_id] = data[inverse_indices == i]
    
    return grid_dict, bin_edges

def reduce_points_nd_grid(data, L=0.1, hg=3, K_input = None, hull_func=None):
    n_dims = data.shape[1]
    if n_dims < 2:
        raise ValueError("Dữ liệu phải có ít nhất 2 chiều")
    
    grid_dict, bin_edges = n_dimensional_grid(data, hg)
    all_reduced_points = []
    
    for cell_id, cell_points in grid_dict.items():
        cell_points_array = np.array(cell_points)
        if len(cell_points_array) == 0:
            continue
        reduced_points = reduce_points_nd(cell_points_array, L, K_input, hull_func)
        if len(reduced_points) > 0:
            all_reduced_points.append(reduced_points)
    if all_reduced_points:
        all_reduced_points = np.vstack(all_reduced_points)
        unique_points = np.unique(all_reduced_points, axis=0)
    else:
        unique_points = np.array([])
    
    return unique_points

def reduce_points_nd_using_centers(data, L=0.1, hg=3, K_input = None, hull_func=None):
    n_dims = data.shape[1]
    if n_dims < 2:
        raise ValueError("Dữ liệu phải có ít nhất 2 chiều")

    centers_per_dim = []
    for dim in range(n_dims):
        _, centers = cluster_1d_feature(data, dim, L)
        centers_per_dim.append(centers)


    grids = np.array(np.meshgrid(*centers_per_dim, indexing='ij'))
    grid_points = grids.reshape(n_dims, -1).T
    grid_dict, bin_edges = n_dimensional_grid(grid_points, hg)
    print(f"Tạo lưới với {len(grid_points)}")
    all_reduced_points = []

    for cell_id, cell_points in grid_dict.items():
        cell_points_array = np.array(cell_points)
        if len(cell_points_array) == 0:
            continue
        reduced_points = reduce_points_nd(cell_points_array, L, K_input, hull_func)
        if len(reduced_points) > 0:
            all_reduced_points.append(reduced_points)
    if all_reduced_points:
        all_reduced_points = np.vstack(all_reduced_points)
        unique_points = np.unique(all_reduced_points, axis=0)
    else:
        unique_points = np.array([])
    
    return unique_points


