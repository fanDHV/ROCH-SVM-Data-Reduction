import numpy as np
from itertools import combinations
from sklearn.neighbors import KDTree
import time
import os
import sys

# Thêm thư mục cha vào sys.path để import được Function
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Function.function import *

def assign_to_grid(X, n_bins_per_dim):
    n_samples, n_dim = X.shape
    grid_indices = np.zeros((n_samples, n_dim), dtype=int)
    grid_edges = []
    
    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)
    ranges = maxs - mins
    ranges = np.where(ranges == 0, 1, ranges)
    
    for d in range(n_dim):
        edges = np.linspace(mins[d], maxs[d], n_bins_per_dim[d] + 1)
        indices = ((X[:, d] - mins[d]) / ranges[d] * n_bins_per_dim[d]).astype(int)
        indices = np.clip(indices, 0, n_bins_per_dim[d] - 1) 
        grid_indices[:, d] = indices
        grid_edges.append(edges)

    return grid_indices, grid_edges

def group_indices(indices_fine, bins_coarse):
    bins_fine = np.max(indices_fine, axis=0) + 1
    ratio = bins_fine / np.array(bins_coarse)
    indices_coarse = (indices_fine / ratio).astype(int)
    return indices_coarse


def map_coords_to_indices(points2d, hull_coords):
    if hull_coords.shape[0] == 0:
        return np.array([], dtype=int)
    
    tree = KDTree(points2d)
    distances, local_indices = tree.query(hull_coords, k=1)
    
    # Flatten and get unique values (numpy is faster than set for small arrays)
    flat = local_indices.flatten()
    return np.unique(flat).astype(int)

def reduce_example_with_hg(X, ha, hb, hg,L = None, K_input = None, hull_func = None):

    n_dim = X.shape[1]
    bins_hg = [2**hg] * n_dim
    # print("HG bins:", bins_hg)
    indices_hg, _ = assign_to_grid(X, bins_hg)

    def part(indices):
        if indices.shape[0] == 0:
            return {}
        tuples = [tuple(row) for row in indices]
        partitions = {}
        for i, key in enumerate(tuples):
            if key not in partitions:
                partitions[key] = []
            partitions[key].append(i)
        return partitions
    partitions = part(indices_hg)
    points_ha = reduce_points_nd_hg(X, partitions, K_input = K_input, hull_func = hull_func)
    return points_ha

def reduce_points_nd_hg(X, partitions, K_input=None, hull_func=None):
    """Extract boundary points at ha level (finest resolution).
    Optimized with vectorized operations and cleaner logic."""
    boundary_indices = set()
    n_dim = X.shape[1]
    
    for part_key, part_indices in partitions.items():
        part_indices_arr = np.array(part_indices, dtype=int)
        X_part = X[part_indices_arr]
        
        for d1, d2 in combinations(range(n_dim), 2):
            points2d = X_part[:, [d1, d2]]
            
            try:
                if hull_func is not None:
                    if K_input is None:
                        hull_coords = hull_func(points2d)
                        local_hull_indices = map_coords_to_indices(points2d, hull_coords)
                    else:
                        local_hull_indices = hull_func(points2d, K=K_input)
                    
                else:
                    hull = ConvexHull(points2d)
                    local_hull_indices = hull.vertices
            except:
                local_hull_indices = np.arange(len(part_indices_arr))  
            global_indices = part_indices_arr[local_hull_indices]
            boundary_indices.update(global_indices)

    # Return sorted unique indices as 1D array
    return np.array(sorted(boundary_indices), dtype=int)
