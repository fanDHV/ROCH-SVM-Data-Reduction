import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import pandas as pd
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

def reduce_example(X, ha, hb, hg,L = None, K_input = None, hull_func = None):

    n_dim = X.shape[1]
    bins_hb = [2**hb] * n_dim
    indices_hb, _ = assign_to_grid(X, bins_hb)

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
    partitions_hb = part(indices_hb)
    ratio = [2 ** (hb - ha)] * n_dim
    partitions_hb = reduce_points_nd_hb(X, partitions_hb, ratio, K_input = K_input, hull_func = hull_func)  
    points_ha = reduce_points_nd_ha(X, partitions_hb, K_input = K_input, hull_func = hull_func)
    return points_ha



def reduce_points_nd_hb(X, partitions_hb, ratio, min_part_size=2, K_input=None, hull_func=None):
    n_dim = X.shape[1]
    new_partitons = {}
    for part_key, part_indices in partitions_hb.items():
        if len(part_indices) < min_part_size:
            continue
        
        part_indices_arr = np.array(part_indices, dtype=int)
        X_part = X[part_indices_arr]
        boundary_indices = set()

        for d1, d2 in combinations(range(n_dim), 2):
            points2d = X_part[:, [d1, d2]]
            
            try:
                if hull_func is not None:
                    if K_input is None:
                        hull_coords = hull_func(points2d)
                        local_hull_indices = map_coords_to_indices(points2d, hull_coords)
                    else:
                        local_hull_indices = hull_func(points2d, K=K_input)
                        # print("local" , local_hull_indices)
                else:
                    hull = ConvexHull(points2d)
                    local_hull_indices = hull.vertices
            except:
                local_hull_indices = np.arange(len(part_indices_arr))
            
            # Vectorized mapping from local to global indices
            global_indices = part_indices_arr[local_hull_indices]
            boundary_indices.update(global_indices)
        
        # Map partition key from hb to ha level
        key_arr = np.asarray(part_key, dtype=float)
        new_key = tuple(np.floor(key_arr / ratio).astype(int))
        if new_key not in new_partitons:
            new_partitons[new_key] = []
        new_partitons[new_key].extend(list(boundary_indices))

    return new_partitons

def reduce_points_nd_ha(X, partitions_hb, min_part_size=2, K_input=None, hull_func=None):
    """Extract boundary points at ha level (finest resolution).
    Optimized with vectorized operations and cleaner logic."""
    boundary_indices = set()
    n_dim = X.shape[1]
    
    for part_key, part_indices in partitions_hb.items():
        if len(part_indices) < min_part_size:
            continue
        
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

