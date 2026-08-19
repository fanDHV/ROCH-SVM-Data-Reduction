import numpy as np
from Function.o_quickhull_lib import *
from sklearn.decomposition import FastICA, PCA
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
#from Function.Convex_Concave import convex_concave_hull_indices, convex_concave_hull_visualization_v1
from Function.Convex_Concave_fast import convex_concave_hull_indices, convex_concave_hull_visualization_v1
from shapely.geometry import Polygon, MultiPolygon
try:
    import alphashape
except ImportError:
    alphashape = None
import time
import os
try:
    from Function.Convex_Concave_better import (
        convex_concave_hull_visualization_better,
        convex_concave_hull_indices_better,
    )
except ImportError:
    convex_concave_hull_visualization_better = None
    convex_concave_hull_indices_better = None


def _get_plt():
    import matplotlib.pyplot as plt
    return plt

def X_transformed(points, M):
    return np.linalg.solve(M, points.T).T

def X_original(X_transformed, M):
    return (M @ X_transformed.T).T

def nomalized(points):
    return points - np.mean(points, axis=0)

def nomalized_axis(v):
    return v / np.linalg.norm(v)
#sửa ngày 11 tháng 4 năm 2025 by Quyến
def roch_vertices(points, M = None):
    
    if M is None:
        # Nếu không có M, tính toán nó từ các điểm
        M = find2axis_by_convex_hull(points)
    X_transformed_2 = X_transformed(points, M)
    oqh_transformed_edge = O_quickhull(X_transformed_2).arranged_points
    oqh_original_edge = X_original(oqh_transformed_edge, M)
    
    return oqh_original_edge[:-1]

def cch_vertices_indices(points, K = 50):
    hull = convex_concave_hull_indices(points,K)
    return hull 

def cch_vertices(points, K = 50):
    hull = convex_concave_hull_visualization_v1(points,K)
    return hull 

def cch_vertices_better(points, K = 4):
    if convex_concave_hull_visualization_better is None:
        raise ImportError("Function.Convex_Concave_better is required for cch_vertices_better")
    hull = convex_concave_hull_visualization_better(points, K)
    return hull

def cch_vertices_indices_better(points, K = 4):
    if convex_concave_hull_indices_better is None:
        raise ImportError("Function.Convex_Concave_better is required for cch_vertices_indices_better")
    hull = convex_concave_hull_indices_better(points, K)
    return hull

#thêm vào ngày 17 tháng 2 năm 2025 by Quyến
def find2axis_by_convex_hull(X):
    hull = ConvexHull(X)
    cvhull = X[hull.vertices]
    cvhull = np.vstack([cvhull, cvhull[0]])
    # Tính khoảng cách giữa các điểm liền kề
    distances = np.linalg.norm(cvhull[1:] - cvhull[:-1], axis=1)

    # Tìm vị trí có khoảng cách lớn nhất
    max_idx = np.argmax(distances)

    # Lấy hai điểm tạo thành cạnh dài nhất
    point1, point2 = cvhull[max_idx], cvhull[max_idx + 1]
    
    mean = np.mean(X, axis = 0)
    M = np.column_stack((point1 - mean, point2 - mean))
    return M
def S(hull):
    return Polygon(hull).area


#Le them ngày 23/4/2025


def find2axis_by_convex_hull_new(X):
    hull = ConvexHull(X)
    cvhull = X[hull.vertices]
    cvhull = np.vstack([cvhull, cvhull[0]])  # Đảm bảo bao lồi khép kín
    distances = np.linalg.norm(cvhull[1:] - cvhull[:-1], axis=1)
    max_idx = np.argmax(distances)
    point1, point2 = cvhull[max_idx], cvhull[max_idx + 1]
    mean = np.mean(X, axis=0)
    M = np.column_stack((point1 - mean, point2 - mean))
    return M, cvhull, point1, point2, mean


def plot_convex_hull_with_axes(X, save_path=None):
    plt = _get_plt()
    M, cvhull, point1, point2, mean = find2axis_by_convex_hull_new(X)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(cvhull[:, 0], cvhull[:, 1], color='blue', lw=2)
    ax.scatter(X[:, 0], X[:, 1], color='gray', s=5)
    ax.plot([point1[0], point2[0]], [point1[1], point2[1]], color='red', lw=2)
    ax.quiver(mean[0], mean[1], M[0, 0], M[1, 0], angles='xy', scale_units='xy', scale=1, color='green')
    ax.quiver(mean[0], mean[1], M[0, 1], M[1, 1], angles='xy', scale_units='xy', scale=1, color='orange')
    ax.set_aspect('equal')
    ax.set_title("Convex Hull with Longest Edge and Two Axes")

    # Lưu hình vào thư mục fig/
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if save_path:
        plt.savefig(save_path)
        print(f"Saved Oriented Convex Hull plot to {save_path}")
    else:
        plt.show()

#Quyen them ngay 11/5/2025

#Theo dien tich va mean
def find2axis_by_convex_hull_new_S1(X):
    hull = ConvexHull(X)
    cvhull = X[hull.vertices]
    cvhull = np.vstack([cvhull, cvhull[0]])  # Đảm bảo bao lồi khép kín
    mean = np.mean(X, axis=0)  # Tính điểm mean
    
    # Tính diện tích tam giác tạo bởi từng cạnh và điểm mean
    a = cvhull[:-1]
    b = cvhull[1:]
    # Tính cross product để xác định diện tích (không cần giá trị tuyệt đối 0.5)
    cross_products = (b[:, 0] - a[:, 0]) * (mean[1] - a[:, 1]) - (mean[0] - a[:, 0]) * (b[:, 1] - a[:, 1])
    areas = np.abs(cross_products)
    
    max_idx = np.argmax(areas)
    point1, point2 = cvhull[max_idx], cvhull[max_idx + 1]
    print(point1)
    
    M = np.column_stack((point1 - mean, point2 - mean))
    print("M",M)
    return M, cvhull, point1, point2, mean
#Theo dien tich va var
def plot_convex_hull_with_axes_S1(X, save_path=None):
    plt = _get_plt()
    M, cvhull, point1, point2, mean = find2axis_by_convex_hull_new_S1(X)
    print("mean", mean)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(cvhull[:, 0], cvhull[:, 1], color='blue', lw=2)
    ax.scatter(X[:, 0], X[:, 1], color='gray', s=5)
    ax.plot([point1[0], point2[0]], [point1[1], point2[1]], color='red', lw=2)
    ax.quiver(mean[0], mean[1], M[0, 0], M[1, 0], angles='xy', scale_units='xy', scale=1, color='green')
    ax.quiver(mean[0], mean[1], M[0, 1], M[1, 1], angles='xy', scale_units='xy', scale=1, color='orange')
    ax.set_aspect('equal')
    ax.set_title("Convex Hull with Longest Edge and Two Axes")

    # Lưu hình vào thư mục fig/
    os.makedirs("fig", exist_ok=True)
    if save_path:
        plt.savefig(save_path)
        print(f"Saved Oriented Convex Hull plot to {save_path}")
    else:
        plt.show()

#Quyen them ngay 20/4/2025
def find_CCH(X, K = 4):
    start_time = time.time()
    hull = convex_concave_hull_visualization_v1(X,K)
    end_time = time.time()
    time1 = end_time - start_time
    S1 = S(hull)
    length = len(hull) - 1
    return length, S1, time1

def find_OCH(X, M = None):
    start_time = time.time()
    hull = roch_vertices(X, M)
    end_time = time.time()
    time1 = end_time - start_time
    S1 = S(hull)
    length = len(hull) - 1
    return length, S1, time1
def find_alphashape(X, alpha=0.5):
    if alphashape is None:
        raise ImportError("alphashape is required for find_alphashape")
    start_time = time.time()
    hull = alphashape.alphashape(X, alpha)
    end_time = time.time()
    time1 = end_time - start_time
    
    # Kiểm tra loại hình và tính diện tích, chu vi
    if isinstance(hull, Polygon):
        S1 = hull.area
        length = hull.length
    elif isinstance(hull, MultiPolygon):
        S1 = sum(poly.area for poly in hull.geoms)
        length = sum(poly.length for poly in hull.geoms)
    else:
        S1 = length = 0

    #points = np.array(list(hull.exterior.coords))
    #S1 = S(points)
    #length = len(points) - 1
    
    return length, S1, time1

def find_CV(X):
    start_time = time.time()
    hull = ConvexHull(X)
    end_time = time.time()
    vertices = X[hull.vertices]
    hull = np.vstack([vertices, vertices[0]])
    time1 = end_time - start_time
    S1 = S(hull)
    length = len(hull) - 1
    return length, S1, time1


def find_CCH_new(X, K=4, save_path=None):
    plt = _get_plt()
    start_time = time.time()
    hull = convex_concave_hull_visualization_v1(X, K)
    end_time = time.time()
    time1 = end_time - start_time
    S1 = S(hull)
    length = len(hull) - 1

    # Vẽ Concave Hull
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], s=5, color='gray', label='Input Points')

    hull_closed = hull if np.array_equal(hull[0], hull[-1]) else np.vstack([hull, hull[0]])
    ax.plot(hull_closed[:, 0], hull_closed[:, 1], color='green', lw=2, label=f'Convex-Concave Hull (K={K})')

    ax.set_aspect('equal')
    ax.set_title(f"Convex-Concave Hull (K={K})")
    #ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved Concave Hull plot to {save_path}")
    else:
        plt.show()

    return length, S1, time1

def find_CCH_better(X, K=4, save_path=None):
    plt = _get_plt()
    start_time = time.time()
    hull = convex_concave_hull_visualization_better(X, K)
    end_time = time.time()
    time1 = end_time - start_time
    S1 = S(hull)
    length = len(hull) - 1

    # Vẽ Concave Hull
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], s=5, color='gray', label='Input Points')

    hull_closed = hull if np.array_equal(hull[0], hull[-1]) else np.vstack([hull, hull[0]])
    ax.plot(hull_closed[:, 0], hull_closed[:, 1], color='green', lw=2, label=f'Convex-Concave Hull (K={K})')

    ax.set_aspect('equal')
    ax.set_title(f"Convex-Concave Hull (K={K})")
    #ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved Concave Hull plot to {save_path}")
    else:
        plt.show()

    return length, S1, time1

def find_OCH_new(X, M = None, save_path=None):
    plt = _get_plt()
    start_time = time.time()
    hull = roch_vertices(X, M)
    end_time = time.time()
    time1 = end_time - start_time
    S1 = S(hull)
    length = len(hull) - 1

    # Vẽ ROCH
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], s=5, color='gray', label='Input Points')

    hull_closed = hull if np.array_equal(hull[0], hull[-1]) else np.vstack([hull, hull[0]])
    ax.plot(hull_closed[:, 0], hull_closed[:, 1], color='blue', lw=2, label='Restricted Orientation Convex Hull')

    ax.set_aspect('equal')
    ax.set_title("Restricted Orientation Convex Hull")
    #ax.legend()
    plt.tight_layout()  # Đảm bảo rằng không có phần nào bị cắt khi vẽ

    if save_path:
        plt.savefig(save_path)
        print(f"Saved Oriented Convex Hull plot to {save_path}")
    else:
        plt.show()

    return length, S1, time1

def find_alphashape_new(X, alpha=0.5, save_path=None):
    if alphashape is None:
        raise ImportError("alphashape is required for find_alphashape_new")
    plt = _get_plt()
    start_time = time.time()
    hull = alphashape.alphashape(X, alpha)
    end_time = time.time()
    time1 = end_time - start_time

    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], s=5, color='gray', label='Input Points')

    if isinstance(hull, Polygon):
        S1 = hull.area
        length = hull.length
        x, y = hull.exterior.xy
        ax.plot(x, y, color='red', lw=2, label=f'Alpha Shape (alpha={alpha})')

    elif isinstance(hull, MultiPolygon):
        S1 = sum(poly.area for poly in hull.geoms)
        length = sum(poly.length for poly in hull.geoms)
        for poly in hull.geoms:
            x, y = poly.exterior.xy
            ax.plot(x, y, color='red', lw=2, label=f'Alpha Shape (alpha={alpha})')
    else:
        S1 = length = 0

    ax.set_aspect('equal')
    ax.set_title(f"Alpha Shape (alpha = {alpha})")
    #ax.legend()


    if save_path:
        plt.savefig(save_path)
        print(f"Saved alpha shape plot to {save_path}")
    else:
        plt.show()

    return length, S1, time1


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



def reduce_points_cell(data, K_input=None, hull_func=None):
    n_dims = data.shape[1]
    if n_dims < 2:
        raise ValueError("Dữ liệu phải có ít nhất 2 chiều")
    hull_coords = None 
    try:
        if hull_func is not None:
            if K_input is None:
                hull_coords = hull_func(data)
            else:

                hull_coords = hull_func(data, K=K_input)
        else:
            hull = ConvexHull(data)
            hull_coords = data[hull.vertices]
    except Exception as e:
        hull_coords = data
    unique_points = np.unique(np.array(hull_coords), axis=0)
    return unique_points


def reduce(X, cells_hb, K_input=None, hull_func=None):
    all_border_points = []
    for cell in cells_hb:
        samples = cell['samples']
        if len(samples) == 0:
            continue
        cell_data = X[samples]
        try:
            reduced_points = reduce_points_cell(cell_data, K_input=K_input, hull_func=hull_func)
            if reduced_points.size > 0:
                all_border_points.append(reduced_points)
        except Exception:
            all_border_points.append(cell_data)
    
    if len(all_border_points) > 0:
        return np.vstack(all_border_points)
    else:
        return np.array([])


def reduce_points_ha_hb_hg(X, ha, hb, hg, min_points=2, K_input=None, hull_func=None):
    grid_hg, grid_ha, grid_hb = create_all_grids(X, ha=ha, hb=hb, hg=hg, n_dims=min(ha, X.shape[1]))
    
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
    
    X_reduced = reduce(X, grid_hb_smoothed, K_input=K_input, hull_func=hull_func)
    
    if X_reduced.size > 0:
        X_reduced = np.unique(X_reduced, axis=0)

    return X_reduced
