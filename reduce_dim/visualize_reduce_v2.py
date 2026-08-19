import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from reduce_dim_v2 import *
from Function.function import *


def generate_cone_dataset(n_samples=5000, noise=0.01):
    z = np.random.uniform(0, 1, n_samples)
    radius = (1 - z) * 2
    theta = np.random.uniform(0, 2 * np.pi, n_samples)
    
    x = radius * np.cos(theta) + np.random.normal(0, noise, n_samples)
    y = radius * np.sin(theta) + np.random.normal(0, noise, n_samples)
    z = z + np.random.normal(0, noise, n_samples)
    
    X = np.column_stack([x, y, z])
    return X


def generate_sphere_dataset(n_samples=5000, noise=0.01):
    phi = np.random.uniform(0, 2*np.pi, n_samples)
    theta = np.random.uniform(0, np.pi, n_samples)
    r = 5 + np.random.normal(0, noise, n_samples)
    
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    
    return np.column_stack([x, y, z])


def visualize_reduction_3d(X_original, X_reduced, title="3D Reduction"):
    fig = plt.figure(figsize=(20, 6))
    
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(X_original[:, 0], X_original[:, 1], X_original[:, 2], 
               c='black', s=1, alpha=0.5)
    ax1.set_title(f'(a) Original\n{len(X_original)} points', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.view_init(elev=20, azim=45)
    
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], 
               c='red', s=2, alpha=0.7)
    ax2.set_title(f'(b) Border Points\n{len(X_reduced)} points', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.view_init(elev=20, azim=45)
    
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(X_original[:, 0], X_original[:, 1], X_original[:, 2], 
               c='blue', s=1, alpha=0.15, label='Original')
    ax3.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], 
               c='red', s=2, alpha=0.8, label='Reduced')
    
    reduction = (1 - len(X_reduced) / len(X_original)) * 100
    ax3.set_title(f'(c) Overlay\n{reduction:.1f}% reduction', fontsize=14, fontweight='bold')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.legend()
    ax3.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    return fig


def demo_cone():
    print("\n" + "="*70)
    print("DEMO 1: CONE DATASET WITH OQH")
    print("="*70)
    
    n_samples = 40000
    X = generate_cone_dataset(n_samples=n_samples, noise=0.01)
    print(f"Generated cone: {X.shape}")
    
    hg, ha, hb = 12, 4, 9
    print(f"Parameters: hg={hg}, ha={ha}, hb={hb}")
    
    X_reduced = reduce_points_nd_ha_hb_hg(
        X, ha=ha, hb=hb, hg=hg, 
        min_points=2, L=0.005, 
        hull_func=oqh_vertices
    )
    
    fig = visualize_reduction_3d(X, X_reduced, "Cone with OQH")
    plt.savefig('reduce_v2_cone_oqh.png', dpi=150, bbox_inches='tight')
    print("Saved: reduce_v2_cone_oqh.png")
    plt.show()


def demo_cone_cch():
    print("\n" + "="*70)
    print("DEMO 2: CONE DATASET WITH CCH")
    print("="*70)
    
    n_samples = 40000
    X = generate_cone_dataset(n_samples=n_samples, noise=0.01)
    print(f"Generated cone: {X.shape}")
    
    hg, ha, hb = 12, 4, 9
    print(f"Parameters: hg={hg}, ha={ha}, hb={hb}, K=10")
    
    X_reduced = reduce_points_nd_ha_hb_hg(
        X, ha=ha, hb=hb, hg=hg, 
        min_points=2, L=0.005, 
        K_input=10,
        hull_func=cch_vertices_indies
    )
    
    fig = visualize_reduction_3d(X, X_reduced, "Cone with CCH")
    plt.savefig('reduce_v2_cone_cch.png', dpi=150, bbox_inches='tight')
    print("Saved: reduce_v2_cone_cch.png")
    plt.show()


def demo_sphere():
    print("\n" + "="*70)
    print("DEMO 3: SPHERE DATASET WITH OQH")
    print("="*70)
    
    n_samples = 40000
    X = generate_sphere_dataset(n_samples=n_samples, noise=0.1)
    print(f"Generated sphere: {X.shape}")
    
    hg, ha, hb = 12, 4, 9
    print(f"Parameters: hg={hg}, ha={ha}, hb={hb}")
    
    X_reduced = reduce_points_nd_ha_hb_hg(
        X, ha=ha, hb=hb, hg=hg, 
        min_points=2, L=0.01, 
        hull_func=oqh_vertices
    )
    
    fig = visualize_reduction_3d(X, X_reduced, "Sphere with OQH")
    plt.savefig('reduce_v2_sphere_oqh.png', dpi=150, bbox_inches='tight')
    print("Saved: reduce_v2_sphere_oqh.png")
    plt.show()


def demo_comparison():
    print("\n" + "="*70)
    print("DEMO 4: COMPARISON - OQH vs CCH vs ConvexHull")
    print("="*70)
    
    n_samples = 10000
    X = generate_cone_dataset(n_samples=n_samples, noise=0.01)
    print(f"Generated dataset: {X.shape}")
    
    hg, ha, hb = 10, 4, 8
    
    print("\n1. OQH...")
    X_oqh = reduce_points_nd_ha_hb_hg(
        X, ha=ha, hb=hb, hg=hg, min_points=2, L=0.005, 
        hull_func=oqh_vertices
    )
    
    print("\n2. CCH (K=10)...")
    X_cch = reduce_points_nd_ha_hb_hg(
        X, ha=ha, hb=hb, hg=hg, min_points=2, L=0.005, 
        K_input=10, hull_func=cch_vertices_indies
    )
    
    print("\n3. ConvexHull...")
    X_cvh = reduce_points_nd_ha_hb_hg(
        X, ha=ha, hb=hb, hg=hg, min_points=2, L=0.005, 
        hull_func=None
    )
    
    fig = plt.figure(figsize=(24, 6))
    
    ax1 = fig.add_subplot(141, projection='3d')
    ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c='black', s=1, alpha=0.3)
    ax1.set_title(f'Original\n{len(X)} points', fontsize=12, fontweight='bold')
    ax1.view_init(elev=20, azim=45)
    
    ax2 = fig.add_subplot(142, projection='3d')
    ax2.scatter(X_oqh[:, 0], X_oqh[:, 1], X_oqh[:, 2], c='red', s=2, alpha=0.7)
    ax2.view_init(elev=20, azim=45)
    
    ax3 = fig.add_subplot(143, projection='3d')
    ax3.scatter(X_cch[:, 0], X_cch[:, 1], X_cch[:, 2], c='green', s=2, alpha=0.7)
    ax3.view_init(elev=20, azim=45)
    
    ax4 = fig.add_subplot(144, projection='3d')
    ax4.scatter(X_cvh[:, 0], X_cvh[:, 1], X_cvh[:, 2], c='blue', s=2, alpha=0.7)
    ax4.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.savefig('reduce_v2_comparison.png', dpi=150, bbox_inches='tight')
    print("\nSaved: reduce_v2_comparison.png")
    plt.show()


def main():
    print("\n" + "="*70)
    print("VISUALIZATION FOR reduce_points_nd_ha_hb_hg")
    print("="*70)
    
    demo_cone()
    demo_cone_cch()
    demo_sphere()
    demo_comparison()
    
    print("\n" + "="*70)
    print("COMPLETED!")
    print("="*70)


if __name__ == "__main__":
    main()
