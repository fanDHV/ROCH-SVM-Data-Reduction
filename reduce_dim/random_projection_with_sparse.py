from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
# Load dataset
data = load_iris()
X, y = data.data, data.target
feature_names = data.feature_names
target_names = data.target_names
print(feature_names)
print(target_names)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Khởi tạo và fit-transform
from sklearn.random_projection import SparseRandomProjection

# Khởi tạo và fit-transform
srp = SparseRandomProjection(n_components=2)
X_srp = srp.fit_transform(X_scaled)

# Visualize
plt.figure(figsize=(8, 6))
for color, i, target_name in zip(['navy', 'turquoise', 'darkorange'], [0, 1, 2], target_names):
    plt.scatter(
        X_srp[y == i, 0], 
        X_srp[y == i, 1], 
        color=color, 
        label=target_name,
        alpha=0.8
    )
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title('Sparse Random Projection')
plt.legend()
plt.show()