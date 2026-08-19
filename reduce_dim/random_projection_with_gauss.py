import numpy as np
from sklearn.random_projection import SparseRandomProjection
from Function.function import oqh_vertices



X = np.random.rand(100, 5)
transformer = SparseRandomProjection(compute_inverse_components=True, n_components=2)
X_new = transformer.fit_transform(X)
X_new = oqh_vertices(X_new)
print(X_new.shape)
X_new_inversed = transformer.inverse_transform(X_new)
print(X_new_inversed.shape)
X_new_again = transformer.transform(X_new_inversed)
print(np.allclose(X_new, X_new_again))