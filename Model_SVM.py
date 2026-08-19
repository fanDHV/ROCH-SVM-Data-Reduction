from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score
from Function.function import *
from reduce_dim.reduce_h_dim import reduce_example_with_hg
from reduce_dim.reduce_dim_v2 import reduce_points_nd_ha_hb_hg


def train_orginal(X_train,X_test,y_train,y_test, svm):
    
    start_train = time.time()
    svm.fit(X_train, y_train)
    run_time = time.time() - start_train
    y_pred = svm.predict(X_test)
    return accuracy_score(y_test, y_pred),precision_score(y_test, y_pred, zero_division=0),recall_score(y_test, y_pred, zero_division=0),f1_score(y_test, y_pred, zero_division=0) ,run_time

def train_by_ROCH_2d_dataset(X_train, X_test, y_train, y_test,svm, ha = 1, hb = 3, hg = 6, hull_func = roch_vertices):
    start_train = time.time() 
    X_train_ch = []
    y_train_ch = []
    for i in np.unique(y_train):
        X_class_i = X_train[y_train == i]
        hull = reduce_points_ha_hb_hg(X_class_i, ha = ha, hb = hb, hg = hg, hull_func = hull_func)
        X_train_ch.extend(hull)
        y_train_ch.extend([i] * len(hull))

    X_new, y_new = np.array(X_train_ch), np.array(y_train_ch)

    print("ROCH len:" ,len(X_new))
    svm.fit(X_new, y_new)
    run_time = time.time() - start_train
    y_pred = svm.predict(X_test)
    return accuracy_score(y_test, y_pred),precision_score(y_test, y_pred, zero_division=0),recall_score(y_test, y_pred, zero_division=0),f1_score(y_test, y_pred, zero_division=0) ,run_time

def train_by_CCH_2d_dataset(X_train, X_test, y_train, y_test, svm , K_input = 50, ha = 1, hb = 3, hg = 6, hull_func = cch_vertices_better):
    start_train = time.time()
    X_train_ch = []
    y_train_ch = []
    for i in np.unique(y_train):
        X_class_i = X_train[y_train == i]
        hull = reduce_points_ha_hb_hg(X_class_i, ha = ha, hb = hb, hg = hg, K_input = K_input, hull_func = hull_func)

        X_train_ch.extend(hull)
        y_train_ch.extend([i] * len(hull))
    X_new, y_new = np.array(X_train_ch), np.array(y_train_ch)
    print("CCH len:" ,len(X_new))

    svm.fit(X_new, y_new)
    run_time = time.time() - start_train
    y_pred = svm.predict(X_test)
    return accuracy_score(y_test, y_pred),precision_score(y_test, y_pred, zero_division=0),recall_score(y_test, y_pred, zero_division=0),f1_score(y_test, y_pred, zero_division=0) ,run_time

"""Quyen them ngay 10 8 2025"""
"""High dimensional data"""

def train_orginal_high(X_train, X_test, y_train, y_test, svm):
    # Original không có bước reduce → time_reduce = 0.0
    time_reduce = 0.0

    start_fit = time.time()
    svm.fit(X_train, y_train)
    time_fit = time.time() - start_fit

    n_sv = svm.n_support_.sum()

    y_pred = svm.predict(X_test)
    return (
        accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred, zero_division=0),
        recall_score(y_test, y_pred, zero_division=0),
        f1_score(y_test, y_pred, zero_division=0),
        time_reduce,
        time_fit,
        n_sv
    )

def train_by_ROCH_high(X_train, X_test, y_train, y_test, svm,ha = None, hb = None, hg = 4, L = 0.1):
    start_train = time.time() 
    X_train_ch = []
    y_train_ch = []
    for i in np.unique(y_train):
        X_class_i = X_train[y_train == i]
        partitions = reduce_example_with_hg(X_class_i, ha=ha, hb=hb, hg=hg,L=L,  hull_func=roch_vertices)
        reduced_points = X_class_i[partitions]
        X_train_ch.extend(reduced_points)
        y_train_ch.extend([i] * len(reduced_points))

    
    X_new, y_new = np.array(X_train_ch), np.array(y_train_ch)
    svm.fit(X_new, y_new)
    run_time = time.time() - start_train
    y_pred = svm.predict(X_test)
    print("Reduced training points:", X_new.shape)
    return accuracy_score(y_test, y_pred),precision_score(y_test, y_pred, zero_division=0),recall_score(y_test, y_pred, zero_division=0),f1_score(y_test, y_pred, zero_division=0) ,run_time

def train_by_CCH_high(X_train, X_test, y_train, y_test, svm, K = 50,ha = None, hb = None, hg = 4, L = 0.1, K_input = 50):
    start_train = time.time()
    X_train_ch = []
    y_train_ch = []
    for i in np.unique(y_train):
        X_class_i = X_train[y_train == i]
        partitions = reduce_example_with_hg(X_class_i,ha = ha, hb = hb,hg=hg,K_input = K_input,L=L,  hull_func=cch_vertices_indices)
        reduced_points = X_class_i[partitions]
        X_train_ch.extend(reduced_points)
        y_train_ch.extend([i] * len(reduced_points))

    X_new, y_new = np.array(X_train_ch), np.array(y_train_ch)
    print("Reduced training points:", X_new.shape)
    svm.fit(X_new, y_new)
    n_sv = svm.n_support_.sum()
    run_time = time.time() - start_train
    y_pred = svm.predict(X_test)
    return accuracy_score(y_test, y_pred),precision_score(y_test, y_pred, zero_division=0),recall_score(y_test, y_pred, zero_division=0),f1_score(y_test, y_pred, zero_division=0) ,run_time


#Quyen thêm ngày 19 12 2025
def train_by_ROCH_v2(X_train, X_test, y_train, y_test, svm, ha=None, hb=None, hg=4, L=0.1, min_points=2):

    # ── Bước 1: Reduce ────────────────────────────────────────
    start_reduce = time.time()
    X_train_ch = []
    y_train_ch = []
    ha_value = ha if ha is not None else 4
    hb_value = hb if hb is not None else 4

    def add_class_points(points, label):
        X_train_ch.extend(points)
        y_train_ch.extend([label] * len(points))

    for i in np.unique(y_train):
        X_class_i = X_train[y_train == i]

        if len(X_class_i) < 3:
            add_class_points(X_class_i, i)
            continue

        try:
            reduced_points = reduce_points_nd_ha_hb_hg(
                X_class_i,
                ha=ha_value,
                hb=hb_value,
                hg=hg,
                min_points=min_points,
                L=L,
                hull_func=roch_vertices
            )

            if len(reduced_points) == 0:
                add_class_points(X_class_i, i)
            else:
                add_class_points(reduced_points, i)
        except Exception:
            add_class_points(X_class_i, i)

    X_new, y_new = np.array(X_train_ch), np.array(y_train_ch)
    time_reduce = time.time() - start_reduce
    print(f"ROCH_v2 reduced training points: {X_new.shape}")

    reduction_rate = 1 - len(X_new) / len(X_train)

    # ── Bước 2: Fit SVM ───────────────────────────────────────
    start_fit = time.time()
    if len(X_new) == 0 or len(np.unique(y_new)) < 2:
        svm.fit(X_train, y_train)
    else:
        svm.fit(X_new, y_new)
    time_fit = time.time() - start_fit


    n_sv = svm.n_support_.sum()
    y_pred = svm.predict(X_test)
    return (
        accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred, zero_division=0),
        recall_score(y_test, y_pred, zero_division=0),
        f1_score(y_test, y_pred, zero_division=0),
        time_reduce,
        time_fit,
        reduction_rate,
        n_sv
    )

def train_by_CCH_v2(X_train, X_test, y_train, y_test, svm, ha=None, hb=None, hg=4, L=0.1, K_input=50, min_points=2):

    # ── Bước 1: Reduce ────────────────────────────────────────
    start_reduce = time.time()
    X_train_ch = []
    y_train_ch = []

    for i in np.unique(y_train):
        X_class_i = X_train[y_train == i]

        if len(X_class_i) < 3:
            X_train_ch.extend(X_class_i)
            y_train_ch.extend([i] * len(X_class_i))
            continue

        try:
            reduced_points = reduce_points_nd_ha_hb_hg(
                X_class_i,
                ha=ha if ha is not None else 4,
                hb=hb if hb is not None else 4,
                hg=hg,
                min_points=min_points,
                K_input=K_input,
                L=L,
                hull_func=cch_vertices_indices
            )

            if len(reduced_points) == 0:
                X_train_ch.extend(X_class_i)
                y_train_ch.extend([i] * len(X_class_i))
            else:
                X_train_ch.extend(reduced_points)
                y_train_ch.extend([i] * len(reduced_points))
        except Exception as e:
            X_train_ch.extend(X_class_i)
            y_train_ch.extend([i] * len(X_class_i))

    X_new, y_new = np.array(X_train_ch), np.array(y_train_ch)
    time_reduce = time.time() - start_reduce
    print(f"CCH_v2 reduced training points: {X_new.shape}, K={K_input}")

    reduction_rate = 1 - len(X_new) / len(X_train)

    # ── Bước 2: Fit SVM ───────────────────────────────────────
    start_fit = time.time()
    if len(X_new) == 0 or len(np.unique(y_new)) < 2:
        svm.fit(X_train, y_train)
    else:
        svm.fit(X_new, y_new)
    time_fit = time.time() - start_fit

    n_sv = svm.n_support_.sum()
    y_pred = svm.predict(X_test)
    return (
        accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred, zero_division=0),
        recall_score(y_test, y_pred, zero_division=0),
        f1_score(y_test, y_pred, zero_division=0),
        time_reduce,
        time_fit,
        reduction_rate,
        n_sv
    )
