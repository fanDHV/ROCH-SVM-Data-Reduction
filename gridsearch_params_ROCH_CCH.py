import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from Function.function import *
import pandas as pd
from Model_SVM import *
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from collections import Counter
from reduce_dim.reduce_dim_v2 import reduce_points_nd_ha_hb_hg
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"

scaler = StandardScaler()

# Định nghĩa các giá trị để grid search
K_values = [6, 9, 15, 30, 50, 70]
ha_values = [3, 4, 5]
hb_values = [17,18]
#hb_values = [6, 8, 10, 12, 14, 16]
hg_values = [9, 11, 13, 15, 17, 19]

L_factor = 0.05
min_points = 2
n_runs = 50




def validate_h_params(ha, hb, hg):
    """Kiểm tra constraint: ha <= hb <= hg"""
    return ha <= hb <= hg


def balance_with_smote(X_train, y_train):
    class_counts = Counter(y_train)
    if len(class_counts) < 2:
        return X_train, y_train

    min_class = min(class_counts.values())
    max_class = max(class_counts.values())
    imbalance_ratio = min_class / max_class

    if imbalance_ratio < 0.5:
        k_neighbors = min(5, min_class - 1) if min_class > 1 else 1
        if k_neighbors >= 1:
            smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
            X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
            return X_balanced, y_balanced

    return X_train, y_train


def find_best_params(X_train, y_train, cv_folds=3, random_state=42):
    """Tìm hyperparameters tối ưu cho SVM trên tập huấn luyện"""
    param_grid = {
        'C': [6],
        'gamma': [1.2]
    }

    svm = SVC(kernel='rbf', random_state=random_state, class_weight='balanced')

    actual_cv_folds = min(cv_folds, len(np.unique(y_train)), len(y_train) // 2)
    if actual_cv_folds < 2:
        actual_cv_folds = 2

    cv = StratifiedKFold(n_splits=actual_cv_folds, shuffle=True, random_state=random_state)

    grid_search = GridSearchCV(
        svm, param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0,
        return_train_score=True
    )

    try:
        grid_search.fit(X_train, y_train)
        return grid_search.best_params_['C'], grid_search.best_params_['gamma']
    except:
        return 1.0, 'scale'


def get_reduced_training_set_roch(X_train, y_train, ha, hb, hg, L, min_points):
    """Lấy reduced training set sử dụng ROCH"""
    X_train_reduced = []
    y_train_reduced = []

    for i in np.unique(y_train):
        X_class_i = X_train[y_train == i]

        if len(X_class_i) < 3:
            X_train_reduced.extend(X_class_i)
            y_train_reduced.extend([i] * len(X_class_i))
            continue

        try:
            reduced_points = reduce_points_nd_ha_hb_hg(
                X_class_i, ha=ha, hb=hb, hg=hg,
                min_points=min_points, L=L,
                hull_func=roch_vertices
            )

            if len(reduced_points) == 0:
                X_train_reduced.extend(X_class_i)
                y_train_reduced.extend([i] * len(X_class_i))
            else:
                X_train_reduced.extend(reduced_points)
                y_train_reduced.extend([i] * len(reduced_points))
        except Exception as e:
            X_train_reduced.extend(X_class_i)
            y_train_reduced.extend([i] * len(X_class_i))

    return np.array(X_train_reduced), np.array(y_train_reduced)


def get_reduced_training_set_cch(X_train, y_train, ha, hb, hg, L, K_input, min_points):
    """Lấy reduced training set sử dụng CCH"""
    X_train_reduced = []
    y_train_reduced = []

    for i in np.unique(y_train):
        X_class_i = X_train[y_train == i]

        if len(X_class_i) < 3:
            X_train_reduced.extend(X_class_i)
            y_train_reduced.extend([i] * len(X_class_i))
            continue

        try:
            reduced_points = reduce_points_nd_ha_hb_hg(
                X_class_i, ha=ha, hb=hb, hg=hg,
                min_points=min_points, K_input=K_input, L=L,
                hull_func=cch_vertices_indices
            )

            if len(reduced_points) == 0:
                X_train_reduced.extend(X_class_i)
                y_train_reduced.extend([i] * len(X_class_i))
            else:
                X_train_reduced.extend(reduced_points)
                y_train_reduced.extend([i] * len(reduced_points))
        except Exception as e:
            X_train_reduced.extend(X_class_i)
            y_train_reduced.extend([i] * len(X_class_i))

    return np.array(X_train_reduced), np.array(y_train_reduced)


def run(dataset_name):
    import os
    import time as time_module

    # Khởi tạo dictionary để lưu kết quả
    results = {
        'run': [],
        'config': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'time': [],
        'method': [],
        'C': [],
        'gamma': [],
        'ha': [],
        'hb': [],
        'hg': [],
        'K': [],
        'n_training_samples': [],
        'reduction_ratio': [],
        'train_accuracy': [],
        'n_support_vectors': [],
        'time_reduction': [],  # Thời gian chia lưới ha,hb,hg,(K)
        'time_fit': []         # Thời gian fit SVM thuần túy
    }

    csv_path = DATA_DIR / f'{dataset_name}.csv'

    if not csv_path.exists() or csv_path.stat().st_size == 0:
        processed_path = DATA_DIR / f'{dataset_name}_processed.csv'
        if processed_path.exists():
            csv_path = processed_path
        else:
            raise FileNotFoundError(f"Cannot find data file for {dataset_name}")

    df = pd.read_csv(csv_path)
    points = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Đếm số valid configurations
    valid_roch_configs = sum(1 for ha in ha_values for hb in hb_values for hg in hg_values
                             if validate_h_params(ha, hb, hg))
    valid_cch_configs = sum(1 for k in K_values for ha in ha_values for hb in hb_values for hg in hg_values
                            if validate_h_params(ha, hb, hg))

    print(f"\n{'='*80}")
    print(f"Dataset: {dataset_name}")
    print(f"Total Samples: {len(y)} | Features: {points.shape[1]}")
    print(f"Classes: {np.unique(y)} | Distribution: {dict(Counter(y))}")
    print(f"\nGrid Search Parameters:")
    print(f"  K_values: {K_values}")
    print(f"  ha_values: {ha_values}")
    print(f"  hb_values: {hb_values}")
    print(f"  hg_values: {hg_values}")
    print(f"  Constraint: ha <= hb <= hg")
    print(f"  L_factor: {L_factor}")
    print(f"  Number of runs: {n_runs}")
    print(f"\nValid Configurations:")
    print(f"  ROCH_v2: {valid_roch_configs} configs")
    print(f"  CCH_v2:  {valid_cch_configs} configs")
    print(f"{'='*80}\n")

    for run_idx in range(n_runs):
        print(f"\n{'='*80}")
        print(f"RUN {run_idx+1}/{n_runs}")
        print(f"{'='*80}\n")

        # BƯỚC 1: Split data
        X_train, X_test, y_train, y_test = train_test_split(
            points, y, test_size=0.3, random_state=run_idx * 7 + 13, stratify=y
        )

        # BƯỚC 2: Standardization
        scaler_instance = StandardScaler()
        X_train_scaled = scaler_instance.fit_transform(X_train)
        X_test_scaled = scaler_instance.transform(X_test)

        # BƯỚC 3: SMOTE
        X_train_balanced, y_train_balanced = balance_with_smote(X_train_scaled, y_train)

        original_size = len(X_train_balanced)
        '''
        # ==================== ORIGINAL SVM (Grid Search trên C và gamma) ====================
        print("Training Original SVM (with grid search on C and gamma)...")

                
        
        C_org, gamma_org = find_best_params(
            X_train_balanced, y_train_balanced,
            cv_folds=5,
            random_state=42 + run_idx
        )
        
        

        svm_org = SVC(kernel='rbf', C=C_org, gamma=gamma_org, class_weight='balanced')

        start_train = time_module.time()
        svm_org.fit(X_train_balanced, y_train_balanced)
        run_time = time_module.time() - start_train

        y_pred_train = svm_org.predict(X_train_balanced)
        y_pred_test = svm_org.predict(X_test_scaled)

        accuracy_org = accuracy_score(y_test, y_pred_test)
        train_accuracy_org = accuracy_score(y_train_balanced, y_pred_train)
        precision_org = precision_score(y_test, y_pred_test, zero_division=0)
        recall_org = recall_score(y_test, y_pred_test, zero_division=0)
        f1_org = f1_score(y_test, y_pred_test, zero_division=0)
        n_support = len(svm_org.support_vectors_) if hasattr(svm_org, 'support_vectors_') else 0

        results['run'].append(run_idx)
        results['config'].append(f'Original_C{C_org}_g{gamma_org}')
        results['accuracy'].append(accuracy_org)
        results['train_accuracy'].append(train_accuracy_org)
        results['precision'].append(precision_org)
        results['recall'].append(recall_org)
        results['f1'].append(f1_org)
        results['time'].append(run_time)
        results['method'].append('Original')
        results['C'].append(C_org)
        results['gamma'].append(str(gamma_org))
        results['ha'].append(None)
        results['hb'].append(None)
        results['hg'].append(None)
        results['K'].append(None)
        results['n_training_samples'].append(len(X_train_balanced))
        results['reduction_ratio'].append(0.0)
        results['n_support_vectors'].append(n_support)
        results['time_reduction'].append(0.0)
        results['time_fit'].append(run_time)

        print(f"✓ Original (C={C_org}, gamma={gamma_org}): "
              f"Test Acc={accuracy_org:.4f}, Train Acc={train_accuracy_org:.4f}, "
              f"Time={run_time:.2f}s, SVs={n_support}\n")
	    '''
        # ==================== ROCH_v2 GRID SEARCH ====================
        print(f"Grid Search for ROCH_v2 ({valid_roch_configs} valid configurations)...")
        config_count = 0
        skipped_count = 0

        for ha in ha_values:
            for hb in hb_values:
                for hg in hg_values:
                    if not validate_h_params(ha, hb, hg):
                        skipped_count += 1
                        continue

                    config_count += 1
                    print(f"  [{config_count}/{valid_roch_configs}] ha={ha}, hb={hb}, hg={hg}...", end=' ')

                    # ĐO THỜI GIAN TỪ ĐÂY: bao gồm cả thời gian reduction
                    start_train = time_module.time()

                    # BƯỚC 1: Reduction (chia lưới ha, hb, hg)
                    X_train_reduced, y_train_reduced = get_reduced_training_set_roch(
                        X_train_balanced, y_train_balanced,
                        ha, hb, hg, L_factor, min_points
                    )
                    time_reduction = time_module.time() - start_train

                    if len(X_train_reduced) < 10 or len(np.unique(y_train_reduced)) < 2:
                        print(f"Skipped (too few samples: {len(X_train_reduced)})")
                        continue

                    C_roch, gamma_roch = find_best_params(
                        X_train_reduced, y_train_reduced,
                        cv_folds=5,
                        random_state=100 + run_idx + config_count
                    )

                    # BƯỚC 2: SVM fit
                    start_fit = time_module.time()
                    svm_roch = SVC(kernel='rbf', C=C_roch, gamma=gamma_roch, class_weight='balanced')
                    svm_roch.fit(X_train_reduced, y_train_reduced)
                    time_fit = time_module.time() - start_fit

                    run_time = time_reduction + time_fit

                    y_pred_train = svm_roch.predict(X_train_reduced)
                    y_pred_test = svm_roch.predict(X_test_scaled)

                    accuracy_roch = accuracy_score(y_test, y_pred_test)
                    train_accuracy_roch = accuracy_score(y_train_reduced, y_pred_train)
                    precision_roch = precision_score(y_test, y_pred_test, zero_division=0)
                    recall_roch = recall_score(y_test, y_pred_test, zero_division=0)
                    f1_roch = f1_score(y_test, y_pred_test, zero_division=0)
                    n_support = len(svm_roch.support_vectors_) if hasattr(svm_roch, 'support_vectors_') else 0
                    reduction_ratio = 1 - (len(X_train_reduced) / original_size)

                    results['run'].append(run_idx)
                    results['config'].append(f'ROCH_ha{ha}_hb{hb}_hg{hg}_C{C_roch}_g{gamma_roch}')
                    results['accuracy'].append(accuracy_roch)
                    results['train_accuracy'].append(train_accuracy_roch)
                    results['precision'].append(precision_roch)
                    results['recall'].append(recall_roch)
                    results['f1'].append(f1_roch)
                    results['time'].append(run_time)
                    results['method'].append('ROCH_v2')
                    results['C'].append(C_roch)
                    results['gamma'].append(str(gamma_roch))
                    results['ha'].append(ha)
                    results['hb'].append(hb)
                    results['hg'].append(hg)
                    results['K'].append(None)
                    results['n_training_samples'].append(len(X_train_reduced))
                    results['reduction_ratio'].append(reduction_ratio)
                    results['n_support_vectors'].append(n_support)
                    results['time_reduction'].append(time_reduction)
                    results['time_fit'].append(time_fit)

                    print(f"Acc={accuracy_roch:.4f}, C={C_roch}, gamma={gamma_roch}, "
                          f"Samples={len(X_train_reduced)} ({reduction_ratio:.1%}), "
                          f"Time={run_time:.3f}s [reduce={time_reduction:.3f}s + fit={time_fit:.3f}s]")

        if skipped_count > 0:
            print(f"  Skipped {skipped_count} invalid configs (violating ha <= hb <= hg)")

        # ==================== CCH_v2 GRID SEARCH ====================
        print(f"\nGrid Search for CCH_v2 ({valid_cch_configs} valid configurations)...")
        config_count = 0
        skipped_count = 0

        for k in K_values:
            for ha in ha_values:
                for hb in hb_values:
                    for hg in hg_values:
                        if not validate_h_params(ha, hb, hg):
                            skipped_count += 1
                            continue

                        config_count += 1
                        print(f"  [{config_count}/{valid_cch_configs}] K={k}, ha={ha}, hb={hb}, hg={hg}...", end=' ')

                        # ĐO THỜI GIAN TỪ ĐÂY
                        start_train = time_module.time()

                        # BƯỚC 1: Reduction (chia lưới ha, hb, hg, K)
                        X_train_reduced, y_train_reduced = get_reduced_training_set_cch(
                            X_train_balanced, y_train_balanced,
                            ha, hb, hg, L_factor, k, min_points
                        )
                        time_reduction = time_module.time() - start_train

                        if len(X_train_reduced) < 10 or len(np.unique(y_train_reduced)) < 2:
                            print(f"Skipped (too few samples: {len(X_train_reduced)})")
                            continue

                        C_cch, gamma_cch = find_best_params(
                            X_train_reduced, y_train_reduced,
                            cv_folds=5,
                            random_state=200 + run_idx + config_count
                        )

                        # BƯỚC 2: SVM fit
                        start_fit = time_module.time()
                        svm_cch = SVC(kernel='rbf', C=C_cch, gamma=gamma_cch, class_weight='balanced')
                        svm_cch.fit(X_train_reduced, y_train_reduced)
                        time_fit = time_module.time() - start_fit

                        run_time = time_reduction + time_fit

                        y_pred_train = svm_cch.predict(X_train_reduced)
                        y_pred_test = svm_cch.predict(X_test_scaled)

                        accuracy_cch = accuracy_score(y_test, y_pred_test)
                        train_accuracy_cch = accuracy_score(y_train_reduced, y_pred_train)
                        precision_cch = precision_score(y_test, y_pred_test, zero_division=0)
                        recall_cch = recall_score(y_test, y_pred_test, zero_division=0)
                        f1_cch = f1_score(y_test, y_pred_test, zero_division=0)
                        n_support = len(svm_cch.support_vectors_) if hasattr(svm_cch, 'support_vectors_') else 0
                        reduction_ratio = 1 - (len(X_train_reduced) / original_size)

                        results['run'].append(run_idx)
                        results['config'].append(f'CCH_K{k}_ha{ha}_hb{hb}_hg{hg}_C{C_cch}_g{gamma_cch}')
                        results['accuracy'].append(accuracy_cch)
                        results['train_accuracy'].append(train_accuracy_cch)
                        results['precision'].append(precision_cch)
                        results['recall'].append(recall_cch)
                        results['f1'].append(f1_cch)
                        results['time'].append(run_time)
                        results['method'].append('CCH_v2')
                        results['C'].append(C_cch)
                        results['gamma'].append(str(gamma_cch))
                        results['ha'].append(ha)
                        results['hb'].append(hb)
                        results['hg'].append(hg)
                        results['K'].append(k)
                        results['n_training_samples'].append(len(X_train_reduced))
                        results['reduction_ratio'].append(reduction_ratio)
                        results['n_support_vectors'].append(n_support)
                        results['time_reduction'].append(time_reduction)
                        results['time_fit'].append(time_fit)

                        print(f"Acc={accuracy_cch:.4f}, C={C_cch}, gamma={gamma_cch}, "
                              f"Samples={len(X_train_reduced)} ({reduction_ratio:.1%}), "
                              f"Time={run_time:.3f}s [reduce={time_reduction:.3f}s + fit={time_fit:.3f}s]")

        if skipped_count > 0:
            print(f"  Skipped {skipped_count} invalid configs (violating ha <= hb <= hg)")

    # ==================== PHÂN TÍCH KẾT QUẢ (TRUNG BÌNH QUA CÁC RUNS) ====================
    df_results = pd.DataFrame(results)

    print(f"\n{'='*80}")
    print(f"FINAL RESULTS SUMMARY (AVERAGED OVER {n_runs} RUNS)")
    print(f"{'='*80}\n")
    '''
    # ========== Original SVM (mỗi cặp C, gamma có kết quả riêng) ==========
    df_org = df_results[df_results['method'] == 'Original']
    if len(df_org) > 0:
        org_grouped = df_org.groupby(['C', 'gamma']).agg(
            accuracy_mean=('accuracy', 'mean'),
            accuracy_std=('accuracy', 'std'),
            train_accuracy=('train_accuracy', 'mean'),
            precision=('precision', 'mean'),
            recall=('recall', 'mean'),
            f1=('f1', 'mean'),
            time=('time', 'mean'),
            time_std=('time', 'std'),
            n_samples=('n_training_samples', 'mean'),
            n_support_vectors=('n_support_vectors', 'mean'),
        ).reset_index()

        best_org = org_grouped.loc[org_grouped['accuracy_mean'].idxmax()]

        print("=== Best Original SVM Configuration (Averaged over runs) ===")
        print(f"Config: C={best_org['C']}, gamma={best_org['gamma']}")
        print(f"Test Accuracy:   {best_org['accuracy_mean']:.6f} ± {best_org['accuracy_std']:.6f}")
        print(f"Train Accuracy:  {best_org['train_accuracy']:.6f}")
        print(f"Precision:       {best_org['precision']:.6f}")
        print(f"Recall:          {best_org['recall']:.6f}")
        print(f"F1:              {best_org['f1']:.6f}")
        print(f"Time:            {best_org['time']:.4f}s ± {best_org['time_std']:.4f}s")
        print(f"Samples:         {int(best_org['n_samples'])}")
        print(f"Support Vectors: {int(best_org['n_support_vectors'])}")
        print()

        print("All Original SVM Configurations (by average accuracy):")
        top_org = org_grouped.sort_values('accuracy_mean', ascending=False)[
            ['C', 'gamma', 'accuracy_mean', 'accuracy_std', 'n_samples', 'time']
        ]
        top_org.columns = ['C', 'gamma', 'acc_mean', 'acc_std', 'samples', 'time']
        print(top_org.to_string(index=False))
        print()
    '''
    # ========== Best ROCH_v2 (mỗi cặp ha, hb, hg, C, gamma có kết quả riêng) ==========
    df_roch = df_results[df_results['method'] == 'ROCH_v2']
    if len(df_roch) > 0:
        roch_grouped = df_roch.groupby(['ha', 'hb', 'hg', 'C', 'gamma']).agg(
            accuracy_mean=('accuracy', 'mean'),
            accuracy_std=('accuracy', 'std'),
            train_accuracy=('train_accuracy', 'mean'),
            precision=('precision', 'mean'),
            recall=('recall', 'mean'),
            f1=('f1', 'mean'),
            time=('time', 'mean'),
            time_std=('time', 'std'),
            time_reduction=('time_reduction', 'mean'),
            time_fit=('time_fit', 'mean'),
            n_samples=('n_training_samples', 'mean'),
            reduction_ratio=('reduction_ratio', 'mean'),
            reduction_std=('reduction_ratio', 'std'),
            n_support_vectors=('n_support_vectors', 'mean'),
        ).reset_index()

        best_roch = roch_grouped.loc[roch_grouped['accuracy_mean'].idxmax()]

        print("=== Best ROCH_v2 Configuration (Averaged over runs) ===")
        print(f"Config: ha={int(best_roch['ha'])}, hb={int(best_roch['hb'])}, hg={int(best_roch['hg'])}, "
              f"C={best_roch['C']}, gamma={best_roch['gamma']}")
        print(f"Test Accuracy:   {best_roch['accuracy_mean']:.6f} ± {best_roch['accuracy_std']:.6f}")
        print(f"Train Accuracy:  {best_roch['train_accuracy']:.6f}")
        print(f"Precision:       {best_roch['precision']:.6f}")
        print(f"Recall:          {best_roch['recall']:.6f}")
        print(f"F1:              {best_roch['f1']:.6f}")
        print(f"Time (total):    {best_roch['time']:.4f}s  "
              f"[reduce={best_roch['time_reduction']:.4f}s + fit={best_roch['time_fit']:.4f}s]")
        print(f"Samples:         {int(best_roch['n_samples'])} ({best_roch['reduction_ratio']:.1%} of original)")
        print(f"Support Vectors: {int(best_roch['n_support_vectors'])}")
        print(f"Reduction:       {best_roch['reduction_ratio']:.4f} ± {best_roch['reduction_std']:.4f}")
        print()

        print("Top 5 ROCH_v2 Configurations (by average accuracy):")
        top_roch = roch_grouped.nlargest(5, 'accuracy_mean')[
            ['ha', 'hb', 'hg', 'C', 'gamma', 'accuracy_mean', 'accuracy_std', 'n_samples', 'reduction_ratio', 'time']
        ]
        top_roch.columns = ['ha', 'hb', 'hg', 'C', 'gamma', 'acc_mean', 'acc_std', 'samples', 'reduction', 'time']
        print(top_roch.to_string(index=False))
        print()

    # ========== Best CCH_v2 (mỗi cặp K, ha, hb, hg, C, gamma có kết quả riêng) ==========
    df_cch = df_results[df_results['method'] == 'CCH_v2']
    if len(df_cch) > 0:
        cch_grouped = df_cch.groupby(['K', 'ha', 'hb', 'hg', 'C', 'gamma']).agg(
            accuracy_mean=('accuracy', 'mean'),
            accuracy_std=('accuracy', 'std'),
            train_accuracy=('train_accuracy', 'mean'),
            precision=('precision', 'mean'),
            recall=('recall', 'mean'),
            f1=('f1', 'mean'),
            time=('time', 'mean'),
            time_std=('time', 'std'),
            time_reduction=('time_reduction', 'mean'),
            time_fit=('time_fit', 'mean'),
            n_samples=('n_training_samples', 'mean'),
            reduction_ratio=('reduction_ratio', 'mean'),
            reduction_std=('reduction_ratio', 'std'),
            n_support_vectors=('n_support_vectors', 'mean'),
        ).reset_index()

        best_cch = cch_grouped.loc[cch_grouped['accuracy_mean'].idxmax()]

        print("=== Best CCH_v2 Configuration (Averaged over runs) ===")
        print(f"Config: K={int(best_cch['K'])}, ha={int(best_cch['ha'])}, hb={int(best_cch['hb'])}, "
              f"hg={int(best_cch['hg'])}, C={best_cch['C']}, gamma={best_cch['gamma']}")
        print(f"Test Accuracy:   {best_cch['accuracy_mean']:.6f} ± {best_cch['accuracy_std']:.6f}")
        print(f"Train Accuracy:  {best_cch['train_accuracy']:.6f}")
        print(f"Precision:       {best_cch['precision']:.6f}")
        print(f"Recall:          {best_cch['recall']:.6f}")
        print(f"F1:              {best_cch['f1']:.6f}")
        print(f"Time (total):    {best_cch['time']:.4f}s  "
              f"[reduce={best_cch['time_reduction']:.4f}s + fit={best_cch['time_fit']:.4f}s]")
        print(f"Samples:         {int(best_cch['n_samples'])} ({best_cch['reduction_ratio']:.1%} of original)")
        print(f"Support Vectors: {int(best_cch['n_support_vectors'])}")
        print(f"Reduction:       {best_cch['reduction_ratio']:.4f} ± {best_cch['reduction_std']:.4f}")
        print()

        print("Top 5 CCH_v2 Configurations (by average accuracy):")
        top_cch = cch_grouped.nlargest(5, 'accuracy_mean')[
            ['K', 'ha', 'hb', 'hg', 'C', 'gamma', 'accuracy_mean', 'accuracy_std', 'n_samples', 'reduction_ratio', 'time']
        ]
        top_cch.columns = ['K', 'ha', 'hb', 'hg', 'C', 'gamma', 'acc_mean', 'acc_std', 'samples', 'reduction', 'time']
        print(top_cch.to_string(index=False))
        print()

    # ==================== SO SÁNH CÁC PHƯƠNG PHÁP ====================
    print(f"{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}\n")

    comparison_data = []
    '''
    # Best Original
    if len(df_org) > 0:
        comparison_data.append({
            'Method': f"Original (C={best_org['C']}, gamma={best_org['gamma']})",
            'Test Accuracy': f"{best_org['accuracy_mean']:.4f}±{best_org['accuracy_std']:.4f}",
            'Samples': f"{int(best_org['n_samples'])}",
            'Reduction': "0.0%",
            'Time_total': f"{best_org['time']:.3f}s",
            'Time_reduce': "0.000s",
            'Time_fit': f"{best_org['time']:.3f}s"
        })
    '''
    # Best ROCH
    if len(df_roch) > 0:
        comparison_data.append({
            'Method': f"ROCH (ha={int(best_roch['ha'])},hb={int(best_roch['hb'])},hg={int(best_roch['hg'])},"
                      f"C={best_roch['C']},gamma={best_roch['gamma']})",
            'Test Accuracy': f"{best_roch['accuracy_mean']:.4f}±{best_roch['accuracy_std']:.4f}",
            'Samples': f"{int(best_roch['n_samples'])}",
            'Reduction': f"{best_roch['reduction_ratio']:.1%}",
            'Time_total': f"{best_roch['time']:.3f}s",
            'Time_reduce': f"{best_roch['time_reduction']:.3f}s",
            'Time_fit': f"{best_roch['time_fit']:.3f}s"
        })

    # Best CCH
    if len(df_cch) > 0:
        comparison_data.append({
            'Method': f"CCH (K={int(best_cch['K'])},ha={int(best_cch['ha'])},hb={int(best_cch['hb'])},"
                      f"hg={int(best_cch['hg'])},C={best_cch['C']},gamma={best_cch['gamma']})",
            'Test Accuracy': f"{best_cch['accuracy_mean']:.4f}±{best_cch['accuracy_std']:.4f}",
            'Samples': f"{int(best_cch['n_samples'])}",
            'Reduction': f"{best_cch['reduction_ratio']:.1%}",
            'Time_total': f"{best_cch['time']:.3f}s",
            'Time_reduce': f"{best_cch['time_reduction']:.3f}s",
            'Time_fit': f"{best_cch['time_fit']:.3f}s"
        })

    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    print()

    # ==================== LƯU KẾT QUẢ ====================
    RESULTS_DIR.mkdir(exist_ok=True)

    with pd.ExcelWriter(RESULTS_DIR / f'grid_search_results_{dataset_name}.xlsx') as writer:
        # All results
        df_results.to_excel(writer, sheet_name='All_Results', index=False)
        '''
        # Original results
        if len(df_org) > 0:
            df_org.to_excel(writer, sheet_name='Original_Details', index=False)
            org_grouped.to_excel(writer, sheet_name='Original_Averaged', index=False)
        '''
        # ROCH results
        if len(df_roch) > 0:
            df_roch.to_excel(writer, sheet_name='ROCH_v2_Details', index=False)
            roch_grouped.to_excel(writer, sheet_name='ROCH_v2_Averaged', index=False)

        # CCH results
        if len(df_cch) > 0:
            df_cch.to_excel(writer, sheet_name='CCH_v2_Details', index=False)
            cch_grouped.to_excel(writer, sheet_name='CCH_v2_Averaged', index=False)

        # Summary
        comparison_df.to_excel(writer, sheet_name='Comparison', index=False)

    # Lưu text file
    with (RESULTS_DIR / f"grid_search_results_{dataset_name}.txt").open("w", encoding='utf-8') as f:
        f.write(f"{'='*80}\n")
        f.write(f"GRID SEARCH RESULTS FOR: {dataset_name}\n")
        f.write(f"Averaged over {n_runs} runs\n")
        f.write(f"{'='*80}\n\n")
        '''
        if len(df_org) > 0:
            f.write("=== Best Original SVM ===\n")
            f.write(f"Config: C={best_org['C']}, gamma={best_org['gamma']}\n")
            f.write(f"Test Accuracy: {best_org['accuracy_mean']:.6f} ± {best_org['accuracy_std']:.6f}\n")
            f.write(f"Time: {best_org['time']:.4f}s ± {best_org['time_std']:.4f}s\n\n")
        '''
        if len(df_roch) > 0:
            f.write("=== Best ROCH_v2 Configuration ===\n")
            f.write(f"Config: ha={int(best_roch['ha'])}, hb={int(best_roch['hb'])}, hg={int(best_roch['hg'])}, "
                    f"C={best_roch['C']}, gamma={best_roch['gamma']}\n")
            f.write(f"Test Accuracy: {best_roch['accuracy_mean']:.6f} ± {best_roch['accuracy_std']:.6f}\n")
            f.write(f"Samples: {int(best_roch['n_samples'])} ({best_roch['reduction_ratio']:.1%})\n")
            f.write(f"Time: {best_roch['time']:.4f}s\n\n")

        if len(df_cch) > 0:
            f.write("=== Best CCH_v2 Configuration ===\n")
            f.write(f"Config: K={int(best_cch['K'])}, ha={int(best_cch['ha'])}, hb={int(best_cch['hb'])}, "
                    f"hg={int(best_cch['hg'])}, C={best_cch['C']}, gamma={best_cch['gamma']}\n")
            f.write(f"Test Accuracy: {best_cch['accuracy_mean']:.6f} ± {best_cch['accuracy_std']:.6f}\n")
            f.write(f"Samples: {int(best_cch['n_samples'])} ({best_cch['reduction_ratio']:.1%})\n")
            f.write(f"Time: {best_cch['time']:.4f}s\n\n")

        f.write("=== Comparison ===\n")
        f.write(comparison_df.to_string(index=False))

    print(f"\n✓ Saved: Results/grid_search_results_{dataset_name}.xlsx")
    print(f"✓ Saved: Results/grid_search_results_{dataset_name}.txt\n")


if __name__ == "__main__":

    datasets = [
    'Balls_3D100'
    #'magic_gamma',
    #'pima_diabetes',
    #'ionosphere',
    #'heart_statlog',
    #'four_class',
    #'cross',
    #'rotated_cross',
    #'breast_cancer',
    #'haberman',
        # 'synthetic_highdim_200k',
        # 'creditcard_fraud'
        # 'credit_scoring',
        # 'adult',
        # 'sampleEntry'
    ]

    for dataset in datasets:
        run(dataset)
