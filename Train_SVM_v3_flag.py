'''
Train_SVM_v3_flag.py
====================
Hỗ trợ bật/tắt từng method qua 3 flag:
    run_original, run_roch, run_cch

Tách riêng time_reduce và time_fit cho 3 phương pháp:
    - Original : time_reduce = 0.0  (không có bước reduce)
    - ROCH_v2  : time_reduce = thời gian chạy thuật toán ROCH
    - CCH_v2   : time_reduce = thời gian chạy thuật toán CCH

Yêu cầu Model_SVM trả về tuple:
    train_orginal_high → (acc, pre, rec, f1, time_reduce, time_fit, n_sv)
    train_by_ROCH_v2   → (acc, pre, rec, f1, time_reduce, time_fit, rr, n_sv)
    train_by_CCH_v2    → (acc, pre, rec, f1, time_reduce, time_fit, rr, n_sv)
'''
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from Function.function import *
import pandas as pd
from Model_SVM import *
from sklearn.preprocessing import StandardScaler
from collections import Counter
from imblearn.over_sampling import SMOTE


def balance_with_smote(X_train, y_train):
    class_counts = Counter(y_train)
    if len(class_counts) < 2:
        return X_train, y_train
    min_class = min(class_counts.values())
    max_class = max(class_counts.values())
    imbalance_ratio = min_class / max_class
    if imbalance_ratio < 0.3:
        k_neighbors = min(5, min_class - 1) if min_class > 1 else 1
        if k_neighbors >= 1:
            smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
            X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
            return X_balanced, y_balanced
    return X_train, y_train


def run(
    dataset_name,
    # ── Flag bật/tắt method ───────────────────────────────────
    run_original = True,
    run_roch     = True,
    run_cch      = True,
    # ── SVM params ────────────────────────────────────────────
    C_org=5.0,   gamma_org=2.0,
    C_roch=5.0,  gamma_roch=3.5,
    C_cch=5.0,   gamma_cch=3.5,
    # ── ROCH params ───────────────────────────────────────────
    ha_roch=4, hb_roch=10, hg_roch=12,
    # ── CCH params ────────────────────────────────────────────
    ha_cch=5,  hb_cch=10,  hg_cch=12,
    K_values=[9],
    # ── Chung ─────────────────────────────────────────────────
    L_factor=0.05,
    min_points=2,
    n_runs=50,
):
    if not any([run_original, run_roch, run_cch]):
        raise ValueError(f"[{dataset_name}] Phải bật ít nhất 1 method!")

    # ── Load data ─────────────────────────────────────────────
    project_dir = Path(__file__).resolve().parent
    csv_path = project_dir / 'data' / f'{dataset_name}.csv'
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        processed_path = project_dir / 'data' / f'{dataset_name}_processed.csv'
        if processed_path.exists():
            print(f"⚠ Dùng file thay thế: {processed_path}")
            csv_path = processed_path
        else:
            raise FileNotFoundError(f"Không tìm thấy data: {dataset_name}")

    df     = pd.read_csv(csv_path)
    points = df.iloc[:, :-1].values
    y      = df.iloc[:, -1].values

    # ── In thông tin ──────────────────────────────────────────
    active = []
    if run_original: active.append('Original')
    if run_roch:     active.append('ROCH')
    if run_cch:      active.append('CCH')

    print(f"\n{'='*70}")
    print(f"DATASET  : {dataset_name}  |  Shape={points.shape}  |  n_runs={n_runs}")
    print(f"Methods  : {', '.join(active)}")
    if run_original:
        print(f"Original : C={C_org}, gamma={gamma_org}")
    if run_roch:
        print(f"ROCH     : C={C_roch}, gamma={gamma_roch}, ha={ha_roch}, hb={hb_roch}, hg={hg_roch}")
    if run_cch:
        print(f"CCH      : C={C_cch}, gamma={gamma_cch}, ha={ha_cch}, hb={hb_cch}, hg={hg_cch}, K={K_values}")
    print(f"L={L_factor}, min_points={min_points}")
    print(f"{'='*70}")

    # ── Khởi tạo containers ───────────────────────────────────
    time_reduces    = {}
    time_fits       = {}
    reduction_rates = {}
    accuracys       = {}
    precision       = {}
    recall          = {}
    f1              = {}
    support_vectors = {}

    if run_original:
        time_reduces['timeReduce_Original']  = []
        time_fits['timeFit_Original']        = []
        accuracys['accuracy_Original']       = []
        precision['precision_Original']      = []
        recall['recall_Original']            = []
        f1['f1_Original']                    = []
        support_vectors['sv_Original'] = []

    if run_roch:
        time_reduces['timeReduce_ROCH']      = []
        time_fits['timeFit_ROCH']            = []
        reduction_rates['reduction_ROCH']    = []
        accuracys['accuracy_ROCH']           = []
        precision['precision_ROCH']          = []
        recall['recall_ROCH']                = []
        f1['f1_ROCH']                        = []
        support_vectors['sv_ROCH'] = []

    if run_cch:
        for k in K_values:
            time_reduces[f'timeReduce_CCH{k}']   = []
            time_fits[f'timeFit_CCH{k}']         = []
            reduction_rates[f'reduction_CCH{k}'] = []
            accuracys[f'accuracy_CCH{k}']        = []
            precision[f'precision_CCH{k}']       = []
            recall[f'recall_CCH{k}']             = []
            f1[f'f1_CCH{k}']                     = []
            support_vectors[f'sv_CCH{k}'] = []

    # ── Main loop ─────────────────────────────────────────────
    for i in range(n_runs):
        print(f"\n{'='*60}")
        print(f"Run {i+1}/{n_runs}")
        print(f"{'='*60}")

        X_train, X_test, y_train, y_test = train_test_split(
            points, y, test_size=0.3, random_state=i, stratify=y
        )
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)
        X_train, y_train = balance_with_smote(X_train, y_train)

        print(f"Original size: {points.shape}, Classes: {Counter(y)}")
        print(f"(After sampling): {X_train.shape}, Classes: {Counter(y_train)}")

        # ── Original SVM ──────────────────────────────────────
        if run_original:
            svm_org = SVC(kernel='rbf', C=C_org, gamma=gamma_org)
            acc, pre, rec, f1s, t_reduce, t_fit, n_sv = train_orginal_high(
                X_train, X_test, y_train, y_test, svm_org
            )
            accuracys['accuracy_Original'].append(acc)
            precision['precision_Original'].append(pre)
            recall['recall_Original'].append(rec)
            f1['f1_Original'].append(f1s)
            time_reduces['timeReduce_Original'].append(t_reduce)
            time_fits['timeFit_Original'].append(t_fit)
            support_vectors['sv_Original'].append(n_sv)
            print(f"✓ Original : Acc={acc:.4f} | TimeReduce={t_reduce:.4f}s | TimeFit={t_fit:.4f}s | TimeTotal={t_reduce+t_fit:.4f}s")

        # ── ROCH_v2 ───────────────────────────────────────────
        if run_roch:
            svm_roch = SVC(kernel='rbf', C=C_roch, gamma=gamma_roch)
            acc, pre, rec, f1s, t_reduce, t_fit, rr, n_sv = train_by_ROCH_v2(
                X_train, X_test, y_train, y_test, svm_roch, 
                ha=ha_roch, hb=hb_roch, hg=hg_roch, L=L_factor, min_points=min_points
            )
            accuracys['accuracy_ROCH'].append(acc)
            precision['precision_ROCH'].append(pre)
            recall['recall_ROCH'].append(rec)
            f1['f1_ROCH'].append(f1s)
            time_reduces['timeReduce_ROCH'].append(t_reduce)
            time_fits['timeFit_ROCH'].append(t_fit)
            reduction_rates['reduction_ROCH'].append(rr)
            support_vectors['sv_ROCH'].append(n_sv)
            print(f"✓ ROCH_v2  : Acc={acc:.4f} | TimeReduce={t_reduce:.4f}s | TimeFit={t_fit:.4f}s | TimeTotal={t_reduce+t_fit:.4f}s | Reduction={rr:.4f}")

        # ── CCH_v2 ────────────────────────────────────────────
        if run_cch:
            for k in K_values:
                svm_cch = SVC(kernel='rbf', C=C_cch, gamma=gamma_cch)
                acc, pre, rec, f1s, t_reduce, t_fit, rr, n_sv = train_by_CCH_v2(
                    X_train, X_test, y_train, y_test, svm_cch,
                    ha=ha_cch, hb=hb_cch, hg=hg_cch,
                    L=L_factor, K_input=k, min_points=min_points
                )
                accuracys[f'accuracy_CCH{k}'].append(acc)
                precision[f'precision_CCH{k}'].append(pre)
                recall[f'recall_CCH{k}'].append(rec)
                f1[f'f1_CCH{k}'].append(f1s)
                time_reduces[f'timeReduce_CCH{k}'].append(t_reduce)
                time_fits[f'timeFit_CCH{k}'].append(t_fit)
                reduction_rates[f'reduction_CCH{k}'].append(rr)
                
                support_vectors[f'sv_CCH{k}'].append(n_sv)
                print(f"✓ CCH(K={k:2d}): Acc={acc:.4f} | TimeReduce={t_reduce:.4f}s | TimeFit={t_fit:.4f}s | TimeTotal={t_reduce+t_fit:.4f}s | Reduction={rr:.4f}")

        print(f"Run {i+1}/{n_runs} completed.")

    # ── Tổng hợp kết quả ──────────────────────────────────────
    def stats(lst): return np.mean(lst), np.std(lst)

    rows_raw = []
    rows_fmt = []

    method_specs = []

    if run_original:
        method_specs.append(dict(
            label  = 'Original',
            params = f"C={C_org}, gamma={gamma_org}",
            keys   = {
                'Accuracy'   : ('accuracy_Original',      accuracys),
                'Precision'  : ('precision_Original',     precision),
                'Recall'     : ('recall_Original',        recall),
                'F1'         : ('f1_Original',            f1),
                'TimeReduce' : ('timeReduce_Original',    time_reduces),
                'TimeFit'    : ('timeFit_Original',       time_fits),
                'SupportVector' : ('sv_Original',            support_vectors),
                'Reduction'  : (None, None),   # Original không có Reduction
            }
        ))

    if run_roch:
        method_specs.append(dict(
            label  = 'ROCH_v2',
            params = f"C={C_roch}, gamma={gamma_roch}, ha={ha_roch}, hb={hb_roch}, hg={hg_roch}, L={L_factor}",
            keys   = {
                'Accuracy'   : ('accuracy_ROCH',          accuracys),
                'Precision'  : ('precision_ROCH',         precision),
                'Recall'     : ('recall_ROCH',            recall),
                'F1'         : ('f1_ROCH',                f1),
                'TimeReduce' : ('timeReduce_ROCH',        time_reduces),
                'TimeFit'    : ('timeFit_ROCH',           time_fits),
                'Reduction'  : ('reduction_ROCH',         reduction_rates),
                'SupportVector' : ('sv_ROCH',                support_vectors),
            }
        ))

    if run_cch:
        for k in K_values:
            method_specs.append(dict(
                label  = f'CCH_v2-{k}',
                params = (
                    f"C={C_cch}, gamma={gamma_cch}, ha={ha_cch}, hb={hb_cch}, "
                    f"hg={hg_cch}, K={k}, L={L_factor}"
                ),
                keys   = {
                    'Accuracy'   : (f'accuracy_CCH{k}',       accuracys),
                    'Precision'  : (f'precision_CCH{k}',      precision),
                    'Recall'     : (f'recall_CCH{k}',         recall),
                    'F1'         : (f'f1_CCH{k}',             f1),
                    'TimeReduce' : (f'timeReduce_CCH{k}',     time_reduces),
                    'TimeFit'    : (f'timeFit_CCH{k}',        time_fits),
                    'Reduction'  : (f'reduction_CCH{k}',      reduction_rates),
                    'SupportVector' : (f'sv_CCH{k}',             support_vectors),
                }
            ))

    for spec in method_specs:
        row_raw = {'Dataset': dataset_name, 'Method': spec['label'], 'Params': spec['params'], 'n_runs': n_runs}
        row_fmt = {'Dataset': dataset_name, 'Method': spec['label'], 'Params': spec['params'], 'n_runs': n_runs}

        for metric_name, (key, store) in spec['keys'].items():
            if key is None or store is None:
                row_raw[f'{metric_name}_Mean'] = None
                row_raw[f'{metric_name}_Std']  = None
                row_fmt[metric_name]           = 'N/A'
            else:
                m, s = stats(store[key])
                row_raw[f'{metric_name}_Mean'] = m
                row_raw[f'{metric_name}_Std']  = s
                row_fmt[metric_name]           = f"{m:.4f} ± {s:.4f}"

        # TimeTotal = TimeReduce + TimeFit (tính từ list gốc để std chính xác)
        key_reduce = spec['keys']['TimeReduce'][0]
        key_fit    = spec['keys']['TimeFit'][0]
        totals     = [r + f for r, f in zip(time_reduces[key_reduce], time_fits[key_fit])]
        mt, st     = stats(totals)
        row_raw['TimeTotal_Mean'] = mt
        row_raw['TimeTotal_Std']  = st
        row_fmt['TimeTotal']      = f"{mt:.4f} ± {st:.4f}"

        rows_raw.append(row_raw)
        rows_fmt.append(row_fmt)

    df_raw = pd.DataFrame(rows_raw)
    df_fmt = pd.DataFrame(rows_fmt)

    # ── In ra màn hình ────────────────────────────────────────
    print(f"\n{'='*110}")
    print(f"SUMMARY — {dataset_name.upper()}  (n_runs={n_runs})")
    print(f"{'='*110}")
    print(df_fmt.to_string(index=False))
    print(f"{'='*110}\n")

    # ── Lưu file Excel ────────────────────────────────────────
    results_dir = project_dir / 'results'
    results_dir.mkdir(exist_ok=True)
    suffix    = '_'.join(active).lower()
    xlsx_path = results_dir / f"results_{suffix}_{dataset_name}.xlsx"

    # Gộp raw time_reduce + time_fit thành 1 dict để lưu sheet Times_Raw
    times_combined = {}
    times_combined.update(time_reduces)
    times_combined.update(time_fits)

    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        df_fmt.to_excel(writer, sheet_name='Summary',          index=False)
        df_raw.to_excel(writer, sheet_name='Summary_Detailed', index=False)
        if times_combined:
            pd.DataFrame(times_combined).to_excel(writer, sheet_name='Times_Raw',      index=False)
        if accuracys:
            pd.DataFrame(accuracys).to_excel(writer,           sheet_name='Accuracies_Raw', index=False)
        if precision:
            pd.DataFrame(precision).to_excel(writer,           sheet_name='Precision_Raw',  index=False)
        if recall:
            pd.DataFrame(recall).to_excel(writer,              sheet_name='Recall_Raw',     index=False)
        if f1:
            pd.DataFrame(f1).to_excel(writer,                  sheet_name='F1_Raw',         index=False)
        if reduction_rates:
            pd.DataFrame(reduction_rates).to_excel(writer,     sheet_name='Reduction_Raw',  index=False)

    print(f"✅ Saved → {xlsx_path}")

    return df_raw
