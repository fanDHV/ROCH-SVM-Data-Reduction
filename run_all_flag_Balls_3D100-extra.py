"""
run_all.py
==========
Batch runner — có thể bật/tắt từng method (Original, ROCH, CCH)
theo từng dataset riêng biệt.

Cách chạy:
    python run_all.py
"""

import os
import time
import pandas as pd
from datetime import datetime
from pathlib import Path
from Train_SVM_v3_flag import run


# ============================================================
#  ⚙️  KHAI BÁO THAM SỐ TỪNG DATASET — CHỈ SỬA Ở ĐÂY
#
#  Dùng 3 flag để bật/tắt method:
#    run_original = True/False
#    run_roch     = True/False
#    run_cch      = True/False
#
#  Ví dụ chỉ chạy ROCH:
#    run_original=False, run_roch=True, run_cch=False
# ============================================================
DATASET_CONFIGS = [

 # =====================================================
# Baseline

dict(
    dataset_name='Balls_3D100',
    n_runs=20,
    run_original=True, run_roch=True, run_cch=True,
    C_org=8.0, gamma_org=2.0, 
    C_roch=8.0, gamma_roch=2,
    ha_roch=3, hb_roch=16, hg_roch=16, 
    C_cch=8.0, gamma_cch=2,
    ha_cch=3, hb_cch=16, hg_cch=16, K_values=[6],

    L_factor=0.05,
    min_points=2,
),



#CCH – Effect of K

dict(
    dataset_name='Balls_3D100',
    n_runs=20,
    run_original=False, run_roch=False, run_cch=True,

    C_cch=8.0, gamma_cch=2,
    ha_cch=3, hb_cch=16, hg_cch=16,
    K_values=[9],

    L_factor=0.05,
    min_points=2,
),

dict(
    dataset_name='Balls_3D100',
    n_runs=20,
    run_original=False, run_roch=False, run_cch=True,

    C_cch=8.0, gamma_cch=2,
    ha_cch=3, hb_cch=16, hg_cch=16,
    K_values=[12],

    L_factor=0.05,
    min_points=2,
),

dict(
    dataset_name='Balls_3D100',
    n_runs=20,
    run_original=False, run_roch=False, run_cch=True,

    C_cch=8.0, gamma_cch=2,
    ha_cch=3, hb_cch=16, hg_cch=16,
    K_values=[15],

    L_factor=0.05,
    min_points=2,
),

dict(
    dataset_name='Balls_3D100',
    n_runs=20,
    run_original=False, run_roch=False, run_cch=True,

    C_cch=8.0, gamma_cch=2,
    ha_cch=3, hb_cch=16, hg_cch=16,
    K_values=[30],

    L_factor=0.05,
    min_points=2,
),

dict(
    dataset_name='Balls_3D100',
    n_runs=20,
    run_original=False, run_roch=False, run_cch=True,

    C_cch=8.0, gamma_cch=2,
    ha_cch=3, hb_cch=16, hg_cch=16,
    K_values=[50],

    L_factor=0.05,
    min_points=2,
),

#CCH – Effect of hg
# hg = 17,18,19,20

dict(
    dataset_name='Balls_3D100',
    n_runs=20,
    run_original=False, run_roch=False, run_cch=True,

    C_cch=8.0, gamma_cch=2,
    ha_cch=3, hb_cch=16, hg_cch=17,
    K_values=[6],

    L_factor=0.05,
    min_points=2,
),

dict(
    dataset_name='Balls_3D100',
    n_runs=20,
    run_original=False, run_roch=False, run_cch=True,

    C_cch=8.0, gamma_cch=2,
    ha_cch=3, hb_cch=16, hg_cch=18,
    K_values=[6],

    L_factor=0.05,
    min_points=2,
),

dict(
    dataset_name='Balls_3D100',
    n_runs=20,
    run_original=False, run_roch=False, run_cch=True,

    C_cch=8.0, gamma_cch=2,
    ha_cch=3, hb_cch=16, hg_cch=19,
    K_values=[6],

    L_factor=0.05,
    min_points=2,
),

dict(
    dataset_name='Balls_3D100',
    n_runs=20,
    run_original=False, run_roch=False, run_cch=True,

    C_cch=8.0, gamma_cch=2,
    ha_cch=3, hb_cch=16, hg_cch=20,
    K_values=[6],

    L_factor=0.05,
    min_points=2,
),

# CCH – Effect of hb và ha

# =====================================================
# CCH - Effect of hb
# =====================================================

dict(
    dataset_name='Balls_3D100',
    n_runs=20,
    run_original=False, run_roch=False, run_cch=True,

    C_cch=8.0, gamma_cch=2,
    ha_cch=3, hb_cch=15, hg_cch=16, K_values=[6],

    L_factor=0.05,
    min_points=2,
),

dict(
    dataset_name='Balls_3D100',
    n_runs=20,
    run_original=False, run_roch=False, run_cch=True,

    C_cch=8.0, gamma_cch=2,
    ha_cch=3, hb_cch=14, hg_cch=16, K_values=[6],

    L_factor=0.05,
    min_points=2,
),

dict(
    dataset_name='Balls_3D100',
    n_runs=20,
    run_original=False, run_roch=False, run_cch=True,

    C_cch=8.0, gamma_cch=2,
    ha_cch=3, hb_cch=13, hg_cch=16, K_values=[6],

    L_factor=0.05,
    min_points=2,
),

dict(
    dataset_name='Balls_3D100',
    n_runs=20,
    run_original=False, run_roch=False, run_cch=True,

    C_cch=8.0, gamma_cch=2,
    ha_cch=3, hb_cch=12, hg_cch=16, K_values=[6],

    L_factor=0.05,
    min_points=2,
),

# =====================================================
# CCH - Effect of ha
# =====================================================

dict(
    dataset_name='Balls_3D100',
    n_runs=20,
    run_original=False, run_roch=False, run_cch=True,

    C_cch=8.0, gamma_cch=2,
    ha_cch=4, hb_cch=16, hg_cch=16, K_values=[6],

    L_factor=0.05,
    min_points=2,
),

dict(
    dataset_name='Balls_3D100',
    n_runs=20,
    run_original=False, run_roch=False, run_cch=True,

    C_cch=8.0, gamma_cch=2,
    ha_cch=5, hb_cch=16, hg_cch=16, K_values=[6],

    L_factor=0.05,
    min_points=2,
),

dict(
    dataset_name='Balls_3D100',
    n_runs=20,
    run_original=False, run_roch=False, run_cch=True,

    C_cch=8.0, gamma_cch=2,
    ha_cch=2, hb_cch=16, hg_cch=16, K_values=[6],

    L_factor=0.05,
    min_points=2,
),

]







# ── Đường dẫn file Excel tổng hợp ────────────────────────────
COMBINED_OUTPUT = (
    Path(__file__).resolve().parent
    / 'results'
    / 'combined_all_datasets_Balls_3D100-extra_too.xlsx'
)
# ============================================================


def save_combined_excel(all_frames: list, output_path: str | Path):
    """Gộp tất cả kết quả vào 1 file Excel nhiều sheet."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_all = pd.concat(all_frames, ignore_index=True)

    # Xác định các metric có trong kết quả (Reduction chỉ có ở ROCH/CCH)
    base_cols  = ['Dataset', 'Method', 'Params', 'n_runs']
    metrics    = ['Accuracy', 'Precision', 'Recall', 'F1', 'Time']
    has_reduc  = 'Reduction_Mean' in df_all.columns
    if has_reduc:
        metrics.append('Reduction')
    if 'SupportVector_Mean' in df_all.columns:
        metrics.append('SupportVector')

    cols_fmt = base_cols + metrics

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:

        # Sheet 1: mean ± std
        df_fmt = df_all.copy()
        for metric in metrics:
            mean_col = f'{metric}_Mean'
            std_col  = f'{metric}_Std'
            if mean_col in df_fmt.columns:
                df_fmt[metric] = df_fmt.apply(
                    lambda r, mc=mean_col, sc=std_col:
                        'N/A' if pd.isna(r[mc])
                        else f"{r[mc]:.4f} ± {r[sc]:.4f}",
                    axis=1
                )
        # Chỉ giữ cols thực sự tồn tại
        cols_exist = [c for c in cols_fmt if c in df_fmt.columns]
        df_fmt[cols_exist].to_excel(writer, sheet_name='All_Formatted', index=False)

        # Sheet 2: số thô
        df_all.to_excel(writer, sheet_name='All_Raw', index=False)

        # Sheet riêng mỗi dataset
        for ds_name, grp in df_all.groupby('Dataset'):
            grp_fmt = grp.copy()
            for metric in metrics:
                mean_col = f'{metric}_Mean'
                std_col  = f'{metric}_Std'
                if mean_col in grp_fmt.columns:
                    grp_fmt[metric] = grp_fmt.apply(
                        lambda r, mc=mean_col, sc=std_col:
                            'N/A' if pd.isna(r[mc])
                            else f"{r[mc]:.4f} ± {r[sc]:.4f}",
                        axis=1
                    )
            cols_exist = [c for c in cols_fmt if c in grp_fmt.columns]
            sheet = str(ds_name)[:31]
            grp_fmt[cols_exist].to_excel(writer, sheet_name=sheet, index=False)

    print(f"\n{'='*70}")
    print(f"✅ File Excel tổng hợp đã lưu → {output_path}")
    print(f"   Sheets: All_Formatted | All_Raw | (1 sheet/dataset)")
    print(f"{'='*70}")


# ============================================================
if __name__ == '__main__':

    print("=" * 70)
    print("  BATCH RUNNER — SVM Multi-Dataset")
    print(f"  Số dataset : {len(DATASET_CONFIGS)}")
    print(f"  Output     : {COMBINED_OUTPUT}")
    print(f"  Bắt đầu   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    all_frames  = []
    run_summary = []
    t_global    = time.time()

    for idx, cfg in enumerate(DATASET_CONFIGS, 1):
        ds_name = cfg['dataset_name']

        # In thông tin method nào được bật
        flags = []
        if cfg.get('run_original', True):  flags.append('Original')
        if cfg.get('run_roch',     True):  flags.append('ROCH')
        if cfg.get('run_cch',      True):  flags.append('CCH')
        print(f"\n[{idx}/{len(DATASET_CONFIGS)}] ▶  {ds_name}  |  Methods: {', '.join(flags) if flags else '⚠ Không có method nào được bật!'}")

        if not flags:
            print("  ⚠ Bỏ qua vì không có method nào được bật.")
            run_summary.append({'Dataset': ds_name, 'Status': 'SKIPPED (no method)', 'Elapsed': '0s'})
            continue
	
	    # ==================================================
        # Kiểm tra ha <= hb <= hg
        # ==================================================
        '''
        if cfg.get('run_roch', False):
            ha = cfg['ha_roch']
            hb = cfg['hb_roch']
            hg = cfg['hg_roch']

        if not (ha <= hb <= hg):
    	    print(
    	        f"⚠ Bỏ qua ROCH: ha={ha}, hb={hb}, hg={hg} "
    	        f"(không thỏa ha<=hb<=hg)"
    	    )
    	    continue

        if cfg.get('run_cch', False):
            ha = cfg['ha_cch']
            hb = cfg['hb_cch']
            hg = cfg['hg_cch']

        if not (ha <= hb <= hg):
    	    print(
    	        f"⚠ Bỏ qua CCH: ha={ha}, hb={hb}, hg={hg} "
    	        f"(không thỏa ha<=hb<=hg)"
    	    )
    	    continue
        '''
        if cfg.get('run_roch', False):
            if not (
                cfg['ha_roch']
                <= cfg['hb_roch']
                <= cfg['hg_roch']
            ):
                print("⚠ Invalid ROCH parameters")
                continue

        if cfg.get('run_cch', False):
            if not (
                cfg['ha_cch']
                <= cfg['hb_cch']
                <= cfg['hg_cch']
            ):
                print("⚠ Invalid CCH parameters")
                continue

        t0     = time.time()
        status = 'OK'
        try:
            df_result = run(**cfg)
            all_frames.append(df_result)
        except Exception as e:
            print(f"  ❌ Lỗi: {e}")
            status = f'ERROR: {e}'
        elapsed = time.time() - t0
        run_summary.append({
            'Dataset' : ds_name,
            'Status'  : status,
            'Elapsed' : f"{elapsed:.1f}s",
        })

    if all_frames:
        save_combined_excel(all_frames, COMBINED_OUTPUT)
    else:
        print("⚠  Không có kết quả nào để gộp.")

    total = time.time() - t_global
    print(f"\n{'='*70}")
    print(f"  TỔNG KẾT  —  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Tổng thời gian: {total/60:.1f} phút")
    print(f"{'='*70}")
    for r in run_summary:
        icon = '✅' if r['Status'] == 'OK' else ('⏭' if 'SKIPPED' in r['Status'] else '❌')
        print(f"  {icon}  {r['Dataset']:40s}  {r['Elapsed']:>8s}  [{r['Status']}]")
    print(f"{'='*70}\n")
