# ROCH-SVM-Data-Reduction
Research code for **Data Reduction for Support Vector Machine Training Using
Restricted-Orientation Convex Hulls**.

ROCH-SVM reduces a binary-classification training set by extracting local
boundary samples with restricted-orientation convex hulls (ROCHs), then trains
an RBF-kernel support vector machine on the retained samples. The repository
also contains the standard LibSVM baseline and the convex--concave hull
(CCH-SVM) baseline used in the paper.

## Repository layout

```text
.
├── Function/                    # ROCH/OCH and CCH geometry routines
├── reduce_dim/                  # boundary extraction for higher dimensions
├── data/                        # benchmark CSV files
├── results/                     # generated outputs (ignored by Git)
├── Model_SVM.py                 # model and reduction pipelines
├── Train_SVM_v3_flag.py         # repeated evaluation routine
├── run_experiment.py            # command-line entry point
├── gridsearch_params_ROCH_CCH.py
└── requirements.txt
```

Legacy exploratory scripts are retained for traceability. New users should
start with `run_experiment.py`.

## Installation

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## Quick start

Run a one-split standard-SVM smoke test:

```bash
python3 run_experiment.py haberman --methods svm --runs 1
```

Compare all three methods:

```bash
python3 run_experiment.py haberman --methods svm roch cch --runs 5 \
  --c 5 --gamma 2 --ha 4 --hb 10 --hg 12 --k 9
```

The command may be launched from any working directory. Output workbooks are
written to `results/`.

For a full parameter search, edit the search spaces near the bottom of
`gridsearch_params_ROCH_CCH.py`, then run:

```bash
python3 gridsearch_params_ROCH_CCH.py
```

Large experiments can be computationally expensive. The manuscript used 50
stratified 70/30 train--test splits; the CLI defaults to one split so that the
installation can be checked quickly.

## Data format

Each CSV file contains one sample per row. Feature columns come first and the
binary class label is the final column. The loader treats the first row as a
header; see `data/README.md` for dataset notes.

## Reproducibility notes

- Train/test splits use run indices as random seeds.
- Scaling is fitted on each training fold only.
- SMOTE is applied only to sufficiently imbalanced training folds.
- Runtime depends on the machine and installed numerical libraries.
- The exact dataset-specific parameters reported in the article are recorded
  in the experiment scripts and manuscript tables.

## Citation

If this code supports your research, please cite the accompanying article.
Bibliographic metadata can be added to `CITATION.cff` after publication.

