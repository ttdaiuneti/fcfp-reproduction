# FCFP — Fixed-Confidence Fuzzy Purity

Code, data and per-fold results for the paper

> **Monotone Confidence Weights in Weighted Fuzzy-Rough Attribute Reduction: A Characterisation and the Fixed-Confidence Fuzzy Purity Measure**
> Pham Van Duong, Tran Thanh Dai, Nguyen Nhu Son

Every table and figure of the experimental section can be regenerated from the files in `results/r1/` with the two scripts in `scripts/make_*`. The raw results can be regenerated with the experiment scripts listed below.

## Layout

| Path | Content |
|------|---------|
| `algorithms/engine.py` | Row-chunked evaluation of pairwise measures (memory O(n²) per subset, float32 for n > 6000) |
| `algorithms/reducers_r1.py` | FCFP (Algorithms 1–2 of the paper), the confidence weights of Definition 2 (`knn_purity`), and the wrappers of the rough-set baselines |
| `algorithms/fcfp_variants.py` | FCFP with hybrid weights ω_α = α ω^(C) + (1 − α) ω^(B) |
| `algorithms/*_Hu.py`, `*_Cornelis.py`, `*_Xia.py`, `*_Ji.py`, `*_Wang.py`, `rankers.py` | Baselines; see `BASELINES.md` |
| `data/datasets/` | The 27 datasets as CSV files (last column = decision) |
| `data/datasets_r1_manifest.csv` | Source, size, class counts and SHA-256 of the 15 larger datasets |
| `scripts/` | Experiment runners and table generators (below) |
| `results/r1/` | All raw results, one row per method, dataset and fold |
| `tables/r1/`, `figures/r1/` | LaTeX tables and pgfplots figures generated from `results/r1/` |

## Environment

Python 3.9 with the versions in `requirements.txt`. The runs reported in the paper used an Apple M4 Pro (48 GB), one BLAS thread per process (`OMP_NUM_THREADS=1`).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
```

## Protocol

Stratified ten-fold cross-validation (seed 42). In each fold, min–max scaling is fitted on the training part; every method runs on the training part only; the selected attributes are evaluated with an RBF SVM (C = 1, gamma = scale) and a 3-NN classifier. An empty reduct predicts the majority class of the training part. Each method has one hour per fold; a method that exceeds it on the first fold is recorded as `timeout` for the dataset. A fold that exceeded the limit after the first fold had finished was rerun with two hours (`results/r1/superseded_rows.csv` keeps the replaced rows). The selectors mRMR, ReliefF, L1, AFRSFCV, Random and SFRS-eq keep the FCFP reduct size of the same fold.

## Reproducing the results

All runners are resumable and append to their output file in `results/r1/`.

| Experiment | Command | Output |
|------------|---------|--------|
| Main comparison | `python3 scripts/run_r1.py --datasets <names> --timeout 3600 --out raw_long.csv` | `raw_long.csv`, `raw_long_b.csv` (Madelon, Electrical) |
| SFRS at the FCFP size | `python3 scripts/run_r1.py --datasets <names> --methods SFRS-eq --timeout 3600 --out raw_sfrs_eq.csv --seed_from raw_long.csv --seed_methods FCFP` | `raw_sfrs_eq.csv`, `raw_sfrs_eq_b.csv` |
| Monotonicity, random nested pairs and greedy runs | `python3 scripts/monotonicity_r1.py --datasets <names> --tag A` (r ≤ 60); add `--b1_max 30` for r > 60 (`--tag C`) | `monotonicity_pairs_*.csv`, `monotonicity_greedy_*.csv` |
| Search with hybrid weights | `--methods FCFPa@0.0 FCFPa@0.25 FCFPa@0.5 FCFPa@0.75` | `raw_adaptive.csv` |
| Parameter sensitivity | `--methods FCFP@<delta>@<k_omega>[@<rho>]` | `raw_sens.csv` |
| Median-based δ | `--methods FCFP-med` | `raw_delta.csv` |
| ANOVA-F screening | `--methods FCFPscr@50 FCFPscr@100 FCFPscr@200 FCFPscr@500` | `raw_screen.csv` |
| Scalability in n | `python3 scripts/scalability_r1.py --part n --datasets letter magic --methods FCFP NRS FNRS SFRS GBNRS FGB WFRS` | `scalability.csv` |
| Scalability in r | `python3 scripts/scalability_r1.py --part m --datasets smk-can-187 --methods FCFP NRS FNRS FGB WFRS --out scalability_m.csv` | `scalability_m.csv` |

Then:

```bash
python3 scripts/make_tables_r1.py   # accuracy, size, runtime, pairwise tests, datasets
python3 scripts/make_extra_r1.py    # monotonicity, hybrid weights, delta, screening, scalability and sensitivity figures
```

`scripts/check_stop_rule_r1.py` recomputes, for every stored FCFP reduct, the measure and the threshold on the training part and confirms that every reduct reaches the threshold. `scripts/novelty_check_r1.py` compares FCFP with the crisp-decision dependency of Wang et al. (2025) under the same kernel and weights.

## Result files

Each row of `raw_*.csv` has: `dataset, method, fold, n_train, n_test, p, status, size, reduct, acc_svm, acc_knn, acc_mean, time_reduct, time_eval, note`. `reduct` lists the selected attribute indices (0-based columns of the CSV), `time_reduct` is the reduction time in seconds without classifier training, and `status` is `ok` or `timeout`. `isolet_fold0_runtime.csv` holds one fold of Isolet, which is not in the main comparison.

## Data sources

The twelve datasets not listed in the manifest (the small and medium datasets and Arrhythmia) are standard UCI versions. The other datasets come from the UCI repository, OpenML and the scikit-feature repository; `data/datasets_r1_manifest.csv` gives their source names and checksums. Missing values were replaced by the column median and objects without a label were removed.
