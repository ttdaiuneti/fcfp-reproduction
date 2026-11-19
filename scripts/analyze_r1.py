"""Summary tables and tests for the main comparison.

Input : results/r1/raw_long.csv + raw_long_b.csv (per-fold rows).
Output: results/r1/analysis/
  summary_<metric>.csv     mean and std over folds per (dataset, method); 'TO' = time-out
  size.csv, time.csv       mean |R| and mean reduction time (s)
  ranks_<metric>_<group>.csv, tests_<metric>_<group>.csv
Rules
  * empty reduct (NRS with delta = 0.15 on some sets): the classifier predicts the
    majority class of the training part; accuracy is computed on the test fold.
  * a (dataset, method) cell enters a test only if all 10 folds are ok.
  * groups: 'all24' = methods that finish on every dataset; 'sfrs' = + SFRS on the
    datasets where SFRS finishes; 'gbnrs' = + GBNRS (and SFRS) where both finish.
  * Friedman over datasets, Nemenyi CD (alpha = 0.05), Wilcoxon signed-rank of FCFP
    against each method with Holm correction; W/T/L with tie tolerance 0.005.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from run_r1 import load, folds  # noqa: E402

R = os.path.join(ROOT, "results", "r1"); OUT = os.path.join(R, "analysis")
REDUCERS = ["FCFP", "NRS", "FNRS", "FGB", "WFRS"]
RANKERS = ["mRMR", "ReliefF", "L1", "AFRSFCV", "Random"]
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102,
       10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313}


def load_rows():
    d = pd.concat([pd.read_csv(os.path.join(R, f)) for f in ("raw_long.csv", "raw_long_b.csv")]).reset_index(drop=True)
    assert not d.duplicated(["dataset", "method", "fold"]).any()
    empty = (d.status == "ok") & (d["size"] == 0)
    for ds in d.loc[empty, "dataset"].unique():
        X, y = load(ds); fs = folds(y)
        sub = d[empty & (d.dataset == ds)]
        for idx, row in sub.iterrows():
            tr, te = fs[int(row.fold)]
            vals, cnt = np.unique(y[tr], return_counts=True)
            acc = float(np.mean(y[te] == vals[np.argmax(cnt)]))
            m = (d.dataset == ds) & (d.method == row.method) & (d.fold == row.fold)
            d.loc[m, ["acc_svm", "acc_knn", "acc_mean"]] = acc
            d.loc[m, "note"] = "empty reduct: majority class"
    return d


def complete(d, metric):
    ok = d[d.status == "ok"].groupby(["dataset", "method"]).filter(lambda x: len(x) == 10)
    return ok.groupby(["dataset", "method"])[metric].mean().unstack()


def tests(pv, name, metric):
    pv = pv.dropna()
    k, N = pv.shape[1], pv.shape[0]
    ranks = pv.rank(axis=1, ascending=False)
    ar = ranks.mean().sort_values()
    chi, p = stats.friedmanchisquare(*[pv[c] for c in pv.columns])
    cd = Q05.get(k, np.nan) * np.sqrt(k * (k + 1) / (6 * N))
    rows = []
    for c in pv.columns:
        if c == "FCFP":
            continue
        diff = pv["FCFP"] - pv[c]
        try:
            pw = stats.wilcoxon(pv["FCFP"], pv[c], zero_method="wilcox").pvalue
        except ValueError:
            pw = 1.0
        rows.append(dict(method=c, W=int((diff > 0.005).sum()), T=int((diff.abs() <= 0.005).sum()),
                         L=int((diff < -0.005).sum()), mean_diff=diff.mean(), p_wilcoxon=pw,
                         rank_gap=ar[c] - ar["FCFP"]))
    t = pd.DataFrame(rows).sort_values("p_wilcoxon").reset_index(drop=True)
    m = len(t); adj, run = [], 0.0
    for i, pv_ in enumerate(t.p_wilcoxon):
        run = max(run, min(1.0, (m - i) * pv_)); adj.append(run)
    t["p_holm"] = adj
    t["nemenyi_sig"] = t.rank_gap.abs() > cd
    ar.rename("avg_rank").to_frame().assign(N=N, k=k, friedman_chi2=chi, friedman_p=p, CD=cd) \
        .to_csv(os.path.join(OUT, f"ranks_{metric}_{name}.csv"))
    t.to_csv(os.path.join(OUT, f"tests_{metric}_{name}.csv"), index=False)
    return ar, t, dict(N=N, k=k, p=p, CD=cd)


def main():
    os.makedirs(OUT, exist_ok=True)
    d = load_rows()
    ok = d[d.status == "ok"]
    for metric in ("acc_mean", "acc_svm", "acc_knn"):
        g = ok.groupby(["dataset", "method"])[metric]
        s = g.agg(["mean", "std", "count"]).reset_index()
        s.to_csv(os.path.join(OUT, f"summary_{metric}.csv"), index=False)
    ok.groupby(["dataset", "method"])["size"].mean().unstack().to_csv(os.path.join(OUT, "size.csv"))
    ok.groupby(["dataset", "method"])["time_reduct"].agg(["mean", "std"]).to_csv(os.path.join(OUT, "time.csv"))
    for metric in ("acc_mean", "acc_svm", "acc_knn"):
        pv = complete(d, metric)
        base = REDUCERS + RANKERS + ["Full"]
        groups = {"all24": base, "reducers": REDUCERS + ["Full"],
                  "sfrs": REDUCERS + ["SFRS", "Full"], "gbnrs": REDUCERS + ["SFRS", "GBNRS", "Full"]}
        for name, cols in groups.items():
            ar, t, info = tests(pv[cols], name, metric)
            print(f"\n== {metric} / {name}: N={info['N']} k={info['k']} Friedman p={info['p']:.2g} CD={info['CD']:.2f}")
            print(ar.round(2).to_dict())
            print(t.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
