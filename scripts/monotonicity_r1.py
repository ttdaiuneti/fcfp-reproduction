"""Monotonicity violations on real data.

Measures M(B), all on the MinMax-scaled training part of fold 0 (seed 42):
  FCFP-fixed     omega^(C) fixed (the paper's measure; monotone by Theorem)
  FCFP-adaptive  omega^(B) recomputed on B (k_omega-NN purity in B, object excluded)
  FCFP-hybrid-a  alpha*omega^(C) + (1-alpha)*omega^(B), alpha in {0.25, 0.5, 0.75}
  WFRS           Wang et al. (2025) dependency on the fuzzy decision D~ (depends on B)
  SFRS           soft fuzzy dependency of Hu, An, Yu (2010)
Outputs (results/r1/monotonicity_pairs.csv, monotonicity_greedy.csv):
  (i)  random nested pairs B1 subset B2 (n_pairs per dataset, |B1| uniform in
       [1, m-1], B2 = B1 + one random attribute): share with M(B2) < M(B1) - 1e-12
       and the largest drop;
  (ii) forward greedy with the shared threshold rule (theta = 0.95 M(C)) on the
       FCFP variants: steps with negative gain, final |B|, and stop at |B| <= 2.
Large tables are subsampled to n_max objects (stratified) for this diagnostic.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "algorithms")); sys.path.insert(0, HERE)
import engine as E                                   # noqa: E402
from reducers_r1 import knn_purity                   # noqa: E402
from run_r1 import load, folds                       # noqa: E402

DELTA, KW, RHO = 0.3, 5, 0.95


def fcfp_value(X, y, cols, omega):
    meas = E.FCFPMeasure(omega, DELTA)
    return E.score(meas, X, y, E.full_base(meas, X, cols), None, len(cols))


def omega_for(X, y, cols, alpha, omega_C):
    if alpha >= 1.0:
        return omega_C
    oB = knn_purity(X[:, cols], y, KW)
    return alpha * omega_C + (1 - alpha) * oB


def measure_factory(X, y):
    omega_C = knn_purity(X, y, KW)
    out = {"FCFP-fixed": lambda c: fcfp_value(X, y, c, omega_C),
           "FCFP-adaptive": lambda c: fcfp_value(X, y, c, omega_for(X, y, c, 0.0, omega_C))}
    for a in (0.25, 0.5, 0.75):
        out[f"FCFP-hybrid-{a}"] = (lambda a: lambda c: fcfp_value(X, y, c, omega_for(X, y, c, a, omega_C)))(a)
    # WFRS
    from reducers_r1 import WFRS
    wf = WFRS(); n, m = X.shape
    yy = np.unique(y, return_inverse=True)[1]
    scale = np.zeros((m, n))
    for c in np.unique(yy):
        idx = yy == c; v = X[idx].var(axis=0, ddof=1) if idx.sum() > 1 else np.zeros(m)
        scale[:, idx] = np.where(v == 0, 1.0, v)[:, None]
    dA = E.full_base(E.SqEuclid(scale), X, range(m)).astype(float); np.fill_diagonal(dA, np.inf)
    nb = np.argsort(dA, axis=1, kind="stable")[:, :min(5, n - 1)]; w = (yy[nb] == yy[:, None]).mean(axis=1); del dA
    wm = E.WFRSMeasure(scale, w, np.unique(yy))
    out["WFRS"] = lambda c: E.score(wm, X, yy, E.full_base(wm, X, c), None, len(c))
    sm = E.SoftFuzzyMeasure(0.15, 0.1)
    out["SFRS"] = lambda c: E.score(sm, X, y, E.full_base(sm, X, c), None, len(c))
    return out, omega_C


def greedy(X, y, M):
    m = X.shape[1]; full = M(list(range(m))); theta = RHO * full
    B, g, neg = [], 0.0, 0
    while g < theta - 1e-12 and len(B) < m:
        vals = {a: M(B + [a]) for a in range(m) if a not in B}
        a = max(vals, key=vals.get)
        if vals[a] < g - 1e-12:
            neg += 1
        B.append(a); g = vals[a]
    return len(B), neg, g, theta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--pairs", type=int, default=200)
    ap.add_argument("--n_max", type=int, default=3000)
    ap.add_argument("--greedy_m_max", type=int, default=60)
    ap.add_argument("--tag", default="")
    ap.add_argument("--b1_max", type=int, default=0, help="if > 0, |B1| uniform in [1, min(b1_max, m-1)]")
    a = ap.parse_args()
    out = os.path.join(ROOT, "results", "r1"); os.makedirs(out, exist_ok=True)
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    for ds in a.datasets:
        X, y = load(ds); tr, _ = folds(y)[0]
        X, y = MinMaxScaler().fit_transform(X[tr]), y[tr]
        if len(y) > a.n_max:
            X, _, y, _ = train_test_split(X, y, train_size=a.n_max, stratify=y, random_state=0)
        meas, _ = measure_factory(X, y)
        rng = np.random.default_rng(7); m = X.shape[1]
        pairs = []
        for _ in range(a.pairs):
            s = int(rng.integers(1, min(a.b1_max, m - 1) + 1)) if a.b1_max > 0 else int(rng.integers(1, m)); B1 = sorted(rng.choice(m, size=s, replace=False).tolist())
            extra = int(rng.choice([j for j in range(m) if j not in B1])); pairs.append((B1, sorted(B1 + [extra])))
        for name, M in meas.items():
            drops = np.array([M(b1) - M(b2) for b1, b2 in pairs])
            row = dict(dataset=ds, n=len(y), p=m, b1_max=a.b1_max, measure=name, pairs=len(pairs),
                       violations=int((drops > 1e-12).sum()), max_drop=float(max(drops.max(), 0.0)))
            _append(os.path.join(out, f"monotonicity_pairs{a.tag}.csv"), row); print(row, flush=True)
            if name.startswith("FCFP") and m <= a.greedy_m_max:
                size, neg, g, theta = greedy(X, y, M)
                row = dict(dataset=ds, measure=name, size=size, negative_steps=neg, final=g, theta=theta,
                           early_stop=int(size <= 2))
                _append(os.path.join(out, f"monotonicity_greedy{a.tag}.csv"), row); print(row, flush=True)


def _append(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row)); (w.writeheader() if new else None); w.writerow(row)


if __name__ == "__main__":
    main()
