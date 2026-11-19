"""Feature rankers outside rough set theory.  Each returns a full ranking; the shared protocol keeps
the top |R| features, where |R| is the size of the FCFP reduct in the same
outer training fold, so all methods are compared at the same subset size.

* mRMR (Peng, Long and Ding, IEEE TPAMI 27(8):1226-1238, 2005), MID criterion
  max_f [ I(f; c) - mean_{s in S} I(f; s) ], greedy.  Continuous features are
  discretised into three states at mean -/+ std.
   Mutual information from contingency counts (nats).
* ReliefF (Kononenko 1994; implementation skrebate 0.8.3, Urbanowicz et al.
  2018), n_neighbors = 10 (the usual default), all training instances.
* L1-regularised logistic regression (sklearn 1.6.1, solver saga, C = 1,
  max_iter = 5000, multinomial for m > 2); features ranked by the largest
  absolute coefficient over classes; ties among zero coefficients broken by
  index.
"""
from __future__ import annotations

import warnings

import numpy as np


def _disc3(col: np.ndarray) -> np.ndarray:
    mu, sd = col.mean(), col.std()
    return np.digitize(col, [mu - sd, mu + sd])


def _mi(a: np.ndarray, b: np.ndarray) -> float:
    a = np.unique(a, return_inverse=True)[1]; b = np.unique(b, return_inverse=True)[1]
    joint = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(joint, (a, b), 1.0)
    p = joint / joint.sum(); pa = p.sum(1, keepdims=True); pb = p.sum(0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log(p[nz] / (pa @ pb)[nz])).sum())


def mrmr_ranking(X: np.ndarray, y: np.ndarray, k: int | None = None) -> list[int]:
    Xd = np.column_stack([_disc3(X[:, j]) for j in range(X.shape[1])])
    m = Xd.shape[1]; k = m if k is None else min(k, m)
    rel = np.array([_mi(Xd[:, j], y) for j in range(m)])
    sel, red_sum = [], np.zeros(m)
    for _ in range(k):
        score = rel - (red_sum / len(sel) if sel else 0.0)
        score[sel] = -np.inf
        j = int(np.argmax(score)); sel.append(j)
        red_sum += np.array([_mi(Xd[:, t], Xd[:, j]) if t not in sel else 0.0 for t in range(m)])
    return sel


def relieff_ranking(X: np.ndarray, y: np.ndarray, n_neighbors: int = 10) -> list[int]:
    from skrebate import ReliefF
    r = ReliefF(n_neighbors=min(n_neighbors, X.shape[0] - 1), n_jobs=1)
    r.fit(np.asarray(X, dtype=np.float64), np.asarray(y))
    return [int(i) for i in np.argsort(-r.feature_importances_, kind="stable")]


def l1_ranking(X: np.ndarray, y: np.ndarray, C: float = 1.0) -> list[int]:
    from sklearn.linear_model import LogisticRegression
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr = LogisticRegression(penalty="l1", solver="saga", C=C, max_iter=5000, random_state=42)
        lr.fit(X, y)
    score = np.abs(lr.coef_).max(axis=0)
    return [int(i) for i in np.argsort(-score, kind="stable")]


if __name__ == "__main__":
    import pandas as pd, time
    from sklearn.preprocessing import MinMaxScaler, LabelEncoder
    df = pd.read_csv("../data/datasets/wine.csv", header=None)
    X = MinMaxScaler().fit_transform(df.iloc[:, :-1].values); y = LabelEncoder().fit_transform(df.iloc[:, -1])
    for f in (mrmr_ranking, relieff_ranking, l1_ranking):
        t = time.time(); print(f.__name__, f(X, y), f"{time.time()-t:.2f}s")
    # MI sanity: identical variables -> H(a); independent -> ~0
    a = np.repeat([0, 1, 2], 100); print("MI(a,a) =", round(_mi(a, a), 4), "ln 3 =", round(np.log(3), 4))
