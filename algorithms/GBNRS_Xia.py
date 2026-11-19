"""GBNRS baseline: a thin wrapper around the authors'
released code (vendor_gbnrs_xia.py, copied verbatim from
https://github.com/syxiaa/GBNRS).  Replaces GBNRS_AR.py, which used a forward
search with a 0.99 threshold and purity 0.9 (neither appears in the paper or
in the authors' code).

What the authors' code does (and this wrapper keeps):
  * granular balls are split deterministically (generateOmegaCluster / splitGB)
    until purity = 1 or the ball has fewer than `min_sample` objects;
  * the score of an attribute set is the number of objects in pure balls;
  * backward elimination: in each round remove the attribute whose removal
    keeps the largest score, as long as that score is >= the current one;
  * `min_sample` (d2) is chosen from range(2, 2m, max(1, m // 5)) by the
    resubstitution accuracy of a 5-NN classifier on the training data
    (their __main__ block).  This uses training data only.
Deviation from the authors' script: `max(1, m // 5)` guards the step for
m < 5 (their int(m/5) would be 0); our datasets have m >= 7, so it never fires.
The paper (Algorithm 1) describes a k-means-based ball generation; the
released code does not use k-means.  We follow the released code and say so
in the manuscript.
"""
from __future__ import annotations

import contextlib
import io
import time

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from vendor_gbnrs_xia import attribute_reduce


class GBNRSXia:
    def __init__(self, select_min_sample: bool = True, min_sample: int = 2, time_cap: float | None = None):
        self.select_min_sample = select_min_sample
        self.min_sample = int(min_sample)
        self.time_cap = time_cap

    def _reduce(self, U, d2):
        with contextlib.redirect_stdout(io.StringIO()):
            return list(attribute_reduce(U, pur=1, d2=d2))

    def fit_reduction(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.unique(np.asarray(y), return_inverse=True)[1].astype(np.float64)
        U = np.hstack([X, y[:, None]])
        m = X.shape[1]
        t0 = time.perf_counter()
        if not self.select_min_sample:
            self.reduct_ = sorted(int(a) for a in self._reduce(U, self.min_sample))
            self.min_sample_ = self.min_sample
            return self
        best_acc, best_row, best_d2, tried = -1.0, None, None, []
        for d2 in range(2, 2 * m, max(1, m // 5)):
            row = self._reduce(U, d2)
            clf = KNeighborsClassifier(n_neighbors=5).fit(X[:, row], y)
            acc = float((clf.predict(X[:, row]) == y).mean())
            tried.append((d2, len(row), acc))
            if best_acc < acc:
                best_acc, best_row, best_d2 = acc, row, d2
            if self.time_cap and time.perf_counter() - t0 > self.time_cap:
                break
        self.reduct_ = sorted(int(a) for a in best_row)
        self.min_sample_, self.tried_ = best_d2, tried
        return self


if __name__ == "__main__":
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler, LabelEncoder
    df = pd.read_csv("../data/datasets/wine.csv", header=None)
    X = MinMaxScaler().fit_transform(df.iloc[:, :-1].values); y = LabelEncoder().fit_transform(df.iloc[:, -1])
    t = time.time(); g = GBNRSXia().fit_reduction(X, y)
    print("wine GBNRS", g.reduct_, "d2 =", g.min_sample_, g.tried_, f"{time.time()-t:.1f}s")
