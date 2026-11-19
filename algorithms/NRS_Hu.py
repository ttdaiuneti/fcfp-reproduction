"""Neighbourhood rough set reduction of Hu, Yu, Liu and Wu (2008),
"Neighborhood rough set based heterogeneous feature subset selection",
Information Sciences 178(18):3577-3594.
Reimplemented from the paper (the earlier implementation NRS_AR.py, which added a stopping
threshold 0.99*gamma(C) and a backward step that are not in the paper).

Model: features scaled to [0,1], 2-norm distance, neighbourhood
delta_B(x) = {y : ||x-y||_B <= delta}; POS_B = {x : delta_B(x) within the
class of x}; gamma_B = |POS_B| / |U|.  Paper: delta in [0.1, 0.2]; we use 0.15.
Search: Algorithm 1 (NFARNRS) - add the attribute with the largest
SIG(a, red, D) = gamma_{red+a} - gamma_red while SIG > eps.  The paper says
only that eps is "a little positive real number"; gamma moves in steps of
1/|U|, so any eps < 1/|U| gives the same result as eps = 0, which we use.
"""
from __future__ import annotations

import numpy as np


class NRSHu:
    def __init__(self, delta: float = 0.15, eps: float = 0.0):
        self.delta, self.eps = float(delta), float(eps)

    def _gamma(self, D2: np.ndarray, y: np.ndarray) -> float:
        nb = D2 <= self.delta ** 2
        bad = (nb & (y[:, None] != y[None, :])).any(axis=1)
        return float((~bad).mean())

    def fit_reduction(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64); y = np.asarray(y)
        n, m = X.shape
        sq = [(X[:, a][:, None] - X[:, a][None, :]) ** 2 for a in range(m)]
        red, D2, g = [], np.zeros((n, n)), 0.0
        while len(red) < m:
            best, best_g = None, -1.0
            for a in range(m):
                if a in red:
                    continue
                ga = self._gamma(D2 + sq[a], y)
                if ga > best_g:
                    best, best_g = a, ga
            if best_g - g > self.eps:
                red.append(best); D2 = D2 + sq[best]; g = best_g
            else:
                break
        self.reduct_, self.gamma_ = sorted(red), g
        return self


if __name__ == "__main__":
    # Paper Sec. 2 toy: with delta = 0.1 on one attribute, x2's neighbourhood is {x2, x5}
    import pandas as pd, time
    from sklearn.preprocessing import MinMaxScaler, LabelEncoder
    df = pd.read_csv("../data/datasets/wine.csv", header=None)
    X = MinMaxScaler().fit_transform(df.iloc[:, :-1].values); y = LabelEncoder().fit_transform(df.iloc[:, -1])
    t = time.time(); r = NRSHu().fit_reduction(X, y); print("wine NRS", r.reduct_, r.gamma_, f"{time.time()-t:.2f}s")
