"""Noise-aware weighted fuzzy rough set (WFRS) feature selection of Wang, Deng,
An and Huang (2025), IEEE Trans. Fuzzy Systems 33(10):3504-3514.  Closest prior work to FCFP.

Equations (checked against page images 3-6 of the PDF):
  Eq. (9)   d_B(xi -> xj) = sum_{a in B} (a(xi) - a(xj))^2 / (s2(a, l(xj)) + lam)
            s2(a, l(xj)) = sample variance of a in the class of xj;
            lam = 1 if s2 = 0, else 0 (Sec. IV-A).
  Eq. (10)  R_B(xi -> xj) = exp(-d_B(xi -> xj))
  Def. 3.3-3.4  w_xj = |same-class objects among the k nearest neighbours of xj
            under d_A (xj excluded)| / k,  k = 5 (Sec. IV-A).  Computed once on A.
  Eq. (15)  R^w_B(xi -> xj) = w_xj * R_B(xi -> xj)
  Eq. (21)  Dt_j(xi) = sum_u R^w_B(xi->u) * D_j(u) / sum_u R^w_B(xi->u)
  Eq. (17)  lower_B(P)(x) = inf_u max(1 - R^w_B(x->u), P(u))
  Eq. (23)-(24)  POS(x) = max_j lower_B(Dt_j)(x),  gamma_B = mean_x POS(x)
  Algorithm 1: forward greedy; add argmax Sig = gamma_{Red+a} - gamma_Red while
            Sig > delta; paper recommends delta in [0.005, 0.008] (Sec. IV-C); we use 0.005.
Reading choices: Eq. (9) prints the sum over k = 1..m; we sum over the
attributes of B (otherwise d_B would not depend on B).  "Sample variance" is
taken with ddof = 1.  The numerical example (Tables I-III) could not be
reproduced from the printed equations; even Table II (classical fuzzy rough
sets) reports non-zero lower approximations of other classes at an object,
which the classical definition cannot give.  So no numeric test is possible.
"""
from __future__ import annotations

import numpy as np


class WFRSWang:
    def __init__(self, k: int = 5, delta: float = 0.005, lam: float = 1.0, ddof: int = 1):
        self.k, self.delta, self.lam, self.ddof = int(k), float(delta), float(lam), int(ddof)

    def _attr_dist(self, X, y):
        n, m = X.shape
        denom = np.zeros((m, n))
        for c in np.unique(y):
            idx = y == c
            v = X[idx].var(axis=0, ddof=self.ddof) if idx.sum() > self.ddof else np.zeros(m)
            denom[:, idx] = np.where(v == 0, self.lam, v)[:, None]
        # per attribute: (a(xi)-a(xj))^2 / denom(a, xj)   (column = xj)
        return [((X[:, a][:, None] - X[:, a][None, :]) ** 2) / denom[a][None, :] for a in range(m)]

    def _weights(self, dA, y):
        n = y.size
        k = min(self.k, n - 1)
        d = dA.copy(); np.fill_diagonal(d, np.inf)
        nb = np.argsort(d, axis=1, kind="stable")[:, :k]      # row j: neighbours of xj under d_A(xj -> .)
        return (y[nb] == y[:, None]).mean(axis=1)

    def gamma_from_d(self, dB, y, w, classes) -> float:
        Rw = np.exp(-dB) * w[None, :]
        tot = Rw.sum(axis=1)
        pos = np.zeros(y.size)
        for c in classes:
            Dt = np.where(tot > 0, (Rw * (y == c)[None, :]).sum(axis=1) / np.where(tot > 0, tot, 1), 0.0)
            low = np.min(np.maximum(1.0 - Rw, Dt[None, :]), axis=1)
            pos = np.maximum(pos, low)
        return float(pos.mean())

    def fit_reduction(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64); y = np.unique(np.asarray(y), return_inverse=True)[1]
        n, m = X.shape
        classes = np.unique(y)
        parts = self._attr_dist(X, y)
        self.w_ = self._weights(np.sum(parts, axis=0), y)
        red, dB, g = [], np.zeros((n, n)), 0.0
        while len(red) < m:
            best, best_g = None, -np.inf
            for a in range(m):
                if a in red:
                    continue
                ga = self.gamma_from_d(dB + parts[a], y, self.w_, classes)
                if ga > best_g:
                    best, best_g = a, ga
            if best_g - g > self.delta:
                red.append(best); dB = dB + parts[best]; g = best_g
            else:
                break
        self.reduct_, self.gamma_ = sorted(int(a) for a in red), g
        return self


if __name__ == "__main__":
    import pandas as pd, time
    from sklearn.preprocessing import MinMaxScaler, LabelEncoder
    for ds in ["wine", "segment"]:
        df = pd.read_csv(f"../data/datasets/{ds}.csv", header=None)
        X = MinMaxScaler().fit_transform(df.iloc[:, :-1].values); y = LabelEncoder().fit_transform(df.iloc[:, -1])
        t = time.time(); r = WFRSWang().fit_reduction(X, y)
        print(ds, "WFRS", r.reduct_, round(r.gamma_, 4), f"{time.time()-t:.1f}s")
