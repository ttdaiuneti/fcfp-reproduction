"""AFRSFCV of Wang, Zhang, An and Deng (2026), "Adaptive feature selection based
on fuzzy rough set fusion model with class variance", Pattern Recognition 170,
112014 (pdf/Wang2026ClassVariance.pdf; equations read from page images 3-4).
Ranking baseline.

  Eq. (7)  R_k(xi, xj) = exp(-(a_k(xi) - a_k(xj))^2 / (2 s2_{k,t})), t = class of xi
           (the worked example after Eq. (7) uses the class of the first argument).
  Eq. (8)  lower_k(D_t)(xi) = min_j max(1 - R_k(xi, xj), D_t(xj))
           = min_{xj not in D_t} (1 - R_k(xi, xj))
  Eq. (9)  n x r matrix of these values per feature.
  Eq. (12)-(13)  xi_{ik} = 1 if the own class of xi attains max_t lower_k(D_t)(xi).
  Eq. (14) min_w 1/2 ||I - sum_k w_k R_k(D)||_F^2 + l1/2 ||1 - P w||^2 + l2/2 ||w||^2,
           w = softmax(tau).
  Eq. (23) tau <- tau - eta dL/dtau, 200 iterations (Algorithm 1).
  Algorithm 1, step 19: rank features by w (descending).
Deviations, all forced by the paper:
  * the printed gradient (Eqs. 16, 19, 22) is not the derivative of Eq. (14)
    (it uses sum_k (1 - w_k R_k) where the derivative has I_ij - sum_k w_k R_k);
    we use the exact gradient of Eq. (14);
  * l1, l2 were tuned per dataset on test accuracy in the paper (Sec. 6.2); we fix
    l1 = l2 = 1 (inside the ranges the paper reports as best, Sec. 6.7);
  * eta and the stopping ratio theta are not given; eta = 0.5, tau0 = 0, and the
    number of features is set by the shared protocol (|R| of FCFP in the fold).
"""
from __future__ import annotations

import numpy as np


class AFRSFCV:
    def __init__(self, l1: float = 1.0, l2: float = 1.0, eta: float = 0.5, iters: int = 200,
                 ddof: int = 1, eps_var: float = 1e-12):
        self.l1, self.l2, self.eta, self.iters, self.ddof, self.eps_var = l1, l2, eta, int(iters), ddof, eps_var

    def lower_tensor(self, X, y):
        n, m = X.shape
        classes = np.unique(y)
        L = np.zeros((m, n, classes.size))
        for k in range(m):
            s2 = np.zeros(n)
            for c in classes:
                idx = y == c
                s2[idx] = X[idx, k].var(ddof=self.ddof) if idx.sum() > self.ddof else 0.0
            s2 = np.maximum(s2, self.eps_var)
            R = np.exp(-((X[:, k][:, None] - X[:, k][None, :]) ** 2) / (2 * s2[:, None]))
            for t, c in enumerate(classes):
                out = y != c
                L[k, :, t] = (1.0 - R[:, out]).min(axis=1) if out.any() else 1.0
        return L, classes

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64); y = np.unique(np.asarray(y), return_inverse=True)[1]
        n, m = X.shape
        L, classes = self.lower_tensor(X, y)                     # (m, n, r)
        I = (y[:, None] == classes[None, :]).astype(float)       # (n, r)
        own = L[:, np.arange(n), y]                              # (m, n)
        P = (own >= L.max(axis=2) - 1e-12).T.astype(float)       # (n, m)  Eq. 12-13
        tau = np.zeros(m); ones = np.ones(n); hist = []
        # np.errstate: numpy 2 + macOS Accelerate emits spurious matmul warnings; results are checked finite below
        with np.errstate(all='ignore'):
          for _ in range(self.iters):
              w = np.exp(tau - tau.max()); w /= w.sum()
              H = I - np.tensordot(w, L, axes=1)                    # (n, r)
              S = ones - P @ w
              hist.append(0.5 * (H ** 2).sum() + 0.5 * self.l1 * (S ** 2).sum() + 0.5 * self.l2 * (w ** 2).sum())
              gw = -np.tensordot(L, H, axes=([1, 2], [0, 1])) + self.l1 * (P.T @ P @ w - P.T @ ones) + self.l2 * w
              gtau = (np.diag(w) - np.outer(w, w)) @ gw
              tau = tau - self.eta * gtau
        assert np.all(np.isfinite(tau)), 'non-finite tau'
        w = np.exp(tau - tau.max()); w /= w.sum()
        self.w_, self.tau_, self.loss_ = w, tau, hist
        self.ranking_ = [int(i) for i in np.argsort(-tau, kind="stable")]   # same order as w, no underflow ties
        return self


if __name__ == "__main__":
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler, LabelEncoder
    df = pd.read_csv("../data/datasets/wine.csv", header=None)
    X = MinMaxScaler().fit_transform(df.iloc[:, :-1].values); y = LabelEncoder().fit_transform(df.iloc[:, -1])
    a = AFRSFCV().fit(X, y)
    print("loss first/last:", round(a.loss_[0], 4), round(a.loss_[-1], 4), "monotone decrease:", bool(np.all(np.diff(a.loss_) <= 1e-9)))
    print("ranking:", a.ranking_); print("w:", np.round(a.w_, 3))
    # gradient check
    m = AFRSFCV(iters=0); X2, y2 = X[:40, :5], y[:40]
    L, cl = m.lower_tensor(X2, y2); I = (y2[:, None] == cl[None, :]).astype(float); own = L[:, np.arange(40), y2]
    P = (own >= L.max(axis=2) - 1e-12).T.astype(float)
    def loss(t):
        w = np.exp(t - t.max()); w /= w.sum(); H = I - np.tensordot(w, L, axes=1); S = 1 - P @ w
        return 0.5 * (H**2).sum() + 0.5 * (S**2).sum() + 0.5 * (w**2).sum()
    t = np.random.default_rng(0).normal(size=5); w = np.exp(t - t.max()); w /= w.sum()
    H = I - np.tensordot(w, L, axes=1); gw = -np.tensordot(L, H, axes=([1, 2], [0, 1])) + (P.T @ P @ w - P.T @ np.ones(40)) + w
    g = (np.diag(w) - np.outer(w, w)) @ gw
    num = np.array([(loss(t + 1e-6 * e) - loss(t - 1e-6 * e)) / 2e-6 for e in np.eye(5)])
    print("grad check max abs diff:", float(np.abs(g - num).max()))
