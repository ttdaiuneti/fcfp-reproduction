"""Reducers on top of engine.py (row-chunked, memory O(n^2)
for one base matrix).  Measures and parameters are those of the per-method
modules (NRS_Hu, FNRS_Cornelis, SFRS_Hu, WFRS_Wang); test_equivalence() checks
that both give the same reducts on small data.

FCFP follows the manuscript: omega^(C) = share of same-class objects among the
k_omega nearest neighbours in C, the object itself excluded (Definition 2);
Algorithm 2 (forward-backward): forward greedy while FCFP(B) < theta, theta =
rho * FCFP(C); then remove any a with FCFP(B - {a}) >= theta, repeated until no
change.  (The submitted code compared with the current value instead of theta
in the backward step; this module follows the paper.)
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

import engine as E


def knn_purity(X, y, k, chunk=None):
    """omega(y) of Definition 2: share of same-class objects among the k nearest
    neighbours of y, y itself excluded.  Ties at the k-th distance are split
    evenly: objects strictly closer count 1, the t objects tied at the k-th
    distance share the remaining k - (#closer) slots, each with weight
    (k - #closer)/t.  This equals the expected purity under uniform random
    tie-breaking, so omega does not depend on row order or on the search
    structure (on low-cardinality attributes ball_tree and brute force break
    ties differently, which changed results in the submitted code)."""
    X = np.asarray(X, float); y = np.asarray(y)
    n = X.shape[0]; k = min(int(k), max(1, n - 1))
    chunk = chunk or E.chunk_size(n)
    sqn = (X ** 2).sum(axis=1)
    out = np.empty(n)
    for s in range(0, n, chunk):
        rows = np.arange(s, min(n, s + chunk))
        with np.errstate(all="ignore"):                 # numpy 2 + Accelerate: spurious matmul warnings
            D = ((X[rows][:, None, :] - X[None, :, :]) ** 2).sum(axis=2) if X.shape[1] <= 8 else \
                np.maximum(sqn[rows][:, None] + sqn[None, :] - 2.0 * X[rows] @ X.T, 0.0)
        assert np.all(np.isfinite(D)), "non-finite distances in knn_purity"
        if X.shape[1] > 8:                                   # exact zeros for duplicates
            D[np.isclose(D, 0.0, atol=1e-12)] = 0.0
        D[np.arange(len(rows)), rows] = np.inf
        kth = np.partition(D, k - 1, axis=1)[:, k - 1][:, None]
        less, eq = D < kth, D == kth
        frac = (k - less.sum(axis=1)) / eq.sum(axis=1)
        same = y[rows][:, None] == y[None, :]
        out[rows] = ((less & same).sum(axis=1) + frac * (eq & same).sum(axis=1)) / k
    return out


# ---------------------------------------------------------------- searches
def forward_threshold(meas, X, y, target, backward=True, tol=1e-12, stop_no_gain=True):
    # stop_no_gain=False follows Algorithm 1 of the paper (add the best attribute until target or B = C);
    # True keeps the extra stop on a step without gain (used by the FNRS baseline).

    n, m = X.shape
    base, B, g = meas.init_base(n), [], 0.0
    while g < target - tol and len(B) < m:
        best, bg = None, -np.inf
        for a in range(m):
            if a in B:
                continue
            ga = E.score(meas, X, y, base, a, len(B) + 1)
            if ga > bg + tol:
                best, bg = a, ga
        if stop_no_gain and bg <= g + tol and B:
            break
        B.append(best); E.add_to_base(meas, X, base, best); g = bg
    if backward:
        changed = True
        while changed and len(B) > 1:
            changed = False
            for a in list(B):
                gr = E.score(meas, X, y, base, a, len(B) - 1, sign=-1)
                if gr >= target - tol:
                    E.add_to_base(meas, X, base, a, sign=-1); B.remove(a); g = gr; changed = True
                    break
    return sorted(int(a) for a in B), g


def forward_sig(meas, X, y, eps):
    n, m = X.shape
    base, B, g = meas.init_base(n), [], 0.0
    while len(B) < m:
        best, bg = None, -np.inf
        for a in range(m):
            if a in B:
                continue
            ga = E.score(meas, X, y, base, a, len(B) + 1)
            if ga > bg:
                best, bg = a, ga
        if bg - g > eps:
            B.append(best); E.add_to_base(meas, X, base, best); g = bg
        else:
            break
    return sorted(int(a) for a in B), g


def forward_ranking(meas, X, y, tol=1e-12, max_len=None):
    """Greedy forward ranking; with max_len the ranking stops after max_len
    attributes (its prefix is identical to the full ranking's prefix)."""
    n, m = X.shape
    base, rank = meas.init_base(n), []
    while len(rank) < (m if max_len is None else min(m, max_len)):
        best, bg = None, -np.inf
        for a in range(m):
            if a in rank:
                continue
            ga = E.score(meas, X, y, base, a, len(rank) + 1)
            if ga > bg + tol:
                best, bg = a, ga
        rank.append(best); E.add_to_base(meas, X, base, best)
    return rank


# ---------------------------------------------------------------- reducers
class FCFP:
    def __init__(self, delta=0.3, k_omega=5, rho=0.95):
        self.delta, self.k_omega, self.rho = delta, k_omega, rho

    def fit_reduction(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y)
        self.omega_ = knn_purity(X, y, self.k_omega)
        meas = E.FCFPMeasure(self.omega_, self.delta)
        full = E.score(meas, X, y, E.full_base(meas, X, range(X.shape[1])), None, X.shape[1])
        self.theta_ = self.rho * full
        self.reduct_, self.value_ = forward_threshold(meas, X, y, self.theta_, stop_no_gain=False)
        return self


class NRS:
    def __init__(self, delta=0.15):
        self.delta = delta

    def fit_reduction(self, X, y):
        self.reduct_, self.value_ = forward_sig(E.NRSMeasure(self.delta), np.asarray(X, float), np.asarray(y), 0.0)
        return self


class FNRS:
    def __init__(self, alpha=0.95):
        self.alpha = alpha

    def fit_reduction(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y)
        meas = E.CornelisMeasure(X.std(axis=0, ddof=1))
        posA = E.score(meas, X, y, E.full_base(meas, X, range(X.shape[1])), None, X.shape[1])
        if posA <= 0:
            self.reduct_, self.value_ = list(range(X.shape[1])), 0.0
            return self
        # gamma_B = |POS_B| / |POS_A|  ->  stop when mean POS_B >= alpha * mean POS_A
        self.reduct_, v = forward_threshold(meas, X, y, self.alpha * posA)
        self.value_ = v / posA
        return self


class SFRS:
    def __init__(self, delta=0.15, beta=0.1, knn_k=3, cv_folds=10):
        self.delta, self.beta, self.knn_k, self.cv_folds = delta, beta, knn_k, cv_folds

    def fit_reduction(self, X, y):
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.neighbors import KNeighborsClassifier
        X = np.asarray(X, float); y = np.asarray(y)
        rank = forward_ranking(E.SoftFuzzyMeasure(self.delta, self.beta), X, y)
        n_splits = max(2, min(self.cv_folds, int(np.bincount(y).min())))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        accs = [cross_val_score(KNeighborsClassifier(n_neighbors=self.knn_k), X[:, rank[:j]], y, cv=cv).mean()
                for j in range(1, len(rank) + 1)]
        self.ranking_, self.prefix_acc_ = rank, accs
        self.reduct_ = sorted(int(a) for a in rank[:int(np.argmax(accs)) + 1])
        return self


class WFRS:
    def __init__(self, k=5, delta=0.005, lam=1.0, ddof=1):
        self.k, self.delta, self.lam, self.ddof = k, delta, lam, ddof

    def fit_reduction(self, X, y):
        X = np.asarray(X, float); y = np.unique(np.asarray(y), return_inverse=True)[1]
        n, m = X.shape
        scale = np.zeros((m, n))
        for c in np.unique(y):
            idx = y == c
            v = X[idx].var(axis=0, ddof=self.ddof) if idx.sum() > self.ddof else np.zeros(m)
            scale[:, idx] = np.where(v == 0, self.lam, v)[:, None]
        base_meas = E.SqEuclid(scale)
        dA = E.full_base(base_meas, X, range(m)).astype(float)
        np.fill_diagonal(dA, np.inf)
        k = min(self.k, n - 1)
        nb = np.argsort(dA, axis=1, kind="stable")[:, :k]
        self.w_ = (y[nb] == y[:, None]).mean(axis=1)
        del dA
        meas = E.WFRSMeasure(scale, self.w_, np.unique(y))
        self.reduct_, self.value_ = forward_sig(meas, X, y, self.delta)
        return self


def test_equivalence():
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler, LabelEncoder
    from NRS_Hu import NRSHu
    from FNRS_Cornelis import FNRSCornelis
    from SFRS_Hu import SFRSHu
    from WFRS_Wang import WFRSWang
    res = []
    for ds in ["wine", "glass", "ionosphere", "ecoli"]:
        df = pd.read_csv(f"../data/datasets/{ds}.csv", header=None)
        X = MinMaxScaler().fit_transform(df.iloc[:, :-1].values); y = LabelEncoder().fit_transform(df.iloc[:, -1])
        pairs = [("NRS", NRS().fit_reduction(X, y).reduct_, NRSHu().fit_reduction(X, y).reduct_),
                 ("FNRS", FNRS().fit_reduction(X, y).reduct_, FNRSCornelis().fit_reduction(X, y).reduct_),
                 ("SFRS", SFRS().fit_reduction(X, y).reduct_, SFRSHu().fit_reduction(X, y).reduct_),
                 ("WFRS", WFRS().fit_reduction(X, y).reduct_, WFRSWang().fit_reduction(X, y).reduct_)]
        for name, a, b in pairs:
            res.append((ds, name, a == b, a, b))
    return res


if __name__ == "__main__":
    for r in test_equivalence():
        print(r[0], r[1], "SAME" if r[2] else f"DIFF chunked={r[3]} dense={r[4]}")
