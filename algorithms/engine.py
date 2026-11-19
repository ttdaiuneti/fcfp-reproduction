"""Row-chunked evaluation of pairwise measures.

Every distance-based reducer here keeps one n x n "base" matrix for
the current attribute subset B (float64 for n <= 6000, float32 above) and
evaluates a candidate attribute a by adding its contribution chunk by chunk,
without storing one n x n matrix per attribute.  All measures used here are
sums over objects of a per-row quantity, so row chunks give exact results.

A measure object implements:
  init_base(n)                -> base matrix for B = {}
  contrib(X, a, rows)         -> contribution of attribute a for rows x all objects
  combine(base_rows, c, size) -> matrix used by row_score for subset size |B|+1
  row_score(M, rows, y)       -> per-row values (summed and divided by n by the caller)
"""
from __future__ import annotations

import numpy as np


def _dtype(n):
    return np.float64 if n <= 6000 else np.float32


def chunk_size(n, budget_mb=256):
    return max(1, int(budget_mb * 2**20 / (8 * max(n, 1))))


def score(meas, X, y, base, a, size, sign=+1, chunk=None):
    """Mean row score of B +/- {a} given base for B (sign=-1 removes a).
    Measures with a `prepass(M, rows, y)` method get a first pass whose per-row
    outputs are stacked and handed to row_score through `meas.pre` (WFRS needs
    the fuzzy decision of every object before any lower approximation)."""
    n = X.shape[0]
    chunk = chunk or chunk_size(n)
    def blocks():
        for s in range(0, n, chunk):
            rows = np.arange(s, min(n, s + chunk))
            c = meas.contrib(X, a, rows) if a is not None else 0.0
            yield rows, meas.combine(base[rows] + sign * c, size)
    if hasattr(meas, "prepass"):
        meas.pre = np.concatenate([meas.prepass(M, rows, y) for rows, M in blocks()], axis=0)
    tot = 0.0
    for rows, M in blocks():
        tot += float(np.sum(meas.row_score(M, rows, y)))
    return tot / n


def add_to_base(meas, X, base, a, sign=+1, chunk=None):
    n = X.shape[0]
    chunk = chunk or chunk_size(n)
    for s in range(0, n, chunk):
        rows = np.arange(s, min(n, s + chunk))
        base[rows] += sign * meas.contrib(X, a, rows).astype(base.dtype)


def full_base(meas, X, cols):
    base = meas.init_base(X.shape[0])
    for a in cols:
        add_to_base(meas, X, base, a)
    return base


# --------------------------------------------------------------- measures
class SqEuclid:
    """Base = squared Euclidean distance over B (optionally per-column scaled)."""
    def __init__(self, col_scale=None):
        self.col_scale = col_scale          # (m, n) divisor per attribute and column object, or None

    def init_base(self, n):
        return np.zeros((n, n), dtype=_dtype(n))

    def contrib(self, X, a, rows):
        d = (X[rows, a][:, None] - X[None, :, a]) ** 2
        if self.col_scale is not None:
            d = d / self.col_scale[a][None, :]
        return d

    def combine(self, M, size):
        return M


class FCFPMeasure(SqEuclid):
    """FCFP(B) = mean_x min_{y not in class(x)} (1 - omega(y) exp(-d_B(x,y)^2/delta^2))."""
    def __init__(self, omega, delta):
        super().__init__(); self.omega, self.delta2 = np.asarray(omega, float), float(delta) ** 2

    def row_score(self, D2, rows, y):
        other = y[rows][:, None] != y[None, :]
        v = np.where(other, 1.0 - self.omega[None, :] * np.exp(-D2 / self.delta2), 1.0)
        return v.min(axis=1)


class NRSMeasure(SqEuclid):
    def __init__(self, delta):
        super().__init__(); self.delta2 = float(delta) ** 2

    def row_score(self, D2, rows, y):
        bad = ((D2 <= self.delta2) & (y[rows][:, None] != y[None, :])).any(axis=1)
        return (~bad).astype(float)


class CornelisMeasure:
    """Base = sum_a R_a(x,y), R_a = max(0, 1 - |a(x)-a(y)|/s_a); R_B = max(0, S - (|B|-1))."""
    def __init__(self, sigma):
        self.sigma = np.asarray(sigma, float)

    def init_base(self, n):
        return np.zeros((n, n), dtype=_dtype(n))

    def contrib(self, X, a, rows):
        s = self.sigma[a]
        if s <= 0:
            return np.ones((len(rows), X.shape[0]))
        return np.maximum(0.0, 1.0 - np.abs(X[rows, a][:, None] - X[None, :, a]) / s)

    def combine(self, S, size):
        return np.maximum(0.0, S - (size - 1))

    def row_score(self, RB, rows, y):
        # POS_B(x) = min_{z: d(z) != d(x)} (1 - R_B(z, x)); R_B symmetric, so use row x
        other = y[rows][:, None] != y[None, :]
        return np.where(other, 1.0 - RB, 1.0).min(axis=1)


class SoftFuzzyMeasure(SqEuclid):
    """Hu, An, Yu (2010): POS(x) = max over classes of SD(x, U - X), d = 1 - exp(-D2/delta)."""
    def __init__(self, delta, beta):
        super().__init__(); self.delta, self.beta = float(delta), float(beta)

    def row_score(self, D2, rows, y):
        from SFRS_Hu import soft_distance_rows
        dist = 1.0 - np.exp(-np.asarray(D2, float) / self.delta)
        pos = np.zeros(len(rows))
        for c in np.unique(y):
            comp = np.where(y != c)[0]
            pos = np.maximum(pos, soft_distance_rows(dist[:, comp], self.beta))
        return pos


class WFRSMeasure(SqEuclid):
    """Wang et al. (2025): directed distance with class-variance column scaling,
    R^w(x,u) = w_u exp(-d_B(x->u)); fuzzy decision Dt_c(x) (Eq. 21) in a prepass,
    then lower(Dt_c)(x) = min_u max(1 - R^w(x,u), Dt_c(u)) and POS = max_c."""
    def __init__(self, col_scale, w, classes):
        super().__init__(col_scale); self.w, self.classes = np.asarray(w, float), list(classes)

    def _rw(self, dB):
        return np.exp(-np.asarray(dB, float)) * self.w[None, :]

    def prepass(self, dB, rows, y):
        Rw = self._rw(dB); tot = Rw.sum(axis=1); safe = np.where(tot > 0, tot, 1.0)
        return np.stack([np.where(tot > 0, (Rw * (y == c)[None, :]).sum(axis=1) / safe, 0.0)
                         for c in self.classes], axis=1)          # (rows, r)

    def row_score(self, dB, rows, y):
        Rw = self._rw(dB)
        pos = np.zeros(len(rows))
        for t, c in enumerate(self.classes):
            pos = np.maximum(pos, np.min(np.maximum(1.0 - Rw, self.pre[:, t][None, :]), axis=1))
        return pos
