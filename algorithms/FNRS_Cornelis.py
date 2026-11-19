"""Fuzzy-rough dependency of Cornelis, Jensen, Hurtado and Slezak (2010),
"Attribute selection with fuzzy decision reducts", Information Sciences
180(2):209-224.  Reimplemented from the paper (replaces the earlier FNRS_AR.py).

Model (crisp decision, quantitative conditional attributes):
  Eq. (15)  R_a(x,y) = max(min((a(y)-a(x)+s_a)/s_a, (a(x)-a(y)+s_a)/s_a), 0)
                     = max(0, 1 - |a(x)-a(y)|/s_a),   s_a = std of a
  Eq. (16)  R_B(x,y) = T_L over a in B of R_a(x,y)
                     = max(0, sum_a R_a(x,y) - (|B|-1))   (Lukasiewicz t-norm)
  Prop. 1 + Eq. (17) with I_L:
            POS_B(y) = min_{z: d(z) != d(y)} (1 - R_B(z,y))   (1 if no such z)
  Eq. (21)  gamma_B = |POS_B| / |POS_A|   (sigma-counts)
  Def. 1    B is a fuzzy decision superreduct to degree alpha if gamma_B >= alpha.

Search (shared across reducers): forward greedy adding the
attribute with the largest gamma until gamma_B >= alpha; then backward
removal of attributes whose deletion keeps gamma >= alpha.

Standard deviation: the paper does not state the estimator; Example 4 reports
s_a4 = 16.385 and s_a5 = 131.176 on its 7 objects, which test_example4()
uses to fix the choice (see that function).
"""
from __future__ import annotations

import numpy as np


class FNRSCornelis:
    def __init__(self, alpha: float = 0.95, ddof: int = 1, backward: bool = True):
        self.alpha = float(alpha)
        self.ddof = int(ddof)
        self.backward = bool(backward)

    # ----------------------------------------------------------- relations
    def _attr_relation(self, col: np.ndarray, s: float) -> np.ndarray:
        if s <= 0:  # constant attribute: all objects indiscernible on a
            return np.ones((col.size, col.size), dtype=np.float64)
        diff = np.abs(col[:, None] - col[None, :])
        return np.maximum(0.0, 1.0 - diff / s)

    def fit_relations(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        self.sigma_ = X.std(axis=0, ddof=self.ddof)
        self._X = X
        return self

    def _pos_from_sum(self, S: np.ndarray, size: int, y: np.ndarray) -> np.ndarray:
        RB = np.maximum(0.0, S - (size - 1))
        other = y[:, None] != y[None, :]          # z (row) vs y (col)
        vals = np.where(other, 1.0 - RB, 1.0)
        return vals.min(axis=0)

    def pos(self, cols, y: np.ndarray) -> np.ndarray:
        cols = list(cols)
        n = self._X.shape[0]
        if not cols:
            return np.zeros(n)
        S = np.zeros((n, n))
        for a in cols:
            S += self._attr_relation(self._X[:, a], self.sigma_[a])
        return self._pos_from_sum(S, len(cols), np.asarray(y))

    def gamma(self, cols, y) -> float:
        return float(self.pos(cols, y).sum() / self._posA_sum)

    # -------------------------------------------------------------- search
    def fit_reduction(self, X: np.ndarray, y: np.ndarray):
        y = np.asarray(y)
        self.fit_relations(X)
        n, m = self._X.shape
        rel = [self._attr_relation(self._X[:, a], self.sigma_[a]) for a in range(m)]
        S_all = np.sum(rel, axis=0)
        self._posA_sum = float(self._pos_from_sum(S_all, m, y).sum())
        if self._posA_sum <= 0:
            self.reduct_ = list(range(m)); self.gamma_ = 0.0
            return self
        B: list[int] = []
        S = np.zeros((n, n))
        g = 0.0
        while g < self.alpha - 1e-12 and len(B) < m:
            best, best_g = None, -1.0
            for a in range(m):
                if a in B:
                    continue
                ga = self._pos_from_sum(S + rel[a], len(B) + 1, y).sum() / self._posA_sum
                if ga > best_g + 1e-12:
                    best, best_g = a, ga
            B.append(best); S = S + rel[best]; g = best_g
        if self.backward:
            changed = True
            while changed and len(B) > 1:
                changed = False
                for a in list(B):
                    rest = [b for b in B if b != a]
                    Sr = S - rel[a]
                    gr = self._pos_from_sum(Sr, len(rest), y).sum() / self._posA_sum
                    if gr >= self.alpha - 1e-12:
                        B, S, g, changed = rest, Sr, gr, True
                        break
        self.reduct_ = sorted(B)
        self.gamma_ = float(g)
        return self


def test_example4():
    """Cornelis et al. (2010), Examples 1, 4 and 7 (Pima sample, B = {a4, a5})."""
    X = np.array([
        [1, 101, 50, 15, 36, 24.2, 0.526, 26],
        [8, 176, 90, 34, 300, 33.7, 0.467, 58],
        [7, 150, 66, 42, 342, 34.7, 0.718, 42],
        [7, 187, 68, 39, 304, 37.7, 0.254, 41],
        [0, 100, 88, 60, 110, 46.8, 0.962, 31],
        [0, 105, 64, 41, 142, 41.5, 0.173, 22],
        [1, 95, 66, 13, 38, 19.6, 0.334, 25]], dtype=float)
    y = np.array([0, 1, 0, 1, 0, 0, 0])
    out = {}
    for ddof in (0, 1):
        m = FNRSCornelis(ddof=ddof).fit_relations(X)
        out[ddof] = (m.sigma_[3], m.sigma_[4], m.pos([3, 4], y), m.pos(range(8), y))
    return out


if __name__ == "__main__":
    for ddof, (s4, s5, posB, posA) in test_example4().items():
        print(f"ddof={ddof}: s_a4={s4:.3f} s_a5={s5:.3f}")
        print("  POS_B =", np.round(posB, 3), " gamma_B =", round(posB.sum() / posA.sum(), 3))
        print("  POS_A =", np.round(posA, 3))
    print("paper: s_a4=16.385 s_a5=131.176; POS_B=(1,0.808,0.473,0.473,1,1,1); gamma_B=0.822; POS_A = all 1")
