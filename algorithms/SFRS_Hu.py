"""Soft fuzzy dependency of Hu, An and Yu (2010), "Soft fuzzy rough sets for
robust feature evaluation and selection", Information Sciences 180:4384-4400.
Reimplemented from the paper (the earlier implementation SFRS_AR.py, which clipped the
relation to [eps, 1-eps] and is not the published model).

Model (crisp decision):
  Eq. (10)  R(x,y) = exp(-||x-y||^2 / delta),  d(x,y) = 1 - R(x,y)
            paper settings (Sec. 6.2): delta = 0.15, 2-norm.
  Def. 3    SD(x,Y) = d(x,y*) with y* = argmax_{y in Y} { d(x,y) - beta*m_Y(y) },
            m_Y(y) = |{y' in Y : d(x,y') < d(x,y)}|; paper setting beta = 0.1 (Sec. 6.1).
  Eq. (19)-(20) for a crisp class X:  lower_S(X)(x) = SD(x, U - X).
  Def. 5    POS_S(x) = sup_{X in U/D} lower_S(X)(x),  gamma_S(B) = sum_x POS_S(x) / |U|.

POS follows Definition 5 literally (sup over all classes). For a class X that
does not contain x, U - X contains x itself at d = 0, so its contribution is
small unless the nearer objects of U - X are skipped as noise.

Search (default, `search="original"`): the paper's Table 1 - rank all features
by sequential forward selection on SFD, then keep the prefix with the highest
k-NN (k = 3, Sec. 6.3) cross-validated accuracy. The paper does not give the
number of folds; we use stratified 10-fold CV inside the outer training fold
(seed 42), so no test data are used. Ties go to the shortest prefix.
`search="threshold"` keeps the shared-protocol variant (forward until
gamma_S(B) >= theta*gamma_S(C), then backward pruning) for sensitivity checks.
"""
from __future__ import annotations

import numpy as np


def soft_distance_rows(D: np.ndarray, beta: float) -> np.ndarray:
    """Row-wise soft distance (Definition 3). D: (n, k) distances to the set Y."""
    if D.shape[1] == 0:
        return np.ones(D.shape[0])
    S = np.sort(D, axis=1)
    k = S.shape[1]
    j = np.arange(k)[None, :]
    new = np.ones_like(S, dtype=bool)
    new[:, 1:] = S[:, 1:] > S[:, :-1]
    # m = number of strictly smaller distances = index of first occurrence
    m = np.maximum.accumulate(np.where(new, j, 0), axis=1)
    score = S - beta * m
    arg = np.argmax(score, axis=1)          # first maximiser
    return S[np.arange(S.shape[0]), arg]


class SFRSHu:
    def __init__(self, delta: float = 0.15, beta: float = 0.1, theta: float = 0.95,
                 backward: bool = True, search: str = "original", cv_folds: int = 10, knn_k: int = 3):
        self.delta, self.beta, self.theta, self.backward = float(delta), float(beta), float(theta), bool(backward)
        self.search, self.cv_folds, self.knn_k = search, int(cv_folds), int(knn_k)

    def _pos_from_sq(self, D2: np.ndarray, y: np.ndarray) -> np.ndarray:
        dist = 1.0 - np.exp(-D2 / self.delta)
        pos = np.zeros(y.size)
        for c in np.unique(y):
            comp = np.where(y != c)[0]
            low = soft_distance_rows(dist[:, comp], self.beta)
            pos = np.maximum(pos, low)
        return pos

    def gamma_from_sq(self, D2, y) -> float:
        return float(self._pos_from_sq(D2, np.asarray(y)).mean())

    def fit_reduction(self, X: np.ndarray, y: np.ndarray):
        if self.search == "original":
            return self._fit_original(X, y)
        X = np.asarray(X, dtype=np.float64); y = np.asarray(y)
        n, m = X.shape
        sq = [(X[:, a][:, None] - X[:, a][None, :]) ** 2 for a in range(m)]
        target = self.theta * self.gamma_from_sq(np.sum(sq, axis=0), y)
        B, D2, g = [], np.zeros((n, n)), 0.0
        while g < target - 1e-12 and len(B) < m:
            best, best_g = None, -1.0
            for a in range(m):
                if a in B:
                    continue
                ga = self.gamma_from_sq(D2 + sq[a], y)
                if ga > best_g + 1e-12:
                    best, best_g = a, ga
            B.append(best); D2 = D2 + sq[best]; g = best_g
        if self.backward:
            changed = True
            while changed and len(B) > 1:
                changed = False
                for a in list(B):
                    Dr = D2 - sq[a]
                    gr = self.gamma_from_sq(Dr, y)
                    if gr >= target - 1e-12:
                        B = [b for b in B if b != a]; D2, g, changed = Dr, gr, True
                        break
        self.reduct_, self.gamma_, self.target_ = sorted(B), float(g), float(target)
        return self


    def _fit_original(self, X, y):
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.neighbors import KNeighborsClassifier
        X = np.asarray(X, dtype=np.float64); y = np.asarray(y)
        n, m = X.shape
        sq = [(X[:, a][:, None] - X[:, a][None, :]) ** 2 for a in range(m)]
        rank, D2 = [], np.zeros((n, n))
        while len(rank) < m:                                   # Table 1: full forward ranking
            best, best_g = None, -1.0
            for a in range(m):
                if a in rank:
                    continue
                ga = self.gamma_from_sq(D2 + sq[a], y)
                if ga > best_g + 1e-12:
                    best, best_g = a, ga
            rank.append(best); D2 = D2 + sq[best]
        n_splits = max(2, min(self.cv_folds, int(np.bincount(y).min())))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        accs = [cross_val_score(KNeighborsClassifier(n_neighbors=self.knn_k), X[:, rank[:j]], y, cv=cv).mean()
                for j in range(1, m + 1)]
        j = int(np.argmax(accs))                               # first maximiser = shortest prefix
        self.ranking_, self.prefix_acc_ = rank, accs
        self.reduct_ = sorted(int(a) for a in rank[:j + 1])
        self.gamma_ = self.gamma_from_sq(np.sum([sq[a] for a in self.reduct_], axis=0), y)
        return self


if __name__ == "__main__":
    # Example on p. 4389: SD(x, Y) = 0.49 with beta = 0.06
    D = np.array([[0.11, 0.29, 0.49, 0.50, 0.51, 0.50]])
    print("soft distance:", soft_distance_rows(D, 0.06)[0], "(paper: 0.49)")
    print("hard distance (beta large):", soft_distance_rows(D, 10.0)[0], "(should be 0.11)")
