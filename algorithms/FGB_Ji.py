"""FGBERS of Ji, Peng, Zhao and Yao (2023), "Extended rough sets model based on
fuzzy granular ball and its attribute reduction", Information Sciences 640,
119071.  Reimplemented from the paper (the earlier implementation FGB_AR.py, which computed a
per-object fuzzy purity and built no granular balls).

Measure (Defs. 8-14, Eqs. 12-20 of the paper):
  ball k: centre C_k = mean of members (Eq. 12), radius R_k = max dist (Eq. 13),
          purity P_k = max_t |X_t| / |U'_k| (Eq. 14), label = majority class.
  BC_k  = min over members x with label != ball label of dist(x, C_k) / R_k
          (Eq. 15 with BF(x, C_k) = 1 - dist/R_k);  BC_k = 1 if the ball is pure.
  SC(x) = 1 if label(x) = ball label, else BC_k; if R_k = 0 and P_k < 1, SC = 0
          for every member (Eq. 16).
  lower(D_i, C_k) = sum_x SC(x) (Eq. 17); FPOS_B = sum_k lower (Eq. 19);
  FR_B = FPOS_B / |U| (Eq. 20).
Ball generation (Algorithm 1): k = number of classes, initial centres = class
means, k-means; every ball with purity < delta is split again in the same way
using the classes it contains. The pseudo-code shows one level; we recurse
until purity >= delta or the ball cannot be split (identical points, or the
split returns one cluster). These two stopping cases are our reading.
"Generate balls" is read as k-means initialised at the class means and run to
convergence (Lloyd, max 100 iterations).
Search (Algorithm 2): rank attributes by FR of the single attribute (desc);
stage 2 adds a in that order if FR(Red + a) > current FR; stage 3 repeats the
rule once over the attributes not selected.  Paper setting: delta = 0.75.
"""
from __future__ import annotations

import numpy as np


def _kmeans(X: np.ndarray, centres: np.ndarray, iters: int = 100) -> np.ndarray:
    lab = None
    for _ in range(iters):
        d = ((X[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        new = d.argmin(axis=1)
        if lab is not None and np.array_equal(new, lab):
            break
        lab = new
        for j in range(centres.shape[0]):
            pts = X[lab == j]
            if len(pts):
                centres[j] = pts.mean(axis=0)
    return lab


def generate_balls(X: np.ndarray, y: np.ndarray, delta: float, idx=None, depth: int = 0):
    """Return a list of index arrays, one per granular ball."""
    if idx is None:
        idx = np.arange(X.shape[0])
    classes = np.unique(y[idx])
    if len(classes) == 1:
        return [idx]
    centres = np.array([X[idx][y[idx] == c].mean(axis=0) for c in classes])
    lab = _kmeans(X[idx], centres.copy())
    balls = []
    parts = [idx[lab == j] for j in range(len(classes)) if np.any(lab == j)]
    if len(parts) == 1 or depth > 200:          # cannot split further
        return [idx]
    for p in parts:
        cnt = np.bincount(y[p])
        if cnt.max() / p.size >= delta:
            balls.append(p)
        else:
            balls.extend(generate_balls(X, y, delta, p, depth + 1))
    return balls


def fr_measure(X: np.ndarray, y: np.ndarray, delta: float) -> float:
    n = X.shape[0]
    fpos = 0.0
    for b in generate_balls(X, y, delta):
        lab_b = np.bincount(y[b]).argmax()
        same = y[b] == lab_b
        if same.all():
            fpos += b.size
            continue
        C = X[b].mean(axis=0)
        dist = np.sqrt(((X[b] - C) ** 2).sum(axis=1))
        R = dist.max()
        if R == 0:
            continue                             # Eq. 16: SC = 0 for all members
        bc = dist[~same].min() / R
        fpos += same.sum() + bc * (~same).sum()
    return fpos / n


class FGBJi:
    def __init__(self, delta: float = 0.75):
        self.delta = float(delta)

    def fit_reduction(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.unique(np.asarray(y), return_inverse=True)[1]
        m = X.shape[1]
        fr1 = [fr_measure(X[:, [a]], y, self.delta) for a in range(m)]
        order = list(np.argsort(-np.asarray(fr1), kind="stable"))
        red, old = [], -1.0
        for a in order:                                  # stage 2
            fr = fr_measure(X[:, red + [a]], y, self.delta)
            if fr > old:
                red.append(a); old = fr
        for a in [a for a in order if a not in red]:     # stage 3
            fr = fr_measure(X[:, red + [a]], y, self.delta)
            if fr > old:
                red.append(a); old = fr
        self.reduct_, self.fr_, self.fr_single_ = sorted(int(a) for a in red), float(old), fr1
        return self


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # sanity 1: two well-separated pure classes -> FR = 1
    X = np.vstack([rng.normal(0, .05, (20, 2)), rng.normal(1, .05, (20, 2))]); y = np.r_[np.zeros(20, int), np.ones(20, int)]
    print("separated FR =", fr_measure(X, y, .75), "(expect 1.0)")
    # sanity 2: one ball with a foreign point at the centre -> its SC = 0
    X = np.array([[0., 0.], [1., 0.], [-1., 0.], [0., 1.], [0., -1.]]); y = np.array([1, 0, 0, 0, 0])
    print("foreign at centre FR =", fr_measure(X, y, .75), "(expect 4/5 = 0.8)")
    # sanity 3: foreign point on the rim -> BC = 1, FR = 1 (paper Sec. 4.2 case 2)
    X = np.array([[0., 0.], [.5, 0.], [-.5, 0.], [0., .5], [1., 0.]]); y = np.array([0, 0, 0, 0, 1])
    print("foreign on rim FR =", fr_measure(X, y, .75), "(expect 1.0 when centre-to-foreign = R)")
