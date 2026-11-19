"""FCFP against the crisp-decision dependency of Wang et al. (2025, IEEE TFS), with the same
kernel and the same weights (Definition 2: object excluded, tie-sharing; reducers_r1.knn_purity) on the eleven UCI
datasets of the benchmark, loaded and min-max scaled as in run_r1 (whole table)."""
import sys, numpy as np
CODE = __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))
sys.path.insert(0, CODE + "/scripts"); sys.path.insert(0, CODE + "/algorithms")
from run_r1 import load
from reducers_r1 import knn_purity
from sklearn.preprocessing import MinMaxScaler
DELTA, KW = 0.3, 5
rng = np.random.default_rng(0)
def terms(R, w, y):
    n = len(y); own = np.empty(n); union = np.full(n, -np.inf)
    G = 1 - w[None, :] * R
    for c in np.unique(y):
        lower = G[:, y != c].min(1); union = np.maximum(union, lower); own = np.where(y == c, lower, own)
    return own, union
res = []
for name in ["wine", "glass", "heart", "ecoli", "dermatology", "sonar", "ionosphere", "vehicle", "yeast", "segment", "page-blocks"]:
    X, y = load(name)[:2]; X = MinMaxScaler().fit_transform(np.asarray(X, float)); y = np.unique(np.asarray(y), return_inverse=True)[1]
    n, r = X.shape; w = knn_purity(X, y, KW)
    D2 = lambda B: sum((X[:, [a]] - X[:, [a]].T) ** 2 for a in B)
    gd = []
    for _ in range(40):
        B = rng.choice(r, size=rng.integers(1, r + 1), replace=False)
        own, uni = terms(np.exp(-D2(B) / DELTA ** 2), w, y); gd.append(abs(uni.mean() - own.mean()))
    def greedy(kind):
        B = []
        for _ in range(min(5, r)):
            best, bv = None, -1
            base = D2(B) if B else 0
            for a in range(r):
                if a in B: continue
                own, uni = terms(np.exp(-(base + (X[:, [a]] - X[:, [a]].T) ** 2) / DELTA ** 2), w, y)
                v = (own if kind == "own" else uni).mean()
                if v > bv: best, bv = a, v
            B.append(best)
        return B
    g1, g2 = greedy("own"), greedy("union")
    res.append((name, np.mean(gd), g1 == g2)); print(name, n, round(np.mean(gd), 4), g1 == g2, g1, g2, flush=True)
print("max mean diff", round(max(r[1] for r in res), 4), "same top5:", sum(r[2] for r in res), "of", len(res))
