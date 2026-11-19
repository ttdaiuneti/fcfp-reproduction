"""Did the zero-gain stop in reducers_r1.forward_threshold ever end the FCFP search below theta?
For every (dataset, fold) of the main run and of the screening run, recompute on the training
part theta = rho * FCFP(C) and FCFP(R) for the stored reduct R. R below theta means the extra
stop (not in Algorithm 1) was used. Output: results/r1/stop_rule_check.csv"""
import ast, os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "algorithms"))
import engine as E
from run_r1 import load, folds
from reducers_r1 import knn_purity
from analyze_r1 import load_rows
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import f_classif
R = os.path.join(ROOT, "results", "r1"); out = os.path.join(R, "stop_rule_check.csv")
done = set()
if os.path.exists(out):
    o = pd.read_csv(out); done = set(zip(o.dataset, o.method, o.fold))
main = load_rows(); main = main[(main.method == "FCFP") & (main.status == "ok")]
extra = [pd.read_csv(os.path.join(R, f)) for f in ("raw_screen.csv", "raw_sens.csv", "raw_delta.csv")]
extra = pd.concat([e[e.status == "ok"] for e in extra])
rows = pd.concat([main[["dataset", "method", "fold", "reduct"]], extra[["dataset", "method", "fold", "reduct"]]])
order = rows.dataset.map(lambda d: len(load(d)[1]))
rows = rows.assign(_n=order).sort_values(["_n", "dataset", "method", "fold"])
cache = {}
for _, r in rows.iterrows():
    key = (r.dataset, r.method, int(r.fold))
    if key in done: continue
    if r.dataset not in cache:
        X, y = load(r.dataset)[:2]; cache = {r.dataset: (np.asarray(X, float), np.asarray(y), folds(np.asarray(y)))}
    X, y, F = cache[r.dataset]; tr = F[int(r.fold)][0]
    Xt = MinMaxScaler().fit(X[tr]).transform(X[tr]); yt = y[tr]
    red = ast.literal_eval(r.reduct)
    if r.method.startswith("FCFPscr@"):
        q = int(r.method.split("@")[1])
        with np.errstate(all="ignore"):
            Fs = np.nan_to_num(f_classif(Xt, yt)[0], nan=0.0)
        keep = np.sort(np.argsort(-Fs, kind="stable")[:min(q, Xt.shape[1])])
        pos = {int(c): i for i, c in enumerate(keep)}; Xt = Xt[:, keep]; red = [pos[a] for a in red]
    delta, k, rho = 0.3, 5, 0.95
    if r.method.startswith("FCFP@"):
        parts = r.method.split("@")[1:]; delta, k = float(parts[0]), int(parts[1]); rho = float(parts[2]) if len(parts) > 2 else 0.95
    if r.method == "FCFP-med":
        from scipy.spatial.distance import pdist
        idx = np.random.default_rng(0).choice(len(yt), size=1500, replace=False) if len(yt) > 1500 else np.arange(len(yt))
        delta = 0.3 * float(np.median(pdist(Xt[idx])))
    meas = E.FCFPMeasure(knn_purity(Xt, yt, k), delta)
    full = E.score(meas, Xt, yt, E.full_base(meas, Xt, range(Xt.shape[1])), None, Xt.shape[1])
    val = E.score(meas, Xt, yt, E.full_base(meas, Xt, red), None, len(red)) if red else 0.0
    row = dict(dataset=r.dataset, method=r.method, fold=int(r.fold), size=len(red), theta=rho * full, value=val,
               below=int(val < rho * full - 1e-9))
    pd.DataFrame([row]).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
    print(row, flush=True)
