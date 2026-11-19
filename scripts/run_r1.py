"""Runner for all experiments (append-only, resumable).

For every (dataset, fold, method): stratified 10-fold split (seed 42) of the
full table; MinMaxScaler fit on the training part only; the reducer runs on the
training part in its own process with a wall-clock cap; the selected columns
are evaluated with RBF-SVM (C=1, gamma='scale') and 3-NN (Euclidean) trained
on the training part and tested on the held-out fold.

Rows go to results/r1/raw_long.csv (one row per dataset/method/fold):
  dataset, method, fold, n_train, n_test, p, status (ok|timeout|error), size,
  reduct (json list), acc_svm, acc_knn, acc_mean, time_reduct, time_eval, note
A (dataset, method, fold) already present with status ok/timeout is skipped,
so the runner can be stopped and restarted.  Rankers (mRMR, ReliefF, L1,
AFRSFCV) and the Random-|R| control take |R| from the FCFP row of the same
fold, so FCFP must be present first (the scheduler defers them otherwise).

Controls: "Full" (all attributes) and "Random" (|R| attributes drawn uniformly,
seed = 1000 + fold, same |R| as FCFP).
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "algorithms"))
DATA = os.path.join(ROOT, "data", "datasets")
OUTDIR = os.path.join(ROOT, "results", "r1")
FIELDS = ["dataset", "method", "fold", "n_train", "n_test", "p", "status", "size", "reduct",
          "acc_svm", "acc_knn", "acc_mean", "time_reduct", "time_eval", "note"]
RANKERS = {"mRMR", "ReliefF", "L1", "AFRSFCV", "Random", "SFRS-eq"}
ALL_METHODS = ["Full", "FCFP", "NRS", "FNRS", "SFRS", "GBNRS", "FGB", "WFRS",
               "mRMR", "ReliefF", "L1", "AFRSFCV", "Random"]


def load(name):
    df = pd.read_csv(os.path.join(DATA, f"{name}.csv"), header=None)
    X = df.iloc[:, :-1].to_numpy(float)
    y = np.unique(df.iloc[:, -1].to_numpy(), return_inverse=True)[1]
    return X, y


def folds(y, k=10, seed=42):
    from sklearn.model_selection import StratifiedKFold
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return list(StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def run_method(method, Xtr, ytr, size, fold):
    """Return a list of selected column indices."""
    m = Xtr.shape[1]
    if method == "Full":
        return list(range(m))
    if method == "FCFP":
        from reducers_r1 import FCFP; return FCFP().fit_reduction(Xtr, ytr).reduct_
    if method.startswith("FCFP@"):                    # sensitivity: FCFP@delta@k_omega[@rho]
        from reducers_r1 import FCFP
        v = method.split("@")[1:]
        return FCFP(delta=float(v[0]), k_omega=int(v[1]), rho=float(v[2]) if len(v) > 2 else 0.95).fit_reduction(Xtr, ytr).reduct_
    if method.startswith("FCFPa@"):                   # adaptive / hybrid weights, alpha after '@'
        from fcfp_variants import FCFPHybrid
        return FCFPHybrid(alpha=float(method.split("@")[1])).fit_reduction(Xtr, ytr).reduct_
    if method.startswith("FCFPscr@"):                 # ANOVA-F screening to the top q attributes (training part), then FCFP
        from reducers_r1 import FCFP
        from sklearn.feature_selection import f_classif
        q = int(method.split("@")[1])
        with np.errstate(all="ignore"):
            F = np.nan_to_num(f_classif(Xtr, ytr)[0], nan=0.0)
        keep = np.sort(np.argsort(-F, kind="stable")[:min(q, m)])
        return sorted(int(keep[c]) for c in FCFP().fit_reduction(Xtr[:, keep], ytr).reduct_)
    if method == "FCFP-med":                          # delta = 0.3 x median pairwise training distance in C
        from reducers_r1 import FCFP
        from scipy.spatial.distance import pdist
        idx = np.random.default_rng(0).choice(len(ytr), size=1500, replace=False) if len(ytr) > 1500 else np.arange(len(ytr))
        return FCFP(delta=0.3 * float(np.median(pdist(Xtr[idx])))).fit_reduction(Xtr, ytr).reduct_
    if method == "NRS":
        from reducers_r1 import NRS; return NRS().fit_reduction(Xtr, ytr).reduct_
    if method == "FNRS":
        from reducers_r1 import FNRS; return FNRS().fit_reduction(Xtr, ytr).reduct_
    if method == "SFRS":
        from reducers_r1 import SFRS; return SFRS().fit_reduction(Xtr, ytr).reduct_
    if method == "WFRS":
        from reducers_r1 import WFRS; return WFRS().fit_reduction(Xtr, ytr).reduct_
    if method == "GBNRS":
        from GBNRS_Xia import GBNRSXia; return GBNRSXia().fit_reduction(Xtr, ytr).reduct_
    if method == "FGB":
        from FGB_Ji import FGBJi; return FGBJi().fit_reduction(Xtr, ytr).reduct_
    if method in ("mRMR", "ReliefF", "L1"):
        import rankers
        f = {"mRMR": lambda: rankers.mrmr_ranking(Xtr, ytr, k=size),
             "ReliefF": lambda: rankers.relieff_ranking(Xtr, ytr),
             "L1": lambda: rankers.l1_ranking(Xtr, ytr)}[method]
        return sorted(f()[:size])
    if method == "AFRSFCV":
        from AFRSFCV_Wang import AFRSFCV; return sorted(AFRSFCV().fit(Xtr, ytr).ranking_[:size])
    if method == "SFRS-eq":                            # SFRS ranking cut at |R| of FCFP (equal-size comparison)
        from reducers_r1 import forward_ranking
        import engine as E
        return sorted(int(a) for a in forward_ranking(E.SoftFuzzyMeasure(0.15, 0.1), Xtr, ytr, max_len=size))
    if method == "Random":
        return sorted(np.random.default_rng(1000 + fold).choice(m, size=size, replace=False).tolist())
    raise ValueError(method)


PREIMPORT = {"FCFP": "reducers_r1", "FCFP-med": "reducers_r1", "SFRS-eq": "reducers_r1", "NRS": "reducers_r1", "FNRS": "reducers_r1", "SFRS": "reducers_r1",
             "WFRS": "reducers_r1", "GBNRS": "GBNRS_Xia", "FGB": "FGB_Ji", "mRMR": "rankers",
             "ReliefF": "rankers", "L1": "rankers", "AFRSFCV": "AFRSFCV_Wang"}


def _worker(q, method, Xtr, ytr, size, fold):
    try:
        import importlib
        if method.startswith(("FCFP@", "FCFPscr@", "FCFPa@")):
            importlib.import_module("reducers_r1"); importlib.import_module("fcfp_variants"); import sklearn.svm, sklearn.neighbors, sklearn.feature_selection  # noqa: F401,E401
        elif method in PREIMPORT:                      # keep import time out of time_reduct
            importlib.import_module(PREIMPORT[method])
            import sklearn.svm, sklearn.neighbors, sklearn.model_selection, sklearn.linear_model  # noqa: F401
            if method == "ReliefF":
                import skrebate  # noqa: F401
        t = time.perf_counter(); cols = run_method(method, Xtr, ytr, size, fold)
        q.put(("ok", [int(c) for c in cols], time.perf_counter() - t, ""))
    except Exception:
        q.put(("error", [], 0.0, traceback.format_exc()[-500:]))


def evaluate(Xtr, ytr, Xte, yte, cols):
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    if not cols:
        return float("nan"), float("nan")
    a = SVC(C=1.0, kernel="rbf", gamma="scale").fit(Xtr[:, cols], ytr).score(Xte[:, cols], yte)
    b = KNeighborsClassifier(n_neighbors=3).fit(Xtr[:, cols], ytr).score(Xte[:, cols], yte)
    return float(a), float(b)


def done_keys(path):
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    return {(r.dataset, r.method, int(r.fold)): r for r in df.itertuples() if r.status in ("ok", "timeout")}


def task(ds, method, fold, timeout, path, lock):
    from sklearn.preprocessing import MinMaxScaler
    X, y = load(ds)
    tr, te = folds(y)[fold]
    sc = MinMaxScaler().fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te]); ytr, yte = y[tr], y[te]
    if fold > 0:
        r0 = done_keys(path).get((ds, method, 0))
        if r0 is not None and r0.status == "timeout":       # fold 0 did not finish within the cap
            row = dict(dataset=ds, method=method, fold=fold, n_train=len(tr), n_test=len(te), p=X.shape[1],
                       status="timeout", size="", reduct="[]", acc_svm=float("nan"), acc_knn=float("nan"),
                       acc_mean=float("nan"), time_reduct=float("nan"), time_eval=0.0,
                       note="not run: fold 0 exceeded the cap")
            with lock:
                with open(path, "a", newline="") as f:
                    csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
            return "timeout"
        if r0 is None:
            return "defer"
    size = None
    if method in RANKERS:
        r = done_keys(path).get((ds, "FCFP", fold))
        if r is None or r.status != "ok":
            return "defer"
        size = int(r.size)
    ctx = mp.get_context("spawn")
    q = ctx.Queue(); p = ctx.Process(target=_worker, args=(q, method, Xtr, ytr, size, fold)); p.start()
    import queue as _queue
    try:
        status, cols, tr_time, note = q.get(timeout=timeout)   # read before join: large results block the pipe
    except _queue.Empty:
        status, cols, tr_time, note = "timeout", [], float(timeout), f"cap {timeout}s"
    if p.is_alive():
        p.terminate()
    p.join()
    t = time.perf_counter()
    acc_svm, acc_knn = evaluate(Xtr, ytr, Xte, yte, cols) if status == "ok" else (float("nan"),) * 2
    row = dict(dataset=ds, method=method, fold=fold, n_train=len(tr), n_test=len(te), p=X.shape[1],
               status=status, size=len(cols) if status == "ok" else "", reduct=json.dumps(cols),
               acc_svm=acc_svm, acc_knn=acc_knn, acc_mean=np.nanmean([acc_svm, acc_knn]) if status == "ok" else float("nan"),
               time_reduct=tr_time, time_eval=time.perf_counter() - t, note=note.replace("\n", " | "))
    with lock:
        new = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
    return status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--methods", nargs="+", default=ALL_METHODS)
    ap.add_argument("--folds", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--timeout", type=float, default=7200)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--out", default="raw_long.csv")
    ap.add_argument("--seed_from", default=None, help="copy finished rows of these methods from another file")
    ap.add_argument("--seed_methods", nargs="*", default=[])
    a = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True); path = os.path.join(OUTDIR, a.out)
    if a.seed_from:
        src = pd.read_csv(os.path.join(OUTDIR, a.seed_from))
        src = src[src.method.isin(a.seed_methods) & src.dataset.isin(a.datasets) & src.status.isin(["ok", "timeout"])]
        have = done_keys(path)
        src = src[[(r.dataset, r.method, int(r.fold)) not in have for r in src.itertuples()]]
        if len(src):
            src = src.assign(note=src.note.fillna("").astype(str).radd("from " + a.seed_from + "; "))
            src[FIELDS].to_csv(path, mode="a", header=not os.path.exists(path), index=False)
            print("seeded", len(src), "rows from", a.seed_from)
    import threading
    from concurrent.futures import ThreadPoolExecutor
    lock = threading.Lock()
    pending = [(d, m, f) for d in a.datasets for f in a.folds for m in a.methods]
    pending.sort(key=lambda t: (t[2] > 0, t[1] in RANKERS))   # fold 0 first, then reducers before rankers
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        while pending:
            done = done_keys(path)
            todo = [t for t in pending if t not in done]
            if not todo:
                break
            res = list(ex.map(lambda t: _task_safe(t[0], t[1], t[2], a.timeout, path, lock), todo))
            for t, r in zip(todo, res):
                print(t, r, flush=True)
            pending = [t for t, r in zip(todo, res) if r == "defer"]
            if pending and all(r == "defer" for r in res):
                print("deferred tasks need FCFP rows that are missing:", pending[:5]); break


def _task_safe(*args):
    try:
        return task(*args)
    except Exception:
        traceback.print_exc(); return "error"


if __name__ == "__main__":
    main()
