"""Scalability study.

Part n (objects): letter (20000 x 16) and magic (19020 x 10); stratified
subsamples of n in {2500, 5000, 10000, 20000} objects (magic: 19020 at the top
level), seeds rep = 0, 1, 2; MinMax on the subsample; each reducer runs once on
the whole subsample in its own process with a wall-clock cap.
Part m (attributes): smk-can-187 (187 x 19993); random column subsets of
m in {1000, 2500, 5000, 10000, 19993}, seeds rep = 0, 1, 2.

Rule: if a method exceeds the cap at some size for rep 0, larger sizes and the
other reps of that size are recorded as 'not run' (no change of the cap).
Rows: results/r1/scalability.csv
  part, dataset, n, m, rep, method, status, size, time_reduct, note
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import queue as _queue
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from run_r1 import load, _worker, OUTDIR  # noqa: E402

FIELDS = ["part", "dataset", "n", "m", "rep", "method", "status", "size", "time_reduct", "note"]
N_LEVELS = [2500, 5000, 10000, 20000]
M_LEVELS = [1000, 2500, 5000, 10000, 19993]


def subsample(ds, part, level, rep):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    X, y = load(ds)
    if part == "n":
        if level < len(y):
            X, _, y, _ = train_test_split(X, y, train_size=level, stratify=y, random_state=rep)
    else:
        if level < X.shape[1]:
            cols = np.sort(np.random.default_rng(rep).choice(X.shape[1], size=level, replace=False))
            X = X[:, cols]
    return MinMaxScaler().fit_transform(X), y


def done(path):
    if not os.path.exists(path):
        return {}
    import pandas as pd
    df = pd.read_csv(path)
    return {(r.part, r.dataset, int(r.n if r.part == "n" else r.m), int(r.rep), r.method): r.status
            for r in df.itertuples()}


def write(path, lock, row):
    with lock:
        new = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS); (w.writeheader() if new else None); w.writerow(row)


def blocked(d, part, ds, level, rep, method, levels):
    """True if rep 0 of this level or any smaller level timed out."""
    for lv in levels:
        if lv > level:
            break
        st = d.get((part, ds, lv, 0, method))
        if st in ("timeout", "not run") and not (lv == level and rep == 0):
            return True
    return False


def cell(part, ds, level, rep, method, timeout, path, lock, levels):
    d = done(path)
    if (part, ds, level, rep, method) in d:
        return "skip"
    if rep > 0 and (part, ds, level, 0, method) not in d:
        return "defer"
    if blocked(d, part, ds, level, rep, method, levels):
        X, y = subsample(ds, part, level, rep)
        write(path, lock, dict(part=part, dataset=ds, n=len(y), m=X.shape[1], rep=rep, method=method,
                               status="not run", size="", time_reduct="", note="smaller size or rep 0 exceeded the cap"))
        return "not run"
    smaller = [lv for lv in levels if lv < level]
    if smaller and (part, ds, smaller[-1], 0, method) not in d:
        return "defer"
    X, y = subsample(ds, part, level, rep)
    ctx = mp.get_context("spawn"); q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(q, method, X, y, None, 0)); p.start()
    try:
        status, cols, t, note = q.get(timeout=timeout)
    except _queue.Empty:
        status, cols, t, note = "timeout", [], float(timeout), f"cap {timeout}s"
    if p.is_alive():
        p.terminate()
    p.join()
    write(path, lock, dict(part=part, dataset=ds, n=len(y), m=X.shape[1], rep=rep, method=method, status=status,
                           size=len(cols) if status == "ok" else "", time_reduct=t, note=note.replace("\n", " | ")))
    return status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["n", "m"], required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=7200)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--out", default="scalability.csv")
    a = ap.parse_args()
    path = os.path.join(OUTDIR, a.out); lock = threading.Lock()
    levels = N_LEVELS if a.part == "n" else M_LEVELS
    pending = [(a.part, ds, lv, r, m) for ds in a.datasets for lv in levels for r in range(a.reps) for m in a.methods]
    pending.sort(key=lambda t: (t[2], t[3]))
    # dynamic queue: a deferred cell goes back to the queue and is retried after another cell finishes
    import time
    from concurrent.futures import FIRST_COMPLETED, wait
    running = {}
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        while pending or running:
            d = done(path)
            ready = [t for t in pending if not _needs_wait(d, t, levels)]
            for t in ready[: a.jobs - len(running)]:
                pending.remove(t); running[ex.submit(_safe, *t, a.timeout, path, lock, levels)] = t
            if not running:
                print("stuck:", pending[:5]); break
            fin, _ = wait(list(running), timeout=30, return_when=FIRST_COMPLETED)
            for f in fin:
                t = running.pop(f); r = f.result(); print(t, r, flush=True)
                if r == "defer":
                    pending.append(t)


def _needs_wait(d, t, levels):
    part, ds, lv, rep, m = t
    if rep > 0 and (part, ds, lv, 0, m) not in d:
        return True
    smaller = [x for x in levels if x < lv]
    return bool(smaller) and (part, ds, smaller[-1], 0, m) not in d


def _safe(*args):
    import traceback
    try:
        return cell(*args)
    except Exception:
        traceback.print_exc(); return "error"


if __name__ == "__main__":
    main()
