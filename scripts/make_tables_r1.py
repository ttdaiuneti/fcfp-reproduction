"""LaTeX tables, generated from the per-fold rows only.

Inputs : results/r1/raw_long.csv, raw_long_b.csv (main comparison),
         results/r1/monotonicity_pairs_{A,B,C}.csv, monotonicity_greedy_{A,C}.csv
Outputs: <manuscript>/tables/r1/*.tex and results/r1/analysis/pairwise_tests.csv
Conventions
  * accuracy in %, one decimal, mean +- std over the 10 folds (pandas std, ddof = 1);
  * '--' = the method did not finish within 1 h on fold 0 (see the text);
  * empty reduct -> majority class of the training part (analyze_r1.load_rows);
  * bold = best mean among the reducers of the row (Full excluded), after rounding;
  * pairwise tests: FCFP against each method on the datasets where both finished,
    Wilcoxon signed-rank, one Holm correction over all comparisons of a classifier;
    W/T/L with tolerance 0.5 accuracy points.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from analyze_r1 import load_rows  # noqa: E402
from run_r1 import load  # noqa: E402

R = os.path.join(ROOT, "results", "r1")
MS = os.path.join(ROOT, "tables", "r1"); os.makedirs(MS, exist_ok=True)
GROUPS = [("Small", ["wine", "glass", "heart", "ecoli", "dermatology", "sonar", "ionosphere"]),
          ("Medium", ["vehicle", "yeast", "segment", "page-blocks"]),
          ("Large $n$", ["spambase", "waveform", "electrical"]),
          ("Large $r$", ["arrhythmia", "semeion", "mfeat-factors", "madelon", "isolet"]),
          ("Gene expression", ["colon", "leukemia", "lung", "prostate", "gli85", "smk-can-187"])]
ORDER = [d for _, g in GROUPS for d in g]
NAME = {"wine": "Wine", "glass": "Glass", "heart": "Heart", "ecoli": "Ecoli", "dermatology": "Dermatology",
        "sonar": "Sonar", "ionosphere": "Ionosphere", "vehicle": "Vehicle", "yeast": "Yeast", "segment": "Segment",
        "page-blocks": "Page-blocks", "spambase": "Spambase", "waveform": "Waveform", "electrical": "Electrical",
        "arrhythmia": "Arrhythmia", "semeion": "Semeion", "mfeat-factors": "Mfeat-factors", "madelon": "Madelon", "isolet": "Isolet",
        "colon": "Colon", "leukemia": "Leukemia", "lung": "Lung", "prostate": "Prostate", "gli85": "GLI-85",
        "smk-can-187": "SMK-CAN-187"}
RS = ["FCFP", "NRS", "FNRS", "SFRS", "GBNRS", "FGB", "WFRS"]
SEL = ["FCFP", "mRMR", "ReliefF", "L1", "AFRSFCV", "Random"]
CLF = {"acc_svm": "SVM", "acc_knn": "3-NN"}


def fold_stats(d, metric):
    ok = d[d.status == "ok"].groupby(["dataset", "method"]).filter(lambda x: len(x) == 10)
    g = ok.groupby(["dataset", "method"])[metric]
    return (100 * g.mean()).unstack(), (100 * g.std()).unstack()


def acc_table(d, metric, methods, label, caption, with_full=True):
    mu, sd = fold_stats(d, metric)
    cols = (["Full"] if with_full else []) + methods
    lines = [r"\begin{minipage}{\linewidth}", r"\centering", rf"\captionof{{table}}{{{caption}}}", rf"\label{{{label}}}",
             r"\resizebox{\linewidth}{!}{%", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{@{}l*{%d}{c}@{}}" % len(cols), r"\toprule",
             "Dataset & " + " & ".join(cols) + r" \\", r"\midrule"]
    for gi, (_, ds_list) in enumerate(GROUPS):
        for ds in ds_list:
            best = max(round(mu.loc[ds, m], 1) for m in methods if not np.isnan(mu.loc[ds].get(m, np.nan)))
            cells = []
            for m in cols:
                v = mu.loc[ds].get(m, np.nan)
                if np.isnan(v):
                    cells.append("--"); continue
                s = f"{v:.1f}"
                if m != "Full" and round(v, 1) == best:
                    s = rf"\mathbf{{{s}}}"
                cells.append(rf"${s} \pm {sd.loc[ds, m]:.1f}$")
            lines.append(NAME[ds] + " & " + " & ".join(cells) + r" \\")
        if gi < len(GROUPS) - 1:
            lines.append(r"\midrule")
    lines.append(r"\midrule")
    complete = [m for m in cols if mu.loc[ORDER, m].notna().all()]
    avg = ["$%.1f$" % mu.loc[ORDER, m].mean() if m in complete else "--" for m in cols]
    lines.append(r"Average & " + " & ".join(avg) + r" \\")
    rk_cols = [m for m in methods if m in complete]
    rk = mu.loc[ORDER, rk_cols].rank(axis=1, ascending=False).mean()
    lines.append(r"Avg.\ rank & " + " & ".join(("$%.2f$" % rk[m]) if m in rk_cols else "--" for m in cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}%", "}", r"\end{minipage}\par"]
    return "\n".join(lines) + "\n"


def size_table(d):
    ok = d[d.status == "ok"].groupby(["dataset", "method"]).filter(lambda x: len(x) == 10)
    mu = ok.groupby(["dataset", "method"])["size"].mean().unstack()
    sd = ok.groupby(["dataset", "method"])["size"].std().unstack()
    lines = [r"\begin{minipage}{\linewidth}", r"\centering",
             r"\captionof{table}{Number of selected attributes (mean $\pm$ std over ten folds). $r$: number of attributes. The selectors of Table~\ref{tab:acc-sel-svm} keep the number of attributes selected by FCFP in the same fold.}",
             r"\label{tab:size}", r"\resizebox{\linewidth}{!}{%", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{@{}lr*{%d}{c}@{}}" % len(RS), r"\toprule",
             r"Dataset & $r$ & " + " & ".join(RS) + r" \\", r"\midrule"]
    for gi, (_, ds_list) in enumerate(GROUPS):
        for ds in ds_list:
            m_all = int(d[(d.dataset == ds)].p.iloc[0])
            cells = []
            for m in RS:
                v = mu.loc[ds].get(m, np.nan)
                cells.append("--" if np.isnan(v) else rf"${v:.1f} \pm {sd.loc[ds, m]:.1f}$")
            lines.append(NAME[ds] + f" & {m_all} & " + " & ".join(cells) + r" \\")
        if gi < len(GROUPS) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}%", "}", r"\end{minipage}\par"]
    return "\n".join(lines) + "\n"


def fmt_t(v, s):
    if np.isnan(v):
        return "--"
    f = (lambda x: f"{x:.2f}") if v < 10 else (lambda x: f"{x:.1f}") if v < 1000 else (lambda x: f"{x:.0f}")
    return rf"${f(v)} \pm {f(s)}$"


def time_table(d):
    ok = d[d.status == "ok"].groupby(["dataset", "method"]).filter(lambda x: len(x) == 10)
    mu = ok.groupby(["dataset", "method"])["time_reduct"].mean().unstack()
    sd = ok.groupby(["dataset", "method"])["time_reduct"].std().unstack()
    lines = [r"\begin{minipage}{\linewidth}", r"\centering",
             r"\captionof{table}{Reduction time in seconds per fold (mean $\pm$ std over ten folds; classifier training excluded). --: more than 1\,h on the first fold.}",
             r"\label{tab:runtime}", r"\resizebox{\linewidth}{!}{%", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{@{}lrr*{%d}{c}@{}}" % len(RS), r"\toprule",
             r"Dataset & $n$ & $r$ & " + " & ".join(RS) + r" \\", r"\midrule"]
    for gi, (_, ds_list) in enumerate(GROUPS):
        for ds in ds_list:
            X, y = load(ds)
            cells = [fmt_t(mu.loc[ds].get(m, np.nan), sd.loc[ds].get(m, np.nan)) for m in RS]
            lines.append(NAME[ds] + f" & {len(y)} & {X.shape[1]} & " + " & ".join(cells) + r" \\")
        if gi < len(GROUPS) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}%", "}", r"\end{minipage}\par"]
    return "\n".join(lines) + "\n"


def dataset_table(d):
    lines = [r"\begin{minipage}{\linewidth}", r"\centering",
             r"\captionof{table}{Benchmark datasets. $n$: objects; $r=|C|$: conditional attributes; $m$: decision classes; min/max: smallest and largest class.}",
             r"\label{tab:datasets}", r"\small", r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
             r"Group & Dataset & $n$ & $r$ & $m$ & min/max \\", r"\midrule"]
    for gi, (gname, ds_list) in enumerate(GROUPS):
        for i, ds in enumerate(ds_list):
            X, y = load(ds); c = np.bincount(y)
            g = rf"\multirow{{{len(ds_list)}}}{{*}}{{{gname}}}" if i == 0 else ""
            lines.append(f"{g} & {NAME[ds]} & {len(y)} & {X.shape[1]} & {len(c)} & {c.min()}/{c.max()}" + r" \\")
        if gi < len(GROUPS) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{minipage}\par"]
    return "\n".join(lines) + "\n"


def pairwise(d):
    rows = []
    for metric in CLF:
        mu, _ = fold_stats(d, metric)
        res = []
        for m in [x for x in RS + SEL + ["Full"] if x != "FCFP"]:
            if m in [r["method"] for r in res]:
                continue
            t = mu.loc[ORDER, ["FCFP", m]].dropna()
            diff = t.FCFP - t[m]
            p = stats.wilcoxon(t.FCFP, t[m]).pvalue if (diff != 0).any() else 1.0
            res.append(dict(metric=metric, method=m, N=len(t), W=int((diff > 0.5).sum()),
                            T=int((diff.abs() <= 0.5).sum()), L=int((diff < -0.5).sum()),
                            mean_diff=diff.mean(), p=p))
        res = sorted(res, key=lambda r: r["p"]); k = len(res); run = 0.0
        for i, r in enumerate(res):
            run = max(run, min(1.0, (k - i) * r["p"])); r["p_holm"] = run
        rows += res
    return pd.DataFrame(rows)


def tests_table(pw):
    order = [m for m in RS + SEL + ["Full"] if m != "FCFP"]
    order = list(dict.fromkeys(order))
    lines = [r"\begin{minipage}{\linewidth}", r"\centering",
             r"\captionof{table}{FCFP against each method: datasets where both finished ($N$), wins/ties/losses of FCFP (tie: within 0.5 points) and Holm-adjusted Wilcoxon $p$-values, per classifier.}",
             r"\label{tab:pairwise}", r"\small", r"\begin{tabular}{@{}lrcccc@{}}", r"\toprule",
             r" & & \multicolumn{2}{c}{SVM} & \multicolumn{2}{c}{3-NN} \\ \cmidrule(lr){3-4}\cmidrule(lr){5-6}",
             r"Method & $N$ & W/T/L & $p_{\mathrm{Holm}}$ & W/T/L & $p_{\mathrm{Holm}}$ \\", r"\midrule"]
    for m in order:
        a = pw[(pw.method == m) & (pw.metric == "acc_svm")].iloc[0]
        b = pw[(pw.method == m) & (pw.metric == "acc_knn")].iloc[0]
        f = lambda p: "$<0.001$" if p < 0.001 else f"${p:.3f}$"
        lines.append(f"{m} & {a.N} & {a.W}/{a['T']}/{a.L} & {f(a.p_holm)} & {b.W}/{b['T']}/{b.L} & {f(b.p_holm)}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{minipage}\par"]
    return "\n".join(lines) + "\n"


def main():
    os.makedirs(MS, exist_ok=True)
    d = load_rows()
    out = {"datasets.tex": dataset_table(d), "size.tex": size_table(d), "runtime.tex": time_table(d)}
    for metric, cname in CLF.items():
        key = "svm" if metric == "acc_svm" else "knn"
        out[f"acc-rs-{key}.tex"] = acc_table(
            d, metric, RS, f"tab:acc-rs-{key}",
            rf"{cname} test accuracy (\%, mean $\pm$ std over ten folds) of the rough-set reducers. Full: all attributes. Bold: best reducer. --: more than 1\,h on the first fold. The last two rows use only the methods that finished on all datasets.")
        out[f"acc-sel-{key}.tex"] = acc_table(
            d, metric, SEL, f"tab:acc-sel-{key}",
            rf"{cname} test accuracy (\%, mean $\pm$ std over ten folds) of FCFP and of selectors that keep the same number of attributes as FCFP. Bold: best selector.")
    pw = pairwise(d); pw.to_csv(os.path.join(R, "analysis", "pairwise_tests.csv"), index=False)
    out["pairwise.tex"] = tests_table(pw)
    for k, v in out.items():
        with open(os.path.join(MS, k), "w") as f:
            f.write(v)
    print(pw.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
