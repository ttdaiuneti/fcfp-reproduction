"""Further LaTeX tables and pgfplots figures, from raw rows only.

Inputs  (results/r1/):
  monotonicity_pairs_{A,B,C}.csv, monotonicity_greedy_{A,B,C}.csv
  raw_adaptive.csv (FCFPa@alpha), raw_long*.csv (FCFP = alpha 1)
  raw_delta.csv (FCFP-med), raw_sfrs_eq.csv (SFRS-eq)
  scalability.csv, scalability_m.csv
  raw_sens.csv, raw_screen.csv (written only when present)
Outputs (manuscript/tables/r1/, manuscript/figures/r1/):
  mono.tex, adaptive.tex, delta.tex, sfrs_eq.tex, scal.tex, sens.tex, screen.tex
Rules
  * monotonicity pairs: |C| <= 60 -> groups A/B (|B1| uniform in [1, |C|-1]);
    |C| > 60 -> group C (|B1| uniform in [1, 30]);
  * accuracy in %, one decimal, mean +- std over the 10 folds (ddof = 1);
  * runtime: mean over the repetitions of a level; a level enters a curve only
    when all of its repetitions finished.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from analyze_r1 import load_rows  # noqa: E402
from make_tables_r1 import GROUPS, ORDER, NAME  # noqa: E402

R = os.path.join(ROOT, "results", "r1")
MAN = ROOT
TAB, FIG = os.path.join(MAN, "tables", "r1"), os.path.join(MAN, "figures", "r1")
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)


def write(path, lines):
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", os.path.relpath(path, MAN))


def head(caption, label, ncol, spec=None):
    return [r"\begin{minipage}{\linewidth}", r"\centering", rf"\captionof{{table}}{{{caption}}}", rf"\label{{{label}}}",
            r"\footnotesize", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}" + (spec or "l" + "c" * ncol) + "@{}}",
            r"\toprule"]


TAIL = [r"\bottomrule", r"\end{tabular}", r"\end{minipage}\par"]


# ---------------------------------------------------------------- monotonicity
MEAS = [("FCFP-fixed", r"$\alpha{=}1$"), ("FCFP-hybrid-0.75", "$0.75$"), ("FCFP-hybrid-0.5", "$0.5$"),
        ("FCFP-hybrid-0.25", "$0.25$"), ("FCFP-adaptive", "$0$"), ("WFRS", "WFRS"), ("SFRS", "SFRS")]


def mono_table():
    ab = pd.concat([pd.read_csv(os.path.join(R, f"monotonicity_pairs_{g}.csv")) for g in "AB"])
    c = pd.read_csv(os.path.join(R, "monotonicity_pairs_C.csv"))
    rows = []
    for ds in ORDER:
        src = ab if ab[ab.dataset == ds].p.iloc[0] <= 60 else c
        rows.append(src[src.dataset == ds].assign(src="AB" if src is ab else "C"))
    d = pd.concat(rows)
    assert set(d.dataset) == set(ORDER) and (d.pairs == 200).all()
    v = d.pivot(index="dataset", columns="measure", values="violations")
    L = head(r"Monotonicity violations among 200 random nested pairs $B_1\subset B_2$ per dataset. "
             r"The FCFP columns differ only in the weight $\omega_\alpha$; $\alpha=1$ is FCFP and $\alpha=0$ the adaptive scheme.",
             "tab:mono", len(MEAS))
    L += [r" & \multicolumn{5}{c}{FCFP with weights $\omega_\alpha$} & & \\ \cmidrule(lr){2-6}",
          "Dataset & " + " & ".join(h for _, h in MEAS) + r" \\", r"\midrule"]
    for gi, (_, g) in enumerate(GROUPS):
        for ds in g:
            L.append(NAME[ds] + " & " + " & ".join(str(int(v.loc[ds, m])) for m, _ in MEAS) + r" \\")
        if gi < len(GROUPS) - 1:
            L.append(r"\midrule")
    L.append(r"\midrule")
    L.append(r"Datasets with a violation & " + " & ".join(str(int((v.loc[ORDER, m] > 0).sum())) for m, _ in MEAS) + r" \\")
    write(os.path.join(TAB, "mono.tex"), L + TAIL)
    stats = {m: dict(datasets=int((v.loc[ORDER, m] > 0).sum()), total=int(v.loc[ORDER, m].sum()),
                     max=int(v.loc[ORDER, m].max()), argmax=v.loc[ORDER, m].idxmax()) for m, _ in MEAS}
    big = d[d.p > 60]
    gene = ["colon", "leukemia", "lung", "prostate", "gli85", "smk-can-187"]
    stats["gene_adaptive"] = int((v.loc[gene, "FCFP-adaptive"] > 0).sum())
    stats["gene_wfrs"] = int((v.loc[gene, "WFRS"] > 0).sum())
    md = d.pivot(index="dataset", columns="measure", values="max_drop")
    stats["maxdrop_adaptive"] = (md.loc[ORDER, "FCFP-adaptive"].max(), md.loc[ORDER, "FCFP-adaptive"].idxmax())
    return stats


def greedy_stats():
    g = pd.concat([pd.read_csv(os.path.join(R, f"monotonicity_greedy_{x}.csv")) for x in "ABC"]).drop_duplicates(["dataset", "measure"])
    neg = g.pivot(index="dataset", columns="measure", values="negative_steps")
    size = g.pivot(index="dataset", columns="measure", values="size")
    early = g.pivot(index="dataset", columns="measure", values="early_stop")
    return dict(n=len(neg), neg_datasets={m: int((neg[m] > 0).sum()) for m in neg},
                early={m: sorted(early.index[early[m] == 1]) for m in early}, size=size)


# ------------------------------------------------------ adaptive -> accuracy
ALPHAS = [("FCFP", r"$1$"), ("FCFPa@0.75", "$0.75$"), ("FCFPa@0.5", "$0.5$"), ("FCFPa@0.25", "$0.25$"), ("FCFPa@0.0", "$0$")]


def adaptive_table():
    a = pd.read_csv(os.path.join(R, "raw_adaptive.csv"))
    main = load_rows(); main = main[main.method == "FCFP"]
    ds = ["glass", "ecoli", "yeast", "vehicle", "segment", "page-blocks"]
    d = pd.concat([a, main[main.dataset.isin(ds)]])
    d = d[d.status == "ok"]
    cnt = d.groupby(["dataset", "method"]).size()
    assert (cnt == 10).all(), cnt[cnt != 10]
    mu = (100 * d.groupby(["dataset", "method"]).acc_svm.mean()).unstack()
    sd = (100 * d.groupby(["dataset", "method"]).acc_svm.std()).unstack()
    sz = d.groupby(["dataset", "method"])["size"].mean().unstack()
    L = head(r"SVM accuracy (\%, mean $\pm$ std) and mean reduct size (in parentheses) with the weights $\omega_\alpha$. "
             r"Algorithm~\ref{alg:forward-backward} and all other settings are unchanged; $\alpha=1$ is FCFP.",
             "tab:adaptive", len(ALPHAS))
    L += [r" & \multicolumn{5}{c}{$\alpha$} \\ \cmidrule(lr){2-6}",
          "Dataset & " + " & ".join(h for _, h in ALPHAS) + r" \\", r"\midrule"]
    for x in ds:
        L.append(NAME[x] + " & " + " & ".join(rf"${mu.loc[x, m]:.1f} \pm {sd.loc[x, m]:.1f}$ ({sz.loc[x, m]:.1f})"
                                              for m, _ in ALPHAS) + r" \\")
    write(os.path.join(TAB, "adaptive.tex"), L + TAIL)
    knn = (100 * d.groupby(["dataset", "method"]).acc_knn.mean()).unstack()
    return mu, sz, knn


# ------------------------------------------------------------- delta
def delta_table():
    dl = pd.read_csv(os.path.join(R, "raw_delta.csv"))
    main = load_rows(); main = main[main.method == "FCFP"]
    rows = []
    L = head(r"FCFP with the absolute scale $\delta=0.3$ and with $\delta$ set to $0.3$ times the median pairwise distance in $C$: "
             r"SVM accuracy (\%) and mean reduct size. `--': not finished within one hour.",
             "tab:delta", 4)
    L += [r" & \multicolumn{2}{c}{$\delta=0.3$} & \multicolumn{2}{c}{median-based $\delta$} \\ \cmidrule(lr){2-3}\cmidrule(lr){4-5}",
          r"Dataset & Acc. & $|R|$ & Acc. & $|R|$ \\", r"\midrule"]
    for gi, (_, g) in enumerate(GROUPS):
        for ds in g:
            a = main[main.dataset == ds]; b = dl[dl.dataset == ds]
            assert len(a) == 10 and len(b) == 10, ds
            ca = rf"${100 * a.acc_svm.mean():.1f}$ & ${a['size'].mean():.1f}$"
            if (b.status == "ok").sum() == 10:
                cb = rf"${100 * b.acc_svm.mean():.1f}$ & ${b['size'].mean():.1f}$"
                rows.append((ds, 100 * a.acc_svm.mean(), 100 * b.acc_svm.mean(), a["size"].mean(), b["size"].mean(),
                             a.time_reduct.mean(), b.time_reduct.mean()))
            else:
                assert (b.status == "timeout").all(), ds
                cb = r"-- & --"
            L.append(NAME[ds] + " & " + ca + " & " + cb + r" \\")
        if gi < len(GROUPS) - 1:
            L.append(r"\midrule")
    write(os.path.join(TAB, "delta.tex"), L + TAIL)
    return pd.DataFrame(rows, columns=["dataset", "acc_abs", "acc_med", "size_abs", "size_med", "t_abs", "t_med"]).set_index("dataset")


# ---------------------------------------------------------- SFRS at equal size
def sfrs_eq_stats():
    q = pd.concat([pd.read_csv(os.path.join(R, f)) for f in ("raw_sfrs_eq.csv", "raw_sfrs_eq_b.csv") if os.path.exists(os.path.join(R, f))])
    q = q[q.method == "SFRS-eq"]   # the file also holds the seeded FCFP rows
    main = load_rows()
    out = []
    for ds, g in q.groupby("dataset"):
        if (g.status == "ok").sum() < 10:
            continue
        f = main[(main.dataset == ds) & (main.method == "FCFP")]
        s = main[(main.dataset == ds) & (main.method == "SFRS")]
        out.append((ds, 100 * f.acc_svm.mean(), 100 * g.acc_svm.mean(), 100 * s.acc_svm.mean() if len(s) == 10 and (s.status == "ok").all() else np.nan,
                    100 * f.acc_knn.mean(), 100 * g.acc_knn.mean(), f["size"].mean(), s["size"].mean() if len(s) else np.nan))
    return pd.DataFrame(out, columns=["dataset", "fcfp_svm", "sfrseq_svm", "sfrs_svm", "fcfp_knn", "sfrseq_knn", "fcfp_size", "sfrs_size"]).set_index("dataset")


# ------------------------------------------------------------- scalability
STYLE = {"FCFP": "black, very thick, mark=*", "NRS": "blue, mark=square*", "FNRS": "cyan!70!black, mark=triangle*",
         "SFRS": "orange!90!black, mark=diamond*", "GBNRS": "red, mark=pentagon*", "FGB": "green!50!black, mark=x",
         "WFRS": "violet, mark=o"}


def curve(d, key):
    ok = d.groupby([key, "method"]).agg(t=("time_reduct", "mean"), k=("status", lambda s: (s == "ok").all()))
    return ok[ok.k].t.unstack()


def scal_figure():
    s = pd.read_csv(os.path.join(R, "scalability.csv")); m = pd.read_csv(os.path.join(R, "scalability_m.csv"))
    panels = [("(a) Letter, $r=16$", curve(s[s.dataset == "letter"], "n"), r"$n$"),
              ("(b) Magic, $r=10$", curve(s[s.dataset == "magic"], "n"), r"$n$"),
              ("(c) SMK-CAN-187, $n=187$", curve(m, "m"), r"$r$")]
    L = [r"\begin{minipage}{\linewidth}", r"\centering", r"\begin{tikzpicture}",
         r"\begin{groupplot}[group style={group size=3 by 1, horizontal sep=1.1cm}, width=0.37\linewidth, height=0.33\linewidth,",
         r"  xmode=log, ymode=log, log basis x=10, tick label style={font=\scriptsize}, label style={font=\scriptsize},",
         r"  title style={font=\scriptsize}, ylabel style={yshift=-3pt}, grid=major, grid style={gray!20}, every axis plot/.append style={mark size=1.4pt}]"]
    for i, (title, c, xl) in enumerate(panels):
        L.append(rf"\nextgroupplot[title={{{title}}}, xlabel={{{xl}}}" + (r", ylabel={Reduction time (s)}" if i == 0 else "")
                 + (r", legend to name=scallegend, legend columns=7, legend style={font=\scriptsize, draw=none}" if i == 0 else "") + "]")
        for meth in STYLE:
            if meth not in c:
                continue
            pts = c[meth].dropna()
            if len(pts) == 0:
                continue
            L.append(rf"\addplot[{STYLE[meth]}] coordinates {{" + " ".join(f"({x},{y:.3f})" for x, y in pts.items()) + "};")
            if i == 0:
                L.append(rf"\addlegendentry{{{meth}}}")
    L += [r"\end{groupplot}", r"\end{tikzpicture}", r"\\[2pt]\pgfplotslegendfromname{scallegend}",
          r"\captionof{figure}{Mean reduction time against the number of objects $n$ (a, b) and of attributes $r$ (c), "
          r"over three random subsamples per point. A missing point means that the method did not finish within two hours.}",
          r"\label{fig:scal}", r"\end{minipage}\par"]
    write(os.path.join(FIG, "scal.tex"), L)
    cs = {k: curve(s[s.dataset == k], "n") for k in ["letter", "magic"]}
    status = s.groupby(["dataset", "n", "method"]).status.agg(lambda x: ",".join(sorted(set(x))))
    return cs, curve(m, "m"), status


def slope(c, meth):
    p = c[meth].dropna(); x, y = np.log(p.index.values.astype(float)), np.log(p.values)
    return float(np.polyfit(x, y, 1)[0])


# ---------------------------------------------------------- sensitivity
def sens_figure():
    path = os.path.join(R, "raw_sens.csv")
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path)
    d = d[d.method.str.count("@") == 2]
    d = d.assign(delta=d.method.str.split("@").str[1].astype(float), k=d.method.str.split("@").str[2].astype(int))
    done = d.groupby(["dataset", "delta", "k"]).filter(lambda g: (g.status == "ok").sum() == 10)
    acc = (100 * done.groupby(["dataset", "delta", "k"]).acc_svm.mean()).unstack("k")
    sz = done.groupby(["dataset", "delta", "k"])["size"].mean().unstack("k")
    cols = {"wine": "black, mark=*", "ionosphere": "blue, mark=square*", "vehicle": "red, mark=triangle*",
            "segment": "green!50!black, mark=diamond*", "arrhythmia": "orange!90!black, mark=pentagon*", "colon": "violet, mark=o"}
    L = [r"\begin{minipage}{\linewidth}", r"\centering", r"\begin{tikzpicture}",
         r"\begin{groupplot}[group style={group size=2 by 1, horizontal sep=1.4cm}, width=0.5\linewidth, height=0.36\linewidth,",
         r"  xlabel={$\delta$}, xtick={0.1,0.2,0.3,0.4,0.5}, tick label style={font=\scriptsize}, label style={font=\scriptsize},",
         r"  title style={font=\scriptsize}, grid=major, grid style={gray!20}, every axis plot/.append style={mark size=1.4pt}]"]
    for i, (tab, title, yl) in enumerate([(acc, "(a) SVM accuracy", "Accuracy (\\%)"), (sz, "(b) Reduct size", "$|R|$")]):
        L.append(rf"\nextgroupplot[title={{{title}}}, ylabel={{{yl}}}" 
                 + (r", legend to name=senslegend, legend columns=6, legend style={font=\scriptsize, draw=none}" if i == 0 else "") + "]")
        for ds, st in cols.items():
            p = tab.loc[ds][5]
            L.append(rf"\addplot[{st}] coordinates {{" + " ".join(f"({x},{v:.2f})" for x, v in p.items()) + "};")
            if i == 0:
                L.append(rf"\addlegendentry{{{NAME[ds]}}}")
    L += [r"\end{groupplot}", r"\end{tikzpicture}", r"\\[2pt]\pgfplotslegendfromname{senslegend}",
          r"\captionof{figure}{SVM accuracy and mean reduct size of FCFP against the scale $\delta$ ($k_\omega=5$, $\rho=0.95$).}",
          r"\label{fig:sens}", r"\end{minipage}\par"]
    write(os.path.join(FIG, "sens.tex"), L)
    rng = acc.max(axis=1) - acc.min(axis=1)
    print("k_omega range of accuracy per (dataset, delta):"); print(rng.round(1).unstack())
    rho = d0 = pd.read_csv(path)
    rho = rho[rho.method.str.count("@") == 3]
    print((100 * rho.groupby(["dataset", "method"]).acc_svm.mean()).round(1).unstack(), rho.groupby(["dataset", "method"])["size"].mean().round(1).unstack())
    return acc, sz


def screen_table():
    path = os.path.join(R, "raw_screen.csv")
    if not os.path.exists(path):
        return None
    s = pd.read_csv(path)
    main = load_rows()
    ds = ["arrhythmia", "semeion", "mfeat-factors", "madelon", "colon", "leukemia", "lung", "prostate", "gli85", "smk-can-187"]
    qs = [50, 100, 200, 500]
    cnt = s[s.status == "ok"].groupby(["dataset", "method"]).size()
    if not ((cnt == 10).all() and len(cnt) == len(ds) * len(qs)):
        print("screen incomplete:", cnt[cnt < 10].to_dict()); return None
    f = main[(main.method == "FCFP") & main.dataset.isin(ds)]
    full = main[(main.method == "Full") & main.dataset.isin(ds)]
    L = head(r"SVM accuracy (\%) and mean reduct size (in parentheses) of FCFP after keeping the $q$ attributes with the largest "
             r"ANOVA $F$ statistic on the training part. `=': $q\ge r$, so no attribute is removed.", "tab:screen", 6)
    L += [r" & & \multicolumn{4}{c}{$q$} & \\ \cmidrule(lr){3-6}",
          r"Dataset & $r$ & $50$ & $100$ & $200$ & $500$ & no screening \\", r"\midrule"]
    L[L.index(r"\begin{tabular}{@{}lcccccc@{}}")] = r"\begin{tabular}{@{}lrccccc@{}}"
    for x in ds:
        g0 = f[f.dataset == x]; r_ = int(g0.p.iloc[0]); cells = [str(r_)]
        for q in qs:
            if q >= r_:
                cells.append("="); continue
            g = s[(s.dataset == x) & (s.method == f"FCFPscr@{q}")]
            cells.append(rf"${100 * g.acc_svm.mean():.1f}$ ({g['size'].mean():.1f})")
        cells.append(rf"${100 * g0.acc_svm.mean():.1f}$ ({g0['size'].mean():.1f})")
        L.append(NAME[x] + " & " + " & ".join(cells) + r" \\")
    write(os.path.join(TAB, "screen.tex"), L + TAIL)
    acc = (100 * s.groupby(["dataset", "method"]).acc_svm.mean()).unstack()
    acc["FCFP"] = 100 * f.groupby("dataset").acc_svm.mean(); acc["Full"] = 100 * full.groupby("dataset").acc_svm.mean()
    print(acc.round(1)); print(s.groupby(["dataset", "method"])["size"].mean().unstack().round(1))
    return acc


def main():
    st = mono_table(); print("mono", st)
    gs = greedy_stats(); print("greedy neg", gs["neg_datasets"], "early", gs["early"])
    mu, sz, knn = adaptive_table(); print(mu.round(1)); print(sz.round(1)); print(knn.round(1))
    dt = delta_table(); print(dt.round(1))
    print("delta: N", len(dt), "acc up>0.5", int((dt.acc_med > dt.acc_abs + 0.5).sum()), "down>0.5", int((dt.acc_med < dt.acc_abs - 0.5).sum()),
          "mean diff", round((dt.acc_med - dt.acc_abs).mean(), 2), "size ratio median", round((dt.size_med / dt.size_abs).median(), 2))
    se = sfrs_eq_stats(); print(se.round(1)); print("sfrs-eq mean diff svm (fcfp - sfrseq)", round((se.fcfp_svm - se.sfrseq_svm).mean(), 2), "N", len(se),
                                                   "W/T/L", int((se.fcfp_svm > se.sfrseq_svm + .5).sum()), int(((se.fcfp_svm - se.sfrseq_svm).abs() <= .5).sum()), int((se.fcfp_svm < se.sfrseq_svm - .5).sum()))
    cs, cm, status = scal_figure()
    for k, c in cs.items():
        print(k, c.round(1).to_string()); print({m: round(slope(c, m), 2) for m in c if c[m].notna().sum() >= 3})
    print(cm.round(1).to_string()); print({m: round(slope(cm, m), 2) for m in cm})
    print(status[status != "ok"])
    sens_figure()
    screen_table()


if __name__ == "__main__":
    main()
