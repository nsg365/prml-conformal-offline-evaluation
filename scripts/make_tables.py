#!/usr/bin/env python3
"""LaTeX tables for the paper, generated from the launch-study summaries."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "results/launch_study"
OUT = ROOT / "paper/tables"
# The four core estimators, in plain language (see the terms table in the paper).
CONFIGS = [("dm", "logreg", "Click model (LR)"), ("dm", "hgb", "Click model (GBT)"), ("ips", "none", "Reweighting"),
           ("dr", "hgb", "Combined (GBT)")]
SLIDE_TABLES = ROOT / "slides/tables"


def tabular_only(lines: list[str]) -> str:
    """The tabular part of a table float, for \\input inside a Beamer frame."""
    start = next(i for i, l in enumerate(lines) if l.startswith(r"\begin{tabular}"))
    end = next(i for i, l in enumerate(lines) if l.startswith(r"\end{tabular}"))
    return "\n".join(lines[start:end + 1]) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SLIDE_TABLES.mkdir(parents=True, exist_ok=True)
    semi = pd.read_csv(STUDY / "semi_within.csv").query("alpha == 0.05 and n_calib == 50")
    obd = pd.read_csv(STUDY / "obd_temporal.csv").query("alpha == 0.05")

    # Main results table: real data (temporal) and controlled experiments (m = 50).
    blocks = [("Real data (60 decisions, history = earlier windows)", obd),
              ("Controlled experiments (exact truth, $m=50$ past decisions)", semi)]
    rules = [("wald", "Usual"), ("wald_hour", "Hour"), ("emp_bernstein", "Bern."), ("clr", "Ours")]
    metrics = [("coverage", "Coverage"), ("false_launch_rate", "Wrong-launch rate"), ("power", "Power")]
    n_cols = 1 + len(rules) * len(metrics)
    lines = [r"\begin{table*}[t]", r"\centering",
             r"\caption{Main results. Coverage: how often estimate $-$ margin was below the true improvement (target 0.95; "
             r"values at or above it in bold). Wrong-launch rate: share of recommenders that were not better but were launched. "
             r"Power: how often a better recommender was launched. Usual: 1.645 standard errors; Hour: usual margin with "
             r"hour-grouped standard errors (real data only); Bern.: Bernstein bound; Ours: learned margin.}",
             r"\label{tab:results}", r"\small", r"\setlength{\tabcolsep}{4.5pt}",
             r"\begin{tabular}{l" + "r" * (n_cols - 1) + "}", r"\toprule",
             "& " + " & ".join(rf"\multicolumn{{{len(rules)}}}{{c}}{{{name}}}" for _, name in metrics) + r" \\",
             "".join(rf"\cmidrule(lr){{{2 + i * len(rules)}-{1 + (i + 1) * len(rules)}}}" for i in range(len(metrics))),
             "Estimator & " + " & ".join(" & ".join(n for _, n in rules) for _ in metrics) + r" \\"]
    for title, frame in blocks:
        lines += [r"\midrule", rf"\multicolumn{{{n_cols}}}{{l}}{{\emph{{{title}}}}} \\"]
        for est, rm, name in CONFIGS:
            sub = frame[(frame["estimator"] == est) & (frame["reward_model"] == rm)].set_index("rule")
            cells = []
            for metric, _ in metrics:
                for rule, _ in rules:
                    if rule not in sub.index:
                        cells.append("--")
                        continue
                    v = sub.loc[rule, metric]
                    text = f"{v:.3f}".lstrip("0") if v < 1 else "1.00"
                    if metric == "coverage" and v >= 0.95 - 1e-9:
                        text = rf"\textbf{{{text}}}"
                    cells.append(text)
            lines.append(rf"{name} & {' & '.join(cells)} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (OUT / "results.tex").write_text("\n".join(lines) + "\n")
    (SLIDE_TABLES / "results.tex").write_text(tabular_only(lines))

    # Real-data error inflation: realised error spread relative to reported uncertainty.
    exp = pd.read_parquet(ROOT / "results/experiments/obd_experiments.parquet")
    labels = {(est, rm): name for est, rm, name in CONFIGS}
    rows = []
    for key, name in labels.items():
        g = exp[(exp["estimator"] == key[0]) & (exp["reward_model"] == key[1])]
        err = g["estimate"] - g["true_delta"]
        def inflation(se_col: str, truth_col: str) -> float:
            return float(np.sqrt(err.var() / ((g[se_col] ** 2).mean() + (g[truth_col] ** 2).mean())))

        rows.append({"estimator": key[0], "reward_model": key[1], "label": name, "bias": err.mean(),
                     "rmse": np.sqrt((err ** 2).mean()), "mean_se": g["se"].mean(),
                     "inflation": inflation("se", "truth_se"),
                     "inflation_impression": inflation("se_impression", "truth_se_impression"),
                     "inflation_hour": inflation("se_hour", "truth_se_impression")})
    infl = pd.DataFrame(rows)
    infl.to_csv(STUDY / "obd_error_inflation.csv", index=False)
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{How wrong the real estimates are compared with the error they report (84 real decisions; bias and RMSE "
             r"in units of $10^{-4}$ click rate). Inflation $=$ actual spread of errors / reported standard error, using the usual, "
             r"page-grouped or hour-grouped standard error. 1.0 means the reported error is honest.}",
             r"\label{tab:inflation}", r"\small", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"& & & & \multicolumn{3}{c}{Inflation}\\", r"\cmidrule(lr){5-7}",
             r"Estimator & Bias & RMSE & se & Usual & Page & Hour\\", r"\midrule"]
    for r in infl.itertuples():
        lines.append(f"{r.label} & {r.bias * 1e4:+.1f} & {r.rmse * 1e4:.1f} & {r.mean_se * 1e4:.1f} & "
                     f"{r.inflation:.2f} & {r.inflation_impression:.2f} & {r.inflation_hour:.2f}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "inflation.tex").write_text("\n".join(lines) + "\n")
    # Adaptive CLR versus CLR: power difference by size of the history.
    within = pd.read_csv(STUDY / "semi_within.csv").query("alpha == 0.05")
    cross = pd.read_csv(STUDY / "semi_cross_dataset.csv").query("alpha == 0.05").assign(n_calib=-1)
    temporal = pd.read_csv(STUDY / "obd_temporal.csv").query("alpha == 0.05").assign(n_calib=-2)
    both = pd.concat([within, cross, temporal])
    settings = [(20, "$m{=}20$"), (50, "$m{=}50$"), (100, "$m{=}100$"), (-1, "Large"), (-2, "Real")]
    configs = [c for c in CONFIGS if c[0] != "dm" or c[1] == "hgb"]
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Adaptive learned margin: change in power compared with the plain learned margin "
             r"(percentage points). $m$: number of past decisions; Large: history from five other datasets "
             r"($m\approx2{,}240$); Real: real-data history (24--78). Model: difficulty model (GBT = gradient-boosted trees). "
             r"Coverage stays within 0.91--1.00.}",
             r"\label{tab:adaptive}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{ll" + "r" * len(settings) + "}", r"\toprule",
             "Estimator & Model & " + " & ".join(n for _, n in settings) + r"\\", r"\midrule"]
    for est, rm, name in configs:
        for kind, kname in (("aclr_ridge", "Ridge"), ("aclr_gbt", "GBT")):
            cells = []
            for m, _ in settings:
                sub = both[(both["estimator"] == est) & (both["reward_model"] == rm) & (both["n_calib"] == m)]
                diff = 100 * (sub.loc[sub["rule"] == kind, "power"].iloc[0] - sub.loc[sub["rule"] == "clr", "power"].iloc[0])
                cells.append(f"{diff:+.1f}".replace("-", "$-$"))
            lines.append(f"{name if kind == 'aclr_ridge' else ''} & {kname} & " + " & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    # The paper shows this comparison as a figure; the full numbers (ridge too) go on a backup slide.
    (SLIDE_TABLES / "adaptive.tex").write_text(tabular_only(lines))
    print(f"tables written to {OUT}")


if __name__ == "__main__":
    main()
