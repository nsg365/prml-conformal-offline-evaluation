#!/usr/bin/env python3
"""Paper and slide figures from the launch-study summaries."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from scipy import stats  # noqa: E402

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import decision  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments/obd_experiments.parquet"
PROCESSED = ROOT / "data/processed"
USUAL, OURS, BERN, HOUR, MAGENTA = "#eb6834", "#2a78d6", "#eda100", "#1baf7a", "#e87ba4"
HIST, SURFACE = "#9ec5f4", "white"
DARK = False  # set by use_dark_theme(); dark PNGs are only used by the web app
STUDY = ROOT / "reports/launch_study"
OUT = ROOT / "paper/figures"

# Validated categorical palette (fixed slot order) with a marker per rule so
# identity never depends on colour alone.
RULES = {
    "clr": ("Learned margin (ours)", "#2a78d6", "o"),
    "wald": ("Usual margin", "#eb6834", "s"),
    "wald_hour": ("Usual, hour-grouped", "#1baf7a", "^"),
    "emp_bernstein": ("Bernstein bound", "#eda100", "D"),
}
# The four core estimators, in plain language.
CONFIG_LABELS = {
    ("dm", "logreg"): "Click model\n(log. reg.)",
    ("dm", "hgb"): "Click model\n(boosted trees)",
    ("ips", "none"): "Reweighting",
    ("dr", "hgb"): "Combined\n(boosted trees)",
}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


def use_dark_theme() -> None:
    """Switch every colour to its validated dark-surface step (same hues)."""
    global DARK, USUAL, OURS, BERN, HOUR, MAGENTA, HIST, SURFACE, INK, MUTED, GRID
    DARK = True
    USUAL, OURS, BERN, HOUR, MAGENTA = "#d95926", "#3987e5", "#c98500", "#199e70", "#d55181"
    HIST, SURFACE = "#1c5cab", "#1a1a19"
    INK, MUTED, GRID = "#f0efec", "#c3c2b7", "#33332f"
    for key, colour in (("clr", OURS), ("wald", USUAL), ("wald_hour", HOUR), ("emp_bernstein", BERN)):
        label, _, marker = RULES[key]
        RULES[key] = (label, colour, marker)
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "legend.labelcolor": INK,
    })


def save(fig, name: str) -> None:
    """PDF for the paper and slides, PNG for the web app; dark mode writes *_dark.png only."""
    if DARK:
        fig.savefig(OUT / f"{name}_dark.png", bbox_inches="tight", dpi=200)
        return
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=200)


plt.rcParams.update({
    "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42,
})


def dot_panel(ax, frame: pd.DataFrame, metric: str, rules: list[str], target: float | None, ylabel: str) -> None:
    configs = [c for c in CONFIG_LABELS if c in set(zip(frame["estimator"], frame["reward_model"]))]
    offsets = {r: (i - (len(rules) - 1) / 2) * 0.14 for i, r in enumerate(rules)}
    for rule in rules:
        label, colour, marker = RULES[rule]
        sub = frame[frame["rule"] == rule].set_index(["estimator", "reward_model"])
        ys = [sub.loc[c, metric] for c in configs]
        xs = [i + offsets[rule] for i in range(len(configs))]
        ax.scatter(xs, ys, s=26, color=colour, marker=marker, label=label, zorder=3,
                   edgecolors=SURFACE, linewidths=0.6)
    if target is not None:
        ax.axhline(target, color=INK, lw=1, ls=(0, (4, 3)), zorder=2)
        ax.text(len(configs) - 0.55, target - 0.005, f"target {target:.2f}", va="top", ha="right", color=INK, fontsize=6.5)
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels([CONFIG_LABELS[c] for c in configs], fontsize=6.5)
    ax.set_ylabel(ylabel)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def semi_figure() -> None:
    within = pd.read_csv(STUDY / "semi_within.csv").query("alpha == 0.05 and n_calib == 50")
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 3.4), sharex=True)
    rules = ["wald", "emp_bernstein", "clr"]
    dot_panel(axes[0], within, "coverage", rules, 0.95, "Coverage")
    dot_panel(axes[1], within, "power", rules, None, "Power (good versions launched)")
    axes[0].set_ylim(0.6, 1.03)
    axes[1].set_ylim(0, 1.0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.55, 1.04),
               handletextpad=0.1, columnspacing=0.7)
    fig.tight_layout(h_pad=0.6)
    save(fig, "semi_coverage_power")
    plt.close(fig)


def obd_figure() -> None:
    path = STUDY / "obd_temporal.csv"
    if not path.exists():
        return
    temporal = pd.read_csv(path).query("alpha == 0.05")
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 3.4), sharex=True)
    rules = ["wald", "wald_hour", "emp_bernstein", "clr"]
    dot_panel(axes[0], temporal, "coverage", rules, 0.95, "Coverage")
    dot_panel(axes[1], temporal, "power", rules, None, "Power (good versions launched)")
    axes[0].set_ylim(0.55, 1.03)
    axes[1].set_ylim(0, 1.0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.55, 1.08),
               handletextpad=0.1, columnspacing=0.7)
    fig.tight_layout(h_pad=0.6)
    save(fig, "obd_temporal")
    plt.close(fig)


def real_scores(estimator: str = "dr", reward_model: str = "hgb") -> pd.DataFrame:
    exp = pd.read_parquet(EXPERIMENTS)
    exp = exp[(exp["estimator"] == estimator) & (exp["reward_model"] == reward_model)].copy()
    exp["score"] = (exp["estimate"] - exp["true_delta"]) / exp["se"]
    return exp


def error_histogram() -> None:
    """How far real estimates overshot the truth, in standard errors."""
    exp = real_scores()
    q = decision.conformal_quantile(exp["score"].to_numpy(), 0.05)
    fig, ax = plt.subplots(figsize=(3.45, 1.9))
    bins = np.arange(-5, 5.5, 0.5)
    ax.hist(exp["score"], bins=bins, color=HIST, edgecolor=SURFACE, linewidth=1.5)
    ymax = ax.get_ylim()[1]
    ax.set_ylim(0, ymax * 1.25)
    for x, colour, label, ha, dx, y in ((1.645, USUAL, "usual\nmargin 1.645", "right", -0.12, 1.2),
                                        (q, OURS, f"learned\nmargin {q:.2f}", "left", 0.12, 1.2)):
        ax.axvline(x, color=colour, lw=2)
        ax.text(x + dx, ymax * y, label, color=INK, fontsize=6.5, va="top", ha=ha)
    above = (exp["score"] > 1.645).mean()
    ax.set_xlabel("Overshoot: (estimate $-$ truth) / standard error")
    ax.set_ylabel("Real decisions")
    ax.set_title(f"{above:.0%} of real estimates overshoot by more than 1.645 se (expected 5%)", fontsize=7, color=INK)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "error_histogram")
    plt.close(fig)


def decision_chart(campaign: str = "all") -> None:
    """Each real decision for one campaign: estimate minus each margin, and the truth.

    The learned margin is calibrated on the other two campaigns, so every
    window of this campaign gets a margin without using its own outcome.
    """
    exp = real_scores()
    calib = exp[exp["campaign"] != campaign]
    test = exp[exp["campaign"] == campaign].sort_values(["direction", "window"], ascending=[False, True])
    q = decision.conformal_quantile(calib["score"].to_numpy(), 0.05)
    z = stats.norm.ppf(0.95)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 1.9), sharey=False)
    titles = {"random->bts": "New = adaptive recommender (better)",
              "bts->random": "New = random recommender (worse)"}
    for ax, (direction, g) in zip(axes, test.groupby("direction", sort=False)):
        x = g["window"].to_numpy()
        est, se, truth = (g[c].to_numpy() * 1e3 for c in ("estimate", "se", "true_delta"))
        usual_low, ours_low = est - z * se, est - q * se
        ax.vlines(x - 0.12, usual_low, est, color=USUAL, lw=2.5, label="estimate $-$ usual margin")
        ax.vlines(x + 0.12, ours_low, est, color=OURS, lw=2.5, label="estimate $-$ learned margin")
        ax.scatter(x, est, s=10, color=INK, zorder=3, label="estimate")
        ax.scatter(x, truth, s=34, marker="_", color=INK, linewidths=2, zorder=4, label="true improvement")
        miss_usual = truth < usual_low
        miss_ours = truth < ours_low
        ax.scatter(x[miss_usual] - 0.12, usual_low[miss_usual], s=28, facecolors="none", edgecolors=USUAL, lw=1.2, zorder=5)
        ax.scatter(x[miss_ours] + 0.12, ours_low[miss_ours], s=28, facecolors="none", edgecolors=OURS, lw=1.2, zorder=5)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(f"{titles[direction]}\nmissed: usual {miss_usual.sum()}/{len(x)}, learned {miss_ours.sum()}/{len(x)}",
                     fontsize=6.5, color=INK)
        ax.set_xlabel("12-hour window")
        ax.set_xticks(range(0, 14, 2))
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Click-rate change ($\\times10^{-3}$)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    fig.tight_layout(w_pad=1.2)
    save(fig, "decision_chart")
    plt.close(fig)


def hourly_drift(campaign: str = "all") -> None:
    """Hourly click rate of both recommenders over the seven days."""
    fig, ax = plt.subplots(figsize=(3.45, 1.8))
    for policy, colour, label, freq in (("bts", OURS, "adaptive recommender (hourly)", "h"),
                                         ("random", USUAL, "random recommender (3-hourly)", "3h")):
        frame = pq.read_table(PROCESSED / f"{policy}_{campaign}.parquet", columns=["timestamp", "click"]).to_pandas()
        ts = pd.to_datetime(frame["timestamp"]).dt.floor(freq)
        ctr = frame.groupby(ts)["click"].mean() * 100
        hours = (ctr.index - ctr.index.min()).total_seconds() / 3600
        ax.plot(hours, ctr.to_numpy(), color=colour, lw=1.2 if freq == "h" else 1.8, label=label)
    for b in range(12, 168, 12):
        ax.axvline(b, color=GRID, lw=0.8, zorder=0)
    ax.set_xlabel("Hours since start (grey lines: 12-hour windows)")
    ax.set_ylabel("Click rate (%)")
    ax.set_xticks(range(0, 169, 24))
    ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=2, handlelength=1.5)
    fig.tight_layout()
    save(fig, "hourly_drift")
    plt.close(fig)


def coverage_vs_history() -> None:
    within = pd.read_csv(STUDY / "semi_within.csv").query("alpha == 0.05")
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.75), sharey=True)
    for ax, (est, rm, title) in zip(axes, (("dr", "hgb", "Combined (GBT)"), ("dm", "hgb", "Click model (GBT)"))):
        sub = within[(within["estimator"] == est) & (within["reward_model"] == rm)]
        for rule in ("wald", "emp_bernstein", "clr"):
            label, colour, marker = RULES[rule]
            r = sub[sub["rule"] == rule].sort_values("n_calib")
            ax.plot(r["n_calib"], r["coverage"], color=colour, marker=marker, ms=4, lw=2, label=label)
        ax.axhline(0.95, color=INK, lw=1, ls=(0, (4, 3)))
        ax.set_title(title, fontsize=7, color=INK)
        ax.set_xticks([20, 50, 100])
        ax.set_xlabel("Past decisions $m$")
        ax.yaxis.grid(True, color=GRID, lw=0.6)
    axes[0].set_ylabel("Coverage")
    axes[0].set_ylim(0.6, 1.02)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.55, 1.1))
    fig.tight_layout(w_pad=0.6)
    save(fig, "coverage_vs_history")
    plt.close(fig)


def adaptive_gain() -> None:
    """Power of the adaptive margin minus the plain learned margin, by history size."""
    within = pd.read_csv(STUDY / "semi_within.csv").query("alpha == 0.05")
    cross = pd.read_csv(STUDY / "semi_cross_dataset.csv").query("alpha == 0.05").assign(n_calib=2240)
    both = pd.concat([within, cross])
    fig, ax = plt.subplots(figsize=(3.45, 1.8))
    styles = {("dm", "hgb"): ("Click model (GBT)", MAGENTA, "D"), ("ips", "none"): ("Reweighting", HOUR, "^"),
              ("dr", "hgb"): ("Combined (GBT)", OURS, "o")}
    for (est, rm), (label, colour, marker) in styles.items():
        sub = both[(both["estimator"] == est) & (both["reward_model"] == rm)]
        piv = sub.pivot_table(index="n_calib", columns="rule", values="power")
        gain = 100 * (piv["aclr_gbt"] - piv["clr"])
        ax.plot(gain.index, gain.to_numpy(), color=colour, marker=marker, ms=4, lw=2, label=label)
    ax.axhline(0, color=INK, lw=1)
    ax.set_xscale("log")
    ax.set_xticks([20, 50, 100, 2240])
    ax.set_xticklabels(["20", "50", "100", "2,240"])
    ax.set_xlabel("Past decisions $m$ (log scale)")
    ax.set_ylabel("Power gain (points)")
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.legend(fontsize=6.5, loc="upper left")
    fig.tight_layout()
    save(fig, "adaptive_gain")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    semi_figure()
    obd_figure()
    error_histogram()
    decision_chart()
    hourly_drift()
    coverage_vs_history()
    adaptive_gain()
    # Dark copies of the figures shown in the web app.
    use_dark_theme()
    error_histogram()
    decision_chart()
    hourly_drift()
    print(f"figures written to {OUT}")


if __name__ == "__main__":
    main()
