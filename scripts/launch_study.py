#!/usr/bin/env python3
"""Compare launch rules on semi-synthetic and real off-policy experiments.

For every test experiment a rule produces a lower confidence bound (LCB) on the
improvement; the candidate is launched when LCB > 0. We report

* false-launch rate: P(launch | true improvement <= 0) — the Type I error;
* power: P(launch | true improvement > 0);
* coverage: P(true improvement >= LCB).

Rules: Wald, bootstrap percentile, empirical Bernstein (sampling-noise only),
and the proposed conformal launch rules calibrated on "past" experiments:
CLR (all past experiments), CLR-null (past experiments that did not improve),
weighted CLR (recency or covariate-shift weights), and adaptive CLR, where a
learned difficulty model (ridge or gradient-boosted trees on overlap
diagnostics) normalises the conformal scores. On real data, cluster-robust Wald
bounds (by impression and by hour) are added as baselines.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import decision  # noqa: E402

REPORTS = ROOT / "reports"
ALPHAS = (0.05, 0.10)
BASELINE_RULES = ("wald", "bootstrap", "emp_bernstein")
DIAGNOSTICS = ["ess_fraction", "weight_cv", "max_weight", "mean_weight"]


def baseline_lcbs(test: pd.DataFrame, alpha: float) -> dict[str, np.ndarray]:
    z = stats.norm.ppf(1 - alpha)
    return {
        "wald": (test["estimate"] - z * test["se"]).to_numpy(),
        "bootstrap": test[f"boot_q{int(alpha * 100):02d}"].to_numpy(),
        "emp_bernstein": np.array([decision.eb_lcb_from_summary(e, v, r, n, alpha) for e, v, r, n in
                                   test[["estimate", "d_var", "d_range", "n_rounds"]].itertuples(index=False)]),
    }


def scores(frame: pd.DataFrame) -> np.ndarray:
    return ((frame["estimate"] - frame["true_delta"]) / frame["se"].clip(lower=1e-12)).to_numpy()


def clr_lcbs(calib: pd.DataFrame, test: pd.DataFrame, alpha: float,
             weights: np.ndarray | None = None) -> dict[str, np.ndarray]:
    est, se = test["estimate"].to_numpy(), test["se"].to_numpy()
    s_all = scores(calib)
    null = (calib["true_delta"] <= 0).to_numpy()
    out = {"clr": est - decision.conformal_quantile(s_all, alpha) * se}
    # With no past non-improving experiments the quantile is undefined: never launch.
    out["clr_null"] = (est - decision.conformal_quantile(s_all[null], alpha) * se) if null.sum() else np.full(len(test), -np.inf)
    if weights is not None:
        out["clr_weighted"] = est - decision.conformal_quantile(s_all, alpha, weights) * se
    return out


ADAPTIVE_KINDS = ("ridge", "gbt")


class AdaptiveCache:
    """Difficulty-model fits are independent of alpha, so fit once per split."""

    def __init__(self, calib: pd.DataFrame, test: pd.DataFrame, seed: int):
        self.calib_s = scores(calib)
        cx, tx = decision.difficulty_features(calib), decision.difficulty_features(test)
        self.cross = {k: decision.adaptive_scales(cx, self.calib_s, tx, k, seed=seed) for k in ADAPTIVE_KINDS}
        self.split = decision.split_adaptive_scales(cx, self.calib_s, tx, "gbt", seed=seed)
        self.est, self.se = test["estimate"].to_numpy(), test["se"].to_numpy()

    def lcbs(self, alpha: float) -> dict[str, np.ndarray]:
        out = {f"aclr_{k}": decision.adaptive_clr_lcb(self.est, self.se, self.calib_s, sc, st, alpha)
               for k, (sc, st) in self.cross.items()}
        used, sc, st = self.split
        out["aclr_gbt_split"] = decision.adaptive_clr_lcb(self.est, self.se, self.calib_s[used], sc, st, alpha)
        return out


def cluster_lcbs(test: pd.DataFrame, alpha: float) -> dict[str, np.ndarray]:
    z = stats.norm.ppf(1 - alpha)
    out = {}
    for col, name in (("se_impression", "wald_impression"), ("se_hour", "wald_hour")):
        if col in test:
            out[name] = (test["estimate"] - z * test[col]).to_numpy()
    return out


def evaluate(lcbs: dict[str, np.ndarray], test: pd.DataFrame) -> list[dict]:
    truth = test["true_delta"].to_numpy()
    null = truth <= 0
    rows = []
    for rule, lcb in lcbs.items():
        launch = lcb > 0
        rows.append({
            "rule": rule,
            "n_test": len(test), "n_null": int(null.sum()), "n_alt": int((~null).sum()),
            "false_launches": int((launch & null).sum()),
            "true_launches": int((launch & ~null).sum()),
            "covered": int((truth >= lcb).sum()),
        })
    return rows


def summarise(raw: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    g = raw.groupby(keys + ["rule"], sort=False)[["n_test", "n_null", "n_alt", "false_launches", "true_launches", "covered"]].sum()
    g["false_launch_rate"] = g["false_launches"] / g["n_null"].clip(lower=1)
    g["power"] = g["true_launches"] / g["n_alt"].clip(lower=1)
    g["coverage"] = g["covered"] / g["n_test"]
    return g.reset_index()


# --------------------------------------------------------------------------- #
# Semi-synthetic protocols
# --------------------------------------------------------------------------- #

def semisynthetic_within(frame: pd.DataFrame, n_calib: int, permutations: int, seed: int) -> pd.DataFrame:
    """Calibrate on n_calib random experiments of a dataset, test on the rest."""
    rng = np.random.default_rng(seed)
    rows = []
    for dataset, group in frame.groupby("dataset"):
        group = group.reset_index(drop=True)
        for p in range(permutations):
            order = rng.permutation(len(group))
            calib, test = group.iloc[order[:n_calib]], group.iloc[order[n_calib:]]
            adaptive = AdaptiveCache(calib, test, seed + p)
            for alpha in ALPHAS:
                lcbs = {**baseline_lcbs(test, alpha), **clr_lcbs(calib, test, alpha), **adaptive.lcbs(alpha)}
                for r in evaluate(lcbs, test):
                    rows.append({"dataset": dataset, "alpha": alpha, "n_calib": n_calib, "perm": p, **r})
    return pd.DataFrame(rows)


def semisynthetic_cross(frame: pd.DataFrame) -> pd.DataFrame:
    """Calibrate on all other datasets, test on the held-out dataset."""
    rows = []
    for dataset in frame["dataset"].unique():
        calib, test = frame[frame["dataset"] != dataset], frame[frame["dataset"] == dataset]
        weights = diagnostic_weights(calib, test)
        adaptive = AdaptiveCache(calib, test, 0)
        for alpha in ALPHAS:
            lcbs = {**baseline_lcbs(test, alpha), **clr_lcbs(calib, test, alpha, weights), **adaptive.lcbs(alpha)}
            for r in evaluate(lcbs, test):
                rows.append({"dataset": dataset, "alpha": alpha, **r})
    return pd.DataFrame(rows)


def semisynthetic_shift(frame: pd.DataFrame) -> pd.DataFrame:
    """Calibrate on experiments without covariate shift, test on shifted ones."""
    rows = []
    for dataset, group in frame.groupby("dataset"):
        calib, test = group[group["shift"] == 0], group[group["shift"] > 0]
        adaptive = AdaptiveCache(calib, test, 0)
        for alpha in ALPHAS:
            lcbs = {**baseline_lcbs(test, alpha), **clr_lcbs(calib, test, alpha), **adaptive.lcbs(alpha)}
            for r in evaluate(lcbs, test):
                rows.append({"dataset": dataset, "alpha": alpha, **r})
    return pd.DataFrame(rows)


def diagnostic_weights(calib: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Weighted conformal (Tibshirani et al., 2019) with a logistic-regression
    density-ratio estimate on experiment-level overlap diagnostics, averaged
    over the test distribution so a single weight vector serves all test points."""
    x = np.log1p(pd.concat([calib[DIAGNOSTICS], test[DIAGNOSTICS]]).to_numpy())
    y = np.r_[np.zeros(len(calib)), np.ones(len(test))]
    x = StandardScaler().fit_transform(x)
    clf = LogisticRegression(C=1.0, max_iter=1000).fit(x, y)
    p = np.clip(clf.predict_proba(x[: len(calib)])[:, 1], 1e-3, 1 - 1e-3)
    ratio = p / (1 - p) * (len(calib) / len(test))
    return ratio / ratio.mean()


# --------------------------------------------------------------------------- #
# Real Open Bandit protocols
# --------------------------------------------------------------------------- #

def obd_temporal(frame: pd.DataFrame, min_windows: int, decay: float) -> pd.DataFrame:
    """For each window, calibrate on all experiments from strictly earlier windows."""
    rows = []
    for w in sorted(frame["window"].unique()):
        if w < min_windows:
            continue
        calib, test = frame[frame["window"] < w], frame[frame["window"] == w]
        age = (w - calib["window"]).to_numpy()
        weights = decay ** (age - 1)
        adaptive = AdaptiveCache(calib, test, int(w))
        for alpha in ALPHAS:
            lcbs = {**baseline_lcbs(test, alpha), **cluster_lcbs(test, alpha), **clr_lcbs(calib, test, alpha, weights),
                    **adaptive.lcbs(alpha)}
            for r in evaluate(lcbs, test):
                rows.append({"window": w, "alpha": alpha, **r})
    return pd.DataFrame(rows)


def obd_leave_campaign_out(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for campaign in frame["campaign"].unique():
        calib, test = frame[frame["campaign"] != campaign], frame[frame["campaign"] == campaign]
        adaptive = AdaptiveCache(calib, test, 0)
        for alpha in ALPHAS:
            lcbs = {**baseline_lcbs(test, alpha), **cluster_lcbs(test, alpha), **clr_lcbs(calib, test, alpha),
                    **adaptive.lcbs(alpha)}
            for r in evaluate(lcbs, test):
                rows.append({"campaign": campaign, "alpha": alpha, **r})
    return pd.DataFrame(rows)


def select(frame: pd.DataFrame, estimator: str, reward_model: str) -> pd.DataFrame:
    return frame[(frame["estimator"] == estimator) & (frame["reward_model"] == reward_model)].reset_index(drop=True)


def configs(frame: pd.DataFrame) -> list[tuple[str, str]]:
    return list(frame[["estimator", "reward_model"]].drop_duplicates().itertuples(index=False, name=None))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--calib-sizes", type=int, nargs="+", default=[20, 50, 100])
    parser.add_argument("--obd-min-windows", type=int, default=4)
    parser.add_argument("--recency-decay", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--obd-file", default="experiments/obd_experiments.parquet")
    parser.add_argument("--skip-semi", action="store_true")
    args = parser.parse_args()
    out = REPORTS / "launch_study"
    out.mkdir(parents=True, exist_ok=True)

    semi_path = ROOT / "experiments/semisynthetic_experiments.parquet"
    if semi_path.exists() and not args.skip_semi:
        semi = pd.read_parquet(semi_path)
        within, cross, shift = [], [], []
        for est, rm in configs(semi):
            f = select(semi, est, rm)
            for m in args.calib_sizes:
                within.append(semisynthetic_within(f, m, args.permutations, args.seed).assign(estimator=est, reward_model=rm))
            if f["dataset"].nunique() > 1:
                cross.append(semisynthetic_cross(f).assign(estimator=est, reward_model=rm))
            shift.append(semisynthetic_shift(f).assign(estimator=est, reward_model=rm))
        keys = ["estimator", "reward_model", "alpha"]
        summarise(pd.concat(within), keys + ["n_calib"]).to_csv(out / "semi_within.csv", index=False)
        summarise(pd.concat(within), keys + ["n_calib", "dataset"]).to_csv(out / "semi_within_by_dataset.csv", index=False)
        if cross:
            summarise(pd.concat(cross), keys).to_csv(out / "semi_cross_dataset.csv", index=False)
            summarise(pd.concat(cross), keys + ["dataset"]).to_csv(out / "semi_cross_dataset_by_dataset.csv", index=False)
        summarise(pd.concat(shift), keys).to_csv(out / "semi_shift.csv", index=False)
        print("semi-synthetic done", flush=True)

    obd_path = ROOT / args.obd_file
    if obd_path.exists():
        obd = pd.read_parquet(obd_path)
        temporal, loco = [], []
        for est, rm in configs(obd):
            f = select(obd, est, rm)
            temporal.append(obd_temporal(f, args.obd_min_windows, args.recency_decay).assign(estimator=est, reward_model=rm))
            loco.append(obd_leave_campaign_out(f).assign(estimator=est, reward_model=rm))
        keys = ["estimator", "reward_model", "alpha"]
        summarise(pd.concat(temporal), keys).to_csv(out / "obd_temporal.csv", index=False)
        summarise(pd.concat(loco), keys).to_csv(out / "obd_leave_campaign_out.csv", index=False)
        summarise(pd.concat(loco), keys + ["campaign"]).to_csv(out / "obd_leave_campaign_out_by_campaign.csv", index=False)
        print("obd done", flush=True)


if __name__ == "__main__":
    main()
