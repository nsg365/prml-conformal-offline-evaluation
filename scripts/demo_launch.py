#!/usr/bin/env python3
"""Live demo: should we launch the new recommender in this time window?

1. Estimates the improvement live on one real Open Bandit window.
2. Learns the margin from decisions in earlier windows (the "past A/B tests"),
   loaded from experiments/obd_experiments.parquet.
3. Prints each rule's estimate minus margin, its launch decision, and the true
   result measured on real traffic, which the rules never see.

The same functions back the web app in app/app.py.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src import decision  # noqa: E402
from obd_experiments import run_campaign  # noqa: E402

HISTORY = ROOT / "experiments/obd_experiments.parquet"


def run_window(campaign: str, window: int) -> list[dict]:
    """Estimate every estimator and both directions live for one 12-hour window."""
    with contextlib.redirect_stdout(io.StringIO()):  # hide the builder's progress lines
        return run_campaign(campaign, 12, 150_000, seed=0, windows=[window])


def make_decision(rows: list[dict], window: int, direction: str, estimator: str, reward_model: str,
                  alpha: float = 0.05, adaptive: bool = False) -> dict:
    """Apply every launch rule to one live estimate.

    ``direction`` is "random->bts" (log of the random recommender, new = adaptive)
    or "bts->random". The learned margin uses only decisions from earlier windows.
    """
    history = pd.read_parquet(HISTORY)
    history = history[(history["estimator"] == estimator) & (history["reward_model"] == reward_model)]
    past = history[history["window"] < window]
    if len(past) < 10:
        raise ValueError(f"Only {len(past)} past decisions before window {window}; pick window 4 or later.")
    exp = next(r for r in rows if r["direction"] == direction and r["estimator"] == estimator
               and r["reward_model"] == reward_model)

    est, se = exp["estimate"], exp["se"]
    s_past = ((past["estimate"] - past["true_delta"]) / past["se"]).to_numpy()
    z = stats.norm.ppf(1 - alpha)
    q = decision.conformal_quantile(s_past, alpha)
    rules = [
        {"name": f"Usual margin ({z:.3f} x se)", "short": "Usual", "kind": "usual", "lower": est - z * se},
        {"name": "Usual margin, hour-grouped se", "short": "Usual,\nhour-grouped", "kind": "usual",
         "lower": est - z * exp["se_hour"]},
        {"name": "Bootstrap margin", "short": "Bootstrap", "kind": "usual",
         "lower": exp[f"boot_q{int(alpha * 100):02d}"]},
        {"name": "Bernstein bound", "short": "Bernstein", "kind": "cautious",
         "lower": decision.eb_lcb_from_summary(est, exp["d_var"], exp["d_range"], exp["n_rounds"], alpha)},
        {"name": f"Learned margin ({q:.2f} x se, {len(s_past)} past decisions)", "short": "Learned\n(ours)",
         "kind": "ours", "lower": est - q * se},
    ]
    if adaptive:
        # The difficulty model is trained on the past decisions only, then
        # predicts how hard this new decision is.
        past_x = decision.difficulty_features(past)
        new_x = decision.difficulty_features(pd.DataFrame([exp]))
        for kind, name, short in (("ridge", "ridge regression", "Adaptive\n(ridge)"),
                                  ("gbt", "boosted trees", "Adaptive\n(boosted)")):
            sigma_past, sigma_new = decision.adaptive_scales(past_x, s_past, new_x, kind)
            margin = decision.conformal_quantile(s_past / sigma_past, alpha) * se * sigma_new[0]
            rules.append({"name": f"Adaptive margin, {name} ({margin / se:.2f} x se)", "short": short,
                          "kind": "ours", "lower": est - margin})
    for rule in rules:
        rule["launch"] = bool(rule["lower"] > 0)
    return {"estimate": est, "se": se, "truth": exp["true_delta"], "rules": rules, "n_past": len(s_past),
            "q": q, "n_rounds": exp["n_rounds"], "current_ctr": exp["logging_ctr"],
            "max_weight": exp["max_weight"], "past_scores": s_past}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", default="men", choices=["men", "women", "all"])
    parser.add_argument("--window", type=int, default=12, help="12-hour window index, 4-13")
    parser.add_argument("--direction", default="random-to-bts", choices=["random-to-bts", "bts-to-random"],
                        help="whose log is used and which recommender is new (bts = the adaptive recommender)")
    parser.add_argument("--estimator", default="dr")
    parser.add_argument("--reward-model", default="hgb")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--adaptive", action="store_true",
                        help="also show the adaptive learned margin (ridge / boosted-tree difficulty model)")
    args = parser.parse_args()
    direction = args.direction.replace("-to-", "->")

    print(f"Estimating offline, live: campaign={args.campaign}, 12-hour window={args.window}, "
          f"log/new recommender={direction.replace('bts', 'adaptive')}, estimator={args.estimator}/{args.reward_model}\n")
    try:
        result = make_decision(run_window(args.campaign, args.window), args.window, direction, args.estimator,
                               args.reward_model, args.alpha, args.adaptive)
    except ValueError as err:
        sys.exit(str(err))

    print(f"Log size: {result['n_rounds']:,}   current click rate: {result['current_ctr']:.4%}   "
          f"largest weight: {result['max_weight']:.0f}")
    print(f"Estimated improvement in click rate: {result['estimate']:+.5f} (standard error {result['se']:.5f})\n")
    print(f"{'Rule':50s} {'est - margin':>13s}  Decision")
    for rule in result["rules"]:
        print(f"{rule['name']:50s} {rule['lower']:+13.5f}  {'LAUNCH' if rule['launch'] else 'hold'}")
    truth = result["truth"]
    verdict = "new recommender is better" if truth > 0 else "new recommender is NOT better"
    print(f"\nTrue improvement, measured on real traffic (hidden from the rules): {truth:+.5f} -> {verdict}")


if __name__ == "__main__":
    main()
