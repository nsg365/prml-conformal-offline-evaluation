#!/usr/bin/env python3
"""Audit the Open Bandit Dataset: sizes, dates, click rates, and propensities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data/processed"
OUT = ROOT / "reports/obd_audit.json"
COLS = ["timestamp", "item_id", "position", "click", "propensity_score"]


def audit(policy: str, campaign: str) -> dict:
    frame = pq.read_table(PROC / f"{policy}_{campaign}.parquet", columns=COLS).to_pandas()
    ts = pd.to_datetime(frame["timestamp"])
    day = ts.dt.date.astype(str)
    by_day = frame.groupby(day)["click"].agg(["size", "sum", "mean"])
    pscore = frame["propensity_score"]
    # For a non-contextual policy the per-(day, position) propensities of the
    # chosen items should sum to about one over the item set.
    per_item = frame.assign(day=day).groupby(["day", "position", "item_id"])["propensity_score"].median()
    coverage = per_item.groupby(level=["day", "position"]).agg(["sum", "size"])
    return {
        "rows": int(len(frame)),
        "first_timestamp": str(ts.min()),
        "last_timestamp": str(ts.max()),
        "n_items": int(frame["item_id"].nunique()),
        "positions": sorted(int(p) for p in frame["position"].unique()),
        "ctr": float(frame["click"].mean()),
        "clicks": int(frame["click"].sum()),
        "pscore": {"min": float(pscore.min()), "median": float(pscore.median()), "max": float(pscore.max()),
                   "n_unique": int(pscore.nunique())},
        "by_day": {d: {"rows": int(r["size"]), "clicks": int(r["sum"]), "ctr": float(r["mean"])}
                   for d, r in by_day.iterrows()},
        "median_pscore_sum_by_day_position": {"min": float(coverage["sum"].min()), "max": float(coverage["sum"].max())},
        "items_seen_by_day_position": {"min": int(coverage["size"].min()), "max": int(coverage["size"].max())},
    }


def main() -> None:
    report = {}
    for policy in ("random", "bts"):
        for campaign in ("all", "men", "women"):
            report[f"{policy}/{campaign}"] = audit(policy, campaign)
            r = report[f"{policy}/{campaign}"]
            print(f"{policy}/{campaign}: {r['rows']:,} rows, {r['n_items']} items, CTR {r['ctr']:.4%}, "
                  f"{r['first_timestamp']} -> {r['last_timestamp']}, pscore {r['pscore']}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
