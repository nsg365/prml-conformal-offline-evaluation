#!/usr/bin/env python3
"""Build real off-policy launch experiments from the Open Bandit Dataset.

The uniform-random and Bernoulli Thompson Sampling (BTS) policies ran at the
same time on split traffic of the same campaign. For every campaign and time
window we create two experiments:

* ``random->bts``: logs of the random policy; candidate = BTS. The true
  improvement is CTR(BTS) - CTR(random) in that window, measured on-policy.
* ``bts->random``: logs of BTS; candidate = random. True improvement is the
  negative of the above. BTS propensities can be tiny, so this direction has
  poor overlap and heavy importance weights.

BTS action probabilities are the empirical item frequencies per (hour, position)
in the BTS logs, which estimate the hour-averaged policy without the selection
bias of averaging logged propensities. Each row is treated as one round with
position as part of the context, as in the Open Bandit Pipeline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import decision, ope  # noqa: E402

PROC = ROOT / "data/processed"
OUT = ROOT / "results/experiments/obd_experiments.parquet"
LIGHT = ["timestamp", "item_id", "position", "click", "propensity_score",
         "user_feature_0", "user_feature_1", "user_feature_2", "user_feature_3"]
USER = ["user_feature_0", "user_feature_1", "user_feature_2", "user_feature_3"]


def add_time_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """Hour, plus an impression key: the three positions of one impression are
    logged within milliseconds for the same user, so (second, user features)
    identifies an impression for cluster-robust standard errors."""
    ts = pd.to_datetime(frame.pop("timestamp"))
    frame["hour"] = ts.dt.floor("h")
    keys = frame[USER].assign(second=ts.dt.floor("s"))
    frame["impression"] = pd.util.hash_pandas_object(keys, index=False).to_numpy()
    return frame


def read_light(policy: str, campaign: str) -> pd.DataFrame:
    frame = pq.read_table(PROC / f"{policy}_{campaign}.parquet", columns=LIGHT).to_pandas()
    return add_time_keys(frame)


def read_sample(policy: str, campaign: str, rate: float, seed: int) -> pd.DataFrame:
    """Bernoulli row sample including affinity columns, read batch by batch."""
    rng = np.random.default_rng(seed)
    parts = []
    for batch in pq.ParquetFile(PROC / f"{policy}_{campaign}.parquet").iter_batches(batch_size=500_000):
        frame = batch.to_pandas()
        keep = rng.random(len(frame)) < rate
        parts.append(frame.loc[keep])
    return add_time_keys(pd.concat(parts, ignore_index=True))


def item_table(campaign: str) -> pd.DataFrame:
    items = pd.read_csv(PROC / f"item_context_{campaign}.csv", index_col=0).sort_values("item_id")
    for c in ("item_feature_1", "item_feature_2", "item_feature_3"):
        items[c] = items[c].astype("category").cat.codes
    return items.reset_index(drop=True)


class RewardFeatures:
    """Features for (round, item) pairs: user, position, item, affinity."""

    def __init__(self, items: pd.DataFrame, user_levels: dict[str, list]):
        self.items = items
        self.user_levels = user_levels

    def user_codes(self, rows: pd.DataFrame) -> np.ndarray:
        return np.column_stack([pd.Categorical(rows[c], categories=self.user_levels[c]).codes for c in USER])

    def matrix(self, rows: pd.DataFrame, user: np.ndarray, item: np.ndarray) -> np.ndarray:
        affinity = rows.filter(like="user-item_affinity").to_numpy(dtype=np.float32)
        aff = affinity[np.arange(len(rows)), item]
        it = self.items.iloc[item]
        return np.column_stack([
            user, rows["position"].to_numpy(), item,
            it["item_feature_1"].to_numpy(), it["item_feature_2"].to_numpy(), it["item_feature_3"].to_numpy(),
            it["item_feature_0"].to_numpy(), aff,
        ]).astype(np.float32)


N_CAT = 9  # user x4, position, item id, item features 1-3 are categorical; the last two are numeric


def fit_predict_q(rows: pd.DataFrame, feats: RewardFeatures, n_items: int, kind: str, seed: int) -> np.ndarray:
    """Two-fold cross-fitted click model returning q_hat for every item."""
    n = len(rows)
    fold = np.random.default_rng(seed).integers(0, 2, n)
    user = feats.user_codes(rows)
    logged = feats.matrix(rows, user, rows["item_id"].to_numpy())
    y = rows["click"].to_numpy()
    q_hat = np.zeros((n, n_items), dtype=np.float32)
    for k in (0, 1):
        tr, te = fold != k, fold == k
        if kind == "logreg":
            enc = OneHotEncoder(handle_unknown="ignore").fit(logged[tr, :N_CAT])

            def design(m):
                return sparse.hstack([enc.transform(m[:, :N_CAT]), sparse.csr_matrix(m[:, N_CAT:])]).tocsr()

            model = LogisticRegression(C=1.0, max_iter=300).fit(design(logged[tr]), y[tr])
        else:
            model = HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
                categorical_features=list(range(N_CAT)), random_state=seed,
            ).fit(logged[tr], y[tr])

            def design(m):
                return m
        rows_te, user_te = rows.loc[te], user[te]
        for a in range(n_items):
            m = feats.matrix(rows_te, user_te, np.full(te.sum(), a))
            q_hat[te, a] = model.predict_proba(design(m))[:, 1]
    return q_hat


def window_id(hour: pd.Series, hours: int) -> np.ndarray:
    start = hour.min().normalize()
    return ((hour - start) / pd.Timedelta(hours=hours)).astype(int).to_numpy()


def run_campaign(campaign: str, window_hours: int, bts_rows_per_window: int, seed: int,
                 windows: list[int] | None = None) -> list[dict]:
    items = item_table(campaign)
    n_items = len(items)
    rnd_light = read_light("random", campaign)
    bts_light = read_light("bts", campaign)
    n_windows = int(np.ceil(7 * 24 / window_hours))
    rate = min(1.0, bts_rows_per_window * n_windows / len(bts_light))
    rnd = read_sample("random", campaign, 1.0, seed)
    bts = read_sample("bts", campaign, rate, seed)
    user_levels = {c: sorted(set(rnd_light[c]) | set(bts_light[c])) for c in USER}
    feats = RewardFeatures(items, user_levels)

    # Hourly BTS policy: empirical item frequency per (hour, position).
    counts = bts_light.groupby(["hour", "position", "item_id"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=range(n_items), fill_value=0)
    bts_policy = counts.div(counts.sum(axis=1), axis=0)

    w_rnd_light, w_bts_light = window_id(rnd_light["hour"], window_hours), window_id(bts_light["hour"], window_hours)
    w_rnd, w_bts = window_id(rnd["hour"], window_hours), window_id(bts["hour"], window_hours)
    rows = []
    for w in (windows if windows is not None else range(n_windows)):
        truth = {}
        for name, light, wid in (("random", rnd_light, w_rnd_light), ("bts", bts_light, w_bts_light)):
            in_w = wid == w
            clicks = light["click"].to_numpy()[in_w].astype(float)
            truth[name] = (clicks.mean(), clicks.std(ddof=1) / np.sqrt(len(clicks)),
                           decision.cluster_se(clicks, light["impression"].to_numpy()[in_w]))
        true_bts_minus_rnd = truth["bts"][0] - truth["random"][0]
        truth_se = float(np.hypot(truth["bts"][1], truth["random"][1]))
        truth_se_impression = float(np.hypot(truth["bts"][2], truth["random"][2]))

        for direction in ("random->bts", "bts->random"):
            if direction == "random->bts":
                log = rnd.loc[w_rnd == w].reset_index(drop=True)
                key = pd.MultiIndex.from_arrays([log["hour"], log["position"]])
                pi_e = bts_policy.reindex(key).fillna(1.0 / n_items).to_numpy()
                true_delta = true_bts_minus_rnd
            else:
                log = bts.loc[w_bts == w].reset_index(drop=True)
                pi_e = np.full((len(log), n_items), 1.0 / n_items)
                true_delta = -true_bts_minus_rnd
            action = log["item_id"].to_numpy()
            reward = log["click"].to_numpy().astype(float)
            pscore = log["propensity_score"].to_numpy().astype(float)
            diag = ope.overlap_diagnostics(pi_e, action, pscore)
            base = {"campaign": campaign, "window": w, "window_hours": window_hours, "direction": direction,
                    "n_logged": len(log), "n_items": n_items, "true_delta": true_delta, "truth_se": truth_se,
                    "truth_se_impression": truth_se_impression,
                    "logging_ctr": reward.mean(), **diag}
            q = {kind: fit_predict_q(log, feats, n_items, kind, seed + w) for kind in ("logreg", "hgb")}
            boot_rng = np.random.default_rng(seed + 1000 * w + (direction == "bts->random"))
            for est in ope.ESTIMATORS:
                for q_kind in (("none",) if est in ("ips", "snips") else ("logreg", "hgb")):
                    phi = ope.estimate(est, pi_e=pi_e, action=action, reward=reward, pscore=pscore,
                                       q_hat=q.get(q_kind))
                    d = phi - reward
                    rows.append({**base, "estimator": est, "reward_model": q_kind,
                                 **decision.summarize_terms(d, boot_rng),
                                 "se_impression": decision.cluster_se(d, log["impression"].to_numpy()),
                                 "se_hour": decision.cluster_se(d, log["hour"].to_numpy())})
            print(f"  {campaign} window {w} {direction}: n={len(log):,} true={true_delta:+.5f} "
                  f"DR(hgb)={[r for r in rows if r['estimator'] == 'dr' and r['reward_model'] == 'hgb'][-1]['estimate']:+.5f}",
                  flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaigns", nargs="+", default=["men", "women", "all"])
    parser.add_argument("--window-hours", type=int, default=12)
    parser.add_argument("--bts-rows-per-window", type=int, default=150_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    rows = []
    for campaign in args.campaigns:
        print(f"campaign {campaign}", flush=True)
        rows.extend(run_campaign(campaign, args.window_hours, args.bts_rows_per_window, args.seed))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.output, index=False)
    print(f"wrote {len(rows):,} rows to {args.output}")


if __name__ == "__main__":
    main()
