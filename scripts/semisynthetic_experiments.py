#!/usr/bin/env python3
"""Generate semi-synthetic off-policy launch experiments with exact ground truth.

Each multiclass dataset becomes a contextual bandit (Dudik et al., 2011): the
context is the feature vector, actions are classes, and the reward for action a
is Bernoulli(0.05 + 0.85 * 1[a == y]). Because every label is known, the true
value of any policy on any population is computable exactly.

One "experiment" = (logging policy, candidate policy, logged sample, reward
model). We record the paired improvement estimate and standard error for each
OPE estimator, plus the true improvement on the deployment population. Covariate
shift between the logging period and the deployment period is induced by
tilting the sampling probability along the first principal component.
"""

from __future__ import annotations

import argparse
import sys
import zlib
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import decision, ope  # noqa: E402

OUT = ROOT / "experiments/semisynthetic_experiments.parquet"
# OpenML data ids.
DATASETS = {"letter": 6, "pendigits": 32, "optdigits": 28, "satimage": 182, "mfeat-factors": 12, "segment": 36}
BASE_RATE, LIFT = 0.05, 0.85
LOGGING_TEMPERATURES = (0.5, 2.0)      # softmax temperature on the logging classifier's log-probs
LOGGING_EPSILON = 0.05
LOG_SIZES = (1_000, 4_000)
SHIFTS = (0.0, 1.5)                    # tilt strength along the first principal component
REPLICATES = 4


def load(data_id: int) -> tuple[np.ndarray, np.ndarray]:
    data = fetch_openml(data_id=data_id, as_frame=False, parser="auto")
    x = StandardScaler().fit_transform(np.asarray(data.data, dtype=np.float64))
    y = LabelEncoder().fit_transform(data.target)
    return x, y


def class_proba(model, x: np.ndarray, n_actions: int) -> np.ndarray:
    proba = np.full((len(x), n_actions), 1e-12)
    proba[:, model.classes_] = np.clip(model.predict_proba(x), 1e-12, 1.0)
    return proba


def softmax_policy(proba: np.ndarray, temperature: float, epsilon: float = 0.0) -> np.ndarray:
    logits = np.log(proba) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(axis=1, keepdims=True)
    return (1 - epsilon) * p + epsilon / proba.shape[1]


def true_value(pi: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    """Exact expected reward of a policy under a weighted population."""
    return float(np.dot(weights, BASE_RATE + LIFT * pi[np.arange(len(y)), y]))


def tilt_probabilities(score: np.ndarray, strength: float) -> np.ndarray:
    w = 1.0 / (1.0 + np.exp(-strength * score))
    return w / w.sum()


def fit_reward_model(x, action, reward, n_actions, kind: str, seed: int) -> np.ndarray:
    """Two-fold cross-fitted reward model; returns q_hat for every action."""
    n = len(x)
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, 2, n)
    q_hat = np.zeros((n, n_actions))
    onehot = np.eye(n_actions)
    for k in (0, 1):
        train, pred = fold != k, fold == k
        if kind == "logreg":
            # Additive in context and action: no interactions, so deliberately misspecified.
            features = np.hstack([x[train], onehot[action[train]]])
            model = LogisticRegression(C=1.0, max_iter=500).fit(features, reward[train])
        else:
            features = np.hstack([x[train], action[train, None]])
            model = HistGradientBoostingClassifier(max_iter=150, categorical_features=[x.shape[1]],
                                                   random_state=seed).fit(features, reward[train])
        xp = x[pred]
        for a in range(n_actions):
            if kind == "logreg":
                feats = np.hstack([xp, np.repeat(onehot[a][None], len(xp), axis=0)])
            else:
                feats = np.hstack([xp, np.full((len(xp), 1), a)])
            q_hat[pred, a] = model.predict_proba(feats)[:, 1] if len(np.unique(reward[train])) > 1 else reward[train].mean()
    return q_hat


def run_dataset(name: str, data_id: int, seed: int) -> list[dict]:
    x, y = load(data_id)
    n_actions = int(y.max() + 1)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(x))
    policy_idx, pool_idx = perm[: len(x) // 3], perm[len(x) // 3:]
    pc = PCA(n_components=1, random_state=seed).fit(x[pool_idx]).transform(x)[:, 0]
    pc = (pc - pc[pool_idx].mean()) / pc[pool_idx].std()

    # The logging policy is a weak classifier fitted on a small slice. Candidates
    # are mostly incremental changes to it (retraining, temperature, exploration,
    # a different model class on similar data), so many true improvements are
    # small or negative, as in real launch reviews; two large-data models are
    # included as clear wins.
    size = max(200, len(policy_idx) // 10)
    slice_ = lambda start, length: policy_idx[start:start + length]  # noqa: E731
    logging_model = LogisticRegression(C=0.05, max_iter=500).fit(x[slice_(0, size)], y[slice_(0, size)])
    logging_proba = class_proba(logging_model, x, n_actions)

    def fitted(model, rows):
        return class_proba(model.fit(x[rows], y[rows]), x, n_actions)

    # name -> (class probabilities, temperature multiplier, epsilon)
    candidates = {f"temperature_x{m}": (logging_proba, m, LOGGING_EPSILON) for m in (0.5, 0.8, 1.25, 2.0)}
    candidates.update({f"epsilon_{e}": (logging_proba, 1.0, e) for e in (0.0, 0.2)})
    candidates["retrain_same_size"] = (fitted(LogisticRegression(C=0.05, max_iter=500), slice_(size, size)), 1.0, LOGGING_EPSILON)
    candidates["retrain_1.5x"] = (fitted(LogisticRegression(C=0.05, max_iter=500), slice_(size, int(1.5 * size))), 1.0, LOGGING_EPSILON)
    candidates["retrain_3x"] = (fitted(LogisticRegression(C=0.05, max_iter=500), slice_(size, 3 * size)), 1.0, LOGGING_EPSILON)
    candidates["naive_bayes_small"] = (fitted(GaussianNB(), slice_(size, size)), 1.0, LOGGING_EPSILON)
    candidates["knn_small"] = (fitted(KNeighborsClassifier(n_neighbors=15), slice_(size, size)), 1.0, LOGGING_EPSILON)
    candidates["random_forest_small"] = (fitted(RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=seed, n_jobs=1),
                                                slice_(size, size)), 1.0, LOGGING_EPSILON)
    candidates["logreg_full"] = (fitted(LogisticRegression(C=1.0, max_iter=500), policy_idx), 1.0, LOGGING_EPSILON)
    candidates["hgb_full"] = (fitted(HistGradientBoostingClassifier(max_iter=100, random_state=seed), policy_idx), 1.0, LOGGING_EPSILON)

    rows = []
    for log_temp, n_log, shift, rep in product(LOGGING_TEMPERATURES, LOG_SIZES, SHIFTS, range(REPLICATES)):
        exp_seed = zlib.crc32(f"{name}|{log_temp}|{n_log}|{shift}|{rep}".encode())
        r = np.random.default_rng(exp_seed)
        # Logging period over-samples one end of the principal component; the
        # deployment period over-samples the other end.
        p_log = tilt_probabilities(pc[pool_idx], shift)
        p_dep = tilt_probabilities(pc[pool_idx], -shift)
        # The pool is the population; logs are drawn from it with replacement.
        log_rows = pool_idx[r.choice(len(pool_idx), size=n_log, replace=True, p=p_log)]

        pi0_log = softmax_policy(logging_proba[log_rows], log_temp, epsilon=LOGGING_EPSILON)
        action = (pi0_log.cumsum(axis=1) < r.random((len(pi0_log), 1))).sum(axis=1).clip(max=n_actions - 1)
        pscore = pi0_log[np.arange(len(action)), action]
        reward = r.binomial(1, BASE_RATE + LIFT * (action == y[log_rows])).astype(float)
        pi0_pool = softmax_policy(logging_proba[pool_idx], log_temp, epsilon=LOGGING_EPSILON)
        v0_dep = true_value(pi0_pool, y[pool_idx], p_dep)
        v0_log = true_value(pi0_pool, y[pool_idx], p_log)

        q_models = {kind: fit_reward_model(x[log_rows], action, reward, n_actions, kind, exp_seed)
                    for kind in ("logreg", "hgb")}
        terms, metas = [], []
        for c_name, (proba, t_mult, c_eps) in candidates.items():
            pi1_log = softmax_policy(proba[log_rows], log_temp * t_mult, c_eps)
            pi1_pool = softmax_policy(proba[pool_idx], log_temp * t_mult, c_eps)
            v1_dep = true_value(pi1_pool, y[pool_idx], p_dep)
            diag = ope.overlap_diagnostics(pi1_log, action, pscore)
            base = {
                "dataset": name, "n_actions": n_actions, "logging_temperature": log_temp, "n_logged": n_log,
                "shift": shift, "replicate": rep, "candidate": c_name, "true_delta": v1_dep - v0_dep,
                "true_delta_logging_period": true_value(pi1_pool, y[pool_idx], p_log) - v0_log,
                **diag,
            }
            for est_name in ope.ESTIMATORS:
                for q_kind in (("none",) if est_name in ("ips", "snips") else ("logreg", "hgb")):
                    phi = ope.estimate(est_name, pi_e=pi1_log, action=action, reward=reward, pscore=pscore,
                                       q_hat=q_models.get(q_kind))
                    terms.append(phi - reward)
                    metas.append({**base, "estimator": est_name, "reward_model": q_kind})
        summaries = decision.summarize_batch(np.column_stack(terms), r)
        rows.extend({**m, **s} for m, s in zip(metas, summaries))
        print(f"  {name}: T={log_temp} n={n_log} shift={shift} rep={rep} done", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    part_dir = OUT.parent / "semisynthetic_parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        part = part_dir / f"{name}.parquet"
        if part.exists():
            print(f"dataset {name}: cached", flush=True)
            continue
        print(f"dataset {name}", flush=True)
        pd.DataFrame(run_dataset(name, DATASETS[name], args.seed)).to_parquet(part, index=False)
    frame = pd.concat([pd.read_parquet(part_dir / f"{n}.parquet") for n in DATASETS if (part_dir / f"{n}.parquet").exists()],
                      ignore_index=True)
    frame.to_parquet(OUT, index=False)
    n_exp = frame.groupby(["dataset", "logging_temperature", "n_logged", "shift", "replicate", "candidate"]).ngroups
    print(f"wrote {len(frame):,} estimator rows for {n_exp:,} experiments to {OUT}")


if __name__ == "__main__":
    main()
