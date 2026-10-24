"""Off-policy value estimators written as per-round influence terms.

Every estimator returns a vector ``phi`` with one entry per logged round such
that ``phi.mean()`` is the point estimate of the evaluation policy's value.
Writing estimators this way gives a common route to standard errors and to the
paired launch statistic: when the baseline is the logging policy itself, its
on-policy value is ``r.mean()``, so the improvement estimate is
``(phi - r).mean()`` and both terms share the same rounds.

Notation (n rounds, K actions):
    pi_e   : (n, K) evaluation-policy action probabilities for each round
    pscore : (n,)  logging-policy probability of the logged action
    action : (n,)  logged action index
    reward : (n,)  observed reward
    q_hat  : (n, K) estimated expected reward for every action (DM/DR family)
"""

from __future__ import annotations

import numpy as np

ESTIMATORS = ("dm", "ips", "snips", "dr", "sndr", "switch_dr")


def importance_weights(pi_e: np.ndarray, action: np.ndarray, pscore: np.ndarray) -> np.ndarray:
    return pi_e[np.arange(len(action)), action] / pscore


def phi_dm(pi_e, action, reward, pscore, q_hat, **_):
    return np.sum(pi_e * q_hat, axis=1)


def phi_ips(pi_e, action, reward, pscore, q_hat=None, **_):
    return importance_weights(pi_e, action, pscore) * reward


def phi_snips(pi_e, action, reward, pscore, q_hat=None, **_):
    # Linearisation of sum(w r) / sum(w): mean equals the SNIPS estimate and
    # the sample variance gives the delta-method standard error.
    w = importance_weights(pi_e, action, pscore)
    value = np.sum(w * reward) / np.sum(w)
    return value + w * (reward - value) / np.mean(w)


def phi_dr(pi_e, action, reward, pscore, q_hat, **_):
    w = importance_weights(pi_e, action, pscore)
    q_logged = q_hat[np.arange(len(action)), action]
    return np.sum(pi_e * q_hat, axis=1) + w * (reward - q_logged)


def phi_sndr(pi_e, action, reward, pscore, q_hat, **_):
    w = importance_weights(pi_e, action, pscore)
    q_logged = q_hat[np.arange(len(action)), action]
    return np.sum(pi_e * q_hat, axis=1) + w * (reward - q_logged) / np.mean(w)


def phi_switch_dr(pi_e, action, reward, pscore, q_hat, switch_tau: float = 50.0, **_):
    # Use the importance-weighted correction only where the weight is moderate;
    # fall back to the reward model where weights explode (Wang et al., 2017).
    w = importance_weights(pi_e, action, pscore)
    q_logged = q_hat[np.arange(len(action)), action]
    correction = np.where(w <= switch_tau, w * (reward - q_logged), 0.0)
    return np.sum(pi_e * q_hat, axis=1) + correction


PHI = {
    "dm": phi_dm,
    "ips": phi_ips,
    "snips": phi_snips,
    "dr": phi_dr,
    "sndr": phi_sndr,
    "switch_dr": phi_switch_dr,
}


def estimate(name: str, **kwargs) -> np.ndarray:
    return PHI[name](**kwargs)


def overlap_diagnostics(pi_e: np.ndarray, action: np.ndarray, pscore: np.ndarray) -> dict:
    """Summaries of how far the evaluation policy is from the logging policy."""
    w = importance_weights(pi_e, action, pscore)
    ess = float(np.sum(w) ** 2 / np.sum(w ** 2)) if np.any(w) else 0.0
    return {
        "mean_weight": float(np.mean(w)),
        "max_weight": float(np.max(w)),
        "weight_cv": float(np.std(w) / max(np.mean(w), 1e-12)),
        "ess_fraction": ess / len(w),
    }
