"""Launch decision rules built on paired off-policy improvement estimates.

A launch rule receives the per-round improvement terms ``d = phi - r`` for one
experiment and returns a lower confidence bound (LCB) on the true improvement.
The candidate is launched when the LCB is above zero.

Baseline rules use only within-experiment sampling noise. The proposed
conformal launch rule (CLR) instead calibrates the critical value on past
experiments whose true improvement was later measured online, so that the
false-launch rate is controlled even when the estimator is biased.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def point_and_se(d: np.ndarray) -> tuple[float, float]:
    return float(np.mean(d)), float(np.std(d, ddof=1) / np.sqrt(len(d)))


BOOT_LEVELS = (0.05, 0.10)


def summarize_terms(d: np.ndarray, rng: np.random.Generator, replicates: int = 200) -> dict:
    """Everything the launch rules need from one experiment's per-round terms.

    Stores the mean, standard error, observed range (for empirical Bernstein)
    and lower percentiles of a multinomial bootstrap of the mean.
    """
    est, se = point_and_se(d)
    n = len(d)
    means = np.empty(replicates)
    for b in range(replicates):
        means[b] = d[rng.integers(0, n, n)].mean()
    out = {"estimate": est, "se": se, "d_var": float(np.var(d, ddof=1)), "d_range": float(d.max() - d.min()),
           "n_rounds": n}
    for level in BOOT_LEVELS:
        out[f"boot_q{int(level * 100):02d}"] = float(np.quantile(means, level))
    return out


def summarize_batch(terms: np.ndarray, rng: np.random.Generator, replicates: int = 200) -> list[dict]:
    """``summarize_terms`` for many term vectors (columns) sharing the same rounds.

    One multinomial bootstrap count matrix is drawn and reused for every column,
    so resampling is a single matrix product.
    """
    n, _ = terms.shape
    counts = rng.multinomial(n, np.full(n, 1.0 / n), size=replicates).astype(np.float64)
    boot_means = counts @ terms / n
    est = terms.mean(axis=0)
    var = terms.var(axis=0, ddof=1)
    out = []
    for j in range(terms.shape[1]):
        row = {"estimate": float(est[j]), "se": float(np.sqrt(var[j] / n)), "d_var": float(var[j]),
               "d_range": float(terms[:, j].max() - terms[:, j].min()), "n_rounds": n}
        for level in BOOT_LEVELS:
            row[f"boot_q{int(level * 100):02d}"] = float(np.quantile(boot_means[:, j], level))
        out.append(row)
    return out


def cluster_se(d: np.ndarray, clusters: np.ndarray) -> float:
    """Cluster-robust standard error of mean(d) (Liang-Zeger with G/(G-1) correction).

    Rows in the same cluster may be arbitrarily correlated; clusters are
    treated as independent.
    """
    _, inverse = np.unique(clusters, return_inverse=True)
    g = inverse.max() + 1
    sums = np.bincount(inverse, weights=d - d.mean(), minlength=g)
    return float(np.sqrt(np.sum(sums ** 2) * g / max(g - 1, 1)) / len(d))


def eb_lcb_from_summary(estimate: float, d_var: float, d_range: float, n: int, alpha: float) -> float:
    log_term = np.log(2.0 / alpha)
    return estimate - np.sqrt(2 * d_var * log_term / n) - 7 * d_range * log_term / (3 * (n - 1))


def wald_lcb(d: np.ndarray, alpha: float) -> float:
    est, se = point_and_se(d)
    return est - stats.norm.ppf(1 - alpha) * se


def bootstrap_lcb(d: np.ndarray, alpha: float, replicates: int = 300, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    n = len(d)
    # Poisson(1) weights approximate multinomial resampling and vectorise well.
    means = np.empty(replicates)
    for b in range(replicates):
        w = rng.poisson(1.0, n)
        means[b] = np.dot(w, d) / max(w.sum(), 1)
    return float(np.quantile(means, alpha))


def empirical_bernstein_lcb(d: np.ndarray, alpha: float) -> float:
    """Maurer & Pontil (2009) bound using the observed range of d.

    The bound assumes d lies in a known interval; using the observed range is a
    plug-in heuristic and is reported as such.
    """
    n = len(d)
    est = float(np.mean(d))
    var = float(np.var(d, ddof=1))
    span = float(np.max(d) - np.min(d))
    log_term = np.log(2.0 / alpha)
    return est - np.sqrt(2 * var * log_term / n) - 7 * span * log_term / (3 * (n - 1))


def conformal_quantile(scores: np.ndarray, alpha: float, weights: np.ndarray | None = None) -> float:
    """(1-alpha) conformal quantile of calibration scores.

    Unweighted: the ceil((m+1)(1-alpha))-th smallest score (split conformal).
    Weighted (Barber et al., 2023): the test point receives weight 1 at +inf and
    calibration points receive the given weights, normalised together.
    """
    scores = np.asarray(scores, dtype=float)
    m = len(scores)
    if weights is None:
        k = int(np.ceil((m + 1) * (1 - alpha)))
        if k > m:
            return float("inf")
        return float(np.sort(scores)[k - 1])
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(scores)
    s, w = scores[order], weights[order]
    total = w.sum() + 1.0
    cum = np.cumsum(w) / total
    idx = np.searchsorted(cum, 1 - alpha, side="left")
    return float(s[idx]) if idx < m else float("inf")


def studentized_score(est: float, se: float, true_delta: float) -> float:
    """Calibration score: how far the estimate overshoots the truth, in SE units."""
    return (est - true_delta) / max(se, 1e-12)


def clr_lcb(d: np.ndarray, calibration_scores: np.ndarray, alpha: float,
            weights: np.ndarray | None = None) -> float:
    """Conformal launch rule: LCB = estimate - q * SE.

    If the new experiment's score is exchangeable with the calibration scores,
    P(true_delta >= LCB) >= 1 - alpha, so launching only when LCB > 0 keeps the
    probability of launching a non-improving policy at most alpha.
    """
    est, se = point_and_se(d)
    q = conformal_quantile(calibration_scores, alpha, weights)
    return est - q * se


# --------------------------------------------------------------------------- #
# Adaptive CLR: a learned difficulty model normalises the conformal scores.
# --------------------------------------------------------------------------- #

def difficulty_features(frame) -> np.ndarray:
    """Experiment-level features known before the online outcome: log size,
    overlap diagnostics, and the estimate's own standard error."""
    return np.column_stack([
        np.log(frame["n_rounds"].to_numpy(dtype=float)),
        frame["ess_fraction"].to_numpy(dtype=float),
        np.log1p(frame["max_weight"].to_numpy(dtype=float)),
        np.log1p(frame["weight_cv"].to_numpy(dtype=float)),
        np.log(frame["mean_weight"].to_numpy(dtype=float) + 1e-6),
        np.log(frame["se"].to_numpy(dtype=float) + 1e-12),
    ])


def make_difficulty_model(kind: str, seed: int = 0):
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if kind == "gbt":
        return GradientBoostingRegressor(n_estimators=60, max_depth=2, learning_rate=0.1, subsample=0.8,
                                         min_samples_leaf=5, random_state=seed)
    raise ValueError(kind)


def _log_abs(s: np.ndarray) -> np.ndarray:
    # Offset keeps the target finite when a past estimate was almost exact.
    return np.log(np.abs(s) + 0.25)


def adaptive_scales(calib_x: np.ndarray, calib_s: np.ndarray, test_x: np.ndarray, kind: str,
                    folds: int = 5, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Cross-fitted difficulty predictions sigma_hat for calibration and test experiments.

    Each calibration experiment's sigma_hat comes from a model fitted on the
    other folds, so its own score never trains the model that normalises it.
    """
    m = len(calib_s)
    k = min(folds, m)
    fold = np.random.default_rng(seed).permutation(np.arange(m) % k)
    y = _log_abs(calib_s)
    sigma_calib = np.empty(m)
    for f in range(k):
        train, held = fold != f, fold == f
        model = make_difficulty_model(kind, seed).fit(calib_x[train], y[train])
        sigma_calib[held] = np.exp(model.predict(calib_x[held]))
    sigma_test = np.exp(make_difficulty_model(kind, seed).fit(calib_x, y).predict(test_x))
    return sigma_calib, sigma_test


def adaptive_clr_lcb(est: np.ndarray, se: np.ndarray, calib_s: np.ndarray, sigma_calib: np.ndarray,
                     sigma_test: np.ndarray, alpha: float) -> np.ndarray:
    """LCB = estimate - q * se * sigma_hat, with q the conformal quantile of s / sigma_hat."""
    q = conformal_quantile(calib_s / sigma_calib, alpha)
    return est - q * se * sigma_test


def split_adaptive_scales(calib_x: np.ndarray, calib_s: np.ndarray, test_x: np.ndarray, kind: str,
                          fit_fraction: float = 1 / 3, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split variant with the exact finite-sample guarantee: the difficulty
    model is fitted on one part of the history and the quantile is taken on the
    rest. Returns (mask of calibration rows used, sigma_calib, sigma_test)."""
    m = len(calib_s)
    perm = np.random.default_rng(seed).permutation(m)
    n_fit = max(3, int(round(fit_fraction * m)))
    fit_idx, cal_idx = perm[:n_fit], perm[n_fit:]
    model = make_difficulty_model(kind, seed).fit(calib_x[fit_idx], _log_abs(calib_s[fit_idx]))
    used = np.zeros(m, dtype=bool)
    used[cal_idx] = True
    return used, np.exp(model.predict(calib_x[used])), np.exp(model.predict(test_x))
