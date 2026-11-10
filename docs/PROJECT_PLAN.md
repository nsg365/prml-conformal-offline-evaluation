# Project plan — Calibrated off-policy launch decisions

> **Historical document.** This is the original design and literature check, written on 24 Oct 2026 before the experiments were run. Some ideas here (e.g. the drift-aware weighted variant) were tried but left out of the final paper. For what was actually built and found, see the [README](../README.md) and the paper.

**Working title:** *Trust, but Calibrate: Using Past A/B Tests to Make Off-Policy Launch Decisions Reliable*

**Evaluation:** Tuesday, 10 November 2026 (live demo + viva).

## Problem

Recommendation teams want to know whether a new policy beats the production one *before* running an A/B test. Off-policy evaluation (OPE) estimates the new policy's value from logs of the old one. A launch decision is then "ship if the lower confidence bound on the improvement Δ = V(new) − V(old) is above zero."

That rule is only as good as its confidence interval. Standard OPE intervals (Wald, bootstrap, empirical Bernstein) describe *sampling noise* only. They ignore reward-model bias, poor overlap, and drift between the logging period and the deployment period. So the stated 5% false-launch rate can be badly violated.

## Contribution

1. **Conformal launch rule (CLR).** Companies keep a history of past A/B tests, each with an OPE estimate made beforehand and a true outcome measured afterwards. Treat these as calibration data: compute studentized errors s_j = (Δ̂_j − Δ_j)/SE_j, and replace the normal critical value with their conformal (1−α) quantile. If a new experiment is exchangeable with past ones, the false-launch rate is ≤ α in finite samples, without assuming the OPE estimator is unbiased.
2. **Drift-aware variant.** Weight recent experiments more (non-exchangeable weighted conformal, Barber et al. 2023) to handle temporal drift.
3. **Real-data evaluation with genuine ground truth.** On the Open Bandit Dataset, the uniform-random and Bernoulli-TS policies ran *simultaneously* on the same traffic, so each one's on-policy click rate is a true outcome for OPE of the other. Splitting by campaign and day gives dozens of real "historical experiments."
4. **Controlled stress tests.** Semi-synthetic bandits built from real classification datasets give exact ground truth for hundreds of experiments, with controlled reward-model misspecification and covariate shift.

All learning components are classical ML: logistic regression, random forests, gradient boosting, naive Bayes and k-NN reward models and policies; estimators and intervals are statistical.

## Methods

- **Estimators (implemented from scratch, checked against `obp`):** DM, IPS, SNIPS, DR, SNDR, Switch-DR; cross-fitted reward models.
- **Intervals / decision rules:** Wald (CLT), percentile bootstrap, empirical Bernstein, conformal launch rule (proposed), weighted conformal (proposed).
- **Metrics:** false-launch rate (Type I, Δ ≤ 0 but launched), power (Δ > 0 and launched), interval coverage, decision regret.

## Novelty boundary (from literature check, 24 Oct 2026)

- Estimator selection is already studied: PAS-IF (AAAI 2023), AutoOPE (2024), cross-validated OPE (2024), meta off-policy estimation (2025). We do **not** claim estimator selection.
- Risk–return of OPE-based policy selection: SharpeRatio@k (ICLR 2024) — portfolio view of top-k selection, not calibrated pairwise launch tests.
- OPE under non-stationarity: Jagerman et al. (WSDM 2019), OPFV (2025). We use drift as a stress condition and handle it through calibration, not a new estimator.
- Offline A/B testing (Gilotte et al., WSDM 2018) estimates policy differences offline; it does not calibrate decision error rates against historical experiments.
- Using past experiments to debias offline estimates appears in adjacent work (e.g., post-launch evaluation, 2025), but we did not find conformal, finite-sample error-rate control for OPE launch decisions. Re-check before final claims.

