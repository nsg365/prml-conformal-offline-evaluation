# Status — 4 November 2026 (evaluation: Tuesday 10 November 2026)

## Done

- **Design and novelty boundary:** [PROJECT_PLAN.md](PROJECT_PLAN.md). Estimator selection is already published (PAS-IF, AutoOPE), so the contribution is the conformal launch rule (CLR).
- **Data:** Open Bandit Dataset downloaded, converted, audited (`reports/obd_audit.json`).
- **Code:** estimators (`src/ope.py`, verified against obp), launch rules (`src/decision.py`), real and semi-synthetic experiment builders, analysis, figures, tables, live demo. Reproducible through the `Makefile`.
- **Experiments (complete):** 84 real launch experiments (`experiments/obd_experiments.parquet`); 2,688 semi-synthetic experiments (`experiments/semisynthetic_experiments.parquet`); study outputs in `reports/launch_study/`.
- **Paper:** `paper/main.pdf`, 5 pages IEEE conference format, rewritten in plain language with a terms table; RL vocabulary removed; 6 figures (pipeline, error histogram, hourly drift, decision chart, coverage vs history, adaptive gain).
- **Slides:** `slides/main.pdf`, 20 frames, plain language, analogy, pipeline diagram, error histogram, hourly drift, decision chart, "where the ML is" slide.
- **Web demo:** `app/app.py` (`make app`), dark theme, landing page, live decision with reveal.
- **Viva notes:** [DEMO_AND_VIVA.md](DEMO_AND_VIVA.md); **glossary:** [GLOSSARY.md](GLOSSARY.md).
- The null-only and weighted variants were dropped from the paper and slides for simplicity (their code and results remain in `scripts/launch_study.py` and `reports/launch_study/`).

## Headline results (α = 0.05)

- Real data: OPE errors are 1.7–2.6× the reported uncertainty. Wald/bootstrap coverage 0.60–0.80; CLR 0.95–0.98.
- Semi-synthetic: Wald coverage 0.675–0.915; CLR 0.950–0.962 for all ten configurations; DM-GBT false launches 27.7% → 0%.
- Limitation shown: under covariate shift between calibration and test, CLR coverage falls to 0.78–0.88.
- v2 additions: cluster-robust Wald baselines (per impression, per hour) show the real-data excess error comes mainly from hour-to-hour drift; hour-clustered Wald still covers only 0.65–0.90. Adaptive CLR (learned difficulty with ridge / gradient-boosted trees) gains power only with large histories (+9–11 points for DM-GBT at m ≈ 2,240) and loses power with small ones.

## Remaining (before 10 November)

1. Ask a TA whether the project fits the "ML models only" rule (see DEMO_AND_VIVA.md, "Is this reinforcement learning?").
2. Upload `deliverables/paper_overleaf.zip` and `deliverables/slides_overleaf.zip` to Overleaf and check they compile there.
3. Rehearse `make app` and `make demo`, and the viva answers in DEMO_AND_VIVA.md.

Done on 4 November: evaluation date updated on the slides; final literature check (no prior work combining conformal calibration with a history of past launch decisions; two related single-log conformal papers, Zhang et al. 2023 and Foffano et al. 2023, now cited); paper still 5 pages.
