# Learning How Wrong Offline Estimates Are: Reliable Launch Decisions for Recommender Systems

PRML course project — Nihar Sagar G, EECE, IIT Dharwad (2026).

Before a company replaces its recommender system, it estimates from old click logs how the new recommender *would* have performed, and launches it only if the estimated improvement is still positive after subtracting a **safety margin** for estimation error. This project shows that on real data the usual margin is about **half as big as it should be**, and replaces it with a **learned margin**: look at past decisions where both the offline estimate and the real result are known, measure how wrong the estimates actually were, and set the margin from that record using **conformal prediction**, which guarantees the chosen safety level (95%).

Everything uses classical machine learning (logistic regression, gradient-boosted trees, ridge regression, naive Bayes, k-NN, random forests, PCA) — no deep learning and no reinforcement learning.

---

## Contents

1. [Quick start](#1-quick-start)
2. [The idea in plain words](#2-the-idea-in-plain-words)
3. [Results](#3-results)
4. [Repository layout](#4-repository-layout)
5. [Setting up on a new computer](#5-setting-up-on-a-new-computer)
6. [Running the demos](#6-running-the-demos)
7. [Reproducing every result](#7-reproducing-every-result)
8. [How the code works](#8-how-the-code-works)
9. [Where each number in the paper comes from](#9-where-each-number-in-the-paper-comes-from)
10. [Paper and slides](#10-paper-and-slides)
11. [Troubleshooting](#11-troubleshooting)
12. [Data, licence and citations](#12-data-licence-and-citations)
13. [Project history and decisions](#13-project-history-and-decisions)

---

## 1. Quick start

On macOS or Linux, with Python 3.11, `git`, `curl`, `unzip` and `make` installed and about 12 GB of free disk space:

```bash
git clone https://github.com/nsg365/prml-conformal-offline-evaluation.git && cd prml-conformal-offline-evaluation
make setup          # create .venv and install pinned packages (~1 minute)
make data           # download + convert the Open Bandit Dataset (~10-15 minutes, one time)
make demo           # live launch decision in the terminal (~25 seconds)
make app            # interactive web demo -> open http://localhost:8501
```

All experiment results are already in the repository (`experiments/`, `reports/`), so the demos work immediately after `make data`; rerunning the experiments is optional (section 7). `make help` lists every command.

---

## 2. The idea in plain words

**Situation.** A shopping app has a *current* recommender (the model that picks which products to show) and wants to try a *new* one. An online A/B test is slow and risky, so the team first **estimates offline**: from the current recommender's log (who was shown what, and whether they clicked), estimate the click rate the new recommender would have achieved.

**Rule.** Launch only if *estimated improvement − safety margin > 0*.

**Problem.** The usual margin is 1.645 standard errors, where the standard error comes from the spread inside one log. It assumes the estimate is unbiased and the standard error is honest. On real data neither holds: the errors were **1.7–2.6× larger** than the standard error claimed (user behaviour drifts hour to hour, and click models are biased), so a margin meant to be safe 95% of the time was safe only **60–80%** of the time.

**Fix — the learned margin.** A company that does this regularly has a track record: for many past candidates it knows the offline estimate *and* the real result measured later. For each past decision compute

```
overshoot s = (estimate − true improvement) / standard error
```

sort the overshoots, take the value that 95% of them are below (on real data ≈ 2.7, not 1.645), and launch only if *estimate − 2.7 × standard error > 0*.

**Analogy.** A friend says "10 minutes" but, judging by the last 20 times, can be up to 25 minutes late — so you plan for 25. You don't need to know *why* they're late.

**Guarantee.** If the new decision resembles the past ones (statistically: the scores are *exchangeable*), the margin is large enough at least 95% of the time, **whatever the estimator and even if it is biased**. This is the split-conformal guarantee: the new score is equally likely to land anywhere among the m+1 sorted scores, so it exceeds the chosen one at most 5% of the time.

**Adaptive version (optional).** A regression model (ridge or gradient-boosted trees) predicts from simple features (log size, how different the two recommenders are) how hard each decision is, and scales the margin per decision.

Plain-English definitions of every term, with their textbook names: [GLOSSARY.md](GLOSSARY.md).

---

## 3. Results

| | Usual margin | Learned margin |
|---|---|---|
| Real data: how often the margin was big enough (target 95%) | 60–80% | **95–98%** |
| Real data: misses out of 84 decisions | 16 | **4** |
| Real data: actual errors vs reported standard error | 1.7–2.6× too big | (corrected by the margin) |
| Controlled experiments: coverage, 10 estimator variants | 0.675–0.915 | **0.950–0.962** |
| Controlled: wrong launches with a biased click model | 27.7% | **0%** |
| Past decisions needed | – | works from 20 |

Honest limitations:
- Being safe costs launches: on real data the learned margin launches truly better recommenders 13–47% of the time vs 63–83% for the (overconfident) usual margin.
- The real data contains only two recommenders, so the 84 real decisions share structure.
- The guarantee needs new decisions to resemble past ones; with a deliberate population shift, coverage drops to 0.78–0.88.
- The adaptive margin only helps with a large history (≈2,240 past decisions); with 20–50 it loses power.

---

## 4. Repository layout

```
.
├── README.md                  ← this file
├── GLOSSARY.md                ← every term in plain English
├── DEMO_AND_VIVA.md           ← demo script, what to say, likely viva questions and answers
├── PROJECT_PLAN.md            ← original design and novelty boundary (literature check)
├── STATUS.md                  ← what was done
├── Makefile                   ← every command (make help)
├── requirements.txt           ← core packages, pinned (Python 3.11)
├── requirements-verify.txt    ← + obp (reference library, pulls PyTorch) for make test
├── requirements-lock.txt      ← exact versions of all packages in a working environment
├── .streamlit/config.toml     ← web app theme (dark), localhost only
│
├── src/                       ← the method (importable library code)
│   ├── ope.py                 ← offline estimators: click model (DM), reweighting (IPS, SNIPS),
│   │                            combined (DR, SNDR, Switch-DR), as one value per log row
│   └── decision.py            ← launch rules: usual (Wald), bootstrap, Bernstein, clustered se,
│                                conformal quantile, learned margin (CLR), adaptive margin
│
├── scripts/                   ← pipelines that produce data, experiments, results and figures
│   ├── convert_obd.py         ← raw CSV (11 GB) → compact Parquet in data/processed/
│   ├── audit_obd.py           ← dataset facts → reports/obd_audit.json
│   ├── obd_experiments.py     ← 84 real launch decisions → experiments/obd_experiments.parquet
│   ├── semisynthetic_experiments.py ← 2,688 controlled decisions → experiments/semisynthetic_experiments.parquet
│   ├── launch_study.py        ← compares all rules → reports/launch_study/*.csv
│   ├── make_tables.py         ← LaTeX tables → paper/tables/
│   ├── make_figures.py        ← figures → paper/figures/ (PDF + PNG + dark PNG for the app)
│   └── demo_launch.py         ← terminal demo (its functions are reused by the web app)
│
├── app/app.py                 ← Streamlit web demo (make app)
├── tests/test_against_obp.py  ← our estimators vs the obp reference library
│
├── experiments/               ← experiment outputs (committed; small)
├── reports/                   ← audit + launch-study result tables (committed)
├── paper/                     ← IEEE paper: main.tex, references.bib, figures/, tables/, main.pdf
├── slides/                    ← Beamer slides: main.tex, figures/, main.pdf
├── deliverables/              ← final PDFs + Overleaf-ready zips + source-code zip
│
├── data/                      ← NOT in git (created by make data)
│   ├── raw/                   ← downloaded zip + 11 GB of CSV (deletable after conversion)
│   └── processed/             ← ~250 MB of Parquet, used by everything
├── .venv/                     ← NOT in git (created by make setup)
└── logs/                      ← NOT in git
```

---

## 5. Setting up on a new computer

### Prerequisites

| Need | Why | Check |
|---|---|---|
| **Python 3.11** (exactly 3.11) | pandas 1.5.3 has no wheels for 3.12+ | `python3.11 --version` |
| git, curl, unzip, make | clone, download data, run commands | `make --version` |
| ~12 GB free disk | the raw CSVs are 11 GB (can be deleted after conversion) | |
| ~8 GB RAM | the largest file is read in batches | |
| Optional: TeX Live (pdflatex, latexmk, IEEEtran, beamer, tikz) | only to rebuild the paper/slides locally; Overleaf also works | `latexmk -v` |

Installing Python 3.11: macOS `brew install python@3.11`; Ubuntu `sudo apt install python3.11 python3.11-venv`; Windows: use WSL (Ubuntu) and follow the Ubuntu steps. If `python3.11` is not on your PATH, pass its location: `make setup PYTHON311=/path/to/python3.11`.

### Steps

```bash
git clone https://github.com/nsg365/prml-conformal-offline-evaluation.git
cd prml-conformal-offline-evaluation

make setup              # creates .venv/ and installs requirements.txt (~1 minute, ~600 MB)
make data               # downloads open_bandit_dataset.zip (413 MB), unzips it (11 GB) and converts
                        # it to data/processed/*.parquet (~250 MB). About 10-15 minutes, once.

make demo               # check that everything works (~25 s)
```

Optional:

```bash
make setup-verify       # installs obp + PyTorch (~1 GB) so that `make test` can run
make test               # our 6 estimators vs obp: six lines ending in OK
rm -rf data/raw         # frees 11 GB; everything else only reads data/processed/
```

Nothing needs internet after `make data` (except `make semisynthetic`, which downloads six small OpenML datasets the first time). The web app runs entirely on your laptop.

---

## 6. Running the demos

### Terminal demo

```bash
make demo               # men's campaign, window 12: the one window where the new recommender was secretly worse
make demo-adaptive      # same, plus the adaptive margin (ridge / boosted-tree difficulty model)
.venv/bin/python scripts/demo_launch.py --campaign women --window 9                         # a clear win
.venv/bin/python scripts/demo_launch.py --campaign all --window 7 --direction bts-to-random  # a clear loss
.venv/bin/python scripts/demo_launch.py --help                                              # all options
```

Expected output of `make demo`:

```
Estimated improvement in click rate: +0.00307 (standard error 0.00116)

Rule                                                est - margin  Decision
Usual margin (1.645 x se)                               +0.00116  LAUNCH
Usual margin, hour-grouped se                           +0.00054  LAUNCH
Bootstrap margin                                        +0.00120  LAUNCH
Bernstein bound                                         -0.00378  hold
Learned margin (2.73 x se, 72 past decisions)           -0.00010  hold

True improvement, measured on real traffic (hidden from the rules): -0.00035 -> new recommender is NOT better
```

Options: `--campaign all|men|women`, `--window 4..13` (windows 0–3 only serve as history), `--direction random-to-bts|bts-to-random` (`bts` = the adaptive recommender), `--estimator dm|ips|snips|dr|sndr|switch_dr`, `--reward-model logreg|hgb|none` (`none` with ips/snips), `--adaptive`.

### Web demo

```bash
make app                # then open http://localhost:8501 ; stop with Ctrl+C in the terminal
```

- **Landing page** — the headline results and the three-step method.
- **Sidebar** — pick a quick-start case (🎯 Tricky case, ✅ Clear win, ⛔ Clear loss) or any campaign, window and estimator, then **Run decision** (about 25 s the first time; cached afterwards).
- **Live decision tab** — the estimate, one bar per rule (from the estimate down to estimate − margin; launch if it stays above zero), a table, and **Reveal the true result**.
- **All 84 real decisions tab** — decision chart, overshoot histogram, hourly drift.
- **How it works tab** — the method and where the ML is.

The server listens only on `localhost` (set in `.streamlit/config.toml`). A step-by-step demo script with what to say is in [DEMO_AND_VIVA.md](DEMO_AND_VIVA.md).

---

## 7. Reproducing every result

The committed `experiments/` and `reports/` already contain all results. To regenerate them from scratch (after `make setup` and `make data`):

| Step | Command | Time (Apple M-series laptop) | Output |
|---|---|---|---|
| 1. Dataset audit | `make audit` | < 1 min | `reports/obd_audit.json` |
| 2. Real experiments | `make obd` | ~40 min | `experiments/obd_experiments.parquet` (84 decisions × 10 estimator variants) |
| 3. Controlled experiments | `make semisynthetic` | ~30–60 min | `experiments/semisynthetic_experiments.parquet` (2,688 decisions) |
| 4. Compare launch rules | `make study` | ~30 min | `reports/launch_study/*.csv` |
| 5. Tables and figures | `make figures` | ~1 min | `paper/tables/*.tex`, `paper/figures/*` |
| 6. Paper and slides | `make paper slides` | ~1 min | `paper/main.pdf`, `slides/main.pdf` |

Tips:
- Step 3 can be run one dataset at a time; finished datasets are cached in `experiments/semisynthetic_parts/`: `.venv/bin/python scripts/semisynthetic_experiments.py --datasets letter`.
- `make study` recomputes everything; to redo only the real-data analysis: `.venv/bin/python scripts/launch_study.py --skip-semi`.
- All randomness is seeded, so reruns give the same numbers.
- After step 5, copy the figures to the slides: `cp paper/figures/*.pdf paper/figures/pipeline.tex slides/figures/`.

---

## 8. How the code works

**Data.** The Open Bandit Dataset contains ~26M recommendations from ZOZOTOWN (a Japanese fashion site) over 7 days in three campaigns (all items: 80, men's: 34, women's: 46; 3 positions on the page). Two recommenders ran **at the same time** on randomly split traffic: one shows items uniformly at random, the other adapts to recent clicks (Bernoulli Thompson sampling, run by the company — we did not train it). Each row: timestamp, item shown, position, click (0/1), probability the item was shown, 4 hashed user features and user–item affinity scores. `convert_obd.py` streams the CSVs into Parquet.

**Real launch decisions** (`obd_experiments.py`). Each campaign is cut into fourteen 12-hour windows. Each window gives two decisions:
- *random → adaptive*: estimate the adaptive recommender from the random one's log;
- *adaptive → random*: the reverse (importance weights up to ~600).

The **truth** is the other recommender's actual click rate in that window, which is valid because both served the same traffic at the same time. The adaptive recommender's item probabilities are its observed item frequencies per hour and position. Click models (logistic regression and gradient-boosted trees on user, item, position and affinity features) are trained with 2-fold cross-fitting. For each estimator the script stores the estimate, the standard error (plain, per-impression-clustered and per-hour-clustered), bootstrap quantiles and overlap diagnostics.

**Controlled decisions** (`semisynthetic_experiments.py`). Six OpenML classification datasets (letter, pendigits, optdigits, satimage, mfeat-factors, segment) become recommendation problems: classes are items, and the click probability is 0.9 for the correct class and 0.05 otherwise, so every recommender's true click rate is exact. The current recommender is a weak logistic regression; new recommenders are realistic changes (temperature, exploration, retraining, naive Bayes, k-NN, random forest, plus two strong models). Population shift is created by tilting the sampled population along the first principal component (PCA).

**Estimators** (`src/ope.py`). Each returns one value φ per log row; the estimate is their mean. The improvement is mean(φ − r), because the current recommender's value is simply the mean logged click.
- click model (DM): Σₐ π_new(a|x) · q̂(x,a)
- reweighting (IPS): w · r, with w = π_new(a|x) / p
- combined (DR): DM + w · (r − q̂(x,a))
- SNIPS, SNDR and Switch-DR (τ = 50) as variants.

**Launch rules** (`src/decision.py`).
- Usual: estimate − 1.645 · se; plus hour-clustered se, the bootstrap 5% quantile and the empirical Bernstein bound.
- Learned margin (`conformal_quantile`, `clr_lcb`): q̂ = the ⌈(m+1)(1−α)⌉-th smallest past score; launch if estimate − q̂ · se > 0.
- Adaptive (`difficulty_features`, `adaptive_scales`, `adaptive_clr_lcb`): ridge or gradient-boosted trees predict log|s| from log size, effective sample size, weight statistics and log se; scores are divided by the predicted difficulty; 5-fold cross-fitting (the split version keeps the exact guarantee).

**Evaluation** (`launch_study.py`). Rules are always judged on decisions they did not see.
- Real data: each window uses only earlier windows as its history (temporal); also leave-one-campaign-out.
- Controlled: m ∈ {20, 50, 100} random past decisions, 100 random splits per dataset; also cross-dataset, and "history without shift, test with shift".
- Metrics at α = 0.05: coverage P(true Δ ≥ estimate − margin), wrong-launch rate P(launch | Δ ≤ 0), power P(launch | Δ > 0).

---

## 9. Where each number in the paper comes from

| Paper item | Source file | Produced by |
|---|---|---|
| Table I (terms) | written by hand in `paper/main.tex` | – |
| Table II (error inflation) | `reports/launch_study/obd_error_inflation.csv` → `paper/tables/inflation.tex` | `make_tables.py` |
| Table III (main results) | `reports/launch_study/obd_temporal.csv`, `semi_within.csv` → `paper/tables/results.tex` | `launch_study.py`, `make_tables.py` |
| Fig. 1 (pipeline) | `paper/figures/pipeline.tex` (TikZ) | written by hand |
| Fig. 2 (overshoot histogram) | `experiments/obd_experiments.parquet` | `make_figures.py` |
| Fig. 3 (hourly drift) | `data/processed/*_all.parquet` | `make_figures.py` |
| Fig. 4 (decision chart) | `experiments/obd_experiments.parquet` | `make_figures.py` |
| Fig. 5 (coverage vs history) | `reports/launch_study/semi_within.csv` | `make_figures.py` |
| Fig. 6 (adaptive gain) | `semi_within.csv`, `semi_cross_dataset.csv` | `make_figures.py` |
| "16 vs 4 misses", "19% overshoot" | `experiments/obd_experiments.parquet` (combined estimator, boosted trees) | `make_figures.py` |
| Leave-one-campaign-out, shift and cross-dataset numbers | `obd_leave_campaign_out.csv`, `semi_shift.csv`, `semi_cross_dataset.csv` | `launch_study.py` |

Names used in the CSVs:
- **Rules:** `wald` = usual margin; `wald_hour` / `wald_impression` = clustered se; `bootstrap`; `emp_bernstein` = Bernstein bound; `clr` = learned margin; `clr_null`, `clr_weighted` = variants not used in the paper; `aclr_ridge`, `aclr_gbt`, `aclr_gbt_split` = adaptive margin.
- **Estimators:** `dm` = click model; `ips` = reweighting; `dr` = combined; `snips`, `sndr`, `switch_dr` = variants. Reward (click) model: `logreg`, `hgb` (gradient-boosted trees) or `none`.

---

## 10. Paper and slides

- Final PDFs: `deliverables/paper.pdf` (5-page IEEE conference format) and `deliverables/slides.pdf` (20 Beamer frames).
- **Overleaf:** upload `deliverables/paper_overleaf.zip` or `deliverables/slides_overleaf.zip` as a new project; the main file is `main.tex`.
- **Locally:** `make paper` / `make slides` (needs TeX Live with IEEEtran, beamer and tikz).
- To refresh the zips after editing:
  ```bash
  cd paper  && zip -r ../deliverables/paper_overleaf.zip  main.tex references.bib figures tables && cd ..
  cd slides && zip -r ../deliverables/slides_overleaf.zip main.tex figures && cd ..
  ```

---

## 11. Troubleshooting

| Problem | Fix |
|---|---|
| `python3.11: command not found` | Install Python 3.11 (section 5), or `make setup PYTHON311=/full/path/to/python3.11` |
| pip tries to build pandas from source, or fails | You are not on Python 3.11. Delete `.venv` and rerun `make setup` with 3.11 |
| `FileNotFoundError ... data/processed/...` | Run `make data` |
| `make data` fails half-way | Delete `data/` and rerun; the download URL is https://research.zozo.com/data_release/open_bandit_dataset.zip |
| `ModuleNotFoundError: obp` on `make test` | Run `make setup-verify` (only the test needs obp) |
| obp errors about `DataFrame.drop` | obp 0.4.1 needs pandas < 2; keep the pinned versions |
| Web app: "port 8501 is already in use" | Stop the other app, or run `.venv/bin/python -m streamlit run app/app.py --server.port 8502` |
| Web app shows old results | Refresh the browser, or restart `make app` (results are cached per window) |
| Demo says "pick window 4 or later" | Windows 0–3 have too little history; use 4–13 |
| Need the exact package versions | `.venv/bin/pip install -r requirements-lock.txt` |

---

## 12. Data, licence and citations

- **Open Bandit Dataset** — Saito, Aihara, Matsutani and Narita, *Open Bandit Dataset and Pipeline: Towards Realistic and Reproducible Off-Policy Evaluation*, NeurIPS Datasets and Benchmarks Track, 2021. Licence: CC BY 4.0. Not stored in this repository; downloaded by `make data`.
- **OpenML datasets** (letter, pendigits, optdigits, satimage, mfeat-factors, segment) — downloaded automatically by scikit-learn's `fetch_openml` and cached in `~/scikit_learn_data`.
- Key method references (full list in `paper/references.bib`): conformal prediction (Vovk et al. 2005; Lei et al. 2018; Angelopoulos & Bates 2023), doubly robust estimation (Dudík et al. 2011), conformal prediction under shift (Tibshirani et al. 2019; Barber et al. 2023), cluster-robust standard errors (Cameron & Miller 2015).

---

## 13. Project history and decisions

- **Course requirement:** novelty, technical complexity and a substantial contribution, not just comparing existing models; classical ML only.
- **Earlier ideas dropped:** GPU job scheduling with runtime prediction (too little ML); TikTok breakout prediction (the main result turned out to be circular — a simple ratio rule matched the model); estimator selection for off-policy evaluation (already published: PAS-IF 2023, AutoOPE 2024).
- **Chosen contribution:** calibrating the *launch decision* using past decisions (conformal prediction), tested on real data with real ground truth. Literature boundary: [PROJECT_PLAN.md](PROJECT_PLAN.md).
- **Findings along the way:** real errors are about 2× the reported standard error; clustering by impression does not explain it, clustering by hour mostly does (behaviour drifts within a window); click-model bias remains; the adaptive margin needs a large history.
- **Writing choice:** the paper and slides avoid reinforcement-learning vocabulary ("policy", "reward", "bandit") and use plain terms ("recommender", "click", "margin"). [GLOSSARY.md](GLOSSARY.md) maps each plain term to its textbook name.
- **Tested on:** macOS (Apple M4), Python 3.11.14, packages in `requirements-lock.txt`; setup from `requirements.txt` was verified in a fresh environment.
