# Learning How Wrong Offline Estimates Are: Reliable Launch Decisions for Recommender Systems

PRML course project — Nihar Sagar G, EECE, IIT Dharwad (2026).

Before a company replaces its recommender system, it estimates from old click logs how the new recommender *would* have performed, and launches it only if the estimated improvement is still positive after subtracting a **safety margin** for estimation error. This project shows that on real data the usual margin is about **half as big as it should be**, and replaces it with a **learned margin**: look at past decisions where both the offline estimate and the real result are known, measure how wrong the estimates actually were, and set the margin from that record using **conformal prediction**, which guarantees the chosen safety level (95%).

Everything uses classical machine learning (logistic regression, gradient-boosted trees, ridge regression, naive Bayes, k-NN, random forests, PCA) — no deep learning and no reinforcement learning.

📄 **Paper:** [paper/main.pdf](paper/main.pdf) (5 pages, IEEE format) · 🖥️ **Slides:** [slides/main.pdf](slides/main.pdf) (22 frames) · 🎬 **Demo script:** [docs/DEMO_AND_VIVA.md](docs/DEMO_AND_VIVA.md)

---

## Contents

0. [Coming back to this later? Start here](#0-coming-back-to-this-later-start-here)
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
12. [Ideas if you pick this up again](#12-ideas-if-you-pick-this-up-again)
13. [Data, licence and citations](#13-data-licence-and-citations)

---

## 0. Coming back to this later? Start here

**Status:** finished. Paper, slides, code, live demo and viva were all presented on 10 November 2026. Nothing is half-done.

**What it is, in one sentence:** a way to decide whether to launch a new recommender from offline estimates, where the safety margin is *learned from how wrong past estimates were* instead of taken from a textbook formula, with a 95% guarantee.

**Fastest way to remember the whole project (≈30 minutes):**

1. Read [section 2](#2-the-idea-in-plain-words) of this file (5 min).
2. Flip through [slides/main.pdf](slides/main.pdf) (10 min) — it tells the story with pictures.
3. Run the demo (section 1) and watch the usual margin say LAUNCH and the learned margin say HOLD on a recommender that was secretly worse (5 min).
4. If a word is unfamiliar: [docs/GLOSSARY.md](docs/GLOSSARY.md).
5. For the full detail: [paper/main.pdf](paper/main.pdf), then section 8 of this file.

**The code that matters** (everything else is plumbing):

| File | What to look at |
|---|---|
| [src/ope.py](src/ope.py) | the 6 offline estimators — `phi_dm`, `phi_ips`, `phi_dr` are the important ones |
| [src/decision.py](src/decision.py) | `conformal_quantile` and `clr_lcb` — **the method itself, ~20 lines**; `wald_lcb` is the usual rule it replaces |
| [scripts/demo_launch.py](scripts/demo_launch.py) | `make_decision` — all the rules applied side by side to one real decision |

**Main numbers to remember:** on 84 real decisions the usual margin was safe 60–80% of the time, the learned margin 95–98% (misses: 16 → 4). On 2,688 controlled decisions a biased estimator's wrong launches went from 27.7% to 0%. The price: fewer launches.

---

## 1. Quick start

On macOS or Linux, with Python 3.11, `git`, `curl`, `unzip` and `make` installed and about 12 GB of free disk space:

```bash
git clone https://github.com/nsg365/prml-conformal-offline-evaluation.git && cd prml-conformal-offline-evaluation
make setup          # create .venv and install pinned packages (~1 minute)
make data           # download + convert the Open Bandit Dataset (~10-15 minutes, one time)
make demo           # live launch decision in the terminal (~30 seconds)
make app            # interactive web demo -> open http://localhost:8501 (Ctrl+C to stop)
```

All experiment results are already in the repository (`results/`), so the demos work as soon as `make data` has finished; rerunning the experiments is optional (section 7). `make help` lists every command.

> **On the original laptop** the project lives in `~/Downloads/College/7th_Sem/PRML_LAB/course_project/course_project3`, with `.venv/` and `data/processed/` already in place — just `cd` there and run `make demo` or `make app`.

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

**Worked example (the demo).** Estimate +0.00307, standard error 0.00116.
- Usual: 0.00307 − 1.645 × 0.00116 = **+0.00116 → LAUNCH**.
- Learned, from 72 past decisions: the ⌈73 × 0.95⌉ = 70th smallest overshoot is 2.73, so 0.00307 − 2.73 × 0.00116 = **−0.00010 → HOLD**.
- Truth, measured on real traffic: **−0.00035**. The new recommender was worse; holding was right.

**Adaptive version (optional).** A regression model (ridge or gradient-boosted trees) predicts from simple features (log size, how different the two recommenders are) how hard each decision is, and scales the margin per decision. It only pays off with a large history.

Plain-English definitions of every term, with their textbook names: [docs/GLOSSARY.md](docs/GLOSSARY.md).

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

Why the real errors are too big: grouping the standard error by hour brings the inflation down to 1.1–1.3 for the reweighting and combined estimators (user behaviour drifts within a 12-hour window), but the hour-grouped margin is still only 65–90% safe, and the click model stays near 2× because its error is bias.

Honest limitations:
- Being safe costs launches: on real data the learned margin launches truly better recommenders 13–47% of the time vs 63–83% for the (overconfident) usual margin.
- The real data contains only two recommenders, so the 84 real decisions share structure.
- The guarantee needs new decisions to resemble past ones; with a deliberate population shift, coverage drops to 0.78–0.88.
- The adaptive margin only helps with a large history (≈2,240 past decisions); with 20–50 it loses power.

---

## 4. Repository layout

```
.
├── README.md                  ← this file (start here)
├── Makefile                   ← every command; `make help` lists them
├── requirements.txt           ← core packages, pinned (Python 3.11)
├── requirements-verify.txt    ← + obp (reference library, pulls PyTorch), only for `make test`
├── requirements-lock.txt      ← exact versions of every package in a known-good environment
├── .streamlit/config.toml     ← web app theme (dark) and localhost-only setting
│
├── src/                       ← THE METHOD (small, importable library code)
│   ├── ope.py                 ←   offline estimators: click model (DM), reweighting (IPS, SNIPS),
│   │                              combined (DR, SNDR, Switch-DR), written as one value per log row
│   └── decision.py            ←   launch rules: usual (Wald), clustered se, bootstrap, Bernstein,
│                                  conformal quantile, learned margin (CLR), adaptive margin
│
├── scripts/                   ← THE PIPELINE (each step reads the previous step's output)
│   ├── convert_obd.py         ←   1. raw CSV (11 GB) → compact Parquet in data/processed/
│   ├── audit_obd.py           ←   2. dataset facts → results/obd_audit.json
│   ├── obd_experiments.py     ←   3. 84 real launch decisions → results/experiments/
│   ├── semisynthetic_experiments.py ← 4. 2,688 controlled decisions → results/experiments/
│   ├── launch_study.py        ←   5. compare all launch rules → results/launch_study/*.csv
│   ├── make_tables.py         ←   6. LaTeX tables → paper/tables/, slides/tables/
│   ├── make_figures.py        ←   7. figures → figures/
│   └── demo_launch.py         ←   terminal demo (its functions are reused by the web app)
│
├── app/app.py                 ← THE WEB DEMO (Streamlit; `make app`)
├── tests/test_against_obp.py  ← checks our estimators against the obp reference library
│
├── results/                   ← ALL OUTPUTS (committed; small)
│   ├── experiments/           ←   obd_experiments.parquet (real), semisynthetic_experiments.parquet
│   ├── launch_study/          ←   coverage / wrong-launch rate / power for every rule (CSV)
│   └── obd_audit.json         ←   rows, dates, click rates of the dataset
├── figures/                   ← every figure, shared by paper, slides and app
│                                (*.pdf for LaTeX, *_dark.png for the web app, pipeline.tex = TikZ diagram)
├── paper/                     ← IEEE paper: main.tex, references.bib, tables/, main.pdf
├── slides/                    ← Beamer slides: main.tex, tables/, main.pdf
│
├── docs/
│   ├── GLOSSARY.md            ← every term in plain English (+ its textbook name)
│   ├── DEMO_AND_VIVA.md       ← demo script, what to say, likely questions with answers
│   └── PROJECT_PLAN.md        ← original design and literature check (historical)
│
├── data/                      ← NOT in git (created by `make data`)
│   ├── raw/                   ←   downloaded zip + 11 GB of CSV (delete after conversion)
│   └── processed/             ←   ~250 MB of Parquet; the demos and figures need this
├── build/                     ← NOT in git (Overleaf zips from `make overleaf`)
└── .venv/                     ← NOT in git (created by `make setup`)
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

On macOS, TeX may be installed but not on the PATH; if `latexmk` is not found, run `export PATH=/Library/TeX/texbin:$PATH` first.

### Steps

```bash
git clone https://github.com/nsg365/prml-conformal-offline-evaluation.git
cd prml-conformal-offline-evaluation

make setup              # creates .venv/ and installs requirements.txt (~1 minute, ~600 MB)
make data               # downloads open_bandit_dataset.zip (413 MB), unzips it (11 GB) and converts
                        # it to data/processed/*.parquet (~250 MB). About 10-15 minutes, once.

make demo               # check that everything works (~30 s)
```

Optional:

```bash
make setup-verify       # installs obp + PyTorch (~1 GB) so that `make test` can run
make test               # our 6 estimators vs obp: six lines ending in OK
rm -rf data/raw         # frees 11 GB; everything else only reads data/processed/
```

Nothing needs internet after `make data` (except `make semisynthetic`, which downloads six small OpenML datasets the first time). The web app runs entirely on your laptop.

Tested on macOS (Apple M4) with Python 3.11.14 and the packages in `requirements-lock.txt`; installing from `requirements.txt` was checked in a fresh environment.

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

How to read it: each rule computes *estimate − margin*; LAUNCH if that is above zero. The usual margins all launch; the learned margin holds; the last line (hidden from every rule) shows holding was right. The Bernstein bound also holds, but it holds almost always, so it is useless in practice.

Options: `--campaign all|men|women`, `--window 4..13` (windows 0–3 only serve as history), `--direction random-to-bts|bts-to-random` (`bts` = the adaptive recommender), `--estimator dm|ips|snips|dr|sndr|switch_dr`, `--reward-model logreg|hgb|none` (`none` with ips/snips), `--adaptive`.

### Web demo

```bash
make app                # then open http://localhost:8501 ; stop with Ctrl+C in the terminal
```

- **Landing page** — the headline results and the three-step method.
- **Sidebar** — pick a quick-start case (🎯 Tricky case, ✅ Clear win, ⛔ Clear loss) or any campaign, window and estimator, then **Run decision** (about 25 s the first time, because it really trains the click model; cached afterwards).
- **Live decision tab** — the estimate, one bar per rule (from the estimate down to estimate − margin; launch if it stays above zero), a table, and **Reveal the true result**.
- **All 84 real decisions tab** — decision chart, overshoot histogram, hourly drift.
- **How it works tab** — the method and where the ML is.

Suggested 4-minute flow: 🎯 Tricky case with "Also show the adaptive margin" ticked → Run → ask "which would you trust?" → Reveal → then ✅ Clear win to show the learned margin still launches good recommenders. The full script with what to say is in [docs/DEMO_AND_VIVA.md](docs/DEMO_AND_VIVA.md).

The server listens only on `localhost` (set in `.streamlit/config.toml`).

---

## 7. Reproducing every result

`results/` already contains all results. To regenerate them from scratch (after `make setup` and `make data`):

| Step | Command | Time (Apple M-series laptop) | Output |
|---|---|---|---|
| 1. Dataset audit | `make audit` | < 1 min | `results/obd_audit.json` |
| 2. Real experiments | `make obd` | ~40 min | `results/experiments/obd_experiments.parquet` (84 decisions × 10 estimator variants) |
| 3. Controlled experiments | `make semisynthetic` | ~30–60 min | `results/experiments/semisynthetic_experiments.parquet` (2,688 decisions) |
| 4. Compare launch rules | `make study` | ~30 min | `results/launch_study/*.csv` |
| 5. Tables and figures | `make figures` | ~1 min | `paper/tables/`, `slides/tables/`, `figures/` |
| 6. Paper and slides | `make paper slides` | ~1 min | `paper/main.pdf`, `slides/main.pdf` |

Tips:
- Step 3 can be run one dataset at a time; finished datasets are cached in `results/experiments/semisynthetic_parts/` (not in git): `.venv/bin/python scripts/semisynthetic_experiments.py --datasets letter`.
- `make study` recomputes everything; to redo only the real-data analysis: `.venv/bin/python scripts/launch_study.py --skip-semi`.
- All randomness is seeded, so reruns give the same numbers.
- `make all` runs test → experiments → study → figures → paper → slides in one go (a couple of hours).
- `make clean` deletes Python caches and LaTeX helper files.

---

## 8. How the code works

### Data

The **Open Bandit Dataset** contains ~26M recommendations from ZOZOTOWN (a Japanese fashion site) over 7 days in three campaigns (all items: 80, men's: 34, women's: 46; 3 positions on the page). Two recommenders ran **at the same time** on randomly split traffic: one shows items uniformly at random, the other adapts to recent clicks (Bernoulli Thompson sampling, run by the company — we did not train it). Each row: timestamp, item shown, position, click (0/1), probability the item was shown, 4 hashed user features and user–item affinity scores. `convert_obd.py` streams the CSVs into Parquet.

### Real launch decisions — `scripts/obd_experiments.py`

Each campaign is cut into fourteen 12-hour windows. Each window gives two decisions:
- *random → adaptive*: estimate the adaptive recommender from the random one's log;
- *adaptive → random*: the reverse (importance weights up to ~600).

14 windows × 3 campaigns × 2 directions = **84 decisions**. The **truth** is the other recommender's actual click rate in that window, which is valid because both served the same traffic at the same time (its own measurement error, ≈0.0003, is recorded as `truth_se`). The adaptive recommender's item probabilities are its observed item frequencies per hour and position. Click models (logistic regression and gradient-boosted trees on user, item, position and affinity features) are trained with 2-fold cross-fitting (train on one half, predict the other). For each estimator the script stores the estimate, the standard error (plain, per-impression-clustered and per-hour-clustered), bootstrap quantiles and overlap diagnostics.

### Controlled decisions — `scripts/semisynthetic_experiments.py`

Six OpenML classification datasets become recommendation problems: classes are items, and the click probability is 0.05 + 0.85 = 0.9 for the correct class and 0.05 otherwise, so every recommender's true click rate is exact.

| Dataset | OpenML id | Rows | Features | Classes ("items") |
|---|---|---|---|---|
| letter | 6 | 20,000 | 16 | 26 |
| pendigits | 32 | 10,992 | 16 | 10 |
| optdigits | 28 | 5,620 | 64 | 10 |
| satimage | 182 | 6,430 | 36 | 6 |
| mfeat-factors | 12 | 2,000 | 216 | 10 |
| segment | 36 | 2,310 | 19 | 7 |

A classifier becomes a recommender through a softmax with temperature T and ε-exploration: π(a|x) = (1−ε)·softmax(log P̂(a|x) / T) + ε/K. The current recommender is a weak logistic regression (T ∈ {0.5, 2}, ε = 0.05). The 14 new recommenders are realistic changes: 4 temperature changes, 2 exploration changes, 3 retrains on more data, naive Bayes, k-NN, random forest, and two strong models (logistic regression and boosted trees on lots of data). Population shift is created by over-sampling one side of the first principal component (PCA) in the log and the other side at deployment. 6 datasets × 32 settings (2 temperatures × 2 log sizes × with/without shift × 4 repeats) × 14 candidates = **2,688 decisions**.

### Estimators — `src/ope.py`

Each returns one value φ per log row; the estimate is their mean. The improvement is mean(φ − r), because the current recommender's value is simply the mean logged click. With w = π_new(a|x) / p (p = logged probability of the shown item):
- click model (DM): Σₐ π_new(a|x) · q̂(x,a)
- reweighting (IPS): w · r — unbiased because E[w·r] = Σₐ π_old · (π_new/π_old) · E[r|a] = Σₐ π_new · E[r|a]
- combined (DR): DM + w · (r − q̂(x,a)) — correct if *either* the click model or the probabilities are right
- SNIPS, SNDR (self-normalised) and Switch-DR (correction only where w ≤ 50) as variants.

### Launch rules — `src/decision.py`

Every rule returns a lower bound L; **launch if L > 0**.
- Usual (`wald_lcb`): estimate − 1.645 · se. Also hour- and impression-clustered se (`cluster_se`), the bootstrap 5% quantile and the empirical Bernstein bound.
- Learned margin (`conformal_quantile`, `clr_lcb`): score each past decision s = (estimate − truth)/se; q̂ = the ⌈(m+1)(1−α)⌉-th smallest score (∞ if m < 19, i.e. never launch); L = estimate − q̂ · se.
- Adaptive (`difficulty_features`, `adaptive_scales`, `adaptive_clr_lcb`): ridge or gradient-boosted trees predict log(|s| + 0.25) from log size, effective sample size, weight statistics and log se, with 5-fold cross-fitting; scores are divided by the predicted difficulty σ̂ and L = estimate − q̂ · σ̂_new · se. `split_adaptive_scales` is the variant with the exact guarantee.

### Evaluation — `scripts/launch_study.py`

Rules are always judged on decisions they did not see.
- Real data: each window uses only *earlier* windows as its history (temporal, from window 4 on → 60 test decisions); also leave-one-campaign-out.
- Controlled: m ∈ {20, 50, 100} random past decisions, 100 random splits per dataset; also cross-dataset (≈2,240 past decisions), and "history without shift, test with shift".
- Metrics at α = 0.05 (also 0.10): coverage P(true Δ ≥ L), wrong-launch rate P(launch | Δ ≤ 0), power P(launch | Δ > 0).

### All ML models used

| Model | Role |
|---|---|
| Logistic regression | click model; current recommender and retrained candidates (controlled data) |
| Gradient-boosted trees | stronger click model (used in the demo); a candidate recommender; adaptive-margin difficulty model |
| Ridge regression | adaptive-margin difficulty model |
| Naive Bayes, k-NN, random forest | candidate recommenders (controlled data) |
| PCA | creates the population shift |
| Conformal prediction | turns the record of past errors into a guaranteed margin |

---

## 9. Where each number in the paper comes from

| Paper item | Source file | Produced by |
|---|---|---|
| Table I (terms) | written by hand in `paper/main.tex` | – |
| Table II (error inflation) | `results/launch_study/obd_error_inflation.csv` → `paper/tables/inflation.tex` | `make_tables.py` |
| Table III (main results) | `results/launch_study/obd_temporal.csv`, `semi_within.csv` → `paper/tables/results.tex` | `launch_study.py`, `make_tables.py` |
| Fig. 1 (pipeline) | `figures/pipeline.tex` (TikZ) | written by hand |
| Fig. 2 (overshoot histogram) | `results/experiments/obd_experiments.parquet` | `make_figures.py` |
| Fig. 3 (hourly drift) | `data/processed/*_all.parquet` | `make_figures.py` |
| Fig. 4 (decision chart) | `results/experiments/obd_experiments.parquet` | `make_figures.py` |
| Fig. 5 (coverage vs history) | `results/launch_study/semi_within.csv` | `make_figures.py` |
| Fig. 6 (adaptive gain) | `semi_within.csv`, `semi_cross_dataset.csv` | `make_figures.py` |
| "16 vs 4 misses", "19% overshoot" | `results/experiments/obd_experiments.parquet` (combined estimator, boosted trees) | `make_figures.py` |
| Leave-one-campaign-out, shift and cross-dataset numbers | `obd_leave_campaign_out.csv`, `semi_shift.csv`, `semi_cross_dataset.csv` | `launch_study.py` |

The slides additionally use `figures/obd_temporal.pdf` and `figures/semi_coverage_power.pdf`, and the tables in `slides/tables/`.

Names used in the CSVs:
- **Rules:** `wald` = usual margin; `wald_hour` / `wald_impression` = clustered se; `bootstrap`; `emp_bernstein` = Bernstein bound; `clr` = learned margin; `clr_null`, `clr_weighted` = variants not used in the paper; `aclr_ridge`, `aclr_gbt`, `aclr_gbt_split` = adaptive margin.
- **Estimators:** `dm` = click model; `ips` = reweighting; `dr` = combined; `snips`, `sndr`, `switch_dr` = variants. Reward (click) model: `logreg`, `hgb` (gradient-boosted trees) or `none`.

---

## 10. Paper and slides

- PDFs: [paper/main.pdf](paper/main.pdf) (5-page IEEE conference format) and [slides/main.pdf](slides/main.pdf) (22 Beamer frames, including 3 backup frames).
- **Rebuild locally:** `make paper` / `make slides` (needs TeX Live with IEEEtran, beamer and tikz). Only `main.pdf` is kept; helper files are deleted automatically.
- **Edit on Overleaf:** `make overleaf` writes `build/paper_overleaf.zip` and `build/slides_overleaf.zip`. Each is self-contained (figures copied in, paths fixed) and compiles on its own: Overleaf → New project → Upload project → pick the zip; the main file is `main.tex`. Copy any edits back into `paper/main.tex` or `slides/main.tex` here (figure paths here are `../figures/...`).
- Figures and tables are generated, not hand-made: change `scripts/make_figures.py` / `make_tables.py` and run `make figures`, rather than editing the files in `figures/` or `*/tables/`. The exceptions are `figures/pipeline.tex` (the TikZ diagram) and Table I, which are written by hand.

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
| `latexmk: command not found` (macOS) | `export PATH=/Library/TeX/texbin:$PATH`, or use `make overleaf` and Overleaf |
| Web app: "port 8501 is already in use" | Stop the other app, or run `.venv/bin/python -m streamlit run app/app.py --server.port 8502` |
| Web app shows old results | Refresh the browser, or restart `make app` (results are cached per window) |
| Demo says "pick window 4 or later" | Windows 0–3 have too little history; use 4–13 |
| Need the exact package versions | `.venv/bin/pip install -r requirements-lock.txt` |

---

## 12. Ideas if you pick this up again

These follow directly from the limitations in section 3:

- **Handle shift in the history.** The guarantee breaks when new decisions differ from past ones (coverage 0.78–0.88 under shift). Weighted conformal prediction is already implemented (`conformal_quantile(..., weights=...)`, `clr_weighted` in the CSVs) but was left out of the paper; a proper study of it is the natural next step.
- **More real recommenders.** The real data has only two, so the 84 decisions are related. Any dataset with several recommenders run side by side (or a company's A/B-test history) would make the real-data evaluation much stronger.
- **Better power.** The learned margin launches good recommenders only 13–47% of the time. Using the history to also *correct* the estimate (not just widen the margin), or a sharper difficulty model, could recover launches without losing safety.
- **Write it up.** The literature check (docs/PROJECT_PLAN.md and the paper's related work) found no prior work calibrating launch decisions on a history of past A/B tests; it could become a workshop paper.

---

## 13. Data, licence and citations

- **Open Bandit Dataset** — Saito, Aihara, Matsutani and Narita, *Open Bandit Dataset and Pipeline: Towards Realistic and Reproducible Off-Policy Evaluation*, NeurIPS Datasets and Benchmarks Track, 2021. Licence: CC BY 4.0. Not stored in this repository; downloaded by `make data`.
- **OpenML datasets** (letter, pendigits, optdigits, satimage, mfeat-factors, segment) — downloaded automatically by scikit-learn's `fetch_openml` and cached in `~/scikit_learn_data`.
- Key method references (full list in `paper/references.bib`): conformal prediction (Vovk et al. 2005; Lei et al. 2018; Angelopoulos & Bates 2023), doubly robust estimation (Dudík et al. 2011), conformal prediction under shift (Tibshirani et al. 2019; Barber et al. 2023), cluster-robust standard errors (Cameron & Miller 2015).
