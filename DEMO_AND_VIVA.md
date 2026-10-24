# Live demo and viva preparation

Read [GLOSSARY.md](GLOSSARY.md) alongside this if any word is unfamiliar.

## The whole project in plain words (learn this by heart)

1. **Situation.** A shopping app wants to replace its recommender. Before testing on real users, it estimates from old click logs how the new one would have done, and launches only if *estimate − safety margin > 0*.
2. **Problem.** The usual margin (1.645 standard errors) comes from the data itself. On real data, estimates were about **2× more wrong** than that margin assumed, so a margin meant to be "95% safe" was safe only **60–80%** of the time.
3. **Fix.** Look at past decisions where both the estimate and the real result are known. Measure how wrong the estimates were. Set the margin from that record (the late-friend idea). Conformal prediction picks the number so it is right 95% of the time, guaranteed, as long as the new decision resembles past ones.
4. **Result.** The learned margin was safe **95–98%** of the time on 84 real decisions and **95–96%** on 2,688 controlled ones, and it cut wrong launches of a biased estimator from **27.7% to 0%**. The cost is fewer launches.

**The analogy.** A friend says "10 minutes" but, over the last 20 times, arrived up to 25 minutes late. You plan for 25. You don't need to know why they are late. The learned margin is the "25".

## Web demo (recommended for the viva)

```bash
cd ~/Downloads/College/7th_Sem/PRML_LAB/course_project/course_project3
make app
```

Then open **http://localhost:8501** in your browser (it only runs on your laptop; no internet needed). The landing page shows the headline results; the sidebar on the left has the controls. Stop it with **Ctrl+C** in the terminal.

**Suggested flow (about 4 minutes):**
1. **Live decision tab.** In the sidebar choose *Quick start → 🎯 Tricky case*, tick *"Also show the adaptive margin"*, press **Run decision** (about 25 seconds the first time; it is training the click model live).
2. Point at the chart: "Each bar goes from the estimate down to estimate minus margin; the decision is written above it. Launch only if the bar stays above zero. Orange bars — the usual margins — say LAUNCH. Blue — my learned margin and the ML-based adaptive ones — say HOLD."
3. **Ask the examiner:** "Which would you trust?" Then press **Reveal the true result**: the dashed line shows the truth is below zero, and the table marks the usual margins ❌ and ours ✅.
4. Choose *✅ Clear win* and run again: "When the new recommender really is better, the learned margin still launches it."
5. **All 84 real decisions tab:** the decision chart (16 misses vs 4), the error histogram (why 1.645 is too small) and the hourly drift (why errors are bigger).
6. **How it works tab:** the analogy, the four steps, and the table of ML models — useful if asked "where is the ML?".

Examiners can also pick any campaign, window (4–13), direction and estimator themselves. If the web page fails for any reason, fall back to the terminal demo below.

## Live demo, step by step (about 5 minutes)

### Before the viva (once, the day before)
1. Charge the laptop; the demo needs no internet.
2. Run the full demo once (steps below) to make sure everything works on your machine.
3. Open `deliverables/slides.pdf` and `deliverables/paper.pdf` so they are ready to show.
4. Make the terminal font large (Cmd and + in Terminal) so examiners can read it.

### Starting
Open **Terminal** and go to the project folder:

```bash
cd ~/Downloads/College/7th_Sem/PRML_LAB/course_project/course_project3
```

Everything runs through `make`; the project's own Python environment (`.venv`) is used automatically. If you ever need the list of commands: `make help`.

### Step 1 — show the code is correct (5 seconds)

```bash
make test
```

You will see six lines ending in `OK`.
**Say:** "These are my six estimators, written from scratch. Each one gives exactly the same answer as the standard reference library, obp, to nine decimal places."

### Step 2 — the main demo (about 25 seconds)

```bash
make demo
```

While it runs (it really trains the click model and computes the estimate live), **say:** "This takes the real log of the random recommender for one 12-hour window of the men's campaign, trains a gradient-boosted click model on it, and estimates whether the adaptive recommender would be better. Then it applies each margin. The true answer is hidden from all the rules."

Expected output:

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

**Say, pointing at each line:**
1. "The estimate says the new recommender is better by 2.6 standard errors — it looks like a confident win."
2. "Every usual margin says LAUNCH, even the hour-grouped one."
3. "The Bernstein bound says hold, but it says hold almost always, so it is useless in practice."
4. "My learned margin has looked at 72 earlier decisions and learnt that real estimates overshoot by up to 2.73 standard errors, not 1.645. With that margin, it says hold."
5. "The last line is the truth, measured on real traffic: the new recommender was actually worse. So holding was right."

### Step 3 — only if asked "where is the ML in your method?" (about 30 seconds)

```bash
make demo-adaptive
```

Same output plus two lines:

```
Adaptive margin, ridge regression (3.03 x se)           -0.00045  hold
Adaptive margin, boosted trees (3.56 x se)              -0.00106  hold
```

**Say:** "This version trains a ridge regression and a gradient-boosted regressor on the 72 past decisions to predict how hard each decision is, from features like log size and the size of the weights. Both judged this decision harder than average and gave it a bigger margin. In my experiments this adaptive version only pays off when the history is large, so I use the plain learned margin by default."

### Step 4 — show that it still launches good recommenders (about 25 seconds)
Likely question: "Doesn't your margin just always say hold?" Answer with a window where the new recommender really is better:

```bash
.venv/bin/python scripts/demo_launch.py --campaign women --window 9
```

Expected: the usual margins **and the learned margin** say LAUNCH (learned margin 2.84 × se, 54 past decisions); only the Bernstein bound says hold; the truth is +0.00117, so launching was right.
**Say:** "When the evidence is strong, the learned margin still launches. It is only stricter when the estimate is misleading — and the Bernstein bound shows what being too strict looks like."

Any window from 4 to 13 and any campaign works, e.g. `--campaign all --window 7 --direction bts-to-random`. In most windows the usual and learned margins agree; the difference shows up in the windows where the estimate is misleading. Over all 84 decisions the usual margin missed 16 times and the learned margin 4 times (Fig. 4 in the slides).

Other options: `--estimator ips|dm|dr|snips|sndr|switch_dr`, `--reward-model logreg|hgb|none` (use `none` with `ips`/`snips`).

### If something goes wrong
- **`No such file or directory`** → you are not in the `course_project3` folder; run the `cd` command again.
- **`.venv/bin/python: not found`** → the environment is missing; run `make setup` (needs internet, ~2 minutes).
- **Error about missing `data/processed`** → the processed data was deleted; `make data` rebuilds it (downloads 413 MB, ~10 minutes). Check before the viva.
- **Anything else** → show the slide with the same output (slide "Live demo") and explain from there.

## How to read each figure

| Figure | What it shows | The one sentence to say |
|---|---|---|
| Pipeline (Fig. 1) | Log → estimate → compare with past decisions → learned margin → launch/hold | "Blue boxes are the ML models; the dashed arrow means every A/B test makes the margin smarter." |
| Error histogram (Fig. 2) | For each real decision, how many standard errors the estimate overshot the truth | "The usual margin assumes 5% of estimates overshoot past 1.645; on real data 19% do, so we move the margin to 2.73." |
| Hourly drift (Fig. 3) | Click rate of both recommenders, hour by hour, over 7 days | "Behaviour changes inside each 12-hour window, which the usual standard error ignores." |
| Decision chart (Fig. 4) | Every real decision in the main campaign: estimate, both margins, the truth | "A circle is a miss; orange misses 5, blue misses 0 here, and over all campaigns 16 vs 4 out of 84." |
| Coverage vs history (Fig. 5) | Coverage as the number of past decisions grows | "20 past decisions are already enough to reach 95%." |
| Adaptive gain (Fig. 6) | Extra power from the learned difficulty model, by history size | "Below zero it hurts, above zero it helps; it needs a big history." |

## The ML in this project (for "where is the ML?")

| Model | Used for |
|---|---|
| Logistic regression | Click model; the current recommender in controlled experiments; retrained new recommenders |
| Gradient-boosted trees | Click model (stronger); difficulty model in the adaptive margin |
| Ridge regression | Difficulty model in the adaptive margin |
| Naive Bayes, k-NN, random forest | New recommenders in the controlled experiments |
| PCA | Creates the population shift |
| Cross-fitting | Models never predict on rows they were trained on |
| Conformal prediction | Turns the record of past errors into a guaranteed margin |

**One-sentence answer:** "Classifiers predict clicks inside the estimators, five classifier families act as recommenders in the controlled experiments, PCA creates the shift, regression models learn how hard each decision is, and conformal prediction turns all this into a margin with a guarantee."

## Concepts you must be able to explain

**Why reweighting is unbiased.** The current recommender showed item a with probability p. If the new one would show it with probability p_new, multiply each logged click by w = p_new / p. On average this turns "clicks under the current recommender" into "clicks under the new one", because E[w·r] over the current recommender's choices = Σₐ p(a) · (p_new(a)/p(a)) · E[r|a] = Σₐ p_new(a) · E[r|a]. Problem: a few huge weights make it noisy.

**Why the click model can be biased.** If the classifier is wrong (too simple, or not enough data), its predictions are wrong in a consistent direction, and averaging doesn't fix that.

**Why the combined estimator helps.** It starts from the click model and adds reweighted *corrections* r − q̂. If the classifier is good, corrections are small (low noise). If it is bad, the reweighted correction fixes the bias, because the probabilities p are known.

**The usual margin.** Estimate ± 1.645 standard errors assumes errors are normal *and* the estimator is unbiased *and* the standard error is right. On real data the third assumption failed.

**The learned margin.** For each past decision compute s = (estimate − truth)/se. Sort them. Pick the value q that 95% are below (precisely: the ⌈(m+1)·0.95⌉-th smallest). Launch if estimate − q·se > 0.

**Why the guarantee holds.** The margin is big enough exactly when the new s ≤ q. If the new decision resembles the past ones, its s is equally likely to fall at any position among the m+1 scores, and at most 5% of positions are above q. So it fails at most 5% of the time. Nothing about normality or unbiasedness is needed.

**Why "m+1".** The new decision is one more score. Counting it makes the guarantee exact for small m. If m is too small (fewer than 19 at 95%), q is infinite and the rule never launches.

**Grouped (clustered) standard errors.** If rows in the same hour behave alike, treating them as independent understates the uncertainty. Grouping by hour adds up errors within each hour first.

**Adaptive margin.** A regression model predicts how wrong each estimate will be from features like log size and weight sizes. Each past score is divided by its predicted difficulty, and the new margin is scaled by the new decision's predicted difficulty. It only helps when there are many past decisions.

## Likely viva questions and answers

**Q: Is this reinforcement learning?**
No. Reinforcement learning is an agent learning by trial and error to maximise reward. Here nothing learns by trial and error. The two recommenders in the data were run by the company years ago; we only read their logs. Our work is supervised learning (click classifiers, regression models) plus statistics (estimating a quantity and its uncertainty). The research area is called "off-policy evaluation" and uses some RL words, but the methods are prediction and estimation.

**Q: What is novel?**
Existing uncertainty margins for these estimates use only the variability inside one log and assume the estimator is unbiased. Other work chooses the best estimator but gives no guarantee about the decision. We use a record of past decisions to set the margin, which gives a guarantee even for biased estimators, and we test it on real data where the true answer is known.

**Q: Where would the history of past decisions come from?**
Every candidate that later gets an A/B test produces a pair (offline estimate, real result). Large companies run hundreds of A/B tests per year. Our results show the method is valid from 20 past decisions.

**Q: Why is the real ground truth valid?**
The random and adaptive recommenders ran at the same time on randomly split traffic of the same campaign. So the adaptive one's measured click rate in a window is what it achieves on that population at that time, which is exactly what we try to estimate from the random one's log. The truth has its own small measurement error (≈0.0003), which we account for.

**Q: Why are real estimates twice as wrong as reported?**
Average bias is tiny, so the problem is spread. Grouping the 3 items on one page changed almost nothing (1.80 → 1.76 for the combined estimator), so that's not it. Grouping by hour brought it down to 1.1–1.3: user behaviour drifts from hour to hour, and the usual formula ignores that. Click models stay near 2 because their error is bias from an imperfect classifier.

**Q: Then why not just use hour-grouped standard errors?**
We tried. They were safe only 65–90% of the time, they can't fix bias, and you must guess the right grouping in advance. In the demo they still launch the worse recommender. The learned margin needs no explanation of why errors are large; it measures them.

**Q: The learned margin launches good recommenders less often (13–47% vs 63–83%). Isn't that bad?**
The usual margin launches more because it is overconfident; it would also launch the worse recommender in the demo. The only other safe rule, the Bernstein bound, launches better recommenders at most 3% of the time. Fewer launches is the honest price of real uncertainty.

**Q: When does your method fail?**
When new decisions don't resemble past ones. When we built the history without population shift but tested with shift, it was safe only 78–88% of the time. The history must contain similar decisions.

**Q: Did the learned difficulty model help?**
Only with lots of history. With 20 past decisions it lost 2.5–7 points of power; with about 2,240 it gained 1 point for reweighting and combined estimators and 9–11 points for the biased click model. A learned component adds noise that only pays off with enough data.

**Q: How do you know your estimator code is correct?**
`tests/test_against_obp.py` compares all six estimators with the reference implementations in the `obp` library on synthetic data; they agree to 1e-9.

**Q: Why classical ML only?**
The contribution is about making decisions reliable, not about representation learning. Logistic regression and boosted trees are the standard click models for this problem, and the learned margin works with any estimator.

## Key numbers

| | Value |
|---|---|
| Real decisions | 84 (60 judged using only earlier windows; 30 of them not better) |
| Real error inflation, usual se | 1.7–1.8 (reweighting / combined), 2.6 (click model) |
| Real error inflation, hour-grouped se | 1.1–1.3 (reweighting / combined), ~2.0 (click model) |
| Real coverage: usual / hour-grouped / learned | 0.60–0.80 / 0.65–0.90 / 0.95–0.98 |
| Real power: usual / learned | 0.63–0.83 / 0.13–0.47 |
| Controlled decisions | 2,688 (6 datasets, 27% not better) |
| Controlled coverage: usual / learned | 0.675–0.915 / 0.950–0.962 |
| Biased click model, wrong launches | 27.7% → 0% |
| Shift failure | learned margin safe 78–88% |
| Adaptive margin vs plain | −2.5 to −7 points (20 past) … +9 to +11 for the click model (≈2,240 past) |
