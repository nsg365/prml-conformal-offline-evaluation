# Glossary — every term in plain English

Grouped from most to least important. Paper terms are in **bold**; the names the research literature uses are in *(italics)*, in case an examiner uses them.

## The setting

- **Recommender** *(policy)*: a model that decides which items to show a user.
- **Current recommender** *(logging policy)*: the one already running; its records are our data.
- **New recommender** *(candidate / target policy)*: the one we are thinking of launching.
- **Log**: the current recommender's records. Each row: user features, item shown, position on the page, the probability the item was shown, and whether it was clicked.
- **Click rate** *(policy value)*: fraction of shown items that were clicked. Our measure of how good a recommender is.
- **Improvement Δ**: click rate of new minus click rate of current. Positive means the new one is better.
- **Offline estimate** *(off-policy evaluation, OPE)*: estimating the new recommender's click rate from the current recommender's log, without running the new one.
- **A/B test**: showing the new recommender to real users and measuring directly. Accurate but slow and risky.
- **Launch decision**: launch if the estimated improvement minus a safety margin is above zero.

## The three estimators

- **Click model** *(direct method, DM)*: a classifier (logistic regression or gradient-boosted trees) predicts P(click | user, item, position). Average its predictions over what the new recommender would show. Biased if the classifier is wrong.
- **Reweighting** *(inverse propensity scoring, IPS)*: average the logged clicks with weights w = P_new(item) / P_current(item). Unbiased, but noisy when weights are large.
- **Combined** *(doubly robust, DR)*: click model plus reweighting of the click model's mistakes. Unbiased if the shown-probabilities are known; less noisy than reweighting.
- **Weight**: how much more (or less) often the new recommender would show an item than the current one did. A weight of 600 means 600 times more often.
- **Cross-fitting**: split the log in two; train the click model on one half and predict on the other, and vice versa. Stops a model from grading its own homework.

## Uncertainty and decisions

- **Standard error (se)**: the estimate's own report of how uncertain it is, computed from the spread inside one log.
- **Margin**: the amount subtracted from the estimate before deciding.
- **Usual margin** *(Wald / normal approximation)*: 1.645 × se, which is "95% safe" if errors are normal and the estimator is unbiased.
- **Bootstrap**: resample the log many times and look at how the estimate varies. Gives nearly the same margin as the usual one.
- **Bernstein bound** *(empirical Bernstein)*: a very cautious textbook margin that works under weak assumptions but is usually far too large.
- **Clustered (grouped) standard error**: a standard error that allows rows in the same group (same page, same hour) to be related. We tried grouping by page and by hour.
- **Coverage**: how often the margin was big enough, i.e. estimate − margin ≤ true improvement. Target: 95%.
- **Wrong launch** *(false launch, Type I error)*: launching a recommender that is not actually better.
- **Power**: how often a recommender that *is* better gets launched.
- **Bias**: an estimator that is wrong in the same direction on average (e.g. a click model that always under-predicts).
- **Inflation**: actual spread of errors ÷ reported standard error. 1.0 means honest; 2.0 means errors are twice as big as reported.

## Our method

- **Learned margin** *(conformal launch rule, CLR)*: set the margin from past decisions where both the estimate and the true result are known.
- **Overshoot score s**: (estimate − true improvement) ÷ se, for one past decision. "How many standard errors too optimistic was it?"
- **Conformal prediction**: the rule for turning past scores into a margin with a guarantee: sort the m scores and take the ⌈(m+1)·0.95⌉-th smallest.
- **Exchangeable**: "the new decision looks like the past ones" — no systematic difference between them. The one assumption behind the guarantee.
- **Calibration set / history**: the past decisions used to learn the margin.
- **Adaptive learned margin**: a regression model (ridge or gradient-boosted trees) predicts how hard each decision is, so easy decisions get smaller margins.
- **Difficulty features**: log size, largest/average weight, weight spread, effective sample size, log(se).
- **Effective sample size**: how many "useful" rows the log really has once weights are taken into account. Low means the new recommender is very different from the current one.

## Data

- **Open Bandit Dataset**: 26M real recommendations from a Japanese fashion site (ZOZOTOWN), 7 days, 3 campaigns. Two recommenders ran at the same time on split traffic.
- **Random recommender**: shows items uniformly at random.
- **Adaptive recommender** *(Bernoulli Thompson Sampling)*: shows items that have been clicked more recently. It was run by the company; we did not train it.
- **Window**: a 12-hour slice of the data. One window gives two decisions.
- **Controlled experiments** *(semi-synthetic)*: classification datasets turned into recommendation problems so the true answer is known exactly.
- **Population shift**: the users in the log differ from the users at launch time. We create it with PCA.
