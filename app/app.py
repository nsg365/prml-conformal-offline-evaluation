"""Launch or hold? — interactive demo of the learned margin.

Run from the project folder with `make app`. Everything is computed live from
the local Open Bandit data; no internet connection is needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from demo_launch import make_decision, run_window  # noqa: E402

FIGURES = ROOT / "figures"
# Validated dark-surface palette (same hues as the paper's figures).
COLOURS = {"usual": "#d95926", "cautious": "#c98500", "ours": "#3987e5"}
SURFACE, INK, MUTED, GRID = "#1a1a19", "#f0efec", "#c3c2b7", "#33332f"
CAMPAIGNS = {"All items": "all", "Men's items": "men", "Women's items": "women"}
DIRECTIONS = {
    "Adaptive recommender (estimated from the random recommender's log)": "random->bts",
    "Random recommender (estimated from the adaptive recommender's log)": "bts->random",
}
ESTIMATORS = {
    "Combined (boosted trees)": ("dr", "hgb"),
    "Combined (logistic regression)": ("dr", "logreg"),
    "Click model (boosted trees)": ("dm", "hgb"),
    "Click model (logistic regression)": ("dm", "logreg"),
    "Reweighting": ("ips", "none"),
}
PRESETS = {
    "🎯 Tricky case (secretly worse)": ("Men's items", 12, list(DIRECTIONS)[0]),
    "✅ Clear win (really better)": ("Women's items", 9, list(DIRECTIONS)[0]),
    "⛔ Clear loss (worse)": ("All items", 7, list(DIRECTIONS)[1]),
}

st.set_page_config(page_title="Launch or hold?", page_icon="🚦", layout="wide")

# HTML is written without indentation or blank lines so Markdown does not
# turn it into a code block.
st.markdown("""<style>
:root{--surface:#1a1a19;--surface-2:#22221f;--border:#2e2e2b;--ink:#f0efec;--muted:#c3c2b7;--faint:#8f8e86;--blue:#3987e5;--orange:#d95926;}
[data-testid="stDeployButton"], #MainMenu {display:none;}
.block-container{padding-top:2.2rem;max-width:1200px;}
.hero{padding:2.4rem 2.4rem 2rem;border:1px solid var(--border);border-radius:20px;background:radial-gradient(1200px 400px at 10% -20%, rgba(57,135,229,.22), transparent 60%),radial-gradient(800px 300px at 100% 0%, rgba(217,89,38,.14), transparent 60%),var(--surface);margin-bottom:1.4rem;}
.eyebrow{font-size:.78rem;letter-spacing:.14em;text-transform:uppercase;color:var(--faint);font-weight:600;margin-bottom:.6rem;}
.hero h1{font-size:3.2rem;line-height:1.05;margin:0 0 .9rem;font-weight:800;color:var(--ink);padding:0;}
.hero h1 .accent{background:linear-gradient(90deg,#3987e5,#7fb3f2);-webkit-background-clip:text;background-clip:text;color:transparent;}
.lede{font-size:1.18rem;line-height:1.6;color:var(--muted);max-width:52rem;margin:0;}
.lede b{color:var(--ink);}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:1rem;margin:1.6rem 0 0;}
.stat{background:var(--surface-2);border:1px solid var(--border);border-radius:14px;padding:1.1rem 1.2rem;}
.stat .num{font-size:1.9rem;font-weight:800;color:var(--ink);white-space:nowrap;}
.stat .num .old{color:var(--orange);}
.stat .num .arrow{color:var(--faint);font-weight:400;margin:0 .35rem;}
.stat .num .new{color:var(--blue);}
.stat .cap{font-size:.9rem;color:var(--muted);margin-top:.3rem;line-height:1.4;}
.steps{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:1rem;margin:0 0 1.2rem;}
.step{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:1.1rem 1.2rem;}
.step .badge{display:inline-flex;align-items:center;justify-content:center;width:1.8rem;height:1.8rem;border-radius:50%;background:rgba(57,135,229,.18);color:var(--blue);font-weight:800;margin-right:.55rem;}
.step .title{font-weight:700;color:var(--ink);font-size:1.05rem;}
.step p{color:var(--muted);font-size:.93rem;line-height:1.5;margin:.6rem 0 0;}
.cta{border:1px dashed rgba(57,135,229,.55);background:rgba(57,135,229,.08);border-radius:14px;padding:.9rem 1.2rem;color:var(--ink);font-size:1rem;}
.compact-title{font-size:1.9rem;font-weight:800;margin:0;color:var(--ink);}
.compact-sub{color:var(--muted);margin:.2rem 0 1rem;}
</style>""", unsafe_allow_html=True)

HERO = (
    '<div class="hero">'
    '<div class="eyebrow">PRML course project · IIT Dharwad</div>'
    '<h1>Launch or hold? <span class="accent">Learn the safety margin.</span></h1>'
    '<p class="lede">Before replacing a recommender, companies estimate from old click logs how the new one '
    'would do, and launch only if it still looks better after a <b>safety margin</b>. On real data the usual margin '
    'is about <b>half as big as it should be</b>. This project learns the right margin from past decisions, '
    'with a 95% guarantee.</p>'
    '<div class="stats">'
    '<div class="stat"><div class="num"><span class="old">60–80%</span><span class="arrow">→</span>'
    '<span class="new">95–98%</span></div><div class="cap">how often the margin was big enough on real data '
    '(target 95%)</div></div>'
    '<div class="stat"><div class="num"><span class="old">16</span><span class="arrow">→</span>'
    '<span class="new">4</span></div><div class="cap">misses out of 84 real launch decisions</div></div>'
    '<div class="stat"><div class="num"><span class="old">27.7%</span><span class="arrow">→</span>'
    '<span class="new">0%</span></div><div class="cap">wrong launches with a biased click model '
    '(controlled experiments)</div></div>'
    '</div></div>'
)
STEPS = (
    '<div class="steps">'
    '<div class="step"><span class="badge">1</span><span class="title">Estimate offline</span>'
    '<p>A click model (logistic regression or boosted trees) estimates the new recommender\'s click rate '
    'from the current recommender\'s log.</p></div>'
    '<div class="step"><span class="badge">2</span><span class="title">Learn the margin</span>'
    '<p>From past decisions with known outcomes, measure how wrong the estimates really were. '
    'Conformal prediction turns that record into the margin.</p></div>'
    '<div class="step"><span class="badge">3</span><span class="title">Launch or hold</span>'
    '<p>Launch only if estimate − learned margin &gt; 0. Safe at least 95% of the time if new decisions '
    'resemble past ones.</p></div>'
    '</div>'
)


@st.cache_data(show_spinner=False)
def cached_window(campaign: str, window: int) -> list[dict]:
    return run_window(campaign, window)


def pp(x: float) -> str:
    """Click-rate change in percentage points."""
    return f"{100 * x:+.3f} pp"


def decision_plot(result: dict, revealed: bool):
    rules = result["rules"]
    est = 100 * result["estimate"]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    lows = [100 * r["lower"] for r in rules]
    span = max(est, 0) - min(min(lows + ([100 * result["truth"]] if revealed else [])), 0)
    for i, rule in enumerate(rules):
        lower = 100 * rule["lower"]
        ax.vlines(i, lower, est, color=COLOURS[rule["kind"]], lw=16, alpha=0.95)
        # Decisions sit in one row above the bars, clear of the zero and truth lines.
        ax.text(i, max(est, 0) + 0.1 * span, "LAUNCH" if rule["launch"] else "HOLD", ha="center", va="bottom",
                fontsize=9.5, fontweight="bold", color=COLOURS[rule["kind"]] if rule["launch"] else MUTED)
    ax.scatter(range(len(rules)), [est] * len(rules), s=46, color=INK, zorder=3, label="estimate")
    ax.axhline(0, color=MUTED, lw=1.2)
    if revealed:
        ax.axhline(100 * result["truth"], color=INK, lw=1.8, ls=(0, (5, 3)), label="true improvement")
    ax.set_xticks(range(len(rules)))
    ax.set_xticklabels([r["short"] for r in rules], fontsize=9, color=INK)
    ax.tick_params(axis="y", colors=MUTED)
    ax.set_xlim(-0.6, len(rules) - 0.4)
    ax.set_ylabel("Change in click rate\n(percentage points)", color=INK)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    low = min(min(lows + ([100 * result["truth"]] if revealed else [])), 0)
    ax.set_ylim(low - 0.08 * span, max(est, 0) + 0.26 * span)
    legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2, fontsize=8, frameon=False)
    for text in legend.get_texts():
        text.set_color(INK)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
st.sidebar.title("🚦 Launch or hold?")
st.sidebar.caption("Pick a real 12-hour window from the Open Bandit Dataset and decide whether to launch "
                   "the new recommender.")

preset = st.sidebar.selectbox("Quick start", ["(choose your own below)"] + list(PRESETS))
if preset in PRESETS:
    default_campaign, default_window, default_direction = PRESETS[preset]
else:
    default_campaign, default_window, default_direction = "Men's items", 12, list(DIRECTIONS)[0]

campaign_label = st.sidebar.selectbox("Campaign", list(CAMPAIGNS), index=list(CAMPAIGNS).index(default_campaign),
                                      disabled=preset in PRESETS)
window = st.sidebar.slider("12-hour window", 4, 13, default_window, disabled=preset in PRESETS,
                           help="Windows 0-3 are kept as history; the learned margin only uses earlier windows.")
direction_label = st.sidebar.radio("Which recommender is new?", list(DIRECTIONS),
                                   index=list(DIRECTIONS).index(default_direction), disabled=preset in PRESETS)
estimator_label = st.sidebar.selectbox("How to estimate offline", list(ESTIMATORS))
adaptive = st.sidebar.checkbox("Also show the adaptive margin (ML difficulty model)")
run = st.sidebar.button("▶ Run decision", type="primary", width="stretch")
st.sidebar.caption("Data: Open Bandit Dataset (ZOZOTOWN, CC BY 4.0). Runs fully offline.")

settings = (CAMPAIGNS[campaign_label], window, DIRECTIONS[direction_label], *ESTIMATORS[estimator_label], adaptive)
if run:
    st.session_state["settings"] = settings
    st.session_state["revealed"] = False

# --------------------------------------------------------------------------- #
# Main area
# --------------------------------------------------------------------------- #
if "settings" not in st.session_state:
    st.markdown(HERO, unsafe_allow_html=True)
    st.markdown(STEPS, unsafe_allow_html=True)
    st.markdown('<div class="cta">👈 Try it: choose <b>Quick start → 🎯 Tricky case</b> in the sidebar and press '
                '<b>Run decision</b>. The click model is trained live (about 25 seconds the first time).</div>',
                unsafe_allow_html=True)
    st.write("")
else:
    st.markdown('<p class="compact-title">🚦 Launch or hold?</p>'
                '<p class="compact-sub">Estimate offline, subtract a safety margin, launch only if it still looks better.</p>',
                unsafe_allow_html=True)

tab_live, tab_all, tab_how = st.tabs(["Live decision", "All 84 real decisions", "How it works"])

with tab_live:
    if "settings" not in st.session_state:
        st.caption("Results will appear here after you press Run decision.")
    else:
        campaign, window_, direction, estimator, reward_model, adaptive_ = st.session_state["settings"]
        with st.spinner("Training the click model on the real log and estimating live..."):
            rows = cached_window(campaign, window_)
        try:
            result = make_decision(rows, window_, direction, estimator, reward_model, adaptive=adaptive_)
        except ValueError as err:
            st.error(str(err))
            st.stop()

        new_ctr = result["current_ctr"] + result["estimate"]
        c1, c2, c3, c4 = st.columns(4)
        with c1.container(border=True):
            st.metric("Estimated improvement", pp(result["estimate"]),
                      help="Estimated click rate of the new recommender minus the current one.")
        with c2.container(border=True):
            st.metric("Click rate: current → new", f"{result['current_ctr']:.2%} → {new_ctr:.2%}")
        with c3.container(border=True):
            st.metric("Standard error (se)", f"{100 * result['se']:.3f} pp",
                      help="How uncertain the estimate claims to be, from the spread inside this one log.")
        with c4.container(border=True):
            st.metric("Past decisions used", result["n_past"],
                      help="Decisions from earlier windows whose true result is known. The learned margin comes from these.")

        revealed = st.session_state.get("revealed", False)
        with st.container(border=True):
            st.pyplot(decision_plot(result, revealed), width="stretch")
            st.caption("Each bar runs from the estimate (dot) down to estimate − margin. "
                       "**Launch only if the bottom of the bar is above 0.** Orange: usual margins. "
                       "Yellow: very cautious Bernstein bound. Blue: learned margins (ours).")

        table = pd.DataFrame({
            "Rule": [r["name"] for r in result["rules"]],
            "Estimate − margin (pp)": [round(100 * r["lower"], 3) for r in result["rules"]],
            "Decision": ["LAUNCH" if r["launch"] else "HOLD" for r in result["rules"]],
        })
        if revealed:
            better = result["truth"] > 0
            table["Right call?"] = ["✅" if r["launch"] == better else "❌" for r in result["rules"]]
        st.dataframe(table, hide_index=True, width="stretch")

        if not revealed:
            if st.button("🔍 Reveal the true result", type="primary"):
                st.session_state["revealed"] = True
                st.rerun()
        else:
            truth = result["truth"]
            message = (f"True improvement measured on real traffic: **{pp(truth)}** — "
                       + ("the new recommender **is better**, so launching was right."
                          if truth > 0 else "the new recommender is **not better**, so holding was right."))
            st.success(message)
            ours = next(r for r in result["rules"] if r["short"].startswith("Learned"))
            if ours["launch"] == (truth > 0):
                st.markdown(f"The learned margin made the right call. It used **{result['q']:.2f} × se** "
                            f"instead of 1.645 × se, because in the {result['n_past']} past decisions the estimates "
                            "overshot the truth by that much 5% of the time.")
            else:
                st.markdown("The learned margin got this one wrong. It is designed to be wrong at most 5% of the "
                            "time; over all 84 real decisions it missed 4.")

with tab_all:
    m1, m2, m3 = st.columns(3)
    with m1.container(border=True):
        st.metric("Usual margin: misses", "16 / 84", help="Should be at most about 5% (4 of 84).")
    with m2.container(border=True):
        st.metric("Learned margin: misses", "4 / 84")
    with m3.container(border=True):
        st.metric("Real errors vs reported", "1.7–2.6×",
                  help="How much bigger the real errors were than the standard error said.")
    with st.container(border=True):
        st.subheader("Every real decision in the main campaign")
        st.image(str(FIGURES / "decision_chart_dark.png"), width="stretch")
        st.caption("A circle marks a miss: the true improvement fell below estimate − margin.")
    col_a, col_b = st.columns(2)
    with col_a.container(border=True):
        st.subheader("Why 1.645 is too small")
        st.image(str(FIGURES / "error_histogram_dark.png"), width="stretch")
        st.caption("How far each real estimate overshot the truth, in standard errors. 19% are beyond the usual "
                   "margin, where 5% was expected; the learned margin sits where only 5% are beyond it.")
    with col_b.container(border=True):
        st.subheader("Why the errors are bigger")
        st.image(str(FIGURES / "hourly_drift_dark.png"), width="stretch")
        st.caption("Click rates drift hour by hour inside each 12-hour window. The usual standard error "
                   "assumes every row is independent, so it misses this.")

with tab_how:
    st.subheader("The idea in one sentence")
    st.markdown("A friend says *“10 minutes”* but, judging by the last 20 times, can be up to 25 minutes late — "
                "so you plan for 25. The **learned margin** does the same for offline estimates: it looks at past "
                "decisions where the real result is known and sets the margin from how wrong the estimates actually were.")
    st.markdown(STEPS, unsafe_allow_html=True)
    st.subheader("Where the machine learning is")
    st.dataframe(pd.DataFrame({
        "Model": ["Logistic regression", "Gradient-boosted trees", "Ridge regression",
                  "Naive Bayes, k-NN, random forest", "PCA", "Conformal prediction"],
        "Used for": ["Click model", "Click model; difficulty model (adaptive margin)",
                     "Difficulty model (adaptive margin)", "New recommenders in the controlled experiments",
                     "Creating population shift in the controlled experiments",
                     "Turning past errors into a guaranteed margin"],
    }), hide_index=True, width="stretch")
    st.caption("Data: Open Bandit Dataset (ZOZOTOWN, CC BY 4.0). Two recommenders ran at the same time on split "
               "traffic, so each one's measured click rate is the true answer when estimating it from the other's log.")
