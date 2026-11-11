"""Investigation dashboard: phishing campaign stage inference (sum-product + Viterbi).
Run:  streamlit run app/streamlit_app.py
EDUCATIONAL PROTOTYPE - stage labels are synthetic; not a production security system.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import STATES, OBS, OBS_ID, K, HARVEST_STAGES  # noqa: E402
from src.generate_data import to_sequences  # noqa: E402
from src.hmm import ChainFactorGraph  # noqa: E402
from src.preprocess import URGENCY_RE  # noqa: E402

COLORS = ["#9aa5b1", "#4c78a8", "#f2a541", "#e8743b", "#d1344c", "#7b2d8e"]
ALERT_MODEL = "HMM filtered (ours)"

st.set_page_config(page_title="Phishing Campaign Investigator", page_icon="🎣", layout="wide")


# ------------------------------------------------------------------ cached loaders
@st.cache_data
def load_events():
    ev = pd.read_csv(ROOT / "data/processed/events.csv.gz")
    ev["timestamp"] = pd.to_datetime(ev["timestamp"], utc=True, format="mixed")
    return ev


@st.cache_data
def load_episodes():
    ep = pd.read_csv(ROOT / "data/processed/episodes.csv")
    ep["seed_time"] = pd.to_datetime(ep["seed_time"], utc=True, format="mixed")
    return ep


@st.cache_data
def load_metrics():
    return json.loads((ROOT / "results/metrics.json").read_text())


@st.cache_resource
def load_graph():
    return ChainFactorGraph.load(ROOT / "models/hmm_params.npz")


@st.cache_resource
def infer_split(split):
    ev, g = load_events(), load_graph()
    out = {}
    for s in to_sequences(ev[ev.split == split]):
        filt, smooth, _ = g.infer(s["obs"])
        out[s["id"]] = {"seq": s, "filt": filt, "smooth": smooth, "vit": g.decode(s["obs"])[0]}
    return out


def high(P):
    return P[:, HARVEST_STAGES].sum(axis=1)


def first_alert(p, tau):
    idx = np.where(p > tau)[0]
    return int(idx[0]) if len(idx) else None


# ------------------------------------------------------------------ plotting helpers
def tick_labels(obs):
    return [f"{i}: {OBS[o]}" for i, o in enumerate(obs)]


def stage_area(P, obs, title):
    fig = go.Figure()
    for k in range(K):
        fig.add_trace(go.Scatter(x=list(range(len(obs))), y=P[:, k], name=STATES[k], mode="lines",
                                 stackgroup="one", line=dict(width=0.5, color=COLORS[k]),
                                 fillcolor=COLORS[k]))
    fig.update_layout(title=title, yaxis_title="P(stage)", yaxis_range=[0, 1], height=340,
                      margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.35),
                      xaxis=dict(tickmode="array", tickvals=list(range(len(obs))),
                                 ticktext=tick_labels(obs), tickangle=-40))
    return fig


def alert_line(p, tau, obs, t_alert):
    fig = go.Figure(go.Scatter(x=list(range(len(p))), y=p, mode="lines+markers", name="P(Harvest or Takeover)",
                               line=dict(color="#d1344c", width=3)))
    fig.add_hline(y=tau, line_dash="dash", line_color="#444", annotation_text=f"threshold {tau:.2f}")
    if t_alert is not None:
        fig.add_vline(x=t_alert, line_color="#d1344c", line_dash="dot", annotation_text="ALERT")
    fig.update_layout(title="Alert signal (online / filtered)", yaxis_range=[0, 1.02], height=280,
                      margin=dict(l=10, r=10, t=50, b=10),
                      xaxis=dict(tickmode="array", tickvals=list(range(len(obs))),
                                 ticktext=tick_labels(obs), tickangle=-40))
    return fig


def path_chart(vit, truth, obs):
    x = list(range(len(vit)))
    fig = go.Figure()
    if truth is not None:
        fig.add_trace(go.Scatter(x=x, y=truth, name="True stage (synthetic)", line=dict(shape="hv", width=8, color="rgba(90,90,90,.3)")))
    fig.add_trace(go.Scatter(x=x, y=vit, name="Viterbi path", line=dict(shape="hv", width=3, color="#d1344c")))
    fig.update_layout(title="Most likely stage sequence (Viterbi)", height=300, margin=dict(l=10, r=10, t=50, b=10),
                      yaxis=dict(tickmode="array", tickvals=list(range(K)), ticktext=STATES, range=[-0.3, K - 0.7]),
                      xaxis=dict(tickmode="array", tickvals=x, ticktext=tick_labels(obs), tickangle=-40),
                      legend=dict(orientation="h", y=-0.45))
    return fig


def heat(M, xs, ys, title, fmt=".2f", height=360):
    fig = go.Figure(go.Heatmap(z=M, x=xs, y=ys, colorscale="Blues", text=np.round(M, 2), texttemplate="%{text}",
                               showscale=False, zmin=0, zmax=1))
    fig.update_layout(title=title, height=height, margin=dict(l=10, r=10, t=50, b=10), yaxis_autorange="reversed")
    return fig


# ------------------------------------------------------------------ guard
needed = [ROOT / "data/processed/events.csv.gz", ROOT / "models/hmm_params.npz", ROOT / "results/metrics.json"]
if not all(p.exists() for p in needed):
    st.error("Model artifacts not found. Run `python -m src.pipeline --raw data/raw/CEAS_08.csv` first.")
    st.stop()

metrics, graph, episodes = load_metrics(), load_graph(), load_episodes().set_index("campaign_id")

# ------------------------------------------------------------------ header + sidebar
st.title("🎣 Phishing Campaign Investigator")
st.caption("Hidden campaign stages inferred from chronological events with a chain factor graph "
           "(sum-product + Viterbi). **Educational prototype - stage labels are synthetic; not a production security system.**")

with st.sidebar:
    st.header("Controls")
    split = st.selectbox("Dataset split", ["test", "val", "train"], index=0,
                         help="Chronological split: train = earliest 70%, val = next 15%, test = latest 15%.")
    default_tau = float(metrics["taus"][ALERT_MODEL])
    tau = st.slider("Alert threshold  τ", 0.05, 0.99, float(np.clip(default_tau, 0.05, 0.99)), 0.01,
                    help="Alert when P(Credential Harvest or Takeover | events so far) exceeds τ. "
                         f"Default = {default_tau:.3f}, chosen on the validation split.")
    mode = st.radio("Stage probabilities", ["Smoothed (offline, all events)", "Filtered (online, events so far)"])
    st.markdown("---")

data = infer_split(split)
rows = []
for cid, d in data.items():
    p = high(d["filt"]); e = episodes.loc[cid]
    t_a = first_alert(p, tau)
    pos = bool(e.reaches_harvest)
    outcome = ("TP" if pos else "FP") if t_a is not None else ("FN" if pos else "TN")
    rows.append({"campaign_id": cid, "seed_time": e.seed_time, "sender_domain": e.sender_domain,
                 "subject": e.seed_subject, "max_score": float(p.max()), "alert_step": t_a,
                 "alert": t_a is not None, "true_compromise": pos, "outcome": outcome})
queue = pd.DataFrame(rows).sort_values("max_score", ascending=False).reset_index(drop=True)

with st.sidebar:
    view = st.selectbox("Queue filter", ["All campaigns", "Alerted only", "Missed compromises (FN)", "False alarms (FP)"])
    q = {"All campaigns": queue, "Alerted only": queue[queue.alert],
         "Missed compromises (FN)": queue[queue.outcome == "FN"], "False alarms (FP)": queue[queue.outcome == "FP"]}[view]
    if q.empty:
        st.warning("No campaigns match this filter at the current threshold."); st.stop()
    choice = st.selectbox("Campaign", q.campaign_id.tolist(),
                          format_func=lambda c: f"{c} · score {queue.set_index('campaign_id').loc[c, 'max_score']:.2f}"
                                                f"{' · ALERT' if queue.set_index('campaign_id').loc[c, 'alert'] else ''}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔎 Investigation", "🚨 Alert queue", "📊 Model & metrics",
                                         "🧪 Try your own sequence", "ℹ️ About & disclosure"])

# ------------------------------------------------------------------ tab 1: investigation
with tab1:
    d = data[choice]; s = d["seq"]; e = episodes.loc[choice]; T = len(s["obs"])
    P = d["smooth"] if mode.startswith("Smoothed") else d["filt"]
    p_hi = high(d["filt"]); t_a = first_alert(p_hi, tau)
    z = s["states"]; t_take = int(np.argmax(z == 5)) if (z == 5).any() else None
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status", "🚨 ALERT" if t_a is not None else "✅ No alert")
    c2.metric("First alert", f"step {t_a}" if t_a is not None else "-",
              help="Time step at which the online alert signal first crossed τ.")
    c3.metric("Peak P(Harvest/Takeover)", f"{p_hi.max():.2f}")
    if t_a is not None and t_take is not None:
        mins = (s["ts"][t_take] - s["ts"][t_a]) / np.timedelta64(1, "m")
        c4.metric("Lead before takeover", f"{mins:,.0f} min" if t_a < t_take else "alert after takeover")
    else:
        c4.metric("Lead before takeover", "n/a")
    st.markdown(f"**Seed email (real CEAS-08):** `{e.sender_domain}` — _{e.seed_subject}_  \n"
                f"**Ground truth (synthetic):** {'reaches Credential Harvest' if e.reaches_harvest else 'no compromise'}"
                f"{' and Takeover' if e.reaches_takeover else ''}")
    st.plotly_chart(stage_area(P, s["obs"], f"Stage probabilities — {mode.split(' (')[0].lower()}"), width="stretch")
    colA, colB = st.columns(2)
    colA.plotly_chart(alert_line(p_hi, tau, s["obs"], t_a), width="stretch")
    colB.plotly_chart(path_chart(d["vit"], z, s["obs"]), width="stretch")
    tl = pd.DataFrame({"step": range(T), "timestamp": pd.to_datetime(s["ts"]).strftime("%Y-%m-%d %H:%M:%S"),
                       "event": [OBS[o] for o in s["obs"]], "true_stage (synthetic)": [STATES[k] for k in z],
                       "viterbi_stage": [STATES[k] for k in d["vit"]],
                       "most_likely_stage_prob": P.max(1).round(3), "P(harvest/takeover)": p_hi.round(3)})
    st.subheader("Event timeline")
    st.dataframe(tl, width="stretch", hide_index=True)

# ------------------------------------------------------------------ tab 2: alert queue
with tab2:
    y = queue.true_compromise.to_numpy(); a = queue.alert.to_numpy()
    tp, fp, fn = int((a & y).sum()), int((a & ~y).sum()), int((~a & y).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0; rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precision", f"{prec:.3f}"); m2.metric("Recall", f"{rec:.3f}")
    m3.metric("F1", f"{f1:.3f}"); m4.metric("Alerts raised", f"{int(a.sum())} / {len(queue)}")
    st.caption(f"Live metrics on the **{split}** split at τ = {tau:.2f}. Campaign-level target: the campaign reaches "
               "Credential Harvest or Takeover. Move the slider to see the precision / recall trade-off.")
    show = queue.copy()
    show["seed_time"] = show.seed_time.dt.strftime("%Y-%m-%d %H:%M")
    show["max_score"] = show.max_score.round(3)
    st.dataframe(show[["campaign_id", "seed_time", "sender_domain", "subject", "max_score", "alert", "alert_step",
                       "true_compromise", "outcome"]], width="stretch", hide_index=True, height=480)

# ------------------------------------------------------------------ tab 3: model & metrics
with tab3:
    t = metrics["test"]
    st.subheader("Stage decoding (test split, per time step)")
    st.dataframe(pd.DataFrame([{"method": n, "accuracy": m["accuracy"], "macro_F1": m["macro_f1"]}
                               for n, m in t["stage"].items()]).round(3), width="stretch", hide_index=True)
    st.subheader("Compromise detection (test split, campaign level, τ tuned on validation)")
    st.dataframe(pd.DataFrame([{"method": n, "precision": m["precision"], "recall": m["recall"], "F1": m["f1"],
                                "tau": m["tau"], "avg_precision": m["avg_precision"],
                                "alerted_before_takeover": t["lead"][n]["alerted_before_takeover"],
                                "median_lead_min": t["lead"][n]["median_lead_minutes"]}
                               for n, m in t["detect"].items()]).round(3), width="stretch", hide_index=True)
    rc = metrics["reference_check"]
    st.info(f"**Reference check (hmmlearn, same parameters):** Viterbi path agreement = {rc['viterbi_path_agreement']:.1%}, "
            f"max |posterior difference| = {rc['max_abs_posterior_diff']:.1e}, max |log-likelihood difference| = "
            f"{rc['max_abs_loglik_diff']:.1e}.")
    cm = np.array(t["stage"]["HMM Viterbi (ours)"]["confusion"], float); cm = cm / cm.sum(1, keepdims=True)
    g1, g2 = st.columns(2)
    g1.plotly_chart(heat(cm, STATES, STATES, "Viterbi confusion matrix (row-normalised)", height=400), width="stretch")
    fig = go.Figure()
    for n, (p, r) in metrics["pr_curves"].items():
        fig.add_trace(go.Scatter(x=r, y=p, mode="lines", name=n))
    fig.update_layout(title="Precision-recall (test)", xaxis_title="Recall", yaxis_title="Precision", height=400,
                      margin=dict(l=10, r=10, t=50, b=10), yaxis_range=[0.3, 1.02])
    g2.plotly_chart(fig, width="stretch")
    h1, h2 = st.columns(2)
    sw = pd.DataFrame(metrics["threshold_sweep"])
    fig = go.Figure()
    for col, nm in [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"), ("alerted_before_takeover", "Alerted before takeover")]:
        fig.add_trace(go.Scatter(x=sw.tau, y=sw[col], mode="lines", name=nm))
    fig.update_layout(title="Threshold trade-off (HMM filtered, test)", xaxis_title="τ", height=360, margin=dict(l=10, r=10, t=50, b=10))
    h1.plotly_chart(fig, width="stretch")
    nz = metrics["robustness"]["noise"]; eps = list(nz)
    fig = go.Figure()
    for n in [ALERT_MODEL, "LogReg(window=3)", "Rule (cred/rule/forward)"]:
        fig.add_trace(go.Scatter(x=[float(e) for e in eps], y=[nz[e][n]["detect_f1"] for e in eps], mode="lines+markers", name=n))
    fig.update_layout(title="Robustness: detection F1 vs event corruption", xaxis_title="corruption rate", height=360,
                      margin=dict(l=10, r=10, t=50, b=10))
    h2.plotly_chart(fig, width="stretch")
    st.subheader("Learned parameters (estimated on the training split only)")
    p1, p2 = st.columns(2)
    p1.plotly_chart(heat(graph.A, STATES, STATES, "Transition factor ψ (A)"), width="stretch")
    p2.plotly_chart(heat(graph.B, OBS, STATES, "Emission factor φ (B)", height=360), width="stretch")

# ------------------------------------------------------------------ tab 4: custom sequence
with tab4:
    st.markdown("Build an event sequence and see the inferred stages. Events are comma-separated, in chronological order.")
    presets = {
        "Full compromise": "domain_probe, normal_activity, email_url, link_clicked, landing_page_visit, credential_submitted, new_ip_login, mailbox_rule_created, bulk_forward",
        "Benign day": "normal_activity, email_clean, normal_activity, link_clicked, normal_activity, new_ip_login, normal_activity",
        "Fizzled campaign": "email_url_urgent, normal_activity, normal_activity, normal_activity, normal_activity, normal_activity",
        "Noisy compromise (missing events)": "email_url, normal_activity, normal_activity, credential_submitted, normal_activity, normal_activity, mailbox_rule_created",
    }
    pick = st.selectbox("Preset", list(presets))
    text = st.text_area("Events", presets[pick], key=f"seq_{pick}", height=90)
    with st.expander("Optional: turn a pasted email into the first event"):
        email_txt = st.text_area("Email subject + body", "", height=90)
        if email_txt.strip():
            has_url = bool(re.search(r"https?://|www\.", email_txt, re.I)); has_kw = bool(URGENCY_RE.search(email_txt))
            cat = ("email_url_urgent" if has_kw else "email_url") if has_url else ("email_urgent_kw" if has_kw else "email_clean")
            st.write(f"Detected event: `{cat}` (URL: {has_url}, urgency keywords: {has_kw}) — prepended to the sequence.")
            text = cat + ", " + text
    names = [x.strip() for x in text.replace("\n", ",").split(",") if x.strip()]
    bad = [n for n in names if n not in OBS_ID]
    if bad:
        st.error(f"Unknown event(s): {bad}. Valid events: {', '.join(OBS)}")
    elif len(names) < 2:
        st.warning("Enter at least two events.")
    else:
        obs = np.array([OBS_ID[n] for n in names])
        filt, smooth, ll = graph.infer(obs); vit, _ = graph.decode(obs)
        p_hi = high(filt); t_a = first_alert(p_hi, tau)
        k1, k2, k3 = st.columns(3)
        k1.metric("Status", "🚨 ALERT" if t_a is not None else "✅ No alert")
        k2.metric("First alert step", t_a if t_a is not None else "-")
        k3.metric("Final stage (Viterbi)", STATES[vit[-1]])
        st.plotly_chart(stage_area(smooth, obs, "Stage probabilities (smoothed)"), width="stretch")
        cc1, cc2 = st.columns(2)
        cc1.plotly_chart(alert_line(p_hi, tau, obs, t_a), width="stretch")
        cc2.plotly_chart(path_chart(vit, None, obs), width="stretch")

# ------------------------------------------------------------------ tab 5: about
with tab5:
    ds = metrics["dataset"]
    st.markdown(f"""
### What this shows
A phishing campaign moves through hidden stages. We observe only noisy events. A **chain factor graph**
(prior φ₀, transition ψ, emission φₑ) is estimated on the training split; **sum-product** gives stage
probabilities per step and **Viterbi** gives the single most likely stage path. Both are implemented from scratch
in NumPy (`src/hmm.py`) and verified against brute force and `hmmlearn`.

### Data and disclosure
* **Real data:** CEAS-08 phishing/benign emails — timestamps, sender domains, URL flag and urgency keywords
  ({ds['rows_window']:,} emails in {ds['window'][0]} to {ds['window'][1]}).
* **Synthetic data:** the hidden **stage labels and all non-email events** (probes, clicks, credential submission,
  logins, mailbox rules, forwards) come from a documented generator (`src/config.py`, `src/generate_data.py`).
  No public dataset contains campaign stages. Results therefore measure how well the model recovers the generator's
  structure, and they favour a model that matches it. See the robustness tests for mismatch.
* **Split:** chronological by seed-email time (train {ds['per_split']['train']['episodes']} /
  val {ds['per_split']['val']['episodes']} / test {ds['per_split']['test']['episodes']} episodes). Parameters use the training split only; τ is tuned on validation.
* **Limits:** educational prototype, not a production security system. Real telemetry would differ.
""")
