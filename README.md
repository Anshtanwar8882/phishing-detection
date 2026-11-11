# Intelligent Phishing Campaign Behaviour Detection Using Sequential Models

Project 17 — hidden campaign stages inferred from chronological events with a **chain factor graph**,
**sum-product** (forward-backward) and **Viterbi**, implemented from scratch in NumPy and checked against
`hmmlearn`. Includes an interactive **Streamlit investigation dashboard**.

> **Disclosure.** Emails (timestamps, sender domains, URL flag, urgency keywords) are **real** (CEAS-08).
> The hidden **stage labels and all non-email events** (clicks, credential submission, logins, mailbox rules, forwards)
> are **synthetic**, from a documented generator, because no public dataset contains campaign stages.
> This is an **educational prototype, not a production security system.**

## Quick start
```bash
pip install -r requirements.txt
# put CEAS_08.csv in data/raw/   (see data/raw/README.md)
python -m src.pipeline --raw data/raw/CEAS_08.csv   # builds sequences, trains, evaluates (~15 s)
python -m src.plots                                  # figures -> results/figures
python -m pytest tests -q                            # algorithm tests
streamlit run app/streamlit_app.py                   # dashboard
```
The repo already contains the generated sequences (`data/processed/`), trained parameters
(`models/hmm_params.npz`) and metrics (`results/metrics.json`), so the dashboard runs without the raw CSV.

## Workflow (maps to the required pipeline)
| Stage | Where |
|---|---|
| Dataset | `data/raw/CEAS_08.csv` (39,154 emails; 38,631 in the dense 4–10 Aug 2008 window) |
| Preprocessing | `src/preprocess.py` — date parsing, window, de-duplication, URL/keyword features |
| Sequence construction | `src/generate_data.py`, parameters in `src/config.py` |
| Model formulation | chain factor graph: prior φ₀, transition ψ, emission φₑ (`src/hmm.py`) |
| Algorithm implementation | `sum_product`, `viterbi`, `estimate_parameters` in `src/hmm.py` |
| Inference / estimation | counting MLE + Laplace on **train only**; online (filtered) and offline (smoothed) posteriors |
| Evaluation | `src/evaluate.py`, `src/pipeline.py` |
| Visualization | `src/plots.py`, `app/streamlit_app.py` |
| Interpretation | notebook section 10 and `report/` |

## Hidden stages and events
Stages (K=6): Benign · Reconnaissance · Delivery · Lure/Click · Credential Harvest · Takeover.
Events (M=12): `normal_activity`, 4 **real** email categories (`email_clean`, `email_urgent_kw`, `email_url`,
`email_url_urgent`), and 7 synthetic events (`domain_probe`, `link_clicked`, `landing_page_visit`,
`credential_submitted`, `new_ip_login`, `mailbox_rule_created`, `bulk_forward`).

## Data construction and leakage control
* Each real email **seeds one campaign episode**; the real email event is placed at its real timestamp.
* Phishing seeds follow a documented Markov chain (may progress, skip stages or fizzle). Benign seeds stay Benign, with
  hard negatives (ordinary clicks and logins).
* **Chronological split** by email time: first 70% train, next 15% validation, last 15% test (asserted in the notebook).
* Parameters are estimated on **train only**; the alert threshold τ is tuned on **validation** only.
* Event-category mixes used by the generator are computed within each split; the model never sees validation/test data.

## Evaluation protocol
* **Stage decoding:** per-step accuracy, macro-F1, per-stage P/R/F1, confusion matrix.
* **Detection:** a campaign is positive if it reaches Credential Harvest. Alert = first step where
  P(Harvest or Takeover | events so far) > τ. Reported: precision, recall, F1, average precision, share alerted before takeover, lead time.
* **Comparisons:** reference `hmmlearn` (same parameters); baselines Naive Bayes (event), logistic regression (event / 3-event window), SOC-style rule.
* **Robustness:** event corruption 0–40%; unseen stage skipping (test generator changed).

## Results (test split, seed 42)
| Method | Stage acc. | Macro-F1 | Detect P | R | F1 |
|---|---|---|---|---|---|
| **HMM Viterbi / filtered alert (ours)** | **0.911** | **0.775** | **0.959** | 0.859 | **0.907** |
| LogReg (3-event window) | 0.844 | 0.636 | 0.942 | 0.839 | 0.887 |
| Naive Bayes / LogReg (single event) | 0.807 | 0.464 | 0.729 / 0.793 | 0.911 / 0.776 | 0.810 / 0.784 |
| Rule (credential / mailbox rule / forward) | – | – | 0.729 | 0.911 | 0.810 |

Our Viterbi path agrees with `hmmlearn` on **100%** of test steps (max posterior difference ≈ 5e-15).
With 20% corrupted events, detection F1 is 0.79 (HMM) vs 0.71 (window LR) vs 0.47 (rule).
Under unseen stage skipping the HMM keeps its stage-decoding lead (0.918 vs 0.837) but its detection edge vanishes (0.899 vs 0.901).

**Honest caveats:** the F1-optimal τ is conservative (alerts often arrive at or after credential submission; only 42% of
takeovers are alerted before they occur). Lowering τ trades precision for earlier warning (see dashboard slider).
Because the synthetic data come from a chain, a chain model is favoured; read relative comparisons and robustness trends,
not absolute numbers, as the finding.

## Repository layout
```
app/streamlit_app.py     dashboard (investigation, alert queue, metrics, custom sequence, disclosure)
src/                     config, preprocess, generate_data, hmm (core algorithms), baselines, evaluate, pipeline, plots
tests/test_hmm.py        brute-force and hmmlearn checks
notebooks/               reproducible, executed notebook
data/{raw,processed}/    raw CEAS (not committed) / generated sequences
models/hmm_params.npz    trained π, A, B
results/                 metrics.json and figures
report/  slides/         technical report and presentation
```

## Dashboard
Sidebar: split, threshold τ slider, smoothed/filtered view, queue filter, campaign. Tabs: **Investigation** (timeline,
stage-probability chart, alert signal, Viterbi vs truth), **Alert queue** (live precision/recall), **Model & metrics**,
**Try your own sequence** (including paste-an-email helper), **About & disclosure**.

## Limitations and ethics
Synthetic stages and events; one week of 2008 email; simple features; discrete events only. Real telemetry would be noisier
and adversaries adapt. Educational prototype only.
