"""Figures used by the notebook, report and slides. Run: python -m src.plots"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .config import STATES, OBS, K
from .hmm import ChainFactorGraph
from .generate_data import to_sequences

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "results/figures"
COLORS = ["#9aa5b1", "#4c78a8", "#f2a541", "#e8743b", "#d1344c", "#7b2d8e"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def confusion(res, name="HMM Viterbi (ours)", path=FIG / "confusion_viterbi.png"):
    cm = np.array(res["test"]["stage"][name]["confusion"], dtype=float)
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xticklabels(STATES, rotation=35, ha="right"); ax.set_yticklabels(STATES)
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                    color="white" if cmn[i, j] > .5 else "black", fontsize=8)
    ax.set_xlabel("Predicted stage"); ax.set_ylabel("True stage (row-normalised)")
    ax.set_title("Viterbi stage decoding (test)")
    fig.colorbar(im, fraction=.046); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def pr(res, path=FIG / "pr_curves.png"):
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for n, (p, r) in res["pr_curves"].items():
        ap = res["test"]["detect"][n]["avg_precision"]
        ax.plot(r, p, label=f"{n} (AP={ap:.2f})", lw=2 if "ours" in n else 1.2)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_ylim(0.3, 1.02)
    ax.set_title("Compromise detection: precision-recall (test)"); ax.legend(fontsize=7.5)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def robustness(res, path=FIG / "robustness_noise.png"):
    rob = res["robustness"]["noise"]
    eps = [float(e) for e in rob]
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for n, st in [("HMM filtered (ours)", "-o"), ("LogReg(window=3)", "--s"), ("Rule (cred/rule/forward)", ":^")]:
        ax.plot(eps, [rob[str(e)][n]["detect_f1"] for e in eps], st, label=n)
    ax.set_xlabel("Event corruption rate"); ax.set_ylabel("Detection F1")
    ax.set_title("Robustness to noisy / missing telemetry"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def sweep(res, path=FIG / "threshold_tradeoff.png"):
    d = pd.DataFrame(res["threshold_sweep"])
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.plot(d.tau, d.precision, label="Precision"); ax.plot(d.tau, d.recall, label="Recall")
    ax.plot(d.tau, d.f1, label="F1", lw=2.2)
    ax.plot(d.tau, d.alerted_before_takeover, "--", label="Alerted before takeover")
    ax.set_xlabel("Alert threshold tau"); ax.set_title("Threshold trade-off (HMM filtered, test)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def example(path=FIG / "example_campaign.png"):
    ev = pd.read_csv(ROOT / "data/processed/events.csv.gz")
    ev["timestamp"] = pd.to_datetime(ev["timestamp"], utc=True, format="mixed")
    ep = pd.read_csv(ROOT / "data/processed/episodes.csv")
    g = ChainFactorGraph.load(ROOT / "models/hmm_params.npz")
    cid = ep[(ep.split == "test") & ep.reaches_takeover & (ep.length >= 12)].campaign_id.iloc[0]
    s = to_sequences(ev[ev.campaign_id == cid])[0]
    filt, smooth, _ = g.infer(s["obs"]); vit = g.decode(s["obs"])[0]
    T = len(s["obs"]); x = np.arange(T)
    fig, axs = plt.subplots(2, 1, figsize=(8, 5.4), sharex=True, gridspec_kw={"height_ratios": [3, 1.6]})
    axs[0].stackplot(x, smooth.T, labels=STATES, colors=COLORS, alpha=.9)
    axs[0].set_ylabel("P(stage | all events)"); axs[0].legend(ncol=6, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.suptitle(f"Campaign {cid}: stage probabilities (sum-product)", y=0.995)
    axs[1].step(x, s["states"], where="mid", lw=3, color="#444", alpha=.35, label="True stage")
    axs[1].step(x, vit, where="mid", lw=1.6, color="#d1344c", label="Viterbi")
    axs[1].set_yticks(range(K)); axs[1].set_yticklabels(STATES, fontsize=7)
    axs[1].set_xticks(x); axs[1].set_xticklabels([OBS[o] for o in s["obs"]], rotation=60, ha="right", fontsize=7)
    axs[1].legend(fontsize=7, loc="upper left"); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def make_all():
    FIG.mkdir(parents=True, exist_ok=True)
    res = json.loads((ROOT / "results/metrics.json").read_text())
    confusion(res); pr(res); robustness(res); sweep(res); example()


if __name__ == "__main__":
    make_all(); print("figures written to", FIG)
