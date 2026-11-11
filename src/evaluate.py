"""Evaluation: stage decoding (per step) and campaign-level detection (alerts)."""
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                             confusion_matrix, precision_recall_curve, average_precision_score)
from .config import K, HARVEST_STAGES


def stage_metrics(y_true, y_pred):
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=list(range(K)), zero_division=0)
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=list(range(K)), average="macro", zero_division=0)),
            "per_stage": {"precision": p.tolist(), "recall": r.tolist(), "f1": f.tolist(), "support": s.tolist()},
            "confusion": confusion_matrix(y_true, y_pred, labels=list(range(K))).tolist()}


def high_stage_prob(P):
    """P(stage in {Credential Harvest, Takeover}) for every step. P is (T,K)."""
    return P[:, HARVEST_STAGES].sum(axis=1)


def campaign_score(step_scores):
    """Campaign-level score = strongest evidence at any time step."""
    return float(np.max(step_scores))


def prf(y_true, scores, tau):
    pred = (np.asarray(scores) > tau)
    y = np.asarray(y_true).astype(bool)
    tp, fp, fn = int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn,
            "tn": int((~pred & ~y).sum())}


def choose_threshold(y_true, scores):
    """Pick tau on the VALIDATION set: maximise F1 over candidate thresholds."""
    cands = np.unique(np.round(np.asarray(scores), 4))
    cands = np.concatenate([[0.0], cands[cands < 1.0]])
    best = max(cands, key=lambda t: (prf(y_true, scores, t)["f1"], -t))
    return float(best)


def pr_curve(y_true, scores):
    p, r, _ = precision_recall_curve(np.asarray(y_true).astype(int), scores)
    return p.tolist(), r.tolist(), float(average_precision_score(np.asarray(y_true).astype(int), scores))


def alert_step(step_scores, tau):
    """First time index where the score exceeds tau (None if never)."""
    idx = np.where(np.asarray(step_scores) > tau)[0]
    return int(idx[0]) if len(idx) else None


def lead_time_stats(seqs, step_scores_list, tau):
    """For campaigns that reach Takeover: share alerted BEFORE takeover and median lead (minutes)."""
    leads, n_take, n_before = [], 0, 0
    for s, sc in zip(seqs, step_scores_list):
        z = s["states"]
        if not (z == 5).any():
            continue
        n_take += 1
        t_k = int(np.argmax(z == 5))
        t_a = alert_step(sc, tau)
        if t_a is not None and t_a < t_k:
            n_before += 1
            leads.append((s["ts"][t_k] - s["ts"][t_a]) / np.timedelta64(1, "m"))
    return {"takeover_campaigns": n_take,
            "alerted_before_takeover": n_before / n_take if n_take else 0.0,
            "median_lead_minutes": float(np.median(leads)) if leads else 0.0}


def bootstrap_f1(y_true, score_a, tau_a, score_b, tau_b, n=1000, seed=0):
    """Paired bootstrap over test campaigns: 95% CI of F1(a), F1(b) and F1(a)-F1(b)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true); a = np.asarray(score_a); b = np.asarray(score_b)
    fa, fb = [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        fa.append(prf(y[i], a[i], tau_a)["f1"]); fb.append(prf(y[i], b[i], tau_b)["f1"])
    fa, fb = np.array(fa), np.array(fb)
    q = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    return {"f1_a_ci": q(fa), "f1_b_ci": q(fb), "diff_ci": q(fa - fb), "p_a_better": float(np.mean(fa > fb))}
