"""End-to-end run: dataset -> preprocessing -> sequences -> model -> inference -> evaluation.
Usage:  python -m src.pipeline --raw data/raw/CEAS_08.csv
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

from .config import *
from .preprocess import load_ceas, email_features
from .generate_data import (build_dataset, to_sequences, generate_episodes, build_emission_matrix,
                            email_pool)
from .hmm import ChainFactorGraph, estimate_parameters, sum_product, viterbi
from .baselines import PerEventBayes, WindowLR, rule_scores, HmmlearnReference
from . import evaluate as ev

ROOT = Path(__file__).resolve().parent.parent


def fit_models(train):
    obs = [s["obs"] for s in train]; st = [s["states"] for s in train]
    pi, A, B = estimate_parameters(obs, st, K, M, alpha=1.0)
    graph = ChainFactorGraph(pi, A, B)
    state_prior = np.bincount(np.concatenate(st), minlength=K) / sum(len(x) for x in st)
    base = {"NaiveBayes(event)": PerEventBayes(state_prior, B),
            "LogReg(event)": WindowLR(1).fit(obs, st),
            "LogReg(window=3)": WindowLR(3).fit(obs, st)}
    return graph, base


def run_inference(graph, base, seqs):
    """Return stage predictions and per-step 'high-stage' scores for every method."""
    obs = [s["obs"] for s in seqs]
    out = {"pred": {}, "score": {}}
    filt, smooth, vit = [], [], []
    for o in obs:
        a, g, _ = graph.infer(o); filt.append(a); smooth.append(g); vit.append(graph.decode(o)[0])
    out["pred"]["HMM Viterbi (ours)"] = vit
    out["pred"]["HMM sum-product argmax (ours)"] = [g.argmax(1) for g in smooth]
    out["score"]["HMM filtered (ours)"] = [ev.high_stage_prob(a) for a in filt]
    out["score"]["HMM smoothed (ours)"] = [ev.high_stage_prob(g) for g in smooth]
    out["filtered"], out["smoothed"] = filt, smooth
    for name, m in base.items():
        P = m.predict_proba(np.concatenate(obs)) if isinstance(m, PerEventBayes) else m.predict_proba_seqs(obs)
        if isinstance(m, PerEventBayes):
            P = np.split(P, np.cumsum([len(o) for o in obs])[:-1])
        out["pred"][name] = [p.argmax(1) for p in P]
        out["score"][name] = [ev.high_stage_prob(p) for p in P]
    out["score"]["Rule (cred/rule/forward)"] = [rule_scores(o) for o in obs]
    return out


def evaluate_set(seqs, inf, taus):
    y_true = np.concatenate([s["states"] for s in seqs])
    y_camp = np.array([(s["states"] >= 4).any() for s in seqs])
    res = {"stage": {}, "detect": {}, "lead": {}}
    for name, preds in inf["pred"].items():
        res["stage"][name] = ev.stage_metrics(y_true, np.concatenate(preds))
    for name, sc in inf["score"].items():
        camp = np.array([ev.campaign_score(x) for x in sc])
        res["detect"][name] = ev.prf(y_camp, camp, taus[name])
        res["detect"][name]["tau"] = taus[name]
        res["detect"][name]["avg_precision"] = ev.pr_curve(y_camp, camp)[2]
        res["lead"][name] = ev.lead_time_stats(seqs, sc, taus[name])
    return res


def perturb_obs(seqs, eps, rng):
    """Replace each event with a random one w.p. eps (sensor noise / evasion)."""
    out = []
    for s in seqs:
        o = s["obs"].copy()
        m = rng.random(len(o)) < eps
        o[m] = rng.integers(0, M, m.sum())
        out.append({**s, "obs": o})
    return out


def run(raw_path, out_dir=ROOT, n_per_class=N_PER_CLASS, seed=SEED, save=True):
    out_dir = Path(out_dir)
    # 1-2. dataset -> preprocessing
    ceas = load_ceas(raw_path)
    feat = email_features(ceas)
    # 3. sequence construction + chronological split
    events, episodes, gens, seeds = build_dataset(feat, n_per_class, seed)
    sq = {sp: to_sequences(events[events.split == sp]) for sp in ("train", "val", "test")}
    # 4-5. model formulation + estimation (TRAIN ONLY)
    graph, base = fit_models(sq["train"])
    # 6. inference on val (tune tau) and test
    inf = {sp: run_inference(graph, base, sq[sp]) for sp in ("val", "test")}
    y_val = np.array([(s["states"] >= 4).any() for s in sq["val"]])
    taus = {}
    for name, sc in inf["val"]["score"].items():
        taus[name] = 0.5 if name.startswith("Rule") else ev.choose_threshold(y_val, [ev.campaign_score(x) for x in sc])
    # 7. evaluation on TEST
    res = {"test": evaluate_set(sq["test"], inf["test"], taus), "val": evaluate_set(sq["val"], inf["val"], taus)}
    y_te = np.array([(s["states"] >= 4).any() for s in sq["test"]])
    res["pr_curves"] = {n: ev.pr_curve(y_te, [ev.campaign_score(x) for x in sc])[:2]
                        for n, sc in inf["test"]["score"].items() if not n.startswith("Rule")}
    # paired bootstrap: HMM (online alert) vs 3-event logistic regression
    sc_h = [ev.campaign_score(x) for x in inf["test"]["score"]["HMM filtered (ours)"]]
    sc_l = [ev.campaign_score(x) for x in inf["test"]["score"]["LogReg(window=3)"]]
    res["bootstrap_hmm_vs_logreg3"] = ev.bootstrap_f1(y_te, sc_h, taus["HMM filtered (ours)"], sc_l, taus["LogReg(window=3)"], seed=seed)
    # reference-implementation check (hmmlearn, same parameters)
    ref = HmmlearnReference(graph.pi, graph.A, graph.B)
    te_obs = [s["obs"] for s in sq["test"]]
    ref_path, ref_post = ref.decode_all(te_obs)
    own_path, own_post = inf["test"]["pred"]["HMM Viterbi (ours)"], inf["test"]["smoothed"]
    y_flat = np.concatenate([s["states"] for s in sq["test"]])
    res["reference_check"] = {
        "viterbi_path_agreement": float(np.mean(np.concatenate(own_path) == np.concatenate(ref_path))),
        "max_abs_posterior_diff": float(max(np.abs(a - b).max() for a, b in zip(own_post, ref_post))),
        "max_abs_loglik_diff": float(max(abs(graph.infer(o)[2] - ref.loglik(o)) for o in te_obs[:200])),
        "hmmlearn_viterbi_stage_accuracy": float(np.mean(np.concatenate(ref_path) == y_flat)),
        "hmmlearn_viterbi_macro_f1": float(ev.stage_metrics(y_flat, np.concatenate(ref_path))["macro_f1"]),
    }
    # robustness A: observation noise
    rng = np.random.default_rng(seed + 7)
    rob = {"noise": {}, "stage_skipping": {}}
    for eps in (0.0, 0.1, 0.2, 0.3, 0.4):
        s2 = perturb_obs(sq["test"], eps, rng)
        r = evaluate_set(s2, run_inference(graph, base, s2), taus)
        rob["noise"][str(eps)] = {n: {"stage_acc": r["stage"][n]["accuracy"] if n in r["stage"] else None,
                                       "detect_f1": r["detect"][n]["f1"]} for n in r["detect"]}
        rob["noise"][str(eps)]["_stage_acc"] = {n: m["accuracy"] for n, m in r["stage"].items()}
    # robustness B: unseen stage skipping (generator changed at test time only)
    te_seeds = seeds[seeds.split == "test"]
    B_te = gens["test"]
    ev_sk, _ = generate_episodes(te_seeds, B_te, np.random.default_rng(seed + 9), A=A_SKIP, id_prefix="S")
    s_sk = to_sequences(ev_sk)
    r = evaluate_set(s_sk, run_inference(graph, base, s_sk), taus)
    rob["stage_skipping"] = {n: {"stage_acc": r["stage"][n]["accuracy"] if n in r["stage"] else None,
                                  "detect_f1": r["detect"][n]["f1"]} for n in r["detect"]}
    rob["stage_skipping"]["_stage_acc"] = {n: m["accuracy"] for n, m in r["stage"].items()}
    res["robustness"] = rob
    # threshold sweep (precision / recall / lead-time trade-off) for the online HMM alert
    sweep = []
    for tau in np.round(np.arange(0.05, 0.99, 0.05), 2):
        d = ev.prf(y_te, [ev.campaign_score(x) for x in inf["test"]["score"]["HMM filtered (ours)"]], tau)
        l = ev.lead_time_stats(sq["test"], inf["test"]["score"]["HMM filtered (ours)"], tau)
        sweep.append({"tau": float(tau), **{k: d[k] for k in ("precision", "recall", "f1")}, **l})
    res["threshold_sweep"] = sweep
    # dataset facts
    res["dataset"] = {
        "rows_raw": int(ceas.attrs["rows_raw"]), "rows_window": int(len(ceas)),
        "window": [WINDOW_START, WINDOW_END], "episodes": int(len(episodes)),
        "events": int(len(events)),
        "per_split": {sp: {"episodes": int((episodes.split == sp).sum()),
                           "positive_rate": float(episodes[episodes.split == sp].reaches_harvest.mean()),
                           "time_range": [str(episodes[episodes.split == sp].seed_time.min()),
                                          str(episodes[episodes.split == sp].seed_time.max())]}
                      for sp in ("train", "val", "test")},
        "email_category_mix": {"phishing": email_pool(seeds, 1).round(4).tolist(),
                               "benign": email_pool(seeds, 0).round(4).tolist()},
    }
    res["taus"] = taus
    res["params"] = {"pi": graph.pi.tolist(), "A": graph.A.tolist(), "B": graph.B.tolist()}
    if save:
        (out_dir / "data/processed").mkdir(parents=True, exist_ok=True)
        events.to_csv(out_dir / "data/processed/events.csv.gz", index=False)
        episodes.to_csv(out_dir / "data/processed/episodes.csv", index=False)
        graph.save(out_dir / "models/hmm_params.npz")
        (out_dir / "results").mkdir(exist_ok=True)
        (out_dir / "results/metrics.json").write_text(json.dumps(res, indent=1))
    return res, graph, base, sq, inf, taus


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "data/raw/CEAS_08.csv"))
    ap.add_argument("--n_per_class", type=int, default=N_PER_CLASS)
    a = ap.parse_args()
    res, *_ = run(a.raw, n_per_class=a.n_per_class)
    t = res["test"]
    print("\nTEST stage-decoding accuracy:")
    for n, m in t["stage"].items(): print(f"  {n:34s} acc={m['accuracy']:.3f} macroF1={m['macro_f1']:.3f}")
    print("TEST detection (campaign level):")
    for n, m in t["detect"].items(): print(f"  {n:34s} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} tau={m['tau']:.3f}")
    print("reference check:", res["reference_check"])
