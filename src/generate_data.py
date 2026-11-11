"""Step 3 of the workflow: sequence construction.

Every episode is seeded by ONE REAL CEAS-08 email (its timestamp, sender domain and
feature category are real). A documented synthetic Markov generator supplies the hidden
stage path and the non-email events (clicks, logins, ...). Stage labels are SYNTHETIC.
"""
import numpy as np
import pandas as pd
from .config import (K, M, OBS, OBS_ID, EMAIL_OBS, A_GEN, PHISH_START, BENIGN_START,
                     EMISSION_WEIGHTS, GAP_MEAN_MIN, SEQ_LEN_RANGE, SPLIT_FRACS, N_PER_CLASS)


def email_pool(seed_df, label):
    """Empirical distribution over the 4 email categories for one real class in one split."""
    sub = seed_df[seed_df["label"] == label]["email_obs"]
    counts = np.array([(sub == o).sum() for o in EMAIL_OBS], dtype=float) + 1.0
    return counts / counts.sum()


def build_emission_matrix(p_ph, p_bn):
    """Generator emission matrix B_gen (K x M); email mass is spread by the real pools."""
    B = np.zeros((K, M))
    for s, weights in EMISSION_WEIGHTS.items():
        for name, w in weights.items():
            if name == "EMAIL_PH":
                B[s, EMAIL_OBS] += w * p_ph
            elif name == "EMAIL_BN":
                B[s, EMAIL_OBS] += w * p_bn
            else:
                B[s, OBS_ID[name]] += w
    return B / B.sum(axis=1, keepdims=True)


def _sample_path(rng, A, start, T):
    z = np.zeros(T, dtype=int)
    z[0] = rng.choice(K, p=start)
    for t in range(1, T):
        z[t] = rng.choice(K, p=A[z[t - 1]])
    return z


def generate_episodes(seeds, B_gen, rng, A=A_GEN, id_prefix="C", start_index=0):
    """Turn real seed emails into event sequences. Returns (events_df, episodes_df)."""
    ev_rows, ep_rows = [], []
    for i, row in enumerate(seeds.itertuples(index=False)):
        cid = f"{id_prefix}{start_index + i:05d}"
        T = int(rng.integers(SEQ_LEN_RANGE[0], SEQ_LEN_RANGE[1] + 1))
        phishing = int(row.label) == 1
        if phishing:                                   # rejection: must contain a Delivery step
            while True:
                z = _sample_path(rng, A, PHISH_START, T)
                if (z == 2).any():
                    break
            anchor = int(np.argmax(z == 2))
        else:
            z = _sample_path(rng, A, BENIGN_START, T)
            anchor = 0
        obs = np.array([rng.choice(M, p=B_gen[s]) for s in z])
        obs[anchor] = int(row.email_obs)               # the REAL email event
        # timestamps: anchor = real email time; other events by exponential gaps
        gaps = np.array([max(1.0, rng.exponential(GAP_MEAN_MIN[s])) for s in z]) * 60.0
        t0 = row.timestamp.timestamp()
        ts = np.zeros(T)
        ts[anchor] = t0
        for t in range(anchor + 1, T):
            ts[t] = ts[t - 1] + gaps[t]
        for t in range(anchor - 1, -1, -1):
            ts[t] = ts[t + 1] - gaps[t + 1]
        for t in range(T):
            ev_rows.append((cid, t, pd.Timestamp(ts[t], unit="s", tz="UTC"),
                            OBS[obs[t]], int(obs[t]), int(z[t])))
        ep_rows.append((cid, row.timestamp, row.sender_domain, str(row.subject)[:90],
                        int(row.label), OBS[int(row.email_obs)], T,
                        bool((z >= 4).any()), bool((z == 5).any())))
    events = pd.DataFrame(ev_rows, columns=["campaign_id", "step", "timestamp", "event", "obs", "true_stage"])
    episodes = pd.DataFrame(ep_rows, columns=["campaign_id", "seed_time", "sender_domain", "seed_subject",
                                              "seed_label", "seed_event", "length",
                                              "reaches_harvest", "reaches_takeover"])
    return events, episodes


def chronological_split(df, fracs=SPLIT_FRACS):
    """Assign train/val/test by time order (earliest 70% -> train, next 15% -> val, last 15% -> test)."""
    n = len(df)
    n_tr, n_va = int(round(fracs[0] * n)), int(round(fracs[1] * n))
    split = np.array(["train"] * n_tr + ["val"] * n_va + ["test"] * (n - n_tr - n_va), dtype=object)
    out = df.sort_values("timestamp").reset_index(drop=True)
    out["split"] = split
    return out


def sample_seeds(feat_df, n_per_class=N_PER_CLASS, seed=0):
    """Draw n real emails per class (uniformly over the window), then split chronologically."""
    rng = np.random.default_rng(seed)
    parts = []
    for lab in (0, 1):
        sub = feat_df[feat_df["label"] == lab]
        idx = rng.choice(len(sub), size=min(n_per_class, len(sub)), replace=False)
        parts.append(sub.iloc[idx])
    return chronological_split(pd.concat(parts))


def build_dataset(feat_df, n_per_class=N_PER_CLASS, seed=0):
    """Full construction. Returns events, episodes (both with a 'split' column) and per-split B_gen."""
    seeds = sample_seeds(feat_df, n_per_class, seed)
    rng = np.random.default_rng(seed + 1)
    ev_all, ep_all, gens, k = [], [], {}, 0
    for sp in ("train", "val", "test"):
        s = seeds[seeds["split"] == sp]
        p_ph, p_bn = email_pool(s, 1), email_pool(s, 0)       # split-local real statistics
        B_gen = build_emission_matrix(p_ph, p_bn)
        gens[sp] = B_gen
        ev, ep = generate_episodes(s, B_gen, rng, start_index=k)
        k += len(s)
        ev["split"], ep["split"] = sp, sp
        ev_all.append(ev); ep_all.append(ep)
    events = pd.concat(ev_all, ignore_index=True)
    episodes = pd.concat(ep_all, ignore_index=True)
    return events, episodes, gens, seeds


def to_sequences(events):
    """events DataFrame -> list of dicts (obs, states, ts, id), ordered by campaign and step."""
    seqs = []
    for cid, g in events.sort_values(["campaign_id", "step"]).groupby("campaign_id", sort=True):
        seqs.append({"id": cid, "obs": g["obs"].to_numpy(), "states": g["true_stage"].to_numpy(),
                     "ts": pd.to_datetime(g["timestamp"], utc=True, format="mixed").dt.tz_convert(None).to_numpy()})
    return seqs
