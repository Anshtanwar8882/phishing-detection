"""Baselines and the reference implementation used for comparison."""
import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from .config import K, M, RULE_OBS, HARVEST_STAGES


class PerEventBayes:
    """Non-sequential: P(stage | single event) by Bayes rule with train prior and emissions."""
    def __init__(self, pi_state, B):
        self.prior = pi_state
        self.B = B

    def predict_proba(self, obs):
        p = self.prior[None, :] * self.B[:, np.asarray(obs)].T
        return p / p.sum(axis=1, keepdims=True)


class WindowLR:
    """Non-sequential-model baseline: multinomial logistic regression on a short window of
    the last `window` events (window=1 -> current event only). No chain structure."""
    def __init__(self, window=1, C=1.0):
        self.w, self.C = window, C
        self.enc = OneHotEncoder(categories=[list(range(M + 1))] * window, handle_unknown="ignore")

    def _window_matrix(self, seqs_obs):
        rows = []
        for o in seqs_obs:
            o = np.asarray(o)
            pad = np.concatenate([np.full(self.w - 1, M), o])      # M = "no event yet"
            rows.append(np.stack([pad[i:i + len(o)] for i in range(self.w)], axis=1))
        return np.vstack(rows)

    def fit(self, seqs_obs, seqs_states):
        X = self.enc.fit_transform(self._window_matrix(seqs_obs))
        y = np.concatenate(seqs_states)
        self.clf = LogisticRegression(C=self.C, max_iter=2000).fit(X, y)
        return self

    def predict_proba_seqs(self, seqs_obs):
        X = self.enc.transform(self._window_matrix(seqs_obs))
        P = np.zeros((X.shape[0], K))
        P[:, self.clf.classes_] = self.clf.predict_proba(X)
        out, i = [], 0
        for o in seqs_obs:
            out.append(P[i:i + len(o)]); i += len(o)
        return out


def rule_scores(obs):
    """Simple SOC-style rule: fire on credential submission / mailbox rule / bulk forward."""
    return np.isin(np.asarray(obs), RULE_OBS).astype(float)


class HmmlearnReference:
    """Reference implementation (hmmlearn CategoricalHMM) loaded with the SAME parameters."""
    def __init__(self, pi, A, B):
        from hmmlearn.hmm import CategoricalHMM
        m = CategoricalHMM(n_components=K, n_features=M)
        m.startprob_, m.transmat_, m.emissionprob_ = pi, A, B
        self.m = m

    def decode_all(self, seqs_obs):
        X = np.concatenate(seqs_obs).reshape(-1, 1)
        lengths = [len(o) for o in seqs_obs]
        path = self.m.predict(X, lengths)                 # Viterbi
        post = self.m.predict_proba(X, lengths)           # forward-backward posteriors
        out_p, out_q, i = [], [], 0
        for L in lengths:
            out_p.append(path[i:i + L]); out_q.append(post[i:i + L]); i += L
        return out_p, out_q

    def loglik(self, obs):
        return float(self.m.score(np.asarray(obs).reshape(-1, 1)))
