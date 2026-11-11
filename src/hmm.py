"""Core Unit-4 algorithms, implemented from scratch with NumPy.

Model = chain factor graph over hidden stages Z_1..Z_T and observed events X_1..X_T
    p(z, x) = phi_0(z_1) * prod_t phi_e(z_t, x_t) * prod_{t>1} psi(z_{t-1}, z_t)

  * phi_0  : prior factor            (vector pi, size K)
  * psi    : pairwise transition     (matrix A, K x K)
  * phi_e  : emission (unary) factor (matrix B, K x M), evaluated at the observed x_t

sum_product : forward/backward message passing  -> marginals P(Z_t | X)   (scaled, stable)
viterbi     : max-product message passing (log)  -> single most likely path
"""
from dataclasses import dataclass
import numpy as np

TINY = 1e-300


def unary_factors(B, obs):
    """Local factor for every time step: phi[t, k] = B[k, x_t]  (shape T x K)."""
    return B[:, np.asarray(obs, dtype=int)].T


def sum_product(pi, A, B, obs):
    """Sum-product on the chain.

    Returns
    -------
    filtered : (T,K)  P(Z_t | X_1..X_t)   forward messages normalised  (online / alerting)
    smoothed : (T,K)  P(Z_t | X_1..X_T)   forward x backward           (offline / investigation)
    loglik   : float  log P(X_1..X_T)
    """
    phi = unary_factors(B, obs)
    T, K = phi.shape
    alpha = np.zeros((T, K))
    c = np.zeros(T)                      # scaling constants (normalisers of forward messages)
    a = pi * phi[0]
    c[0] = max(a.sum(), TINY)
    alpha[0] = a / c[0]
    for t in range(1, T):                # variable -> factor -> variable message
        a = (alpha[t - 1] @ A) * phi[t]
        c[t] = max(a.sum(), TINY)
        alpha[t] = a / c[t]
    beta = np.ones((T, K))               # backward messages
    for t in range(T - 2, -1, -1):
        beta[t] = (A @ (phi[t + 1] * beta[t + 1])) / c[t + 1]
    smoothed = alpha * beta
    smoothed /= smoothed.sum(axis=1, keepdims=True)
    return alpha, smoothed, float(np.log(c).sum())


def viterbi(pi, A, B, obs):
    """Max-product (Viterbi) in log space. Returns (path, log-prob of path)."""
    obs = np.asarray(obs, dtype=int)
    T, K = len(obs), len(pi)
    with np.errstate(divide="ignore"):
        lpi, lA, lB = np.log(pi), np.log(A), np.log(B)
    delta = np.full((T, K), -np.inf)
    back = np.zeros((T, K), dtype=int)
    delta[0] = lpi + lB[:, obs[0]]
    for t in range(1, T):
        s = delta[t - 1][:, None] + lA   # s[i, j] = best score ending in i, then i -> j
        back[t] = s.argmax(axis=0)
        delta[t] = s.max(axis=0) + lB[:, obs[t]]
    path = np.zeros(T, dtype=int)
    path[-1] = delta[-1].argmax()
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path, float(delta[-1].max())


def estimate_parameters(seqs_obs, seqs_states, K, M, alpha=1.0):
    """Supervised maximum-likelihood by counting, with Laplace (add-alpha) smoothing."""
    pi_c = np.full(K, alpha)
    A_c = np.full((K, K), alpha)
    B_c = np.full((K, M), alpha)
    for o, z in zip(seqs_obs, seqs_states):
        o, z = np.asarray(o), np.asarray(z)
        pi_c[z[0]] += 1
        if len(z) > 1:
            np.add.at(A_c, (z[:-1], z[1:]), 1)
        np.add.at(B_c, (z, o), 1)
    return (pi_c / pi_c.sum(),
            A_c / A_c.sum(axis=1, keepdims=True),
            B_c / B_c.sum(axis=1, keepdims=True))


@dataclass
class ChainFactorGraph:
    pi: np.ndarray
    A: np.ndarray
    B: np.ndarray

    def infer(self, obs):
        return sum_product(self.pi, self.A, self.B, obs)

    def decode(self, obs):
        return viterbi(self.pi, self.A, self.B, obs)

    def save(self, path):
        np.savez(path, pi=self.pi, A=self.A, B=self.B)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(d["pi"], d["A"], d["B"])
