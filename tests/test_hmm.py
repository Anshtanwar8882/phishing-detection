"""Unit tests: our sum-product / Viterbi vs brute-force enumeration and vs hmmlearn."""
import itertools, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.hmm import sum_product, viterbi, estimate_parameters


def random_hmm(K=3, M=4, seed=0):
    r = np.random.default_rng(seed)
    n = lambda *s: (lambda x: x / x.sum(-1, keepdims=True))(r.random(s) + 0.1)
    return n(K), n(K, K), n(K, M)


def joint(pi, A, B, z, x):
    p = pi[z[0]] * B[z[0], x[0]]
    for t in range(1, len(z)):
        p *= A[z[t - 1], z[t]] * B[z[t], x[t]]
    return p


def test_sum_product_matches_bruteforce():
    pi, A, B = random_hmm()
    x = [0, 2, 1, 3, 0]
    alpha, gamma, ll = sum_product(pi, A, B, x)
    paths = list(itertools.product(range(3), repeat=len(x)))
    w = np.array([joint(pi, A, B, z, x) for z in paths])
    marg = np.zeros((len(x), 3))
    for p, z in zip(w, paths):
        for t, s in enumerate(z):
            marg[t, s] += p
    assert np.allclose(gamma, marg / w.sum())
    assert np.isclose(ll, np.log(w.sum()))
    assert np.allclose(alpha[-1], gamma[-1])          # last filtered == last smoothed


def test_viterbi_matches_bruteforce():
    pi, A, B = random_hmm(seed=3)
    x = [1, 1, 3, 0, 2, 2]
    path, lp = viterbi(pi, A, B, x)
    best = max(itertools.product(range(3), repeat=len(x)), key=lambda z: joint(pi, A, B, z, x))
    assert list(path) == list(best)
    assert np.isclose(lp, np.log(joint(pi, A, B, best, x)))


def test_matches_hmmlearn():
    from hmmlearn.hmm import CategoricalHMM
    pi, A, B = random_hmm(K=4, M=6, seed=5)
    x = np.random.default_rng(1).integers(0, 6, 40)
    m = CategoricalHMM(n_components=4, n_features=6)
    m.startprob_, m.transmat_, m.emissionprob_ = pi, A, B
    X = x.reshape(-1, 1)
    assert np.allclose(sum_product(pi, A, B, x)[1], m.predict_proba(X))
    assert list(viterbi(pi, A, B, x)[0]) == list(m.predict(X))


def test_estimation_rows_normalised():
    pi, A, B = estimate_parameters([[0, 1, 1]], [[0, 1, 1]], 2, 2)
    assert np.isclose(pi.sum(), 1) and np.allclose(A.sum(1), 1) and np.allclose(B.sum(1), 1)
