import numpy as np
import os

import torch
from scipy.special import roots_hermitenorm

# The private stream draws the secret index, the release noise and the channel coins.
# np.random is the public stream, seeded by `seed`.
_PRIVATE = np.random.RandomState()


def init_private_rng(seed=None):
    global _PRIVATE
    if seed is None or str(seed) == "":
        seed = int.from_bytes(os.urandom(4), "little")
    seed = int(seed) % (2 ** 32)
    _PRIVATE = np.random.RandomState(seed)
    return seed


def private_random():
    return float(_PRIVATE.random_sample())


def private_randn():
    return float(_PRIVATE.randn())


def private_randint(n):
    return int(_PRIVATE.randint(n))


# Probabilist's Hermite nodes, with weights normalised so that sum_i w_i f(x_i) ~= E[f(N(0,1))].
_GH_NODES, _GH_WEIGHTS = roots_hermitenorm(60)
_GH_WEIGHTS = _GH_WEIGHTS / np.sqrt(2 * np.pi)


def binary_channel_mi(q_plus, sigma):
    """I(S; S + N(0, sigma^2)) in nats for S in {-1, +1}, P(S=+1) = q_plus, by 60-node
    Gauss-Hermite quadrature. Not an upper bound; sigma_for_binary_mi adds the margin."""
    q_minus = 1.0 - q_plus
    y_plus = sigma * _GH_NODES + 1.0
    log_f_plus_at_yplus = -np.log(sigma) - 0.5 * _GH_NODES**2
    log_f_minus_at_yplus = -np.log(sigma) - 0.5 * ((y_plus + 1.0) / sigma) ** 2
    a = np.log(q_plus + 1e-300) + log_f_plus_at_yplus
    b = np.log(q_minus + 1e-300) + log_f_minus_at_yplus
    m = np.maximum(a, b)
    log_fY_at_yplus = m + np.log(np.exp(a - m) + np.exp(b - m))
    # Under x ~ N(0,1), f_{Y|+1}(y) dy = phi(x) dx, so the GH weights integrate the KL directly.
    kl_plus = np.sum(_GH_WEIGHTS * (log_f_plus_at_yplus - log_fY_at_yplus))

    y_minus = sigma * _GH_NODES - 1.0
    log_f_minus_at_yminus = -np.log(sigma) - 0.5 * _GH_NODES**2
    log_f_plus_at_yminus = -np.log(sigma) - 0.5 * ((y_minus - 1.0) / sigma) ** 2
    a = np.log(q_plus + 1e-300) + log_f_plus_at_yminus
    b = np.log(q_minus + 1e-300) + log_f_minus_at_yminus
    m = np.maximum(a, b)
    log_fY_at_yminus = m + np.log(np.exp(a - m) + np.exp(b - m))
    kl_minus = np.sum(_GH_WEIGHTS * (log_f_minus_at_yminus - log_fY_at_yminus))

    mi = q_plus * kl_plus + q_minus * kl_minus
    return float(max(mi, 0.0))


def sigma_for_binary_mi(q_plus, beta):
    """Smallest sigma found by bisection with I(S; S + N(0, sigma^2)) <= beta, conservatively: bisects log sigma
    against 0.999 * beta and returns the feasible endpoint, and raises when no finite sigma meets the budget."""
    # A 1e-3 relative margin covers the measured quadrature error; below beta=1e-10 the analytic bound is used instead.
    target = beta * (1.0 - 1e-3)
    tol = min(1e-6, 1e-3 * target)

    def mi_for_solve(sigma):
        if beta <= 1e-10:
            # I(S;Y) <= Var(S)/(2 sigma^2) = 2q(1-q)/sigma^2, an analytic upper bound.
            return (2 * q_plus * (1 - q_plus) / sigma) / sigma
        mi = binary_channel_mi(q_plus, sigma)
        if not np.isfinite(mi):
            raise RuntimeError("Non-finite binary-channel MI; abort the release")
        return mi

    log_sigma_lo = np.log(1e-4)
    log_sigma_hi = np.log(1e6)
    log_sigma_max = np.log(1e12)
    mi_hi = mi_for_solve(np.exp(log_sigma_hi))
    while mi_hi >= target:
        if log_sigma_hi >= log_sigma_max:
            raise RuntimeError("No feasible sigma <= 1e12; abort the release")
        log_sigma_hi = min(log_sigma_hi + np.log(10.0), log_sigma_max)
        mi_hi = mi_for_solve(np.exp(log_sigma_hi))

    # MI decreases in sigma, so returning the hi endpoint keeps mi(hi) <= target even when the iterations run out.
    for _ in range(64):
        log_mid = 0.5 * (log_sigma_lo + log_sigma_hi)
        mi_mid = mi_for_solve(np.exp(log_mid))
        if mi_mid > target:
            log_sigma_lo = log_mid
        else:
            log_sigma_hi = log_mid
            if target - mi_mid < tol:
                break
    return float(np.exp(log_sigma_hi))


def update_p_binary(p, signs, noisy_real_release, sigma):
    diff = signs - noisy_real_release
    log_lik = -0.5 * (diff**2) / (sigma**2)
    log_p = np.log(p + 1e-300) + log_lik
    c = log_p.max()
    return np.exp(log_p - c - np.log(np.sum(np.exp(log_p - c))))


_LN2 = float(np.log(2.0))


def binary_entropy_nats(x):
    x = np.clip(x, 1e-15, 1.0 - 1e-15)
    return -x * np.log(x) - (1.0 - x) * np.log(1.0 - x)


def binary_entropy_inv_lower(y, n_iter=100):
    y = float(np.clip(y, 0.0, _LN2))
    lo, hi = 0.0, 0.5
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        if float(binary_entropy_nats(mid)) < y:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def channel_for_binary_mi(p, beta_t):
    """Distortion-optimal asymmetric channel (alpha, beta, D) with I(X;Y) <= beta_t.

    beta_t >= h(p) gives the identity channel (0, 0, 0); beta_t <= 0 gives the constant
    majority channel, (1, 0) or (0, 1), or (0.5, 0.5) at p = 1/2; otherwise D = h^-1(h(p) - beta_t).
    alpha = Pr[Y=1 | X=0], beta = Pr[Y=0 | X=1].
    """
    Hp = binary_entropy_nats(p)
    if beta_t >= Hp:
        return 0.0, 0.0, 0.0
    if beta_t <= 0.0:
        D = min(p, 1.0 - p)
    else:
        D = binary_entropy_inv_lower(Hp - beta_t)
    D = float(np.clip(D, 0.0, min(p, 1.0 - p)))
    denom = 1.0 - 2.0 * D
    if denom <= 1e-12:
        return 0.5, 0.5, 0.5
    # Through Pr[Y=1] of the backward channel, because the direct closed form cancels near p = 1/2.
    q = float(np.clip((p - D) / denom, 0.0, 1.0))
    alpha = float(np.clip(q * D / (1.0 - p), 0.0, 1.0))
    beta = float(np.clip((1.0 - q) * D / p, 0.0, 1.0))
    return alpha, beta, D


def sample_binary_channel(xi, alpha, beta):
    if xi > 0:
        return -1.0 if private_random() < beta else 1.0
    else:
        return 1.0 if private_random() < alpha else -1.0


def update_p_channel(p_vec, signs, Y, alpha, beta):
    if Y > 0:
        lik = np.where(signs > 0, 1.0 - beta, alpha)
    else:
        lik = np.where(signs > 0, beta, 1.0 - alpha)
    post = p_vec * np.maximum(lik, 1e-300)
    return post / post.sum()


class SubsetAwareDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, m=128):
        self.dataset = dataset
        self.membership_matrix = torch.zeros((len(dataset), m), dtype=torch.bool)
        for element in range(len(dataset)):
            sampled = np.random.choice(m, m // 2, replace=False)
            self.membership_matrix[element, sampled] = True

    def __getitem__(self, index):
        item = self.dataset[index]
        vector = self.membership_matrix[index]
        if isinstance(item, list):
            return [dict(d, membership_vector=vector) for d in item]
        return dict(item, membership_vector=vector)

    def __len__(self):
        return len(self.dataset)


def compute_subset_ghats(ghat_per_sample, membership_vectors):
    membership_float = membership_vectors.astype(np.float64)
    counts = np.maximum(membership_float.sum(axis=0), 1.0)
    weighted = membership_float.T @ ghat_per_sample
    return weighted / counts
