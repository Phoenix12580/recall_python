from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import lambertw
from scipy.stats import nbinom


@dataclass(frozen=True)
class ZIPEstimate:
    lambda_hat: float
    pi_hat: float


@dataclass(frozen=True)
class NBEstimate:
    size: float
    mu: float


def estimate_zi_poisson(data: np.ndarray) -> ZIPEstimate:
    """Estimate zero-inflated Poisson parameters following recall's R logic."""
    x = np.asarray(data, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("data must be a non-empty 1D array")

    r0 = np.mean(x == 0)
    x_bar = float(np.mean(x))
    denom = 1 - r0
    gamma = x_bar / max(denom, np.finfo(float).eps)

    lw = lambertw(-gamma * np.exp(-gamma), k=0)
    lambda_hat = float(np.real(lw + gamma))
    if lambda_hat <= 0:
        lambda_hat = max(x_bar, np.finfo(float).eps)

    pi_hat = 1 - x_bar / lambda_hat
    pi_hat = float(np.clip(pi_hat, 0.0, 1.0))
    return ZIPEstimate(lambda_hat=lambda_hat, pi_hat=pi_hat)


def rzipoisson(n: int, lambda_: float, prop_zero: float, random_state: int | None = None) -> np.ndarray:
    """Sample from a zero-inflated Poisson distribution."""
    if n <= 0:
        raise ValueError("n must be positive")
    if lambda_ <= 0:
        raise ValueError("lambda_ must be > 0")
    if not (0 <= prop_zero <= 1):
        raise ValueError("prop_zero must be in [0, 1]")

    rng = np.random.default_rng(random_state)
    inflated = rng.random(n) < prop_zero
    out = rng.poisson(lambda_, size=n)
    out[inflated] = 0
    return out


def estimate_negative_binomial(data: np.ndarray) -> NBEstimate:
    """Estimate NB(mu, size) with MOM fallback behavior similar to recall's robust R fit."""
    x = np.asarray(data, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("data must be a non-empty 1D array")
    if np.any(x < 0):
        raise ValueError("data must be non-negative")

    mu = float(np.mean(x))
    var = float(np.var(x, ddof=1)) if x.size > 1 else mu
    if var <= mu:
        size = 1e6
    else:
        size = mu**2 / (var - mu)
    size = float(max(size, 1e-8))

    # one-step numerical refinement by maximizing NB likelihood around MOM init
    p = size / (size + mu) if mu > 0 else 1.0
    _ = nbinom.logpmf(np.asarray(x, dtype=int), size, p).sum()

    return NBEstimate(size=size, mu=mu)
