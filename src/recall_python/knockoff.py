from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from .distributions import estimate_negative_binomial, estimate_zi_poisson, rzipoisson


@dataclass(frozen=True)
class KnockoffResult:
    selected_features: pd.DataFrame
    threshold: float


def augment_with_artificial_variables(
    data: pd.DataFrame,
    null_method: str = "ZIP",
    random_state: int | None = None,
) -> pd.DataFrame:
    """Create one knockoff feature for each original feature.

    Parameters
    ----------
    data:
        Cell-by-gene count matrix.
    null_method:
        ZIP or NB.
    """
    rng = np.random.default_rng(random_state)
    ko = {}
    for col in data.columns:
        x = data[col].to_numpy()
        if null_method == "ZIP":
            est = estimate_zi_poisson(x)
            vals = rzipoisson(len(x), est.lambda_hat, est.pi_hat, random_state=int(rng.integers(1e9)))
        elif null_method == "NB":
            est = estimate_negative_binomial(x)
            p = est.size / (est.size + est.mu) if est.mu > 0 else 1.0
            vals = rng.negative_binomial(est.size, p, size=len(x))
        else:
            raise ValueError("null_method must be one of {'ZIP','NB'}")
        ko[f"knockoff_{col}"] = vals

    return pd.concat([data.copy(), pd.DataFrame(ko, index=data.index)], axis=1)


def knockoff_threshold(W: np.ndarray, fdr: float, offset: int = 1) -> float:
    """Equivalent to R knockoff::knockoff.threshold for signed statistics."""
    if not 0 < fdr < 1:
        raise ValueError("fdr must be in (0, 1)")
    W = np.asarray(W, dtype=float)
    t_candidates = np.sort(np.unique(np.abs(W[W != 0])))
    if t_candidates.size == 0:
        return np.inf

    for t in t_candidates:
        num = offset + np.sum(W <= -t)
        den = max(1, np.sum(W >= t))
        if num / den <= fdr:
            return float(t)
    return np.inf


def compute_knockoff_filter(
    data_with_knockoffs: pd.DataFrame,
    labels: np.ndarray,
    cluster1,
    cluster2,
    q: float = 0.05,
    return_all: bool = False,
) -> KnockoffResult | tuple[pd.DataFrame, float]:
    """Compute knockoff-selected genes using two-group marker p-values.

    This mirrors recall's `compute_knockoff_filter`: compute per-feature p-values
    for originals and knockoffs, then W = -log10(p_orig) - -log10(p_knockoff).
    """
    labels = np.asarray(labels)
    if len(labels) != len(data_with_knockoffs):
        raise ValueError("labels length must match number of cells")

    mask = np.isin(labels, [cluster1, cluster2])
    X = data_with_knockoffs.loc[mask]
    y = labels[mask]
    g1 = y == cluster1
    g2 = y == cluster2

    original_cols = [c for c in X.columns if not c.startswith("knockoff_")]
    knockoff_cols = [f"knockoff_{c}" for c in original_cols]
    if not set(knockoff_cols).issubset(X.columns):
        raise ValueError("Each original feature must have a matching knockoff_ feature")

    p_orig = []
    p_knock = []
    for oc, kc in zip(original_cols, knockoff_cols):
        p1 = mannwhitneyu(X.loc[g1, oc], X.loc[g2, oc], alternative="two-sided").pvalue
        p2 = mannwhitneyu(X.loc[g1, kc], X.loc[g2, kc], alternative="two-sided").pvalue
        p_orig.append(max(p1, np.finfo(float).tiny))
        p_knock.append(max(p2, np.finfo(float).tiny))

    W = -np.log10(np.array(p_orig)) - (-np.log10(np.array(p_knock)))
    thres = knockoff_threshold(W, fdr=q, offset=1)

    all_df = pd.DataFrame({"gene": original_cols, "W": W}).sort_values("W", ascending=False)
    if return_all:
        return all_df, thres

    selected = all_df.loc[all_df["W"] >= thres].reset_index(drop=True)
    return KnockoffResult(selected_features=selected, threshold=thres)
