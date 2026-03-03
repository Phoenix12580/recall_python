import numpy as np
import pandas as pd

from recall_python import (
    augment_with_artificial_variables,
    compute_knockoff_filter,
    estimate_zi_poisson,
    knockoff_threshold,
    rzipoisson,
)


def test_zip_estimation_basic():
    x = np.array([0, 0, 1, 2, 0, 3, 0, 1, 0, 2])
    est = estimate_zi_poisson(x)
    assert est.lambda_hat > 0
    assert 0 <= est.pi_hat <= 1


def test_rzipoisson_shape():
    s = rzipoisson(20, lambda_=1.5, prop_zero=0.3, random_state=1)
    assert len(s) == 20
    assert (s >= 0).all()


def test_knockoff_threshold_runs():
    W = np.array([3.2, 1.1, -0.2, -2.0, 0.5])
    t = knockoff_threshold(W, fdr=0.2)
    assert np.isfinite(t) or np.isinf(t)


def test_end_to_end_compute_filter():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "g1": rng.poisson(2.0, 80),
            "g2": rng.poisson(1.0, 80),
            "g3": rng.poisson(0.8, 80),
        }
    )
    labels = np.array([0] * 40 + [1] * 40)
    X_aug = augment_with_artificial_variables(X, null_method="ZIP", random_state=0)
    res = compute_knockoff_filter(X_aug, labels, 0, 1, q=0.5)
    assert hasattr(res, "selected_features")
    assert "W" in res.selected_features.columns or res.selected_features.empty
