"""Fit metrics for the harmonic model."""
from __future__ import annotations

import numpy as np


def r2(y, fitted) -> float:
    y = np.asarray(y, float)
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def rmse(y, fitted) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, float) - fitted) ** 2)))


def mae(y, fitted) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - fitted)))


def mape(y, fitted) -> float:
    y = np.asarray(y, float)
    mask = y != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y[mask] - fitted[mask]) / y[mask])) * 100.0)


def bic(n, rss, n_params) -> float:
    """Gaussian BIC with the MLE plug-in variance (sigma^2 = rss/n)."""
    sigma2 = rss / n
    return n_params * np.log(n) + n * np.log(2 * np.pi * np.e * sigma2)


def aic(n, rss, n_params) -> float:
    sigma2 = rss / n
    return 2 * n_params + n * np.log(2 * np.pi * np.e * sigma2)


def summary_metrics(y, fitted, n_pairs, params_per_pair=4, count_noise_var=True) -> dict:
    y = np.asarray(y, float)
    resid = y - fitted
    n = len(y)
    rss = float(np.sum(resid ** 2))
    k = params_per_pair * n_pairs + (1 if count_noise_var else 0)
    return {
        "n_obs": n,
        "n_pairs": n_pairs,
        "n_params": k,
        "r2": r2(y, fitted),
        "rmse": rmse(y, fitted),
        "mae": mae(y, fitted),
        "mape": mape(y, fitted),
        "resid_std": float(resid.std(ddof=0)),
        "bic": bic(n, rss, k),
        "aic": aic(n, rss, k),
    }
