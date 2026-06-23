"""
Frozen reference copy of the original harmonic-pair functions, kept as a regression
fixture so the library's results can be checked against the source behavior.

This module exists ONLY for the tests. The library implementation lives in src/harmonix;
audited divergences are documented in docs/knowledge/07-qa-audit.md. Do not build on this.
"""
import numpy as np
from scipy.optimize import differential_evolution


def estimate_harmonic_pair(t1, t2, y):
    """Fit two harmonics at fixed periods t1, t2 by ordinary least squares."""
    t = np.arange(len(y))
    cos_T1 = np.cos(2 * np.pi * t / t1)
    sin_T1 = np.sin(2 * np.pi * t / t1)
    cos_T2 = np.cos(2 * np.pi * t / t2)
    sin_T2 = np.sin(2 * np.pi * t / t2)

    R1 = np.sum(y * cos_T1); I1 = np.sum(y * sin_T1)
    R2 = np.sum(y * cos_T2); I2 = np.sum(y * sin_T2)

    c1 = np.sum(cos_T1 ** 2); s1 = np.sum(sin_T1 ** 2)
    c2 = np.sum(cos_T2 ** 2); s2 = np.sum(sin_T2 ** 2)
    c12 = np.sum(cos_T1 * cos_T2); s12 = np.sum(sin_T1 * sin_T2)
    m11 = np.sum(cos_T1 * sin_T1); m22 = np.sum(cos_T2 * sin_T2)
    m12 = np.sum(cos_T1 * sin_T2); m21 = np.sum(cos_T2 * sin_T1)

    M = np.array([
        [c1,  m11, c12, m12],
        [m11, s1,  m21, s12],
        [c12, m21, c2,  m22],
        [m12, s12, m22, s2],
    ])
    Y_vec = np.array([R1, I1, R2, I2])
    a1, b1, a2, b2 = np.linalg.solve(M, Y_vec)

    harmonic1 = a1 * cos_T1 + b1 * sin_T1
    harmonic2 = a2 * cos_T2 + b2 * sin_T2
    fitted = harmonic1 + harmonic2
    residuals = y - fitted
    sigma2 = np.mean(residuals ** 2)
    total_variance = np.var(y, ddof=0)
    return {
        'fitted': fitted, 'harmonic1': harmonic1, 'harmonic2': harmonic2,
        'residuals': residuals, 'SE': np.sqrt(sigma2),
        'total_variance': total_variance, 'remaining_variance': sigma2,
        'partial_r2': 1 - sigma2 / total_variance,
        'a1': a1, 'b1': b1, 'a2': a2, 'b2': b2,
    }


def optimize_harmonic_pair(y, seed=42069):
    """Find (T1, T2) minimizing residual variance via differential evolution."""
    N = len(y)
    bounds = [(2, N * 10), (2, N * 10)]

    def objective(x):
        T1, T2 = x
        if T1 < 2 or T2 <= T1:
            return 1e20
        try:
            return estimate_harmonic_pair(T1, T2, y)['remaining_variance']
        except np.linalg.LinAlgError:
            return 1e20

    res = differential_evolution(
        objective, bounds, strategy='best2bin', popsize=25, maxiter=500,
        tol=1e-7, mutation=(0.7, 1.5), recombination=0.9, polish=True, seed=seed,
    )
    T1, T2 = res.x
    out = estimate_harmonic_pair(T1, T2, y)
    out['T1'] = T1; out['T2'] = T2
    return out


def decompose_harmonics_until_bic(y, verbose=False):
    """Peel optimized pairs from the residual until BIC stops decreasing."""
    N = len(y)
    residuals = y.copy()
    total_model = np.zeros_like(y, dtype=float)
    BIC_prev = np.inf
    results = []
    j = 0
    while True:
        pair = optimize_harmonic_pair(residuals)
        total_model = total_model + pair['fitted']
        residuals = y - total_model
        sigma2_total = np.mean((y - total_model) ** 2)
        total_r2 = 1 - sigma2_total / np.var(y, ddof=0)
        p = 4 * (j + 1)
        BIC_new = p * np.log(N) + N * np.log(2 * np.pi * np.e * sigma2_total)
        if BIC_new > BIC_prev:
            break
        results.append({
            'pair_index': j, 'T1': pair['T1'], 'T2': pair['T2'],
            'a1': pair['a1'], 'b1': pair['b1'], 'a2': pair['a2'], 'b2': pair['b2'],
            'partial_r2': pair['partial_r2'], 'total_r2': total_r2, 'BIC': BIC_new,
            'fitted': pair['fitted'].copy(),
            'harmonic1': pair['harmonic1'].copy(), 'harmonic2': pair['harmonic2'].copy(),
        })
        BIC_prev = BIC_new
        j += 1
    return results


def forecast_from_harmonic_coeffs(results, n_ahead, target_r2=None):
    """Continue every stored harmonic past the training length."""
    if not results:
        raise ValueError("Results list is empty")
    N = len(results[0]['harmonic1'])
    t_future = np.arange(N, N + n_ahead)
    forecast = np.zeros(n_ahead)
    r2_threshold = 0
    for r in results:
        T1, T2 = r['T1'], r['T2']
        h1 = r['a1'] * np.cos(2 * np.pi * t_future / T1) + r['b1'] * np.sin(2 * np.pi * t_future / T1)
        h2 = r['a2'] * np.cos(2 * np.pi * t_future / T2) + r['b2'] * np.sin(2 * np.pi * t_future / T2)
        forecast += h1 + h2
        r2_threshold = r['total_r2']
        if target_r2 is not None and r['total_r2'] >= target_r2:
            break
    return forecast, r2_threshold


def classify_period(T, n):
    """Paper's three-way classification (uses 'grey zone', not 'noise')."""
    if T > n:
        return "trend"
    elif T < (2 / 3) * n:
        return "cycle"
    else:
        return "grey zone"
