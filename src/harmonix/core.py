"""Core math: harmonic-pair estimation, period search, BIC-guarded decomposition, forecast.

Faithful to the original analysis: these keep the source functions' math, defaults, and
result shapes. Optimizations are opt-in flags only (e.g. ``solver='lstsq'``,
``min_r2_gain>0``); with the defaults the output is unchanged. See
docs/knowledge/01-methodology.md and 07-qa-audit.md.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import differential_evolution


# ---------------------------------------------------------------------------
# 1. estimate
# ---------------------------------------------------------------------------

def estimate_harmonic_pair(t1: float, t2: float, y: np.ndarray, solver: str = "normal") -> dict:
    """Fit two harmonics at fixed periods T1, T2 (your normal-equations solve).

    ``solver='normal'`` reproduces the notebook exactly. ``solver='lstsq'`` is the opt-in
    numerically-stable variant (same answer when well conditioned).
    """
    y = np.asarray(y, dtype=float)
    t = np.arange(len(y))
    cos_T1 = np.cos(2 * np.pi * t / t1); sin_T1 = np.sin(2 * np.pi * t / t1)
    cos_T2 = np.cos(2 * np.pi * t / t2); sin_T2 = np.sin(2 * np.pi * t / t2)

    if solver == "lstsq":
        X = np.column_stack([cos_T1, sin_T1, cos_T2, sin_T2])
        a1, b1, a2, b2 = np.linalg.lstsq(X, y, rcond=None)[0]
    else:
        R1 = np.sum(y * cos_T1); I1 = np.sum(y * sin_T1)
        R2 = np.sum(y * cos_T2); I2 = np.sum(y * sin_T2)
        c1 = np.sum(cos_T1 ** 2); s1 = np.sum(sin_T1 ** 2)
        c2 = np.sum(cos_T2 ** 2); s2 = np.sum(sin_T2 ** 2)
        c12 = np.sum(cos_T1 * cos_T2); s12 = np.sum(sin_T1 * sin_T2)
        m11 = np.sum(cos_T1 * sin_T1); m22 = np.sum(cos_T2 * sin_T2)
        m12 = np.sum(cos_T1 * sin_T2); m21 = np.sum(cos_T2 * sin_T1)
        M = np.array([[c1, m11, c12, m12],
                      [m11, s1, m21, s12],
                      [c12, m21, c2, m22],
                      [m12, s12, m22, s2]])
        a1, b1, a2, b2 = np.linalg.solve(M, np.array([R1, I1, R2, I2]))

    harmonic1 = a1 * cos_T1 + b1 * sin_T1
    harmonic2 = a2 * cos_T2 + b2 * sin_T2
    fitted = harmonic1 + harmonic2
    residuals = y - fitted
    sigma2 = float(np.mean(residuals ** 2))
    total_variance = float(np.var(y, ddof=0))
    return {
        "fitted": fitted, "harmonic1": harmonic1, "harmonic2": harmonic2,
        "residuals": residuals, "SE": float(np.sqrt(sigma2)),
        "total_variance": total_variance, "remaining_variance": sigma2,
        "partial_r2": 1 - sigma2 / total_variance if total_variance > 0 else 0.0,
        "a1": float(a1), "b1": float(b1), "a2": float(a2), "b2": float(b2),
    }


# ---------------------------------------------------------------------------
# 2. optimize  (your differential-evolution settings)
# ---------------------------------------------------------------------------

def optimize_harmonic_pair(y: np.ndarray, *, max_period_factor: float = 10.0,
                           popsize: int = 25, maxiter: int = 500, seed: int = 42069,
                           solver: str = "normal") -> dict:
    """Find optimal (T1, T2) minimizing residual variance. Defaults match the notebook
    (bounds [2, 10N], best2bin, popsize 25, maxiter 500, seed 42069)."""
    N = len(y)
    hi = max(4.0, max_period_factor * N)
    bounds = [(2, hi), (2, hi)]

    def objective(x):
        T1, T2 = x
        if T1 < 2 or T2 <= T1:
            return 1e20
        try:
            return estimate_harmonic_pair(T1, T2, y, solver)["remaining_variance"]
        except np.linalg.LinAlgError:
            return 1e20

    res = differential_evolution(
        objective, bounds, strategy="best2bin", popsize=popsize, maxiter=maxiter,
        tol=1e-7, mutation=(0.7, 1.5), recombination=0.9, polish=True, seed=seed,
    )
    out = estimate_harmonic_pair(res.x[0], res.x[1], y, solver)
    out["T1"], out["T2"] = float(res.x[0]), float(res.x[1])
    return out


# ---------------------------------------------------------------------------
# 3. decompose until BIC  (your stopping rule; optimizations opt-in)
# ---------------------------------------------------------------------------

def decompose_harmonics_until_bic(y: np.ndarray, *, max_pairs: int | None = None,
                                  min_r2_gain: float = 0.0, hard_cap: int = 200,
                                  max_period_factor: float = 10.0, popsize: int = 25,
                                  maxiter: int = 500, seed: int = 42069,
                                  solver: str = "normal", verbose: bool = True) -> list[dict]:
    """Iteratively extract harmonic pairs until BIC increases (your rule).

    Defaults reproduce the notebook: BIC = 4(j+1)*ln(N) + N*ln(2*pi*e*sigma^2), stop when it
    rises. Opt-in optimizations: ``min_r2_gain>0`` also stops when a pair's R2 gain is tiny
    (the audit's anti-overfit guard); ``max_pairs`` caps the count. ``hard_cap`` is a runaway
    backstop only.
    """
    y = np.asarray(y, dtype=float)
    N = len(y)
    residuals = y.copy()
    total_model = np.zeros_like(y)
    BIC_prev = np.inf
    prev_total_r2 = 0.0
    results: list[dict] = []
    cap = min(max_pairs, hard_cap) if max_pairs is not None else hard_cap

    j = 0
    while j < cap:
        pair = optimize_harmonic_pair(residuals, max_period_factor=max_period_factor,
                                      popsize=popsize, maxiter=maxiter, seed=seed, solver=solver)
        cand = total_model + pair["fitted"]
        sigma2_total = float(np.mean((y - cand) ** 2))
        total_r2 = 1 - sigma2_total / np.var(y, ddof=0)
        p = 4 * (j + 1)
        BIC_new = p * np.log(N) + N * np.log(2 * np.pi * np.e * sigma2_total)

        if BIC_new > BIC_prev:
            if verbose:
                print("=== BIC increased -> stopping decomposition ===")
            break
        if min_r2_gain > 0 and j > 0 and (total_r2 - prev_total_r2) < min_r2_gain:
            if verbose:
                print(f"=== R2 gain {total_r2 - prev_total_r2:.2e} < {min_r2_gain:.1e} -> stop ===")
            break

        total_model = cand
        residuals = y - total_model
        results.append({
            "pair_index": j, "T1": pair["T1"], "T2": pair["T2"],
            "a1": pair["a1"], "b1": pair["b1"], "a2": pair["a2"], "b2": pair["b2"],
            "partial_r2": pair["partial_r2"], "total_r2": total_r2, "BIC": BIC_new,
            "fitted": pair["fitted"].copy(),
            "harmonic1": pair["harmonic1"].copy(), "harmonic2": pair["harmonic2"].copy(),
        })
        if verbose:
            print(f" Pair {j+1}: T1={pair['T1']:.1f}, T2={pair['T2']:.1f}  "
                  f"R2={total_r2:.4f}  BIC={BIC_new:.1f}")
        BIC_prev = BIC_new
        prev_total_r2 = total_r2
        j += 1
    return results


# ---------------------------------------------------------------------------
# 4. forecast  (your forward continuation)
# ---------------------------------------------------------------------------

def forecast_from_harmonic_coeffs(results: list[dict], n_ahead: int, target_r2=None):
    """Forecast n_ahead steps from stored harmonic coefficients. Returns (forecast, r2)."""
    if not results:
        raise ValueError("Results list is empty")
    N = len(results[0]["harmonic1"])
    t_future = np.arange(N, N + n_ahead)
    forecast = np.zeros(n_ahead)
    r2_threshold = 0
    for r in results:
        forecast += (r["a1"] * np.cos(2 * np.pi * t_future / r["T1"]) + r["b1"] * np.sin(2 * np.pi * t_future / r["T1"])
                     + r["a2"] * np.cos(2 * np.pi * t_future / r["T2"]) + r["b2"] * np.sin(2 * np.pi * t_future / r["T2"]))
        r2_threshold = r["total_r2"]
        if target_r2 is not None and r["total_r2"] >= target_r2:
            break
    return forecast, r2_threshold


# ---------------------------------------------------------------------------
# 5. analyze  (your full per-pair table)
# ---------------------------------------------------------------------------

def analyze_results(y1: np.ndarray, results1: list[dict]):
    """Per-pair amplitudes, phases, modulation conditions, and trend/cycle flags.

    Faithful to the notebook's analyze_results: the three variable-amplitude conditions and
    the T1/T2 trend & cycle interpretation columns.
    """
    import pandas as pd
    n = len(y1)
    rows = []
    for r in results1:
        T1, T2 = r["T1"], r["T2"]
        A1 = np.hypot(r["a1"], r["b1"]); A2 = np.hypot(r["a2"], r["b2"])
        phi1 = np.arctan2(r["a1"], r["b1"]); phi2 = np.arctan2(r["a2"], r["b2"])
        T = (2 * T1 * T2) / (T1 + T2)
        TA = (2 * T1 * T2) / abs(T1 - T2) if T1 != T2 else np.inf
        T_over_TA = abs(T1 - T2) / (T1 + T2) if T1 != T2 else 0
        cond1 = (T1 * T2) / (n * (T1 + T2)) < 1 / 3
        cond2 = (T1 + T2) / abs(T1 - T2) > 3 if T1 != T2 else False
        cond3 = (T1 * T2) / (abs(T1 - T2) * n) > 1 if T1 != T2 else False
        rows.append({
            "j": r["pair_index"], "T1": T1, "A1": A1, "phi1": phi1,
            "T2": T2, "A2": A2, "phi2": phi2,
            "2T1T2/(T1+T2)/n": T / n,
            "Condition 1": "Yes" if cond1 else "No",
            "(T1+T2)/|T1-T2|": (T1 + T2) / abs(T1 - T2) if T1 != T2 else np.inf,
            "Condition 2": "Yes" if cond2 else "No",
            "T/TA": T_over_TA,
            "Condition 3": "Yes" if cond3 else "No",
            "Variable Amplitude": "Yes" if (cond1 and cond2 and cond3) else "No",
            "Partial R2": r["partial_r2"], "R2": r["total_r2"],
            "T1 is Trend": "Yes" if T1 > n else "No",
            "T2 is Trend": "Yes" if T2 > n else "No",
            "T1 is Cycle": "Yes" if T1 < (2 / 3) * n else "No",
            "T2 is Cycle": "Yes" if T2 < (2 / 3) * n else "No",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. variable-amplitude decomposition  (your final closed form, vectorized)
# ---------------------------------------------------------------------------

def decompose_variable_amplitude_pair(A1_raw, A2_raw, T1_raw, T2_raw,
                                      phi1_raw=0.0, phi2_raw=0.0, return_freq=True, t=None):
    """Split a beat pair into envelope and carrier (your final notebook formula).

    Reverses inputs so A1 >= A2; returns carrier period new_T1=T, envelope period 'fun 1'=TA,
    envelope amplitude 'fun'=2*A2, carrier amplitude new_A2=A1-A2, phases, and (if t given)
    the envelope_component, residual_component, and their sum new_harm.
    """
    if A2_raw > A1_raw:
        A1, A2 = A2_raw, A1_raw
        T1, T2 = T2_raw, T1_raw
        phi1, phi2 = phi2_raw, phi1_raw
        carrier_T2, carrier_phi2 = T2_raw, phi2_raw
    else:
        A1, A2 = A1_raw, A2_raw
        T1, T2 = T1_raw, T2_raw
        phi1, phi2 = phi1_raw, phi2_raw
        carrier_T2, carrier_phi2 = T1_raw, phi1_raw

    T = (2 * T1 * T2) / (T1 + T2)
    TA = (2 * T1 * T2) / abs(T1 - T2)
    envelope_amp = 2 * A2
    carrier_amp = A1 - A2
    phi_env = (phi1 + phi2) / 2
    phi_new = np.mod((-phi2_raw + phi1_raw + np.pi) / 2, 2 * np.pi)

    envelope_component = residual_component = new_harm = None
    if t is not None:
        t = np.asarray(t, dtype=float)
        envelope_component = 2 * A2 * np.sin(2 * np.pi * t / TA + phi_new) * np.sin(2 * np.pi * t / T + phi_env)
        residual_component = (A1 - A2) * np.sin(2 * np.pi * t / T1 + phi1_raw)
        new_harm = envelope_component + residual_component

    out = {
        "new_T1": T, "fun": envelope_amp, "fun 1": TA, "fun 2": phi_new, "new_F1": phi_env,
        "new_T2": carrier_T2, "new_A2": carrier_amp, "phi2": carrier_phi2,
        "new_harm": new_harm, "envelope_component": envelope_component,
        "residual_component": residual_component,
    }
    if return_freq:
        out["freq"] = 1 / T
    return out


# ---------------------------------------------------------------------------
# 7. classify  (your trend/cycle/noise counts; grey-zone variant offered)
# ---------------------------------------------------------------------------

def classify_single(T: float, n: int, grey_zone_label: str = "noise") -> str:
    """T>n trend, T<2n/3 cycle, else the in-between band. The notebook labels it 'noise';
    pass ``grey_zone_label='grey zone'`` for the paper's term (see 07-qa-audit.md)."""
    if T > n:
        return "trend"
    if T < (2 / 3) * n:
        return "cycle"
    return grey_zone_label


def count_trend_cycle_noise(analyze_df) -> dict:
    """Counts from an analyze_results table (your count_trend_cycle_noise)."""
    trend = analyze_df[(analyze_df["T1 is Trend"] == "Yes") | (analyze_df["T2 is Trend"] == "Yes")].shape[0]
    cycle = analyze_df[
        (analyze_df["T1 is Cycle"] == "Yes") & (analyze_df["T1 is Trend"] == "No") &
        (analyze_df["T2 is Cycle"] == "Yes") & (analyze_df["T2 is Trend"] == "No")
    ].shape[0]
    noise = analyze_df.shape[0] - trend - cycle
    return {"trend": trend, "cycle": cycle, "noise": noise}


# ---- helpers --------------------------------------------------------------

def carrier_period(T1: float, T2: float) -> float:
    return 2 * T1 * T2 / (T1 + T2)


def envelope_period(T1: float, T2: float) -> float:
    return 2 * T1 * T2 / abs(T1 - T2) if T1 != T2 else np.inf
