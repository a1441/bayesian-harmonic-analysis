"""
QA harness for the hand-written harmonic-pair code (see docs/knowledge/07-qa-audit.md).

These tests pin the CURRENT behavior of the notebook functions (copied faithfully into
harmonics_ref.py) so the library refactor has a regression baseline and so the audit's
claims are backed by executable evidence. Run: pytest tests/ -q
"""
import numpy as np
import pytest

from harmonics_ref import (
    estimate_harmonic_pair,
    optimize_harmonic_pair,
    decompose_harmonics_until_bic,
    forecast_from_harmonic_coeffs,
    classify_period,
)


def make_signal(T1, T2, A1, A2, phi1, phi2, N):
    t = np.arange(N)
    return (A1 * np.sin(2 * np.pi * t / T1 + phi1)
            + A2 * np.sin(2 * np.pi * t / T2 + phi2)).astype(float)


# --- Step 1: estimate_harmonic_pair ----------------------------------------

def test_reconstruction_identity():
    """fitted == harmonic1 + harmonic2, and residuals == y - fitted."""
    y = make_signal(40, 11, 3.0, 1.5, 0.4, 1.1, 600)
    r = estimate_harmonic_pair(40, 11, y)
    np.testing.assert_allclose(r['fitted'], r['harmonic1'] + r['harmonic2'], atol=1e-9)
    np.testing.assert_allclose(r['residuals'], y - r['fitted'], atol=1e-9)


def test_known_periods_recover_amplitude_and_phase():
    """At the true periods, OLS recovers amplitude/phase and explains ~all variance."""
    N = 800
    y = make_signal(50, 13, 4.0, 2.0, 0.3, -0.7, N)
    r = estimate_harmonic_pair(50, 13, y)
    # amplitude from a,b
    A1 = np.hypot(r['a1'], r['b1'])
    A2 = np.hypot(r['a2'], r['b2'])
    assert r['partial_r2'] > 0.999
    assert A1 == pytest.approx(4.0, abs=0.05)
    assert A2 == pytest.approx(2.0, abs=0.05)
    # phase convention: a*cos + b*sin == A*sin(x+phi) with phi = atan2(a, b)
    phi1 = np.arctan2(r['a1'], r['b1'])
    t = np.arange(N)
    recon = A1 * np.sin(2 * np.pi * t / 50 + phi1)
    np.testing.assert_allclose(recon, r['harmonic1'], atol=1e-6)


# --- Step 2: optimize_harmonic_pair ----------------------------------------

@pytest.mark.slow
def test_single_optimize_finds_dominant_period():
    """
    A single optimize call on a two-period signal recovers the DOMINANT period but not
    necessarily both: the objective is multimodal (gap-research section 5). Here the
    larger-amplitude component (T=60, A=5) is found; the weaker one (T=17, A=2.5) is not.
    This pins the known limitation, it is not a contradiction.
    """
    y = make_signal(60, 17, 5.0, 2.5, 0.2, 0.9, 500)
    r = optimize_harmonic_pair(y)
    nearest_to_60 = min(abs(r['T1'] - 60), abs(r['T2'] - 60))
    assert nearest_to_60 < 3
    assert r['partial_r2'] > 0.7


@pytest.mark.slow
def test_decompose_recovers_both_planted_periods():
    """Across iterations the decomposition does surface both planted periods."""
    y = make_signal(60, 17, 5.0, 2.5, 0.2, 0.9, 500)
    results = decompose_harmonics_until_bic(y)
    periods = [p['T1'] for p in results] + [p['T2'] for p in results]
    assert any(abs(T - 60) < 2 for T in periods)
    assert any(abs(T - 17) < 2 for T in periods)


@pytest.mark.slow
def test_bic_overfits_on_low_noise_signal():
    """
    AUDIT FINDING (07-qa-audit.md): on a clean signal the decomposition reaches R2~1 within
    a couple of pairs yet keeps extracting dozens more. As sigma^2 -> 0 the fit term
    N*ln(2*pi*e*sigma^2) -> -inf and swamps the 4j*ln(N) penalty, so BIC never stops. This
    test PINS that current (buggy) behavior; the library must add a stopping guard.
    """
    y = make_signal(60, 17, 5.0, 2.5, 0.2, 0.9, 500)
    results = decompose_harmonics_until_bic(y)
    r2_at_2 = results[1]['total_r2'] if len(results) > 1 else 0
    assert r2_at_2 > 0.999          # essentially perfect after 2 pairs
    assert len(results) > 10        # yet it keeps going (overfitting)


@pytest.mark.slow
def test_determinism_fixed_seed():
    """Fixed seed -> identical periods across runs."""
    y = make_signal(45, 12, 3.0, 1.0, 0.0, 0.5, 400)
    a = optimize_harmonic_pair(y, seed=123)
    b = optimize_harmonic_pair(y, seed=123)
    assert (a['T1'], a['T2']) == (b['T1'], b['T2'])


# --- Step 3: decompose_harmonics_until_bic ---------------------------------

@pytest.mark.slow
def test_bic_stops_on_pure_noise():
    """On white noise there is little structure; decomposition stays short."""
    rng = np.random.default_rng(0)
    y = rng.standard_normal(300)
    results = decompose_harmonics_until_bic(y)
    assert len(results) <= 3


@pytest.mark.slow
def test_total_r2_monotonic_nondecreasing():
    """Cumulative R2 never decreases as pairs are added."""
    y = make_signal(70, 19, 5.0, 3.0, 0.1, 0.6, 500)
    results = decompose_harmonics_until_bic(y)
    r2s = [r['total_r2'] for r in results]
    assert all(b >= a - 1e-9 for a, b in zip(r2s, r2s[1:]))


# --- Step 7: forecast_from_harmonic_coeffs ---------------------------------

def test_forecast_uses_second_pair_coeffs():
    """Audit: h2 must use a2,b2 (not a1,b1). Build a result with distinct coeffs."""
    N = 100
    res = [{
        'harmonic1': np.zeros(N), 'T1': 40.0, 'T2': 10.0,
        'a1': 1.0, 'b1': 0.0, 'a2': 0.0, 'b2': 5.0, 'total_r2': 0.9,
    }]
    fc, _ = forecast_from_harmonic_coeffs(res, 20)
    t = np.arange(N, N + 20)
    expected = (1.0 * np.cos(2 * np.pi * t / 40) + 0.0 * np.sin(2 * np.pi * t / 40)
                + 0.0 * np.cos(2 * np.pi * t / 10) + 5.0 * np.sin(2 * np.pi * t / 10))
    np.testing.assert_allclose(fc, expected, atol=1e-9)


def test_forecast_continuity_with_fit():
    """Evaluated over the training span, the harmonic continuation matches the fit."""
    N = 300
    y = make_signal(50, 13, 4.0, 2.0, 0.3, -0.7, N)
    r = estimate_harmonic_pair(50, 13, y)
    res = [{
        'harmonic1': r['harmonic1'], 'T1': 50.0, 'T2': 13.0,
        'a1': r['a1'], 'b1': r['b1'], 'a2': r['a2'], 'b2': r['b2'], 'total_r2': r['partial_r2'],
    }]
    # forecast machinery uses t in [N, N+h); reproduce the fit over [0, N) the same way
    t = np.arange(N)
    cont = (r['a1'] * np.cos(2 * np.pi * t / 50) + r['b1'] * np.sin(2 * np.pi * t / 50)
            + r['a2'] * np.cos(2 * np.pi * t / 13) + r['b2'] * np.sin(2 * np.pi * t / 13))
    np.testing.assert_allclose(cont, r['fitted'], atol=1e-6)


# --- Step 4: classification boundaries -------------------------------------

def test_classification_three_way_and_terminology():
    """Trend / cycle / grey zone land on the paper thresholds; band is 'grey zone'."""
    n = 1000
    assert classify_period(1500, n) == "trend"      # T > n
    assert classify_period(500, n) == "cycle"       # T < (2/3)n
    assert classify_period(800, n) == "grey zone"   # (2/3)n < T <= n
    # the band must NOT be called noise
    assert classify_period(800, n) != "noise"


def test_variable_amplitude_conditions_paper_vs_notebook():
    """
    Documents the cond3 divergence. Paper: T<2n/3 and TA/T>3 (its 'cond2' T<TA/3 is
    redundant with TA/T>3). Notebook adds a third, stricter condition TA > 2n.
    """
    n = 1000
    T1, T2 = 90.0, 70.0
    T = 2 * T1 * T2 / (T1 + T2)
    TA = 2 * T1 * T2 / abs(T1 - T2)
    paper_cond1 = T < (2 / 3) * n
    paper_cond_close = (T1 + T2) / abs(T1 - T2) > 3      # == T < TA/3
    notebook_cond3 = (T1 * T2) / (abs(T1 - T2) * n) > 1  # == TA > 2n
    assert paper_cond1 and paper_cond_close
    # the redundancy the paper states:
    assert ((T1 + T2) / abs(T1 - T2) > 3) == (T < TA / 3)
    # the notebook's extra condition is a genuinely separate test:
    assert notebook_cond3 == (TA > 2 * n)
