"""Tests for the harmonix library: HarmonicModel + faithful core.

Verifies the sklearn-style surface (fit/predict/summary/analyze/plot) and that the core
math is faithful to the frozen notebook reference in harmonics_ref.py.
"""
import numpy as np
import pandas as pd
import pytest

import harmonix
from harmonix import HarmonicModel, Series
from harmonix import core

import harmonics_ref as ref


def make_signal(T1, T2, A1, A2, p1, p2, N):
    t = np.arange(N)
    return (A1 * np.sin(2 * np.pi * t / T1 + p1) + A2 * np.sin(2 * np.pi * t / T2 + p2)).astype(float)


def make_multi(periods, amps, N):
    """Sum of several sinusoids; needs ceil(len(periods)/2) pairs to fit."""
    t = np.arange(N)
    return sum(A * np.sin(2 * np.pi * t / T) for T, A in zip(periods, amps)).astype(float)


# ---- faithfulness to the notebook reference -------------------------------

def test_estimate_matches_reference_exactly():
    """core.estimate_harmonic_pair (normal solver) == frozen notebook estimate."""
    y = make_signal(50, 13, 4.0, 2.0, 0.3, -0.7, 600)
    a = core.estimate_harmonic_pair(50, 13, y, solver="normal")
    b = ref.estimate_harmonic_pair(50, 13, y)
    for k in ("a1", "b1", "a2", "b2", "partial_r2"):
        assert a[k] == pytest.approx(b[k], rel=1e-9, abs=1e-9)
    np.testing.assert_allclose(a["fitted"], b["fitted"], atol=1e-9)


def test_lstsq_solver_matches_normal_when_well_conditioned():
    y = make_signal(40, 11, 3.0, 1.5, 0.4, 1.1, 500)
    n = core.estimate_harmonic_pair(40, 11, y, solver="normal")
    l = core.estimate_harmonic_pair(40, 11, y, solver="lstsq")
    np.testing.assert_allclose(n["fitted"], l["fitted"], atol=1e-6)


# ---- Series loaders -------------------------------------------------------

def test_series_from_array_and_dataframe():
    y = np.arange(10.0)
    s = Series.from_array(y)
    assert len(s) == 10 and s.ds[0] == 0 and s.ds[-1] == 9
    df = pd.DataFrame({"ds": range(10), "y": y})
    s2 = Series.from_dataframe(df)
    np.testing.assert_array_equal(s.y, s2.y)


def test_series_concat_and_validate():
    a = Series.from_array(np.ones(5))
    b = Series.from_array(np.zeros(3))
    c = a.concat(b)
    assert len(c) == 8
    c.validate()
    with pytest.raises(ValueError):
        Series.from_array(np.array([1.0, np.nan, 3.0])).validate()


# ---- fit / predict / summary ----------------------------------------------

@pytest.fixture(scope="module")
def fitted():
    # a richer signal (6 sinusoids) so more than one pair is warranted
    y = make_multi([200, 150, 90, 60, 40, 25], [6, 5, 4, 3, 2, 1.5], 600)
    return HarmonicModel(max_pairs=3, popsize=12, maxiter=80).fit(y)


def test_fit_attributes(fitted):
    assert 1 <= fitted.n_pairs_ <= 3
    assert fitted.fitted_.shape == fitted.series_.y.shape
    assert 0.0 <= fitted.r2_ <= 1.0
    assert {"r2", "rmse", "mae", "mape", "bic", "aic"} <= set(fitted.metrics_)


def test_predict_shapes_and_intervals(fitted):
    fc = fitted.predict(50, level=[80, 95])
    assert fc["yhat"].shape == (50,)
    assert fc["ds"][0] == len(fitted.series_.y)
    assert "lo_95" in fc and "hi_95" in fc
    assert np.all(fc["hi_95"] >= fc["lo_95"])


def test_analyze_table_columns(fitted):
    df = fitted.analyze()
    assert "Variable Amplitude" in df.columns
    assert {"T1", "T2", "A1", "A2", "phi1", "phi2", "Partial R2", "R2"} <= set(df.columns)
    assert len(df) == fitted.n_pairs_


def test_count_components_keys(fitted):
    c = fitted.count_components()
    assert set(c) == {"trend", "cycle", "noise"}
    c2 = fitted.count_components(grey_zone_label="grey zone")
    assert "grey zone" in c2 and "noise" not in c2


def test_summary_returns_table(fitted):
    table = fitted.summary(returns=True)
    assert isinstance(table, pd.DataFrame) and len(table) == fitted.n_pairs_


def test_not_fitted_raises():
    with pytest.raises(harmonix.NotFittedError):
        HarmonicModel().predict(10)


# ---- pair-count methodology -----------------------------------------------

@pytest.mark.slow
def test_guard_stops_after_clean_fit():
    """With the guard on, a clean 2-sinusoid signal is captured in a few pairs at R2 ~ 1 and
    then stops, instead of overfitting to 20+ pairs like the bare rule."""
    y = make_signal(60, 17, 5.0, 2.5, 0.2, 0.9, 400)
    m = HarmonicModel(popsize=20, maxiter=300, min_r2_gain=1e-3).fit(y)
    assert m.n_pairs_ <= 3
    assert m.r2_ > 0.99


@pytest.mark.slow
def test_bare_bic_overfits_clean_signal():
    """Faithful default (no guard) keeps adding pairs after a perfect fit (audit finding)."""
    y = make_signal(60, 17, 5.0, 2.5, 0.2, 0.9, 400)
    m = HarmonicModel(popsize=20, maxiter=300).fit(y)
    assert m.n_pairs_ > 5          # overfitting, matches the paper's many-pairs regime


def test_max_pairs_caps():
    """With 8 sinusoids the bare rule wants several pairs; the cap binds at 2."""
    y = make_multi([300, 220, 160, 110, 75, 50, 33, 22], [8, 7, 6, 5, 4, 3, 2, 1.5], 700)
    m = HarmonicModel(max_pairs=2, popsize=12, maxiter=80).fit(y)
    assert m.n_pairs_ == 2


@pytest.mark.slow
def test_min_r2_gain_guard_reduces_pairs():
    """The opt-in guard stops earlier than the bare BIC rule on a clean signal."""
    y = make_signal(60, 17, 5.0, 2.5, 0.2, 0.9, 300)
    bare = HarmonicModel(popsize=12, maxiter=80, min_r2_gain=0.0).fit(y)
    guarded = HarmonicModel(popsize=12, maxiter=80, min_r2_gain=1e-3).fit(y)
    assert guarded.n_pairs_ <= bare.n_pairs_


# ---- variable-amplitude decomposition -------------------------------------

def test_plot_methods_return_axes(fitted):
    """Every figure is a callable method and returns a matplotlib Axes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from harmonix import plot_comparison
    axes = [
        fitted.plot_series(window=200),
        fitted.plot_fit(window=200),
        fitted.plot_components(window=200),
        fitted.plot_residuals(window=200),
        fitted.plot_forecast(20, level=[90]),
        fitted.plot_variable_amplitude(window=200),
        fitted.plot_pair_selection(),
        fitted.plot_pair(1, window=200),
        fitted.plot_pair(1, window=200, separate=True),
        fitted.plot_components(window=200, pairs=[1, 2]),
        fitted.plot_evolution(window=200),
        fitted.series_.plot(window=200),
        plot_comparison([fitted, fitted], labels=["a", "b"], window=200),
    ]
    for ax in axes:
        assert ax is not None
    plt.close("all")


def test_pair_signal_and_cumulative(fitted):
    import numpy as np
    assert fitted.pair_signal(1).shape == fitted.series_.y.shape
    # cumulative over all pairs equals the full fit
    np.testing.assert_allclose(fitted.cumulative(fitted.n_pairs_), fitted.fitted_, atol=1e-9)
    # cumulative grows toward the fit
    assert fitted.cumulative(1).shape == fitted.series_.y.shape


def test_variable_amplitude_decomp_shape():
    # a deliberate beat: two close periods
    y = make_signal(95, 70, 3.0, 3.0, 0.0, 0.0, 1000)
    m = HarmonicModel(max_pairs=2, popsize=12, maxiter=80).fit(y)
    vap = m.variable_amplitude_pairs()
    for d in vap:
        assert "envelope_component" in d and len(d["envelope_component"]) == len(y)
        assert d["fun 1"] > d["new_T1"]   # envelope period longer than carrier
