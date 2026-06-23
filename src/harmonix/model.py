"""HarmonicModel: an sklearn-style estimator wrapping the notebook pipeline faithfully.

    model = HarmonicModel().fit(y)     # your decompose_harmonics_until_bic, your defaults
    model.summary()                    # metrics + your analyze_results table + counts
    fc = model.predict(100)            # your forecast_from_harmonic_coeffs, forward
    model.plot()                       # fitted vs actual

Defaults reproduce the notebook. Optimizations are opt-in (``solver='lstsq'``,
``min_r2_gain>0``, ``max_pairs=...``). See src/README.md and docs/knowledge/01-methodology.md.
"""
from __future__ import annotations

import numpy as np

from . import core
from .metrics import summary_metrics
from .series import Series


class NotFittedError(RuntimeError):
    pass


class HarmonicModel:
    """Decompose a series into harmonic pairs, then summarize, forecast, and plot.

    Pair-count methodology: by default the decomposition runs until BIC increases (the
    notebook rule, ~50+ pairs on the example composites). Override with ``max_pairs`` to cap,
    or set ``min_r2_gain>0`` to also stop when a pair's R2 gain is negligible (the audit's
    anti-overfit guard). Period search uses bounds ``[2, max_period_factor * N]`` (factor 10).
    """

    def __init__(self, *, max_pairs: int | None = None, min_r2_gain: float = 0.0,
                 max_period_factor: float = 10.0, popsize: int = 25, maxiter: int = 500,
                 solver: str = "normal", params_per_pair: int = 4, count_noise_var: bool = True,
                 random_state: int = 42069, verbose: bool = False):
        self.max_pairs = max_pairs
        self.min_r2_gain = min_r2_gain
        self.max_period_factor = max_period_factor
        self.popsize = popsize
        self.maxiter = maxiter
        self.solver = solver
        self.params_per_pair = params_per_pair
        self.count_noise_var = count_noise_var
        self.random_state = random_state
        self.verbose = verbose

    # ---- fit --------------------------------------------------------------

    def fit(self, y) -> "HarmonicModel":
        series = y if isinstance(y, Series) else Series.from_array(np.asarray(y, float))
        series.validate()
        self.series_ = series
        yv = series.y
        self.results_ = core.decompose_harmonics_until_bic(
            yv, max_pairs=self.max_pairs, min_r2_gain=self.min_r2_gain,
            max_period_factor=self.max_period_factor, popsize=self.popsize,
            maxiter=self.maxiter, seed=self.random_state, solver=self.solver,
            verbose=self.verbose,
        )
        self.pairs_ = self.results_   # alias
        self.n_pairs_ = len(self.results_)
        self.fitted_ = sum((p["fitted"] for p in self.results_), np.zeros_like(yv))
        self.residuals_ = yv - self.fitted_
        self.metrics_ = summary_metrics(
            yv, self.fitted_, self.n_pairs_, self.params_per_pair, self.count_noise_var)
        self.r2_ = self.metrics_["r2"]
        return self

    def _check_fitted(self):
        if not hasattr(self, "results_"):
            raise NotFittedError("call fit() before this method")

    # ---- predict (forecast forward) ---------------------------------------

    def predict(self, h: int, level=None, target_r2=None):
        """Forecast ``h`` steps forward (your forward continuation).

        Returns ``{ds, yhat}``; with ``level`` (e.g. ``[80, 95]``) adds residual-based
        intervals ``lo_<L>``/``hi_<L>``.
        """
        self._check_fitted()
        yhat, _ = core.forecast_from_harmonic_coeffs(self.results_, h, target_r2=target_r2)
        N = len(self.series_.y)
        out = {"ds": np.arange(N, N + h), "yhat": yhat}
        if level:
            from scipy.stats import norm
            sd = self.residuals_.std(ddof=0)
            for L in (level if isinstance(level, (list, tuple)) else [level]):
                z = norm.ppf(0.5 + L / 200.0)
                out[f"lo_{L}"] = yhat - z * sd
                out[f"hi_{L}"] = yhat + z * sd
        return out

    # ---- analyze (your full per-pair table) -------------------------------

    def analyze(self):
        """Your analyze_results table: amplitudes, phases, conditions, trend/cycle flags."""
        self._check_fitted()
        return core.analyze_results(self.series_.y, self.results_)

    def count_components(self, grey_zone_label: str = "noise") -> dict:
        """Your count_trend_cycle_noise (use grey_zone_label='grey zone' for the paper term)."""
        self._check_fitted()
        counts = core.count_trend_cycle_noise(self.analyze())
        if grey_zone_label != "noise":
            counts[grey_zone_label] = counts.pop("noise")
        return counts

    def pair_signal(self, i: int):
        """The i-th pair's contribution (1-based): harmonic1 + harmonic2."""
        self._check_fitted()
        p = self.pairs_[i - 1]
        return p["harmonic1"] + p["harmonic2"]

    def cumulative(self, k: int):
        """The fitted model using only the first k pairs (the reconstruction history)."""
        self._check_fitted()
        out = np.zeros_like(self.series_.y)
        for p in self.pairs_[:k]:
            out = out + p["fitted"]
        return out

    def variable_amplitude_pairs(self):
        """Decompose every pair flagged Variable Amplitude into envelope + carrier."""
        self._check_fitted()
        df = self.analyze()
        t = np.arange(len(self.series_.y))
        out = []
        for _, row in df[df["Variable Amplitude"] == "Yes"].iterrows():
            d = core.decompose_variable_amplitude_pair(
                row["A1"], row["A2"], row["T1"], row["T2"], row["phi1"], row["phi2"], t=t)
            d["pair"] = int(row["j"]) + 1
            out.append(d)
        return out

    # ---- summary ----------------------------------------------------------

    def summary(self, returns: bool = False, grey_zone_label: str = "noise"):
        """Print a statsmodels-style report (metrics + counts + per-pair table)."""
        self._check_fitted()
        m = self.metrics_
        counts = self.count_components(grey_zone_label)
        table = self.analyze()
        name = self.series_.name or "series"
        gz = grey_zone_label
        lines = [
            "=" * 66,
            f"HarmonicModel summary  -  {name}",
            "=" * 66,
            f"observations   {m['n_obs']:>10}      pairs        {m['n_pairs']:>8}",
            f"parameters     {m['n_params']:>10}      solver       {self.solver:>8}",
            f"R2             {m['r2']:>10.4f}      RMSE      {m['rmse']:>11.4f}",
            f"MAE            {m['mae']:>10.4f}      MAPE      {m['mape']:>10.2f}%",
            f"resid std      {m['resid_std']:>10.4f}",
            f"BIC            {m['bic']:>10.1f}      AIC       {m['aic']:>11.1f}",
            "-" * 66,
            f"components:  trend {counts['trend']}   cycle {counts['cycle']}   "
            f"{gz} {counts[gz]}   "
            f"(var-amp pairs {(table['Variable Amplitude'] == 'Yes').sum()})",
            "=" * 66,
        ]
        print("\n".join(lines))
        with _pd_display():
            print(table.to_string(index=False))
        return table if returns else None

    # ---- plot -------------------------------------------------------------

    def plot(self, kind: str = "fit", *, h: int = 0, level=None, window=None, ax=None):
        """Plot ``kind`` in {'series','fit','components','forecast','residuals',
        'variable_amplitude','pair_selection'}. Needs the [viz] extra."""
        self._check_fitted()
        from . import plotting
        return plotting.plot_model(self, kind=kind, h=h, level=level, window=window, ax=ax)

    # named convenience wrappers (every figure as a one-call method) ---------

    def plot_series(self, window=None, ax=None):
        return self.plot("series", window=window, ax=ax)

    def plot_fit(self, window=None, ax=None):
        return self.plot("fit", window=window, ax=ax)

    def plot_components(self, window=None, pairs=None, ax=None):
        """Plot all harmonic pairs, or a chosen subset via ``pairs=[1, 3, ...]`` (1-based)."""
        self._check_fitted()
        from . import plotting
        return plotting.plot_model(self, kind="components", window=window, ax=ax, indices=pairs)

    def plot_pair(self, i: int, window=None, separate=False, ax=None):
        """Plot any single pair (1-based). ``separate=True`` shows its two harmonics apart."""
        self._check_fitted()
        from . import plotting
        return plotting.plot_model(self, kind="pair", index=i, window=window,
                                   separate=separate, ax=ax)

    def plot_evolution(self, steps=None, window=None, ax=None):
        """Show the fit building up as pairs are added (the reconstruction history)."""
        self._check_fitted()
        from . import plotting
        return plotting.plot_model(self, kind="evolution", steps=steps, window=window, ax=ax)

    def plot_residuals(self, window=None, ax=None):
        return self.plot("residuals", window=window, ax=ax)

    def plot_forecast(self, h, level=None, ax=None):
        return self.plot("forecast", h=h, level=level, ax=ax)

    def plot_variable_amplitude(self, window=None, ax=None):
        return self.plot("variable_amplitude", window=window, ax=ax)

    def plot_pair_selection(self, ax=None):
        return self.plot("pair_selection", ax=ax)

    def compare_to(self, other: "HarmonicModel", labels=None, window=4000, ax=None):
        """Overlay this model's fit against another's (e.g. before vs after)."""
        self._check_fitted()
        from . import plotting
        return plotting.plot_comparison([self, other], labels=labels, window=window, ax=ax)


class _pd_display:
    """Widen pandas output for the summary table, then restore."""
    def __enter__(self):
        import pandas as pd
        self._pd = pd
        self._opts = {k: pd.get_option(k) for k in
                      ("display.max_columns", "display.width", "display.float_format")}
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 220)
        pd.set_option("display.float_format", lambda v: f"{v:.3f}")
        return self

    def __exit__(self, *exc):
        for k, v in self._opts.items():
            self._pd.set_option(k, v)
