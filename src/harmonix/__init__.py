"""harmonix - Bayesian harmonic-pair decomposition for time series.

    from harmonix import HarmonicModel, Series
    model = HarmonicModel().fit(Series.from_json("data/before_1.json"))
    model.summary()
    model.predict(100)
    model.plot()

See docs/knowledge/ for the methodology and src/README.md for the design.
"""
from .series import Series
from .model import HarmonicModel, NotFittedError
from . import core, metrics


def plot_comparison(models, labels=None, window=4000, ax=None):
    """Overlay the fitted models of several series (e.g. before vs after)."""
    from . import plotting
    return plotting.plot_comparison(models, labels=labels, window=window, ax=ax)


__all__ = ["Series", "HarmonicModel", "NotFittedError", "plot_comparison", "core", "metrics"]
__version__ = "0.1.0"
