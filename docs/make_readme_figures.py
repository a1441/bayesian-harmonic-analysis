"""Generate the README figures from the harmonix library on the example data.

Every figure is a one-call library method (model.plot_*, Series.plot, plot_comparison).
Reproducible: `python docs/make_readme_figures.py` writes PNGs to docs/images/. Small pair
caps / DE budgets keep it to a couple of minutes; the figures are illustrative.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
from harmonix import HarmonicModel, Series, plot_comparison  # noqa: E402

IMG = os.path.join(HERE, "images")
os.makedirs(IMG, exist_ok=True)
DATA = os.path.join(ROOT, "data")
# stop on R2 gain (not a hardcoded pair count); 0.02 gives ~7 pairs on this data
FIT_KW = dict(min_r2_gain=0.02, solver="lstsq", maxiter=120, popsize=15)


def composite(a, b, name):
    return Series.from_json(os.path.join(DATA, a)).concat(
        Series.from_json(os.path.join(DATA, b)), name=name)


def save(ax, fname):
    fig = ax.figure
    fig.savefig(os.path.join(IMG, fname), dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("wrote", fname)


def main():
    # primary use is a single time series
    series = composite("before_1.json", "before_2.json", "series")

    print("fitting ...")
    model = HarmonicModel(**FIT_KW).fit(series)

    save(series.plot(window=2000), "01_input_series.png")          # Series.plot()
    save(model.plot_fit(window=2000), "02_fit.png")                # model.plot_fit()
    save(model.plot_components(window=2000), "03_components.png")   # model.plot_components()
    save(model.plot_evolution(window=2000), "08_evolution.png")     # model.plot_evolution()
    save(model.plot_forecast(600, level=[80, 95]), "04_forecast.png")
    save(model.plot_variable_amplitude(), "05_variable_amplitude.png")

    # pair-selection curve: fit further (no guard) so the diminishing returns are visible
    mc = HarmonicModel(max_pairs=12, solver="lstsq", maxiter=120, popsize=15).fit(series)
    save(mc.plot_pair_selection(), "07_pair_selection.png")

    # before/after as one application of the same library (not the only use)
    print("fitting a second series for comparison ...")
    after = composite("after_1.json", "after_2.json", "after")
    model2 = HarmonicModel(**FIT_KW).fit(after)
    save(plot_comparison([model, model2], labels=["before", "after"]), "06_before_after.png")

    print("done ->", IMG)


if __name__ == "__main__":
    main()
