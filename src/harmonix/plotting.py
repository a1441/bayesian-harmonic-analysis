"""Optional plotting (needs the [viz] extra: matplotlib). The compute path never imports this."""
from __future__ import annotations

import numpy as np


# show a per-pair legend only up to this many pairs; beyond that it is noise
MAX_LEGEND_PAIRS = 20


def _window(n, window):
    if window is None:
        return slice(0, n)
    if isinstance(window, int):
        return slice(0, min(window, n))
    return slice(*window)


def _legend_outside(ax, fontsize=8, ncol=1):
    """Place the legend just outside the right edge of the axes."""
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=fontsize,
              frameon=False, ncol=ncol, borderaxespad=0.0)


def plot_model(model, kind="fit", *, h=0, level=None, window=None, ax=None,
               index=None, indices=None, separate=False, steps=None):
    import matplotlib.pyplot as plt

    y = model.series_.y
    w = _window(len(y), window)

    if kind == "pair":
        if index is None:
            raise ValueError("pass index=<pair number, 1-based> to plot a single pair")
        p = model.pairs_[index - 1]
        ax = ax or plt.subplots(figsize=(11, 3.4))[1]
        if separate:
            ax.plot(p["harmonic1"][w], lw=1.0, label=f"harmonic 1 (T={p['T1']:.0f})")
            ax.plot(p["harmonic2"][w], lw=1.0, label=f"harmonic 2 (T={p['T2']:.0f})")
        else:
            ax.plot((p["harmonic1"] + p["harmonic2"])[w], color="#1b7837", lw=1.0,
                    label=f"pair {index}")
        ax.set_title(f"pair {index}: T1={p['T1']:.0f}, T2={p['T2']:.0f}  "
                     f"(partial R2={p['partial_r2']:.3f})")
        ax.set_xlabel("ds"); ax.set_ylabel("contribution"); _legend_outside(ax)
        return ax

    if kind == "components" and indices is not None:
        ax = ax or plt.subplots(figsize=(11, 3.6))[1]
        for i in indices:
            p = model.pairs_[i - 1]
            ax.plot((p["harmonic1"] + p["harmonic2"])[w], lw=0.9,
                    label=f"pair {i} (T1={p['T1']:.0f}, T2={p['T2']:.0f})")
        ax.set_title(f"selected harmonic pairs: {list(indices)}")
        ax.set_xlabel("ds"); ax.set_ylabel("contribution"); _legend_outside(ax)
        return ax

    if kind == "evolution":
        ax = ax or plt.subplots(figsize=(11, 3.8))[1]
        N = model.n_pairs_
        if steps is None:
            steps = list(range(1, N + 1)) if N <= 8 else \
                sorted(set(np.linspace(1, N, 8).round().astype(int).tolist()))
        cmap = plt.get_cmap("viridis")
        ax.plot(y[w], color="#bbbbbb", lw=0.6, label="actual", zorder=0)
        for k in steps:
            cum = model.cumulative(k)
            r2 = model.pairs_[k - 1]["total_r2"]
            ax.plot(cum[w], lw=1.0, color=cmap((k - 1) / max(1, N - 1)),
                    label=f"{k} pair{'s' if k > 1 else ''} (R2={r2:.3f})")
        ax.set_title("fit evolution as harmonic pairs are added")
        ax.set_xlabel("ds"); ax.set_ylabel("y"); _legend_outside(ax)
        return ax

    if kind == "series":
        ax = ax or plt.subplots(figsize=(11, 3.4))[1]
        ax.plot(y[w], color="#2c7fb8", lw=0.8, label=model.series_.name or "series")
        ax.set_title("input series"); ax.set_xlabel("ds"); ax.set_ylabel("y")
        _legend_outside(ax)
        return ax

    if kind == "variable_amplitude":
        from scipy.signal import hilbert
        ax = ax or plt.subplots(figsize=(11, 3.4))[1]
        # the most beat-like pair = closest periods = largest (T1+T2)/|T1-T2|
        def beatiness(p):
            d = abs(p["T1"] - p["T2"]) or 1e-9
            return (p["T1"] + p["T2"]) / d
        p = max(model.pairs_, key=beatiness)
        TA_full = 2 * p["T1"] * p["T2"] / (abs(p["T1"] - p["T2"]) or 1e-9)
        if window is None:   # show ~one and a bit envelope cycles so the beat is visible
            w = slice(0, min(len(y), int(1.8 * TA_full)))
        comp = (p["harmonic1"] + p["harmonic2"])[w]
        env = np.abs(hilbert(comp))
        TA = 2 * p["T1"] * p["T2"] / (abs(p["T1"] - p["T2"]) or 1e-9)
        ax.plot(comp, color="#444", lw=0.8, label=f"pair {p['pair_index']+1} (T1={p['T1']:.0f}, T2={p['T2']:.0f})")
        ax.plot(env, color="#cc3311", lw=1.3, label="amplitude envelope")
        ax.plot(-env, color="#cc3311", lw=1.3)
        ax.set_title(f"variable amplitude (beat): envelope period T_A = {TA:.0f}")
        ax.set_xlabel("ds"); ax.set_ylabel("contribution"); _legend_outside(ax)
        return ax

    if kind == "pair_selection":
        ax = ax or plt.subplots(figsize=(11, 3.4))[1]
        r2s = [p["total_r2"] for p in model.pairs_]
        gains = [r2s[0]] + list(np.diff(r2s)) if r2s else []
        elbow = next((i + 1 for i, g in enumerate(gains) if i > 0 and g < 0.02), len(r2s))
        xs = np.arange(1, len(r2s) + 1)
        ax.plot(xs, r2s, "o-", color="#333", lw=1.2)
        if r2s:
            ax.axvline(elbow, color="#cc3311", ls="--", lw=1.0, label=f"elbow ~ pair {elbow}")
            _legend_outside(ax)
        ax.set_title("pair selection: cumulative R2 per pair")
        ax.set_xlabel("pairs"); ax.set_ylabel("cumulative R2")
        return ax

    if kind == "fit":
        ax = ax or plt.subplots(figsize=(11, 3.6))[1]
        ax.plot(y[w], color="#444", lw=0.8, label="actual")
        ax.plot(model.fitted_[w], color="#cc3311", lw=1.0, label="fitted")
        ax.plot(model.residuals_[w], color="#999", lw=0.6, label="residual")
        ax.set_title(f"fit: actual vs fitted vs residual (R2={model.r2_:.3f}, {model.n_pairs_} pairs)")
        ax.set_xlabel("ds"); ax.set_ylabel("y"); _legend_outside(ax)
        return ax

    if kind == "components":
        ax = ax or plt.subplots(figsize=(11, 3.6))[1]
        many = model.n_pairs_ > MAX_LEGEND_PAIRS
        for p in model.pairs_:
            comp = p["harmonic1"] + p["harmonic2"]
            label = None if many else f"pair {p['pair_index']+1} (T1={p['T1']:.0f}, T2={p['T2']:.0f})"
            ax.plot(comp[w], lw=0.8, label=label)
        title = f"harmonic pairs ({model.n_pairs_})"
        if many:
            title += "  -  legend omitted, too many pairs"
        ax.set_title(title); ax.set_xlabel("ds"); ax.set_ylabel("contribution")
        if not many:
            _legend_outside(ax)
        return ax

    if kind == "residuals":
        ax = ax or plt.subplots(figsize=(11, 3.0))[1]
        ax.plot(model.residuals_[w], color="#777", lw=0.7)
        ax.axhline(0, color="#cc3311", lw=0.8)
        ax.set_title("residuals"); ax.set_xlabel("ds"); ax.set_ylabel("residual")
        return ax

    if kind == "forecast":
        if h <= 0:
            raise ValueError("pass h > 0 to plot a forecast")
        fc = model.predict(h, level=level)
        ax = ax or plt.subplots(figsize=(11, 3.6))[1]
        N = len(y)
        ax.plot(np.arange(N), y, color="#444", lw=0.7, label="actual")
        ax.plot(np.arange(N), model.fitted_, color="#cc3311", lw=0.9, alpha=0.7, label="fitted")
        ax.plot(fc["ds"], fc["yhat"], color="#1b7837", lw=1.3, label="forecast")
        if level:
            for L in (level if isinstance(level, (list, tuple)) else [level]):
                ax.fill_between(fc["ds"], fc[f"lo_{L}"], fc[f"hi_{L}"], color="#1b7837",
                                alpha=0.15, label=f"{L}% interval")
        ax.axvline(N, color="#999", ls="--", lw=0.8)
        ax.set_title(f"forecast ({h} steps)"); ax.set_xlabel("ds"); ax.set_ylabel("y")
        _legend_outside(ax)
        return ax

    raise ValueError(f"unknown kind: {kind!r}")


def plot_comparison(models, labels=None, window=4000, ax=None):
    """Overlay the fitted models of several series (e.g. before vs after)."""
    import matplotlib.pyplot as plt
    colors = ["#2c7fb8", "#d95f0e", "#1b7837", "#7b3294", "#999999"]
    ax = ax or plt.subplots(figsize=(11, 3.6))[1]
    for i, m in enumerate(models):
        c = colors[i % len(colors)]
        name = (labels[i] if labels else None) or m.series_.name or f"series {i+1}"
        w = _window(len(m.series_.y), window)
        ax.plot(m.series_.y[w], color=c, lw=0.4, alpha=0.35)
        ax.plot(m.fitted_[w], color=c, lw=1.4, label=f"{name} (fitted)")
    ax.set_title("comparison: fitted models (faint = actual)")
    ax.set_xlabel("ds"); ax.set_ylabel("y"); _legend_outside(ax)
    return ax
