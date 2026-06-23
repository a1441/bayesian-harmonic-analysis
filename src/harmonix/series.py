"""The input contract: a single time series of (ds, y).

One series at a time, two columns only. Load from JSON, CSV, Excel, a DataFrame, or a
bare array. See docs/knowledge/08-input-schema.md.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Series:
    """An immutable (ds, y) series. The math only ever sees this object."""

    ds: np.ndarray
    y: np.ndarray
    name: str | None = None

    # ---- construction -----------------------------------------------------

    def __post_init__(self):
        object.__setattr__(self, "ds", np.asarray(self.ds))
        object.__setattr__(self, "y", np.asarray(self.y, dtype=float))

    @classmethod
    def from_array(cls, y, ds=None, name=None) -> "Series":
        y = np.asarray(y, dtype=float)
        if ds is None:
            ds = np.arange(len(y))
        return cls(np.asarray(ds), y, name)

    @classmethod
    def from_dataframe(cls, df, ds_col="ds", y_col="y", name=None) -> "Series":
        return cls(df[ds_col].to_numpy(), df[y_col].to_numpy(dtype=float), name)

    @classmethod
    def from_json(cls, path, name=None) -> "Series":
        with open(path, "r") as f:
            d = json.load(f)
        ds = d.get("ds", list(range(len(d["y"]))))
        return cls(np.asarray(ds), np.asarray(d["y"], dtype=float), name or _stem(path))

    @classmethod
    def from_csv(cls, path, ds_col="ds", y_col="y", name=None) -> "Series":
        import pandas as pd
        return cls.from_dataframe(pd.read_csv(path), ds_col, y_col, name or _stem(path))

    @classmethod
    def from_excel(cls, path, sheet=0, ds_col="ds", y_col="y", name=None) -> "Series":
        import pandas as pd
        df = pd.read_excel(path, sheet_name=sheet)
        return cls.from_dataframe(df, ds_col, y_col, name or _stem(path))

    # ---- operations -------------------------------------------------------

    def __len__(self) -> int:
        return len(self.y)

    def concat(self, other: "Series", name=None) -> "Series":
        """Join two series end to end (the paper's before_1 + before_2 aggregation)."""
        y = np.concatenate([self.y, other.y])
        return Series(np.arange(len(y)), y, name)

    def plot(self, window=None, ax=None):
        """Plot the raw series (needs the [viz] extra)."""
        import matplotlib.pyplot as plt
        w = slice(0, window) if isinstance(window, int) else (slice(*window) if window else slice(None))
        ax = ax or plt.subplots(figsize=(11, 3.4))[1]
        ax.plot(self.y[w], color="#2c7fb8", lw=0.8, label=self.name or "series")
        ax.set_title("input series"); ax.set_xlabel("ds"); ax.set_ylabel("y")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8)
        return ax

    def validate(self) -> "Series":
        """Check the contract; raise a precise error otherwise. Returns self."""
        if self.y.ndim != 1:
            raise ValueError("y must be one-dimensional")
        if len(self.ds) != len(self.y):
            raise ValueError("ds and y must have the same length")
        if not np.isfinite(self.y).all():
            raise ValueError("y contains NaN or inf; clean or interpolate before fitting")
        if np.issubdtype(self.ds.dtype, np.number):
            steps = np.diff(self.ds.astype(float))
            if len(steps) and not np.allclose(steps, steps[0]):
                raise ValueError("ds is not regularly spaced")
        return self


def _stem(path) -> str:
    import os
    return os.path.splitext(os.path.basename(str(path)))[0]
