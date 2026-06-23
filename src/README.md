# harmonix (src/)

The library, implemented under `src/harmonix/`. Install editable with `pip install -e .` (or add
`src/` to `sys.path`), then:

    from harmonix import HarmonicModel, Series
    model = HarmonicModel().fit(Series.from_json("data/before_1.json"))
    model.summary(); model.predict(100); model.plot()

The methodology is in [../docs/knowledge/](../docs/knowledge/00-index.md); correctness notes are in the
[QA audit](../docs/knowledge/07-qa-audit.md).

**Faithful by default.** `core.py` is a direct port of the notebook functions
(`estimate_harmonic_pair`, `optimize_harmonic_pair`, `decompose_harmonics_until_bic`,
`forecast_from_harmonic_coeffs`, `analyze_results`, `decompose_variable_amplitude_pair`,
`count_trend_cycle_noise`) with their math, defaults (DE popsize 25 / maxiter 500 / bounds 10N), and
result shapes. Optimizations are **opt-in flags only**: `solver='lstsq'` (stable solve),
`min_r2_gain>0` (anti-overfit stop guard), `max_pairs=N` (cap). With defaults the output matches the
notebook.

## Design rules

- **OOP, `apply()`-style components, kwargs-driven, immutable returns, verbose logging.** Each
  estimator is a class configured at construction and run with `apply()` (or the sklearn-style
  `fit`/`predict` facade below). Returns are new value objects, not in-place mutations.
- **Input is a single `Series` of `(ds, y)`, never a raw file.** Loaders (json/csv/excel/dataframe)
  build it; the math only ever sees the validated object. No multi-series frames, no group column. See
  [08-input-schema.md](../docs/knowledge/08-input-schema.md).
- **No plotting in the compute path.** The core imports no matplotlib. Plotting is an optional layer
  (`viz/`) behind a `[viz]` install extra. Nothing plots unless asked.
- **Corrected math, not the notebook's.** Grey-zone terminology, a real BIC stopping guard,
  configurable variable-amplitude condition. The notebooks stay as a historical record.

## Module map

```
src/harmonix/
  __init__.py    exports HarmonicModel, Series, core, metrics
  series.py      Series (ds, y) value object + from_json/csv/excel/dataframe/array + validate + concat
  core.py        faithful notebook math: estimate / optimize / decompose / forecast / analyze /
                 decompose_variable_amplitude_pair / count_trend_cycle_noise (+ classify helpers)
  model.py       HarmonicModel: fit / predict / summary / analyze / count_components /
                 variable_amplitude_pairs / plot
  metrics.py     r2, rmse, mae, mape, bic, aic, summary_metrics
  plotting.py    optional ([viz] extra): plot_model(kind in fit/components/forecast/residuals)
```

Still on the roadmap (designed, not built): `BeforeAfterComparison`, `SpectralAnalyzer` (FFT suite),
and the documented failed baselines (`ArimaBaseline`, `DifferencingBaseline`, `NaiveTwoSampleTest`),
plus the Bretthorst periodogram period-search backend.

## Components

| Class | `apply()` input | returns | maps to |
|-------|-----------------|---------|---------|
| `Series` + loaders | a json/csv/excel/dataframe | validated `(ds, y)` | [08](../docs/knowledge/08-input-schema.md) |
| `HarmonicPairEstimator` | `T1, T2, y` | `PairFit` (a1,b1,a2,b2, fitted, harmonics, partial_r2) | [01](../docs/knowledge/01-methodology.md) step 1 |
| `HarmonicDecomposer` | `Series`, backend, stop params | `Decomposition` (pairs + R2/BIC trace) | steps 2-3 |
| `HarmonicAnalyzer` | `Decomposition` | `AnalysisTable` (A, phi, trend/cycle/grey, var-amp) | steps 4-5 |
| `HarmonicForecaster` | `Decomposition, h, level` | `Forecast` (yhat + optional intervals) | step 7 |
| `BeforeAfterComparison` | two `Series` (before, after) | `ComparisonReport` | step 6, the headline |
| `SpectralAnalyzer` | `Series` | spectral metrics table | [04](../docs/knowledge/04-spectral-diagnostics.md) |

### Corrections the components must apply (from the audit)

- `HarmonicDecomposer`: add a **stopping guard** (partial-R2-gain tolerance and/or total-R2 target
  and/or epsilon floor on sigma^2). The bare paper BIC overfits on low-noise data. Expose
  `bic_params_per_pair` and a noise-variance `+1` toggle.
- `HarmonicAnalyzer`: classify the `(2/3)n < T <= n` band as **"grey zone"**, not "noise". Make the
  third variable-amplitude condition configurable (paper's two-test set as default, the notebook's
  `T_A > 2n` as an option).
- `search.py`: ship `diff_evolution` (current) and `bayesian_periodogram` (Bretthorst, the papers'
  intended method) behind one interface; seed the search from periodogram peaks.
- `HarmonicForecaster`: intervals from [09](../docs/knowledge/09-gap-research.md); none are in the
  papers.
- `BeforeAfterComparison`: implement an explicit decision mechanism (the open question in
  [05](../docs/knowledge/05-before-after-comparison.md)); default to the autocorrelation-aware
  resampling test pending author confirmation.

## The facade: fit / predict

`HarmonicModel` wraps the core for the common path.

```python
from harmonix import HarmonicModel, Series, BeforeAfterComparison

series = Series.from_json("data/before_1.json")   # or from_csv / from_excel / from_dataframe

model = HarmonicModel(
    period_search="diff_evolution",   # or "bayesian_periodogram"
    stop="r2_target", r2_target=0.99, # the audit-driven stopping guard
    energy_threshold=0.005,
    random_state=42, verbose=True,
)
model.fit(series)
model.summary()                        # AnalysisTable: periods, A, phi, class, var-amp, R2
fc = model.predict(h=100, level=[80, 95])

before = Series.from_json("data/before_1.json").concat(Series.from_json("data/before_2.json"))
after  = Series.from_json("data/after_1.json").concat(Series.from_json("data/after_2.json"))
cmp = BeforeAfterComparison(decision="bootstrap")
cmp.fit(before, after)
cmp.report()                           # architecture diff + verdict
```

- `HarmonicModel.fit` -> `HarmonicDecomposer` + `HarmonicAnalyzer`
- `HarmonicModel.predict` -> `HarmonicForecaster`
- `BeforeAfterComparison` -> aggregate, decompose both, compare, decide

## Build order (when implementation starts)

1. `Series` (ds, y) + loaders + `validate()` + `concat()` (everything depends on it).
2. `HarmonicPairEstimator` (port verbatim, it is correct) + tests already in `tests/`.
3. `HarmonicDecomposer` with the stopping guard (the real fix).
4. `HarmonicAnalyzer` with grey-zone terminology and configurable conditions.
5. `BeforeAfterComparison` once the decision mechanism is chosen.
6. `HarmonicForecaster`, then `SpectralAnalyzer`, then `viz/`, then baselines.

Defer `pyproject.toml`/packaging to a later pass (per the agreed scope).
