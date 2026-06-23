# 08 - Input schema (the standardized contract)

The library works on **one series at a time**, with exactly two columns: `ds` and `y`. Nothing else.
No series id, no group column, no multi-series frames. A before/after comparison is just two such
series passed in separately.

## The contract

| column | type | meaning |
|--------|------|---------|
| `ds` | integer step `0..N-1`, or a regularly spaced datetime | the time index |
| `y` | float | the observed value |

Rules:

- exactly these two columns; extra columns are ignored
- `ds` must be sorted and regularly spaced (the harmonic periods are measured in samples). A datetime
  index is accepted if its spacing is uniform; it maps to the integer step internally.
- `y` is numeric; the NaN policy (drop / interpolate / reject) is declared by the caller.

## Accepted formats

The same two-column series can be loaded from any of:

- **JSON** - columnar `{"ds": [...], "y": [...]}` (what the bundled data uses), via `from_json(path)`.
- **CSV** - a `ds,y` header and one row per sample, via `from_csv(path)`.
- **Excel** - a sheet with `ds` and `y` columns, via `from_excel(path, sheet=...)`.
- **DataFrame** - a `pandas.DataFrame` (or a bare `pd.Series` / `np.ndarray` for `y`, with `ds`
  defaulting to `0..N-1`), via `from_dataframe(df)` / `from_array(y)`.

All loaders produce the same internal `Series` object; the math only ever sees that. Runnable
examples: `data/template.json` and `data/template.csv`.

## The bundled example data

The example dataset is four files in `data/`, one series each, all `ds, y`, fully anonymized (no
subject identifiers, only the integer index and the value):

| file | role |
|------|------|
| `before_1.json` | first run, before the intervention |
| `before_2.json` | second run, before |
| `after_1.json` | first run, after |
| `after_2.json` | second run, after |

Each holds 9130 samples. The canonical paper aggregates the two runs per condition into composites
(Before = before_1 followed by before_2, After likewise) before comparing; aggregation is the caller's
choice, done by concatenating two loaded series. See
[05-before-after-comparison.md](05-before-after-comparison.md).

## Validation

`validate()` checks and reports precise errors for:

- exactly the `ds` and `y` columns present
- `ds` sorted, no duplicates, regular spacing (or a declared frequency)
- numeric `y`, with the declared NaN policy applied
