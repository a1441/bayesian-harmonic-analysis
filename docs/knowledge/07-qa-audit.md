# 07 - QA audit

The notebooks were written by hand, so every formula was checked against the papers and every
implementation against its formula before the library inherits the math. This doc is the verdict
table. The executable evidence is in `tests/` (12 passing tests; run `pytest tests/ -q`), which copies
the core notebook functions verbatim into `tests/harmonics_ref.py` and pins their current behavior.

## Verdict table

| Item | Paper | Notebook code | Verdict |
|------|-------|---------------|---------|
| Pair fit (normal equations) | `f = a1 cos+b1 sin + a2 cos+b2 sin`, OLS | 4x4 Gram solve in `estimate_harmonic_pair` | **Match.** Reconstruction identity holds to 1e-9. |
| Amplitude | `A = sqrt(a^2+b^2)` | same | **Match.** |
| Phase | `phi = arctan(a/b)`, sin basis, quadrant fix | `phi = arctan2(a, b)` | **Match.** `a cos+b sin = A sin(x+phi)` gives `phi = atan2(a,b)`; verified numerically. |
| BIC fit term | `N ln(2 pi e sigma^2)` = `-2 ln Lmax` | `N*ln(2*pi*e*sigma2_total)`, `sigma2 = RSS/N` | **Match**, conditional on the MLE plug-in, which the code uses. |
| BIC penalty | `p ln(N)`, "four per pair" | `p = 4*(j+1)` | **Faithful to paper.** Strict count is `4j+1` (adds shared noise variance) or `2j+1` (periods selected, not fitted). Direction-of-change only, so minor. See [02](02-bic-model-selection.md). |
| Trend / cycle thresholds | `T>n`, `T<2n/3` | `T1>n`, `T1<(2/3)n` | **Match.** |
| Grey-zone band name | "grey zone" | `classify_single` returns `"noise"` | **Bug (terminology).** The plot labels already say `'Grey Zone'`; only the classifier string is wrong. Library uses "grey zone". |
| Variable-amplitude cond 1 | `T = 2T1T2/(T1+T2) < 2n/3` | `T1T2/(n(T1+T2)) < 1/3` | **Match** (algebraically identical). |
| Variable-amplitude cond (close) | `T_A/T = (T1+T2)/|T1-T2| > 3` | `(T1+T2)/|T1-T2| > 3` | **Match.** (The paper's `T < T_A/3` is the same inequality, redundant.) |
| Variable-amplitude cond 3 | (paper's third is redundant with the above) | `T1T2/(|T1-T2| n) > 1`, i.e. `T_A > 2n` | **Divergence.** The notebook replaces the redundant paper condition with a stricter, genuinely separate one: the envelope must be slower than twice the record. Defensible; confirm intent. Library makes it configurable. |
| Beat / envelope period | `T_A = 2T1T2/|T1-T2|` | same | **Match.** This is the envelope period; the intensity-beat period `T1T2/|T1-T2|` is half as long. Label which you mean. |
| Forecast | continue each harmonic past `N` | `h2` uses `a2,b2` (NOT reused `a1,b1`) | **Match, no bug.** The earlier suspicion of coefficient reuse was wrong; verified by `test_forecast_uses_second_pair_coeffs`. |
| BIC stopping rule | add pairs until BIC rises | `while BIC_new <= BIC_prev` | **Bug (overfitting on low noise).** See finding below. |
| Period search | Bretthorst Bayesian periodogram (posterior over omega) | differential evolution on residual variance | **Methodological divergence**, not a bug. Same BIC on top. Library offers both backends. See [09](09-gap-research.md). |

## Headline finding: the BIC stopping rule overfits

On a clean two-harmonic signal the decomposition reaches `total_r2 ~ 1.0` by the **second** pair, then
keeps extracting pairs all the way to **48** (`test_bic_overfits_on_low_noise_signal`). The cause is in
the BIC formula itself: as the residual variance `sigma^2 -> 0`, the fit term
`N * ln(2 pi e sigma^2) -> -infinity` and dominates the bounded penalty `4j * ln(N)`, so adding another
pair almost always lowers BIC even when it is fitting floating-point noise. The loop only stops when the
optimizer can no longer reduce `sigma^2` at all.

This is not just a synthetic artifact. The canonical paper's Table 1 reports 46 to 50 pairs per series
(190 across four series), which is the same regime. So the published decompositions likely include many
spurious pairs past the point of a near-perfect fit. The R2 and the trend/cycle/grey-zone counts that
the paper compares are dominated by the first handful of pairs, so the qualitative conclusions can still
hold, but the pair counts themselves should be read with this in mind.

Recommended fixes for the library (any one helps, combine them):

- Stop when the **partial R2 gain** of a new pair falls below a tolerance (e.g. `< 1e-4`).
- Stop when **total R2** crosses a target (e.g. `0.99`), exposed as a parameter.
- Floor `sigma^2` at a small epsilon so the fit term cannot run to `-infinity`.
- Use the corrected parameter count and consider a heavier penalty (AIC/BIC variants) for low-noise data.

## Secondary finding: single-pair period search is multimodal

A single `optimize_harmonic_pair` on a two-period signal recovers the dominant period but not always
both (`test_single_optimize_finds_dominant_period`): it found `T=60` (amplitude 5) but missed `T=17`
(amplitude 2.5), landing both estimates near 60. The full iterative decomposition does surface both
(`test_decompose_recovers_both_planted_periods`). This is the known multimodality of harmonic
regression ([09](09-gap-research.md) section 5): seed from periodogram peaks, prefer joint
orthonormal estimation for close frequencies.

## What the library inherits

- Corrected: "grey zone" terminology; a real stopping guard; configurable variable-amplitude cond 3;
  the documented BIC parameter-count option.
- Kept as-is (confirmed correct): the pair fit, amplitude/phase, the forecast continuation, the BIC fit
  term.
- Offered as an alternative backend: the Bretthorst Bayesian periodogram for period search.

The notebooks stay unchanged as the historical record. `tests/harmonics_ref.py` is a frozen copy for
regression only; do not build on it.
