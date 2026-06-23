# 05 - Before/after comparison (the headline difference test)

This is what the canonical paper is for: deciding whether a single subject's physiology differs before
and after an intervention, when classical pre/post tests cannot be used.

## Why classical tests fail here

The motivating design has one subject, two HRV runs before a coaching intervention and two after. Two
problems block the standard toolkit:

- **Autocorrelation.** Successive points in a physiological series are dependent. Treating each time
  point as an independent observation is pseudoreplication; it inflates the effective sample size and
  the Type I error rate, so a naive t-test or rank test reports significance that is not real.
- **Tiny N of true units.** If you instead treat each run as one independent unit, you have N = 2
  before and N = 2 after. No test has power at that sample size.

The paper documents five approaches that fail for these reasons: descriptive statistics (no
inference), the naive two-sample test (pseudoreplication), cross-run differencing (reuses each run's
data, inflates degrees of freedom), segment-wise analysis (segments from one run are still
correlated), and ARIMA (needs a long stationary series; two short runs are not enough). The library
keeps these as documented baselines so a user can see why they are wrong, not as recommended methods.

## The approach: compare harmonic architecture

Instead of testing raw values, decompose each condition and compare structure.

1. **Aggregate.** Combine the per-run series into two composites by concatenation. Canonical paper:
   Before = before_1 then before_2, After = after_1 then after_2 (after truncating to a common length
   and z-scoring to a common scale). Each series is loaded on its own (`ds, y`) and joined with
   `concat()`; see [08-input-schema.md](08-input-schema.md).
2. **Decompose** each composite with the harmonic-pair pipeline ([01](01-methodology.md)).
3. **Compare the architecture.** Tabulate, for before vs after:
   - count of trend, cycle, and grey-zone components
   - number of variable-amplitude (beat) pairs
   - total R2 and residual variance
   - number of pairs needed (spectral compaction)
4. **Per-harmonic similarity** (from the harmonics draft): for every harmonic in before against every
   harmonic in after, both normal and inverted, score residual sum, MAE, and MAPE. Matching harmonics
   absorb one another; the share of harmonics with no match measures how differently the two states are
   built.

In the paper's data this surfaced a clear, interpretable difference: the before state had a single
dominant trend, the after state showed three trend components (a triadic hierarchy) plus a new
grey-zone wave and rare beat-like modulation, read as a shift toward multi-scale autonomic regulation.
Both models explained over 96% of variance with negligible noise.

## The open question: what is the actual decision rule?

The paper's abstract claims the method "controls Type I error without sacrificing power," but the
inference shown is structural and descriptive (trend count 1 to 3, R2, spectral compaction). There is
no explicit test statistic, null distribution, or p-value in the presented work. For the library to
deliver "statistical testing for differences" honestly, `BeforeAfterComparison` needs a concrete,
defensible decision mechanism. Three candidates, in [09-gap-research.md](09-gap-research.md):

- **BIC / Bayes-factor model comparison.** Fit before, after, and pooled; compare evidence for "two
  states differ" against "one state explains both". Stays inside the method's own BIC machinery.
- **Per-harmonic similarity metrics.** Threshold the RSS / MAE / MAPE matching above into a
  difference score. Simple, already in the draft, but the thresholds are ad hoc.
- **Autocorrelation-aware resampling.** A block bootstrap or surrogate-series permutation test on the
  decomposition statistic, which respects the dependence that broke the classical tests and yields a
  real p-value.

This is a methodological choice for the authors, not something to invent silently. The library will
implement one as the default (`decision=`) and expose the others, and document the assumptions of
whichever is chosen. Recommendation pending author input: the resampling route, because it is the only
one of the three that produces a calibrated Type I error rate, which is the property the abstract
claims.
