# 09 - Gap research

The notebooks are an incomplete realization of the method. This doc records the pieces the papers
describe but the code skipped or stubbed, plus the pieces no paper covers, with enough math and sources
to implement them properly. Equations are in ASCII so they port straight to code.

## 1. The Bretthorst Bayesian periodogram (the intended period search)

The notebooks find each pair's periods with differential evolution minimizing residual variance. The
papers' intended method is Bretthorst's Bayesian spectrum analysis, which gives a posterior over the
period rather than a point optimum. The library offers it as a pluggable `period_search` backend.

Model the data as model functions plus Gaussian noise:

```
d(t_i) = sum_{j=1..m} a_j * G_j(omega, t_i) + e(t_i)
```

For one sinusoid `G_1 = cos(omega t)`, `G_2 = sin(omega t)`, so `m = 2`. The amplitudes and noise
variance are nuisance parameters. The key trick is to orthonormalize the model functions at each trial
frequency. Form the metric `g_{jk} = sum_i G_j G_k`, diagonalize it (eigenvectors or Gram-Schmidt) to
get orthonormal `H_j` with `sum_i H_j H_k = delta_{jk}`, project the data `h_j = sum_i d(t_i) H_j`, and
form the sufficient statistic

```
h2_bar = ( sum_{j=1..m} h_j^2 ) / m
```

For one sinusoid this is the Schuster periodogram. The posterior over frequency, noise variance
unknown (Jeffreys prior on sigma, uniform on amplitudes, both integrated out), is Student-t:

```
P(omega | D, I)  ∝  [ 1 - m * h2_bar / (N * d2_bar) ]^((m - N)/2)
```

where `d2_bar = (1/N) sum_i d(t_i)^2`. The bracket is one minus the fraction of data power explained,
raised to a negative power, so it sharpens as the fit improves and self-penalizes complexity. For `r`
frequencies stack all model functions (`m = 2r`), orthonormalize the full set jointly, and the same
expression becomes the joint posterior. Joint orthonormalization is why a joint fit beats greedy
peeling when frequencies are close.

Implementation notes that bite: orthonormalize at every trial omega (the `H_j` depend on omega);
evaluate the exponent `(m-N)/2` in log space (it is huge); rotate orthonormal coefficients back through
the eigenvector matrix to recover physical amplitudes.

Sources: Bretthorst 1988 (https://bayes.wustl.edu/glb/book.pdf, https://bayes.wustl.edu/glb/Lomb1.pdf);
BaSAR restatement (https://pmc.ncbi.nlm.nih.gov/articles/PMC3462997/).

## 2. BIC algebra and parameter count

Standard Schwarz: `BIC = k*ln(n) - 2*ln(L_max)`. For iid Gaussian errors,
`-2 ln L_max = N*ln(2*pi) + N*ln(sigma_hat^2) + N` with `sigma_hat^2 = RSS/N`, which equals
`N*ln(2*pi*e*sigma_hat^2)`. So the working forms are `BIC = N*ln(sigma_hat^2) + k*ln(n)` or
`N*ln(RSS/n) + k*ln(n)`. In the Gaussian case the variance counts as a parameter, so
`k = regression_params + 1`. A known-frequency cos+sine pair = 2 params; `j` such harmonics + noise =
`2j + 1`; if frequencies are fitted, `3` per harmonic = `3j + 1`. The notebook's `4j` matches an
"amplitudes + estimated periods" count (4 per pair) but drops the shared `+1`. Faithful to the paper,
mildly over/under-counted versus the strict form; see [02](02-bic-model-selection.md). Sources:
Wikipedia BIC; Neath & Cavanaugh (https://wires.onlinelibrary.wiley.com/doi/10.1002/wics.199).

## 3. Forecast confidence intervals (no paper covers these)

Three routes, in increasing assumption-freedom:

**(a) Delta method / Gaussian closed form.** Conditional on frequencies the model is linear in
amplitudes, so `y_hat(t*) = x(t*)' beta_hat` with prediction variance
`sigma_hat^2 + x(t*)' Cov(beta_hat) x(t*)`. Interval `y_hat(t*) +/- c * sqrt(...)`, `c = 1.96` (95%),
`1.28` (80%). Propagate phase/frequency uncertainty with the gradient of the sinusoid times its
covariance. Cheapest; assumes local linearity and normality.

**(b) Residual bootstrap.** Resample fitted residuals, roll forward
`y*_{T+h} = y_hat_{T+h} + e*_{T+h}`, take per-horizon percentiles over many paths. Robust to non-normal
residuals; refit per resample to capture parameter uncertainty.

**(c) Posterior predictive.** With the Bretthorst backend (section 1), draw
`(amplitudes, frequencies, sigma)` from the posterior, generate `y(t*)` per draw, read predictive
quantiles. The principled choice; folds in frequency and amplitude uncertainty without linearization.

For a deterministic harmonic mean the interval does not widen like `sqrt(h)` (that is an
AR/random-walk property); it widens through parameter uncertainty, so phase and frequency error
dominate at long horizons. Sources: Hyndman & Athanasopoulos FPP3
(https://otexts.com/fpp3/prediction-intervals.html).

## 4. The difference test (the method's headline claim, under-specified in the paper)

The canonical paper claims Type I error control but shows only structural comparison (trend count, R2,
spectral compaction), with no explicit test statistic or null distribution. Candidate mechanisms for
`BeforeAfterComparison`:

- **BIC / Bayes-factor model comparison.** Decompose before, after, and pooled; compare evidence for
  "two distinct states" against "one shared state". Stays inside the method's own machinery; gives a
  model-selection verdict, not a calibrated p-value.
- **Per-harmonic similarity thresholds.** Score every before-vs-after harmonic pair (normal and
  inverted) by RSS, MAE, MAPE (the harmonics draft). Simple, but the thresholds are ad hoc.
- **Autocorrelation-aware resampling.** Block bootstrap or surrogate-series (e.g. phase-randomized or
  IAAFT surrogates) permutation test on a decomposition statistic. This respects the dependence that
  invalidated the classical tests and yields a real, calibrated p-value.

Recommendation, pending author confirmation: the resampling route, because it is the only one that
delivers the calibrated Type I error rate the abstract claims. The other two are useful descriptive
companions. This is a methodological decision for the authors; the library will default to one
(`decision=`) and expose the rest, documenting assumptions. Surrogate-data background: Theiler et al.
(1992) and the IAAFT method are the standard for autocorrelation-preserving null series.

## 5. Harmonic regression pitfalls to guard against

From the standard literature, relevant to whichever period search is used: the frequency objective is
multimodal (one local optimum per periodogram bump), so seed from periodogram peaks; greedy peeling is
biased when `|f1 - f2|` approaches the Rayleigh resolution `1/T_span`, so prefer joint estimation for
close frequencies; cos/sin are not orthogonal on short or irregular spans, which couples amplitudes
(handle via orthonormalization); frequencies above Nyquist `1/(2 dt)` are unidentifiable; spectral
leakage from a non-integer cycle count spreads power into neighbors (window or model the trend).
Sources: VanderPlas Lomb-Scargle (https://arxiv.org/pdf/1703.09824); Harmonic Regression Models review.
