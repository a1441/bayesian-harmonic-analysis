# 01 - Methodology: the harmonic-pair decomposition pipeline

The method models a time series as a sum of **harmonic pairs**. A pair is two sinusoids at two
estimated periods `T1` and `T2`. Pairs are extracted one at a time from the running residual until
adding another pair stops improving the model (BIC, see [02](02-bic-model-selection.md)). Each pair is
then classified and, if its two periods are close enough, interpreted as one amplitude-modulated
component (see [03](03-amplitude-modulation.md)).

Notation: `y` is the series, `N` (or `n`) its length, `t = 0, 1, ..., N-1` the integer time index.

## Step 1 - Estimate one pair at fixed periods

Given two periods `T1, T2` and the data `y`, fit the four-coefficient model

```
f(t) = a1*cos(2*pi*t/T1) + b1*sin(2*pi*t/T1) + a2*cos(2*pi*t/T2) + b2*sin(2*pi*t/T2)
```

Build the four basis vectors `cos(2*pi*t/T1), sin(2*pi*t/T1), cos(2*pi*t/T2), sin(2*pi*t/T2)`, form
the 4x4 Gram matrix `M` of their inner products and the right-hand side `Y = [<y,cos_T1>, <y,sin_T1>,
<y,cos_T2>, <y,sin_T2>]`, and solve `M [a1,b1,a2,b2]' = Y` (ordinary least squares).

Returns: the fitted signal `f`, the two harmonics `h1 = a1*cos+b1*sin` (period T1) and `h2` (period
T2), residuals `y - f`, total variance `var(y)`, remaining variance `var(y - f)`, and
**partial R2** `= 1 - var(residual)/var(y)`.

Notebook function: `estimate_harmonic_pair(t1, t2, y)`.

## Step 2 - Search for the best periods

`T1, T2` are unknown, so search for the pair that explains the most variance of the current residual.
The notebooks use `scipy.optimize.differential_evolution`:

- bounds `T1, T2 in [2, 10*N]`
- constraint `T2 > T1` (and `T1 >= 2`), enforced by a large penalty
- objective: residual variance from Step 1
- settings: strategy `best2bin`, `popsize=25`, `maxiter=500`, `tol=1e-7`, `mutation=(0.7,1.5)`,
  `recombination=0.9`, `polish=True`, fixed seed

This is a separable fit: the outer global search finds the periods, the inner OLS (Step 1) finds the
amplitudes in closed form. The papers' intended alternative is the **Bretthorst Bayesian periodogram**
(a posterior over the period); see [09](09-gap-research.md). The library exposes the period search as a
pluggable backend so both are available.

Notebook function: `optimize_harmonic_pair(y)`.

## Step 3 - Peel pairs until BIC stops improving

```
residual <- y
total_model <- 0
for j = 1, 2, ...:
    pair <- optimize_harmonic_pair(residual)      # Step 2 on the current residual
    total_model <- total_model + pair.fitted
    residual <- y - total_model
    BIC_j <- 4*j*ln(N) + N*ln(2*pi*e*sigma^2)     # sigma^2 = mean((y - total_model)^2)
    if BIC_j > BIC_{j-1}: stop
    record pair
```

Each iteration removes one pair's fit from the residual and adds it to the cumulative model. **Total
R2** is the cumulative variance explained against the original `y`; **partial R2** is the gain from the
newest pair against the residual it was fit on. The loop halts when a new pair raises BIC. See
[02](02-bic-model-selection.md) for the BIC formula, its parameter count, and the audit.

Output: a list of per-pair result dicts, each with `T1, T2, a1, b1, a2, b2, partial_r2, total_r2, BIC,
fitted, harmonic1, harmonic2`.

Notebook function: `decompose_harmonics_until_bic(y)`.

## Step 4 - Amplitude, phase, and classification

For each pair convert coefficients to amplitude and phase:

```
A1 = sqrt(a1^2 + b1^2)      phi1 = atan2(a1, b1)
A2 = sqrt(a2^2 + b2^2)      phi2 = atan2(a2, b2)
```

(The notebook uses `atan2(a, b)`. The papers write `phi = arctan(a/b)` with a quadrant correction on a
sin basis. The conventions agree; see the phase-convention note in [07](07-qa-audit.md).)

**Classify each component by its period relative to the series length `n`** (canonical paper):

```
T > n            -> Trend        (longer than the whole record: a long-term movement)
T < (2/3) n      -> Cycle        (at least ~1.5 periods fit inside the record)
(2/3) n < T <= n -> Grey zone    (ambiguous: a weak trend or a very long cycle)
```

Use the term **grey zone**, not "noise". The notebook code labels this band "Noise"; the canonical
paper reserves "noise" for the negligible residual left after decomposition. The library uses the
paper's term. See [07](07-qa-audit.md).

**Flag a pair as variable (fluctuating) amplitude** when all three conditions hold (see
[03](03-amplitude-modulation.md) for the derivation):

```
1)  T   = 2*T1*T2/(T1+T2)      < (2/3) n        # the carrier period is a true cycle
2)  T   < (1/3) T_A                              # the modulation is slower than the carrier
3)  T_A / T = (T1+T2)/|T1-T2|  > 3               # the two periods are close enough to beat
        where T_A = 2*T1*T2/|T1-T2|
```

Notebook function: `analyze_results(y, results, t)` returns a table with one row per pair (periods,
amplitudes, phases, the three condition flags, the variable-amplitude verdict, the trend/cycle/grey
classification, partial and total R2, and the component arrays).

## Step 5 - Split variable-amplitude pairs

When a pair is flagged variable-amplitude, it represents one oscillation whose amplitude rises and
falls, expressed as a product of a slow envelope and a fast carrier:

```
f(t) = [A_A * sin(2*pi*t/T_A + phi_A)] * sin(2*pi*t/T + phi)
```

with carrier period `T = 2*T1*T2/(T1+T2)` and modulating-amplitude (envelope) period
`T_A = 2*T1*T2/|T1-T2|`. The product-to-sum identity `sin(x)sin(y) = 1/2 [cos(x-y) - cos(x+y)]` shows
this equals the original two-harmonic pair. See [03](03-amplitude-modulation.md).

Notebook functions: `decompose_variable_amplitude_pair(...)`, `classify_harmonic_components(...)`.

## Step 6 - Aggregate and compare (the headline deliverable)

The point of the method is the before/after comparison. Aggregate the per-run series into two
composites (canonical paper: Before = b1 + b2, After = a1 + a2), decompose each composite, and compare
their harmonic architecture: number of trend / cycle / grey-zone components, presence of
variable-amplitude pairs, total R2, residual variance, and spectral compaction (whether the after
state needs fewer pairs). In the paper's HRV data the after state showed a triadic trend hierarchy and
a new grey-zone wave that the before state lacked. The full comparison and the decision mechanism are
in [05](05-before-after-comparison.md).

## Step 7 - Forecast (secondary)

For the forecasting use-case (the Sofia transport paper, not the difference-testing paper), continue
every stored harmonic past the training length:

```
forecast(t) = sum over pairs [ a1*cos(2*pi*t/T1) + b1*sin(2*pi*t/T1)
                             + a2*cos(2*pi*t/T2) + b2*sin(2*pi*t/T2) ],  t = N, N+1, ...
```

Confidence intervals are not given in any paper; the options are in [09](09-gap-research.md).

Notebook function: `forecast_from_harmonic_coeffs(results, n_ahead, target_r2=None)`. Note the audit
flags a likely coefficient-reuse bug in the notebook version; see [07](07-qa-audit.md).
