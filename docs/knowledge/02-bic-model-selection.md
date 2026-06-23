# 02 - BIC model selection

The decomposition does not run forever. After each new harmonic pair it computes the Bayesian
Information Criterion and stops when BIC rises. BIC trades fit against complexity, so it halts before
the model starts fitting noise.

## The formula used

The canonical paper and the notebooks use

```
BIC = p * ln(N) + N * ln(2*pi*e*sigma^2)
```

where `N` is the number of observations, `p` the number of parameters, and `sigma^2 = RSS/N` the MLE
residual variance at the current iteration. The paper states **four parameters per harmonic pair**, so
after `j` pairs `p = 4j` and

```
BIC_j = 4*j*ln(N) + N*ln(2*pi*e*sigma_j^2)
```

The decomposition halts at the first `j` where `BIC_j > BIC_{j-1}`.

## Why this is the right shape

For iid Gaussian errors the maximized log-likelihood is

```
ln L_max = -(N/2) ln(2*pi) - (N/2) ln(sigma^2) - N/2
```

so

```
-2 ln L_max = N*ln(2*pi) + N*ln(sigma^2) + N = N*ln(2*pi*e*sigma^2)
```

since `ln(e) = 1` reproduces the trailing `+N`. The standard Schwarz form `BIC = p*ln(N) - 2 ln L_max`
therefore becomes exactly `p*ln(N) + N*ln(2*pi*e*sigma^2)`. The fit half of the notebook formula is a
full-constant BIC and is correct **as long as sigma^2 is the MLE plug-in `RSS/N`** (which the code
uses), not an unbiased `RSS/(N-p)`.

## The parameter-count caveat

The paper charges `4` per pair. A strict count is different:

- two amplitudes per harmonic, two harmonics per pair = `2j` amplitude parameters
- plus one shared noise variance = `+1`
- plus, if the two periods are themselves estimated (they are, via the period search), `2` more per
  pair

So a rigorous count is `p = 4j + 1` (amplitudes + frequencies + noise) or `2j + 1` if you treat the
periods as selected rather than fitted. The paper's `4j` happens to match the "amplitudes + estimated
periods" count but drops the `+1` for the shared variance.

This matters little in practice because only the **direction** of the BIC change drives the stopping
rule, and the dropped `+1` is constant across iterations on the same data. The library keeps the
paper's `4j` as the default for reproducibility and exposes the parameter count as an option
(`bic_params_per_pair`, plus a `+1` noise-variance toggle) so a stricter penalty can be selected.

## The stopping rule overfits on low-noise data

There is a more serious problem than the parameter count. As the residual variance `sigma^2 -> 0`, the
fit term `N*ln(2*pi*e*sigma^2) -> -infinity` and overwhelms the bounded penalty `4j*ln(N)`. So once the
model fits well, almost any extra pair lowers BIC, and the decomposition keeps adding pairs that fit
floating-point noise. On a clean two-harmonic signal it reaches R2 ~ 1.0 by the second pair yet
extracts 48 pairs total. This matches the canonical paper's reported 46 to 50 pairs per series. The
library must add a stopping guard (partial-R2-gain tolerance, a total-R2 target, or an epsilon floor on
sigma^2). Full finding and the test that pins it: [07-qa-audit.md](07-qa-audit.md).

## Verdict

- Fit term: correct, conditional on `sigma^2 = RSS/N` (the code uses the MLE plug-in, confirmed).
- Penalty term: faithful to the paper, but the paper undercounts by the shared `+1`. Minor.
- Stopping rule: **overfits on low-noise signals.** This is the headline audit finding; needs a guard.

See the audit in [07-qa-audit.md](07-qa-audit.md).

See [06-glossary-and-sources.md](06-glossary-and-sources.md) for the Schwarz and Neath-Cavanaugh
references, and [09-gap-research.md](09-gap-research.md) for the full BIC algebra with sources.
