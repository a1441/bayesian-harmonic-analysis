# Knowledge base: Bayesian Harmonic Analysis

This folder is the methodology spec for the library. It distills the published method and the
exploratory notebooks into one place so the code has a clean contract to build against. Read it in
order.

## The method in one paragraph

You have a before series and an after series from a single subject (the example is heart rate
variability, two runs before a coaching intervention and two after). You want to test whether the two
states differ. Classical pre/post tests fail because the observations are autocorrelated, not
independent. So instead of testing raw values, you decompose each series into pairs of harmonic
components, classify each component as trend, cycle, or grey zone, detect components whose amplitude
fluctuates over time, and then compare the harmonic architecture of before against after. The shift in
that architecture is the result.

## Reading order

| Doc | What it covers |
|-----|----------------|
| [01-methodology.md](01-methodology.md) | The full pipeline: estimate a pair, search periods, BIC stopping, classify, aggregate, compare. The equations. |
| [02-bic-model-selection.md](02-bic-model-selection.md) | How BIC decides when to stop adding pairs, the parameter-count convention, and the audit. |
| [03-amplitude-modulation.md](03-amplitude-modulation.md) | Variable-amplitude (beat) components, the product-to-sum split, and the decay models. |
| [04-spectral-diagnostics.md](04-spectral-diagnostics.md) | The FFT cumulative-energy decomposition and the spectral metric suite (supporting, not core). |
| [05-before-after-comparison.md](05-before-after-comparison.md) | The headline difference test: aggregation, architecture comparison, and the decision mechanism. |
| [06-glossary-and-sources.md](06-glossary-and-sources.md) | Symbols, the papers, and the external references. |
| [07-qa-audit.md](07-qa-audit.md) | Every equation checked paper-vs-code, the verdicts, and the test harness. |
| [08-input-schema.md](08-input-schema.md) | The standardized input contract the library accepts. |
| [09-gap-research.md](09-gap-research.md) | The pieces the papers left open: Bretthorst periodogram, forecast intervals, the difference test. |

## The sources

The canonical reference is `docs/papers/bayesian-harmonic-longitudinal-difference-testing.pdf` (Markov,
Marchev, Haralampiev). The harmonic-pair math comes from `harmonic-oscillations-fluctuating-amplitude.pdf`
and `state-capitalism-kondratiev.pdf` (Haralampiev, Naydenov). The forecasting variant is
`bayesian-fourier-travel-times-sofia.pdf`. The implementation is the `harmonix` library in
`src/harmonix/`.

## Status

The library is implemented in `src/harmonix/` (design notes in `src/README.md`). The QA audit
in `07-qa-audit.md` flags where the original hand-written analysis diverged from the papers; the
library reproduces it faithfully by default and exposes the corrected behavior as opt-in flags.
