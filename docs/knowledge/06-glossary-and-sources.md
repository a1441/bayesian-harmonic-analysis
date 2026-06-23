# 06 - Glossary and sources

## Symbols

| Symbol | Meaning |
|--------|---------|
| `y`, `y(t)` | the observed time series |
| `N`, `n` | series length (number of observations) |
| `t` | integer time index, `0 .. N-1` |
| `T1`, `T2` | the two periods of a harmonic pair |
| `a1, b1, a2, b2` | cosine and sine coefficients of the two harmonics |
| `A = sqrt(a^2 + b^2)` | harmonic amplitude |
| `phi = atan2(a, b)` | harmonic phase |
| `T = 2*T1*T2/(T1+T2)` | carrier period of a pair |
| `T_A = 2*T1*T2/|T1-T2|` | modulating-amplitude (envelope) period of a pair |
| partial R2 | variance of the current residual explained by the newest pair |
| total R2 | cumulative variance of the original series explained so far |
| BIC | Bayesian Information Criterion, the stopping rule |
| Trend | component with `T > n` |
| Cycle | component with `T < (2/3) n` |
| Grey zone | component with `(2/3) n < T <= n` |
| Noise | the negligible residual after decomposition (not a component class) |

## The papers

The source PDFs are not redistributed in this repository (they live only in the authors' local
working copy). They are cited below by title and author; request them from the authors or the
publisher.


- **bayesian-harmonic-longitudinal-difference-testing.pdf** - Markov, Marchev, Haralampiev. *Bayesian
  Harmonic Analysis solving longitudinal statistical testing for differences of experimental
  time-series data.* The canonical reference. Single-subject before/after HRV, harmonic-pair
  decomposition, trend/cycle/grey-zone classification, architecture comparison.
- **harmonic-oscillations-fluctuating-amplitude.pdf** - Haralampiev. *Bayesian approach for estimation
  of harmonic oscillations with fluctuating amplitude.* The amplitude-modulation identity, the
  three-condition test (Eq. 25), the decay models, BIC stopping.
- **state-capitalism-kondratiev.pdf** - Naydenov, Haralampiev. Application of the same harmonic method
  to long economic waves; the trend/cycle boundary rules.
- **bayesian-fourier-travel-times-sofia.pdf** - Haralampiev (and Marchev, Lomev). The forecasting
  variant: harmonic decomposition plus seasonal/weekday/hour corrections to predict travel times.
- **time-series-analysis.pptx** - the method presentation (periodogram, Bayesian estimation, decay
  models, forecast intervals).
- **harmonics-draft.docx** - the working notes that became the canonical paper: the pre/post HRV idea,
  the 0.5% energy threshold, and the per-harmonic RSS/MAE/MAPE comparison.

## External references

Method foundation:

- G. L. Bretthorst, *Bayesian Spectrum Analysis and Parameter Estimation*, Springer Lecture Notes in
  Statistics 48 (1988). Book PDF: https://bayes.wustl.edu/glb/book.pdf ; frequency-estimation chapter:
  https://bayes.wustl.edu/glb/Lomb1.pdf
- BaSAR (Granqvist, Hartley, Denby) restates Bretthorst's equation set in readable form:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC3462997/
- G. Schwarz, *Estimating the Dimension of a Model*, Annals of Statistics 6(2), 461-464 (1978). The
  BIC.

BIC and parameter counting:

- Wikipedia, Bayesian information criterion (Gaussian form, variance counts as a parameter):
  https://en.wikipedia.org/wiki/Bayesian_information_criterion
- Neath & Cavanaugh, The Bayesian information criterion: background, derivation, and applications:
  https://wires.onlinelibrary.wiley.com/doi/10.1002/wics.199

Harmonic regression, beats, intervals, input schemas:

- Harmonic Regression Models, a comparative review:
  https://www.researchgate.net/publication/23756808
- VanderPlas, Understanding the Lomb-Scargle Periodogram: https://arxiv.org/pdf/1703.09824
- Beats (carrier, beat, envelope periods): https://math.mit.edu/classes/18.03/sup/sup7.pdf
- Hyndman & Athanasopoulos, FPP3, prediction intervals: https://otexts.com/fpp3/prediction-intervals.html
- Nixtla input format (`unique_id / ds / y`):
  https://nixtlaverse.nixtla.io/neuralforecast/examples/data_format.html
- sktime datatypes: https://www.sktime.net/en/stable/examples/AA_datatypes_and_datasets.html
- darts TimeSeries guide: https://unit8co.github.io/darts/userguide/timeseries.html

The full equation-by-equation research notes, with the derivations and every source mapped to a claim,
are summarized in [09-gap-research.md](09-gap-research.md).
