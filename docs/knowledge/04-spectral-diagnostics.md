# 04 - Spectral diagnostics (FFT suite)

This is the supporting FFT analysis, separate from the harmonic-pair core. It is an FFT-based view used
to characterize and compare signals, and the library keeps it in a separate `SpectralAnalyzer` rather
than in the core.

## Preprocessing

Before any FFT the signals are conditioned to reduce spectral leakage and remove trend bias:

```
normalize:   z = (y - mean(y)) / std(y)              # z-score
detrend:     scipy.signal.detrend(z)                 # remove linear trend
demean:      subtract the mean of the detrended series
window:      multiply by a Hann window (np.hanning)  # taper the edges
zero-pad:    pad to the next power of two             # FFT efficiency / finer bin spacing
```

## FFT cumulative-energy decomposition

`decompose_fourier_cumulative_energy(y, target_total_pct=90.0, sampling_rate=1.0)`:

1. FFT the signal, keep positive frequencies.
2. Amplitude per bin `amp_k = |FFT_k| / N` (doubled for non-DC, non-Nyquist bins).
3. Energy per bin `E_k = amp_k^2`.
4. Sort bins by energy descending, keep the smallest set whose cumulative energy reaches the target
   percentage.
5. Reconstruct the kept harmonics; return the components, the reconstruction, and a summary table
   (frequency, amplitude, phase = `angle(FFT_k)`, energy contribution %, period).

Related helpers: `decompose_harmonics_array_with_energy` (one isolated harmonic per index via
conjugate-symmetric masking and IFFT), `reconstruct_from_harmonics`, `sort_and_filter` (drop harmonics
below `min_energy=0.005`, i.e. the 0.5% energy threshold from the harmonics draft), and
`forecast_with_fft` (project kept bins forward).

## The metric suite

Each metric is computed on the kept-harmonic spectrum of every signal (before1, before2, after1,
after2), then compared across the before/after split.

| Metric | Computation | Reads as |
|--------|-------------|----------|
| Dominant frequency | `argmax(E)` | location of the strongest harmonic |
| Dominant energy % | `max(E_contribution)` | how concentrated the peak is |
| Weighted mean frequency | `sum(f*E)/sum(E)` | spectral centroid |
| Spectral entropy | `entropy(E/sum(E))` | spread of power: low = concentrated, high = diffuse |
| Spectral slope | `linregress(log f, log E)` | power-law exponent: steep = low-frequency dominance |
| Phase coherence | `|mean(exp(i*phi))|` in [0,1] | alignment of phases across harmonics |
| Frequency range | `max(f) - min(f)` | bandwidth of kept harmonics |
| Top-10% energy | sum of the top decile of sorted energies | concentration in the strongest bins |
| Reconstructed variance | `var(reconstruction)` | power retained |
| Residual variance | `var(y - reconstruction)` | power lost |
| Correlation | `corr(y, reconstruction)` | reconstruction fidelity |

## Pairwise before/after comparison

Merge the two signals' summary tables on frequency and compute per-harmonic amplitude, phase, and
energy differences plus their correlations. Flags used in the notebook: amplitude difference > 0.1,
phase difference > 0.1, energy difference > 1.0%. This FFT comparison is a cross-check on the
harmonic-pair comparison in [05-before-after-comparison.md](05-before-after-comparison.md); the
harmonic-pair method is the primary one because it handles amplitude modulation, which a fixed-bin FFT
does not.
