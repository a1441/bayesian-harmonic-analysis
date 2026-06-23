# 03 - Amplitude modulation (variable-amplitude components)

Classical Fourier models assume each component has a constant amplitude. Real signals often have
oscillations whose strength rises and falls over time. The method captures this without abandoning
sinusoids: a single oscillation with a fluctuating amplitude is mathematically identical to a pair of
constant-amplitude sinusoids at two close frequencies. That pair is exactly the harmonic pair the
decomposition extracts.

## The identity

A harmonic with a fluctuating amplitude is the product of a slow envelope and a fast carrier:

```
f(t) = [A_A * sin(2*pi*t/T_A + phi_A)] * sin(2*pi*t/T + phi)
```

- `T` is the period of the carrier (the oscillation you observe)
- `T_A` is the period of the modulating amplitude (the envelope)
- `A_A`, `phi_A` are the amplitude and phase of the modulation

Apply the product-to-sum identity

```
sin(x) * sin(y) = 1/2 [ cos(x - y) - cos(x + y) ]
```

and the product becomes the sum of two cosines at the difference and sum frequencies:

```
f(t) = (A_A/2) cos( 2*pi*t*(1/T - 1/T_A) + (phi - phi_A) )
     - (A_A/2) cos( 2*pi*t*(1/T + 1/T_A) + (phi + phi_A) )
```

These are two constant-amplitude harmonics at two nearby periods `T1`, `T2`. Read in reverse: whenever
the decomposition finds two harmonics close in frequency, their sum is a beat, and you can re-express
it as one modulated oscillation.

## The period relations

Given the pair `(T1, T2)` the carrier and envelope periods are

```
T   = 2*T1*T2/(T1 + T2)        # carrier  (the observable cycle)
T_A = 2*T1*T2/|T1 - T2|        # modulating-amplitude / envelope period
```

`T_A` is the **envelope** period. The stricter "intensity beat" period is `T1*T2/|T1-T2|`, half as
long, because the envelope magnitude peaks twice per envelope cycle. The notebooks and the canonical
paper use the envelope period `2*T1*T2/|T1-T2|`. Label which one you mean; do not mix the two. See the
beat note in [07-qa-audit.md](07-qa-audit.md).

## When a pair counts as variable-amplitude

Not every pair is a beat. Three conditions must all hold (canonical paper):

```
1)  T = 2*T1*T2/(T1+T2) < (2/3) n      # the carrier is a genuine cycle (fits inside the record)
2)  T < (1/3) T_A                       # the envelope is much slower than the carrier
3)  T_A/T = (T1+T2)/|T1-T2| > 3         # the two periods are close enough to produce a beat
```

If any fails, treat the pair as two independent harmonics rather than one modulated component.

Note on the paper vs the notebook: the paper's conditions 2 and 3 (`T < T_A/3` and `T_A/T > 3`) are the
same inequality written twice, so the paper really states two independent tests. The notebook keeps
condition 1 (`T < 2n/3`) and the close-frequency test (`(T1+T2)/|T1-T2| > 3`) but uses a different
third condition, `T_A > 2n` (the envelope must be slower than twice the record), which is stricter than
the paper. This divergence is flagged in [07-qa-audit.md](07-qa-audit.md); the library makes the third
condition configurable and defaults to the paper's two-test set.

The notebook computes these in `analyze_results` and splits qualifying pairs with
`decompose_variable_amplitude_pair`. Condition 1 is algebraically the `2*T1*T2/(n(T1+T2)) < 2/3` form
from Eq. 25 of the fluctuating-amplitude paper.

## Decay models (extension, not in the notebook core)

The fluctuating-amplitude paper also gives three parametric envelopes for amplitudes that decay rather
than oscillate. They are alternatives to the sinusoidal envelope above and are candidates for a future
library extension:

```
Lorentzian decay in frequency:   f(t) = [A * e^{-a t}]       * sin(2*pi*t/T + phi)
Lorentzian decay in time:        f(t) = [A / (1 + a t^2)]    * sin(2*pi*t/T + phi)
Gaussian decay in time:          f(t) = [A * e^{-a t^2}]     * sin(2*pi*t/T + phi)
```

Each multiplies the carrier by a non-oscillating envelope governed by a single decay parameter `a`.
The notebooks do not implement these; they are recorded here for completeness and traced to Bretthorst
(see [06-glossary-and-sources.md](06-glossary-and-sources.md)).
