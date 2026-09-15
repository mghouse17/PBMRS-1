# Design — a test that cannot return "insufficient events"

## Diagnosis

Nothing is wrong with the machinery. The problem is the **estimand**.

The current primary asks: *is Ĵ elevated before labelled short-covering episodes, versus
volatility-matched controls?* That requires discrete, rare, correctly-labelled events — and every
filter in the chain costs sample:

| constraint | cost |
|---|---|
| split at 2018 | 65% of history spent on training |
| net short ≥5% OI **and** weekly fall ≥5pp **and** appreciation ≥1.5σ | very few windows qualify |
| 42-session separation | clustered episodes collapse to one |
| 5 same-year vol-matched controls, ≤0.25 log distance | fails precisely when episodes occur |
| ≥3 matched episodes to report | 2 is not 3 |

Each is individually defensible. Multiplied, they leave **2 usable events from 23 years of data**.
That is why all 12 specifications returned nothing, and it will recur on any re-run.

**A candid note on how we got here.** Every round of review in this project pushed toward more
conservatism — stability screens, pre-registration, multiple-comparison discipline, careful language.
Each was right in isolation. Cumulatively they produced a design optimised to never yield a false
positive, at the cost of never yielding any positive. A test that cannot produce a result is not
rigorous; it is inert. The correction is not to abandon the discipline, it is to point it at an
estimand that uses the whole sample.

---

## The fix: a continuous association, not an event study

Same question — *does the fragility statistic precede stress?* — asked so that **every rolling
window is an observation** rather than only the handful that clear an episode filter.

```
rolling 500-session window, 21-session step, 2003–2026  ->  ~252 estimates
forward outcome = realized stress over the NEXT 21 sessions, stepped 21 -> non-overlapping outcomes
```

There is no configuration in which this returns "insufficient events."

### The specification

**Predictor:** `Ĵ_t` — the minimum-distance estimate already implemented, on the window ending at
`t`. Keep the confidence set as a secondary descriptor; the point estimate is what enters the test.

**Outcomes** (run all three, declare one primary):
- forward 21-session realized volatility
- forward 21-session maximum drawdown
- forward 21-session absolute terminal return

**The control that makes this non-trivial.** `Ĵ` is estimated from the ACF of squared returns, so it
is mechanically related to volatility. "High Ĵ predicts high forward vol" could be nothing but
volatility clustering, which is a known fact about every financial series and would be worthless as a
finding.

So the primary test is: **does Ĵ predict forward stress after controlling for current realized
volatility?**

```
forward_stress_t = a + b·Ĵ_t + c·log(realized_vol_t) + e_t
```

`b` is the estimand. If `b > 0` with a credible interval excluding zero, the fragility statistic
carries information beyond vol clustering — a real and publishable result. If `b ≈ 0`, Ĵ is a
repackaged volatility measure — also a real result, cleanly stated, and far more informative than
"insufficient events."

**Secondary, more legible presentation:** sort windows into Ĵ quintiles *within* volatility terciles,
and tabulate mean forward stress. Same content, immediately readable by a non-technical reviewer.

### Inference

Overlapping predictor windows mean the naive n = 252 overstates precision. Forward outcomes are
non-overlapping by construction, but `Ĵ_t` is highly persistent — roughly **12 effectively
independent window-spans** across the sample.

Use a **block bootstrap with block length ≈ 24 steps** (one window length) and report the resulting
interval. Expect it to be considerably wider than the naive one. That is the honest cost of a
persistent predictor, and stating it is the point — the test still produces a number.

### Falsification, carried over

- **EUR** run identically. If `b` is the same in EUR as JPY, the statistic is not carry-specific.
- **Gaussian control**: generate matched-SD noise, run the whole pipeline, confirm `b ≈ 0`. This is
  the sharp version of the negative control — much more informative than the current non-rejection
  rate, because it tests the *association* rather than set width.

---

## Keep the event study; demote it

Do not delete it. Report it as a secondary descriptive exhibit with its two matched episodes, and
note that 2026-08-04 — the deck's intervention window — was detected and excluded. The detection is
worth showing even though the test could not run.

---

## Pre-registration

This is a **new design**, not a repair of the failed one. So it can be pre-registered cleanly:
freeze the specification, hash it, timestamp it, then run — exactly the discipline already
implemented in `fx_study.json`. Nothing is laundered, because the new estimand was not chosen by
looking at results from the old one; it was chosen because the old one had no sample.

State that explicitly in the notebook: the event study is reported unchanged as registered, and the
continuous test is a separately registered design motivated by the event study's sample failure.

---

## What a null means here, and why it is worth having

If `b ≈ 0` after the volatility control, the conclusion is: *this fragility statistic, at this
scale, on this pair, adds nothing beyond volatility clustering.* That is a genuine, quantified,
defensible finding about the model — and it is the finding PBMRS has been circling all project
without ever quite stating.

It is also actionable: it points directly at whether the herding channel, which the scale work
showed contributes 24–41% of simulated return variance, leaves any measurable trace in real data.

Either way you get a number with an interval, which is what the last three notebooks have been
missing.
