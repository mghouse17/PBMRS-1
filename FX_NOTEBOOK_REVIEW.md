# Review — 03_fx_regime_calibration.ipynb + GMSG FX deck

Notebook read in full (32 cells, 15 code cells, execution counts 1–15 strictly increasing, zero
errors, 10 exhibits). Deck read: "FX — Asia Becomes Dollar Pressure Point", Varen Desai & Hojin Choi,
31 July 2026.

---

## Execution quality: excellent, and better than the WTI build

- **Pre-registration with cryptographic proof.** `registered_at_utc < started_at_utc <
  computed_at_utc` asserted in cell 1, with a design SHA-256. That is stronger discipline than most
  published applied work, and it is the notebook's most valuable asset.
- **All twelve specifications reported**, primary flagged, nulls and exclusions in Appendix A. No
  specification hunting is possible from this record.
- **Calibration re-verified at FX scale**: 8/8 controls cover nominal 10% (rates 0.078–0.108). Not
  assumed from the WTI work, as the brief required.
- **σ_ε solved, not assigned**: JPY 0.005456 → matched SD 0.006195 against target 0.006183
  (0.19% relative error); EUR 0.50%.
- **Minimum-distance estimator, not `argmax p`** — stated explicitly in Exhibit 5's prose. This was
  the trap I was most worried about and it was avoided cleanly.
- **Quote convention handled**: both series converted to USD-per-foreign-unit, JPY inverted from
  `DEXJPUS`, stated in prose. The sign trap was avoided.
- **Kurtosis gap reported**: JPY 4.21 [3.10, 5.44] vs model ≈0.002; EUR 1.83 vs ≈0.002.

---

## The headline problem: nothing ran

**All 12 specifications returned `insufficient_events`. Zero effect estimates. Zero p-values.**

The primary JPY test labelled 5 validation episodes and matched only 2, below the 3-episode
minimum. Every secondary, every EUR falsification, both training and validation segments: the same.

The notebook is right that "insufficient events is not a statistically tested null." But the
Conclusion then says *"A null leaves the hypothesis unsupported under this design."* Those are
different things and the conclusion blurs them. **There is no null here — the test never executed.**
The hypothesis is neither supported nor refuted; the design failed to produce a testable sample.
Say that precisely, because a reviewer who reads "null" will assume you tested and found nothing,
which is a stronger claim than you earned.

---

## The buried lead: your detector found the deck's headline event and threw it away

This is the most important thing in the review.

Appendix A's exclusion table, JPY validation, third row:

```
date 2026-08-04 | interval_start 2026-07-28 | net_short_before 0.2359
net_short_fall 0.0909 | appreciation_z 3.44 | reason: insufficient prehistory or matched controls
```

The GMSG FX deck, slide 23, is entirely about that window:

> First joint US-Japan FX intervention since 2011: Japan bought yen unilaterally on July 30 (~$59B),
> and the U.S. Treasury joined on July 31 … snapping from ¥163 to ¥158 in under an hour …
> **Vol confirms the surprise:** 1-month implied vol spiked to its high for the month, consistent
> with **a market caught offside** rather than repricing gradually.

**Your episode detector independently identified the deck's centrepiece event from CFTC positioning
and price data alone, with no knowledge of the report — and then discarded it for want of matched
controls.** It is sitting in an appendix exclusion table.

That is a genuine validation of the labelling procedure and it belongs in the main line. It also
gives you the deck connection the notebook says it lacks: the header currently reads *"GMSG's FX
report was unavailable: no claim from an unseen report is attributed to the group."* That is no
longer true, and the deck is unusually well-matched to this work — "caught offside" is a
crowded-positioning claim, which is a fragility assertion in words, and your positioning data
independently corroborates the setup (net short 0.236 before the interval, falling 0.091 through it).

**The honest framing, since the trigger was an intervention:** PBMRS has no mechanism for a policy
shock, so it cannot speak to the trigger — the same limitation as the Saudi pipeline in the WTI work.
What it can speak to is whether the *setup* was fragile. The deck asserts it was; your positioning
series corroborates that independently; and the model could not test it because the episode was
excluded. That is a clean, honest, interesting paragraph and it costs no new machinery.

---

## The cause is diagnosable, and it is structural rather than bad luck

Three of five validation episodes were excluded for "insufficient prehistory or matched controls":
2020-03-10 (z=4.08), 2026-01-27 (z=2.25), 2026-08-04 (z=3.44).

The matching rule requires five non-episode dates **in the same calendar year** with log-volatility
difference ≤0.25, excluding dates within 42 sessions of an episode.

**That criterion fights the event definition.** Episodes are, by construction, high-volatility
periods. Requiring volatility-matched controls from the same calendar year means searching for calm
comparators inside the year whose volatility the episode itself defines. For 2020 and for events near
the end of the sample (2026, where the year is only ~9 months long) the pool is close to empty. This
will recur on any re-run; it is not a one-off.

**Fixes, in order of how much they cost:**

1. Widen the control pool to ±1 year rather than same-year.
2. Match on **pre-event** volatility (the 250-session SD already computed for the z-score) rather
   than contemporaneous volatility, which removes the circularity directly.
3. Use a caliper — nearest available controls within a distance budget — instead of a hard ≤0.25 cut
   with a fixed count of five.

### But you cannot simply apply these and re-run as primary

Pre-registration is the notebook's strongest feature. Changing the design after seeing that it
produced no result and then presenting the new version as primary destroys exactly what makes this
work credible — and a CFA-charterholder reader will spot it immediately.

**The correct move:** report the pre-registered result unchanged as primary (`insufficient_events`),
then add a clearly labelled **post-hoc exploratory arm** with the widened matching, stating plainly
that it was specified after observing the primary failure and that its p-values do not carry
pre-registered interpretation. That is honest, it is standard practice, and it gives the notebook an
actual number while preserving the integrity of the registered test.

---

## Two other findings

**The T=250 rolling sets carry almost no information.** The Gaussian negative control is
non-rejected in **80.2%** of T=250 windows and 48.4% of T=500 windows. The notebook reports this and
says it "limits the meaning of broad confidence sets," which is right but understated — at T=250,
noise passes four times in five, so the shaded confidence bands in Exhibit 5 are close to
uninformative at that horizon. Say so at the exhibit rather than only in the preamble, and consider
whether T=250 earns its place in the main line at all.

**The 5% drawdown threshold is inherited from WTI and mis-scaled.** At JPY's daily SD of 0.59%, a
21-session SD is 2.70%, so a 5% drawdown is a 1.85σ event — versus 0.26σ at WTI's volatility. The
result is 148–234 qualifying paths but only 2–5 recoveries, giving rates of 1.4–2.1% with heavily
overlapping intervals. The notebook flags the small denominators; the underlying cause is that the
threshold was not rescaled with the asset. Set it in σ units rather than percentage points.

---

## Assessment as a work sample

Technically this is the best-executed piece of the three. Pre-registration with hash verification,
full specification disclosure, verified calibration, the correct estimator, careful language
throughout — it reads like someone who has internalised how research goes wrong.

**But it currently contains no findings.** Every effect column is `None`. The WTI notebook at least
produced real discoveries — the instability artifact, the single-session sensitivity, the volatility
understatement. This one has a procedural non-result and a scale table carried over from the brief.

A reviewer will ask what you found. Right now the answer is "the test could not be run," which
demonstrates method without demonstrating judgment about markets. The three fixes above — surface the
deck event, add the labelled exploratory arm, sharpen the null language — would change that
materially, and none of them require touching the simulator.

**Priority order:**

1. Surface the 2026-08-04 episode into the main line beside the deck's intervention slide. Highest
   value, lowest cost, no new computation.
2. Update the header — the FX report is no longer unavailable.
3. Add the post-hoc exploratory arm with widened matching, clearly labelled as post-hoc.
4. Fix the Conclusion's "null" language to "the test did not execute."
5. Move the Gaussian 80.2% implication next to Exhibit 5; reconsider T=250 in the main line.
6. Rescale the drawdown threshold to σ units.
