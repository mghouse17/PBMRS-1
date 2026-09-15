# Out-of-Sample Test — the deck's forward call against realized August–September

**Status: preliminary.** Levels below come from public reporting, not a pulled series. This
sandbox blocks FRED/Yahoo/EIA/Stooq by egress policy, so Claude Code must pull the actual daily
series and regenerate every number. The direction is unambiguous; the decimals are not yet.

---

## The claim being tested

From the deck's "Forward-Looking Tail Risk" bullet, 31 July 2026:

> Thin positioning cuts both ways: it **limits how far a rally can extend**, but limits how much
> needs to unwind if tensions ease. **The more likely path into Q3 is another fast round-trip, not
> a durable break to a higher plateau.**

Two separable claims. An **outcome** claim (round-trip, not a durable break) and a **mechanism**
claim (thin positioning caps rally extension). They fail differently and should be scored
separately.

Deck marks at 31 July: WTI $83.59, Brent $89.03. Net spec length 0.065 on 28 July, rebuilding off
a 0.033 cycle low — thin by the deck's own characterisation.

---

## What happened

| Date | Event | WTI | vs 31 Jul |
|---|---|---|---|
| 2026-09-01 | closes above $90 | ~$90.00 | +7.7% |
| 2026-09-08 | closes above $94 | ~$94.00 | +12.5% |
| 2026-09-11 | **Saudi East–West pipeline shut** after drone attacks from Iraq | — | — |
| 2026-09-14 | "nears $108–110" on supply fears | ~$108 | ~+29% |
| 2026-09-15 | spot | $101.89 | **+21.9%** |

Monthly gain +20.6%, year-on-year +57.9%, described as a four-month high.

**Round-trip test.** A round-trip is a large excursion with the terminal value back near the start.
A durable break is a terminal value at or near the excursion. WTI never returned toward $83.59 at
any point in the window and sits 21.9% above it. This is the durable break to a higher plateau —
the branch the deck named as *less* likely.

---

## The timing detail that decides how much this counts

**The first +12.5% happened before the pipeline attack.** WTI closed above $90 on 1 September and
above $94 on 8 September; the Saudi East–West pipeline was shut on 11 September.

That matters enormously for scoring the call fairly:

- Through 8 September — roughly the 21-session horizon the notebook models, and squarely inside
  the deck's "into Q3" window — WTI was up 12.5% **with no new shock of that magnitude**, and had
  not round-tripped at any point. The call was already failing on its own terms.
- The pipeline attack explains the final leg ($94 → $108 → $102). It does not explain the first.

So the exogenous-shock defence covers part of the miss but not the part that matters for judging
the forecast.

**The mechanism claim failed more cleanly than the outcome claim.** "Thin positioning limits how
far a rally can extend" is a structural assertion, not a probabilistic one. Positioning was thin
(0.065, just off a cycle low) and the rally extended 21.9%. Thin positioning did not cap it.

---

## PBMRS would have been wrong too, in the same direction

Against the notebook's J = 0.78 forward distribution (mean terminal 21-session log return −0.0515,
per-step SD 0.0123 → horizon SD 0.0564):

| Date | realized log return | z vs model |
|---|---|---|
| 2026-09-01 | +0.0739 | **+2.22** |
| 2026-09-08 | +0.1174 | **+3.00** |
| 2026-09-15 | +0.1980 | **+4.43** |

The model's near-critical regime pointed at persistent drawdowns. Realized was a sustained rally
three to four and a half standard deviations into its right tail. **Both the published call and the
simulator missed, and missed the same way** — neither contains a mechanism for an exogenous supply
shock, which is limitation (5)/(7) in the notebook restated against out-of-sample data instead of
in-sample.

## The scale finding replicates out-of-sample

| Date | z under PBMRS | z under WTI's own realized vol (21-session SD 19.2%) |
|---|---|---|
| 2026-09-01 | +2.22 | +0.39 |
| 2026-09-08 | +3.00 | +0.61 |
| 2026-09-15 | +4.43 | +1.03 |

A 3–4σ event for the model is a 0.6–1.0σ event for actual WTI. That is the same ~3.7× volatility
understatement the in-sample work found, now confirmed on data the model never saw. This is
arguably the most robust result in the entire project: it holds in-sample, out-of-sample, and by
two independent routes.

---

## What this does and does not establish

**Does:** the specific forward call did not come to pass; the stated mechanism (thin positioning
caps rally extension) was contradicted directly; PBMRS's forward distribution was badly
miscentred and mis-scaled; and the volatility understatement replicates out-of-sample.

**Does not:** refute the deck's reasoning as a probabilistic statement. "More likely" is a
distributional claim and one realization is weak evidence against it — if the round-trip had 60%
probability, observing the other branch once is unremarkable. It also does not establish that
anyone should have foreseen a drone attack on Saudi infrastructure on 11 September. Nobody
forecasts that.

The honest scoring: **the outcome claim lost one draw and should be treated as such; the mechanism
claim failed on the evidence.**

---

## How to put this in the work sample

This needs care, because you are submitting it to the group whose analyst made the call.

**Frame it as forecast evaluation, not as correction.** Testing published calls against realized
outcomes is standard research practice and the single most valuable habit a research group can
have. Lead with that.

**Give the fairness points prominence, not a footnote.** The probabilistic-claim caveat, the
11 September pipeline attack, and the fact that this is one realization all belong in the main
text, before the scoring.

**Lead with the symmetry.** The strongest version of this section is not "the deck was wrong." It
is: *both the published call and my simulator missed the August–September move, in the same
direction, for the same reason — neither represents exogenous supply shocks. That shared failure is
the finding.* That is generous, accurate, and considerably more interesting than a scorecard.

**Then it earns the next step.** The natural conclusion writes itself: a jump or heavy-tail
innovation is not a generic improvement to PBMRS, it is the specific missing piece that both the
model and the narrative needed, identified by two independent tests.

---

## For the build

1. Pull FRED `DCOILWTICO` and `DCOILBRENTEU` for 2026-08-01 → present; commit with the same
   SHA-256/manifest contract as the existing cache.
2. Recompute the table above from the real series; the milestone levels here are for direction only.
3. Add a round-trip metric symmetric with Exhibit 9's: max favourable excursion vs terminal return,
   applied identically to realized data and to model paths. Reusing `compute_horizon_stats` on the
   realized path makes the comparison exact rather than rhetorical.
4. Mark 11 September on the chart and split the window at it, so the pre-shock and post-shock legs
   are separately visible.
5. Report both z-columns — model-scaled and WTI-scaled — since the gap between them is the finding.
