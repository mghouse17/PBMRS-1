# Pivot Plan — Commodities → FX

## Why this is the right move, in one paragraph

Measured, not asserted. At EURUSD volatility (σ_ε ≈ 0.0045, a 0.39× rescale from base) PBMRS runs
with **0% pathological paths at both J = 0.40 and J = 0.78**, and the structural flow channel rises
to **24–41% of return variance** against ~20% at base and effectively nothing at oil's scale. The
failure mode that killed the commodities work — needing to scale *up* into numerical instability —
reverses on FX: you scale *down*, which is free, and the herding mechanism explains more rather than
less. The near-critical regime becomes testable for the first time in this project.

---

## Phase 0 — Decisions and the one blocking dependency

**Blocking: get GMSG's latest FX report.** The commodities notebook's single best feature was that
it tested a claim from the group's own published deck. Without the FX equivalent you lose that, and
the work sample becomes a generic model demonstration. Get it before anything else — site, a current
member, or ask when you submit.

**Confirm the application mechanics.** Whether team preference is stated on the form, whether
applications are still open, and whether you can express a first and second choice.

**Pick your scope.** Honest costs:

| Option | Work | What it is |
|---|---|---|
| **A. Addendum** | ~2 days | Submit the commodities notebook plus 2–3 FX exhibits showing the stability/flow-share evidence and a preliminary FX fit. Narrative: "I tested my tool on your commodities deck, it failed, I diagnosed why, here is the asset class where it works." |
| **B. Full rebuild** | ~1 week | Rebuild the whole analysis on FX with a multi-year window. Strictly better work sample; costs a week you may not have. |
| **C. Submit as-is** | ~1 day | Commodities notebook to the FX team unchanged, with a paragraph on why FX is next. Weakest targeting. |

**A is the best value** unless your deadline is comfortable, in which case B produces materially
better work — see Phase 3 for why.

---

## Phase 1 — Data layer

Most of `commodities.py` transfers unchanged: `fetch_fred_series`, `bootstrap_statistic`,
`acf_r2_profile`, `stationary_bootstrap_indices`, `reconcile`, the manifest/caching contract, all of
`calibration.py`, all of `pbmrs_core`. What changes is narrow.

**FRED FX series.** Same CSV endpoint, same header conventions, same `.` missing-value handling as
the oil series — the parser needs no changes. Likely IDs: `DEXUSEU` (USD/EUR), `DEXJPUS` (JPY/USD),
`DEXUSUK` (USD/GBP), `DEXCAUS`, `DEXSZUS`. **Verify each rather than trusting this list** — same rule
as the CFTC dataset IDs last time.

**CFTC FX futures positioning.** COT covers EUR, JPY, GBP futures, so the positioning pipeline
transfers — but **discover the contract codes from the catalogue; I do not have verified values** and
a wrong code produces a plausible-looking wrong series. The four definition variants you already
test (legacy/disaggregated × futures-only/combined) apply identically.

**Quoting convention matters.** `DEXJPUS` is JPY per USD while `DEXUSEU` is USD per EUR — the sign
of "carry crowding" flips between them. Fix a convention explicitly and assert it, because a
sign error here silently inverts the positioning interpretation.

**Window: multi-year, not 145 sessions.** This is the single biggest upgrade available and it is
free — FRED FX history runs to 1999.

---

## Phase 2 — Recalibration

1. Re-run the stability screen at FX scale. Preliminary (250 paths, T=2000): σ_ε = 0.0045 gives
   0% pathological at J = 0.40 and J = 0.78. Extend across the full J grid and confirm where the
   boundary actually sits — it should be much higher than the J ≈ 0.80 wall you hit on oil.
2. Set σ_ε from the target pair's realized daily SD rather than a fixed constant, and record it.
3. Re-verify test calibration at the new scale — the positive control (rejection rate ≈ nominal α on
   true-model pseudo-data) must be re-run; do not assume it carries over from the commodities
   configuration.
4. Re-measure excess kurtosis. **Expect this to still fail** — the Gaussian innovation defect is
   scale-invariant, and FX majors carry excess kurtosis of roughly 2–5. This limitation survives the
   pivot and should be carried forward honestly, not quietly dropped.

---

## Phase 3 — The analysis commodities could not support

This is what makes B worth a week.

**The power problem dissolves.** Your own curve put 80% power between 500 and 1,000 sessions. Six
years of daily FX is ~1,500. For the first time the test can actually discriminate, which means a
*positive* result is possible rather than another honest null.

**But a long window buys a new problem: non-stationarity.** J is not constant over six years, so a
single global estimate is the wrong object. That points at the right design:

**Rolling-window J estimation.** Estimate J over a rolling 250-session window across the sample and
plot the path. The question becomes: *does estimated J rise ahead of carry unwinds?* That is a real,
falsifiable, potentially positive finding, and it is exactly what a fragility model should be able to
do if it works at all.

**Event study on carry unwinds.** These are the canonical reflexive cascades — crowded carry,
positioning extreme, then a violent unwind. That is PBMRS's mechanism in the wild, unlike oil where
the dominant driver was physical supply shocks the model excludes by construction. Identify the
episodes in your window, and test whether estimated J is elevated in the run-up.

**Genuine out-of-sample validation.** Fit on data through a cutoff, validate after. With decades of
history you can do this repeatedly rather than the single post-hoc check you managed on WTI.

**The honest risk to state up front:** FX has its own exogenous-shock problem — central bank
intervention and peg breaks (SNB January 2015 is the extreme case). An endogenous-fragility model
will miss those the same way it missed the Saudi pipeline. The difference is one of degree: FX
volatility is more positioning-driven than oil's, not purely so.

---

## Phase 4 — Notebook and application

Structure carries over almost entirely from `02_wti_regime_calibration.ipynb`: stability-before-
inference, adequacy with Wilson intervals and borderline flags, per-lag influence anatomy, controls,
power, horizon contrast, full provenance. Keep the discipline; swap the subject.

**Fix the five items from the notebook review first** — they apply to any rebuild: report the
recovery contrast, connect the Gaussian negative control to the main result, state that the deck's
mechanism is not what was tested, correct "stable grid through J=0.80" → 0.78, and note that the
selected statistic excludes tail behaviour.

**Carry the commodities work forward rather than discarding it.** "I built this on your commodities
report, it failed, here is precisely why, and here is the measured evidence that FX is where the tool
belongs" is a stronger application narrative than a clean FX result with no history. It demonstrates
that you follow evidence instead of sunk cost, which is the rarer trait.

---

## Critical path

1. Obtain the GMSG FX report *(blocking — start today)*
2. Confirm application status and team-selection mechanics
3. Decide A / B / C against your actual deadline
4. Verify FRED FX series IDs and discover CFTC FX contract codes
5. Stability screen at FX scale across the full J grid
6. Re-verify test calibration at the new scale
7. Build the analysis; rolling-J is the centerpiece if you take option B
8. Apply the five notebook-review fixes
9. Execute, verify, commit
