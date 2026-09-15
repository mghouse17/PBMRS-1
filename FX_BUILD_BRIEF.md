# Build Brief — PBMRS on FX (GMSG application, FX coverage)

**Hand this to Claude Code.** Launch from `C:\Users\Mghou\PBMRS-1`. Self-contained: assume no memory
of prior sessions.

---

## 1. What this is

Salman is applying to York's Global Macro Strategy Group, **FX coverage team**. The deliverable is a
Jupyter notebook work sample.

A previous build (`Public Repo/notebooks/02_wti_regime_calibration.ipynb`) applied PBMRS — the
agent-based market-fragility simulator in this repo — to WTI crude, testing a claim from GMSG's
commodities deck. **It produced an honest null and a diagnosis: PBMRS cannot be run at oil's
volatility, cannot reach its own critical point, and has no mechanism for the exogenous supply
shocks that dominate crude.** That notebook stays in the repo; this is a companion, not a
replacement.

This build moves the same machinery to FX, where measurement says the model actually works. The
narrative arc — *tested it on commodities, it failed, diagnosed why, here is the asset class where
the mechanism applies* — is deliberate and is a selling point. Do not hide the commodities result.

---

## 2. What exists and what transfers

Already in the repo and **reusable unchanged**:

- `pbmrs_core` — canonical package, version `0.2.2`, public imports (`from pbmrs_core import
  SimConfig, run_ensemble, phase_map, acf_r2_profile, load_config`). No `sys.path` hacking.
- `pbmrs_core/calibration.py` — `evaluate_profile_bank`, `wilson_interval`, `lag_contributions`,
  `load_npz_cache`, `save_npz_cache`, `sim_cache_key`, horizon stats. **Entirely asset-agnostic.**
- `pbmrs_core/commodities.py` — `fetch_fred_series`, `bootstrap_statistic`,
  `stationary_bootstrap_indices`, `acf_r2_profile`, `reconcile`, `read_manifest`, the
  SHA-256/manifest caching contract, `CFTCDefinition` and `fetch_cftc_positioning`.
- Chart styling, provenance pattern, cache-key discipline, seed-ledger pattern.

**What changes is narrow: the series IDs, the contract codes, σ_ε, the window length, and the
analysis design.** Do not rewrite the infrastructure.

**Module naming.** `commodities.py` is now misnamed for shared helpers. Either rename it
`marketdata.py` with a thin `commodities.py` shim (preferred), or add `fx.py` importing from it. Do
not duplicate the fetch/bootstrap/ACF code — a second copy that drifts is how the empirical and
simulated paths diverge.

---

## 3. Verified facts — do not re-derive, do not contradict

All measured against this exact codebase.

**Model behaviour by volatility scale** (250 paths, T=2000, pathological = return SD > 5·σ_ε):

| σ_ε | asset analogue | J=0.40 pathological | J=0.78 pathological | flow-share of variance |
|---|---|---|---|---|
| 0.0045 | EURUSD (~0.45%/day) | **0.0%** | **0.0%** | 24% / 41% |
| 0.0110 | SPX (~1.10%/day) | 0.0% | 4.8% | 20% / broken |
| 0.0418 | WTI (~4.18%/day) | 99.2% | 100.0% | meaningless |

This is the whole case for the pivot: **at FX scale the model is stable in the near-critical regime
for the first time, and the structural flow channel explains 24–41% of return variance instead of
being drowned by noise.** Reproduce this table as an early exhibit.

**Stability at base σ_ε = 0.010** (500 paths): J=0.75 → 0.0%, J=0.78 → 1.0%, J=0.80 → 8.2%,
J=0.82 → 28.0%, J=1/β → 50.8%, J=0.85 → 78.8%. The commodities notebook's 1% eligibility rule cut
between 0.78 and 0.80. **Re-run this screen at FX σ_ε — the boundary should move substantially
higher, and where it lands is a finding.**

**Power curve** (J=0.40 null vs J=0.78 truth, α=0.10, 500 pseudo-samples): T=144 → 29.6%,
T=250 → 47.4%, T=500 → 69.2%, T=1000 → 88.0%, T=2000 → 99.0%. **This is why the window must be
multi-year.** It also sets the rolling-window length (see §6).

**Known model defects that survive the pivot** — carry these into Limitations, do not quietly drop:
- Excess kurtosis ≈ 0 across the usable parameter range. FX majors run 2–5. The Gaussian innovation
  caps tail thickness and this is scale-invariant.
- `J·β = 1` is a mean-field *agent-layer* reference, not a demonstrated critical boundary of the
  coupled system.
- One simulator step = one trading session is an assumption, not a derivation.

---

## 4. Phase 1 — data layer

### FRED FX series

Same CSV endpoint, header conventions and `.` missing-value handling as the oil series. The parser
needs no changes.

Likely IDs — **verify each against FRED before use, do not trust this list**: `DEXUSEU` (USD per
EUR), `DEXJPUS` (JPY per USD), `DEXUSUK` (USD per GBP), `DEXCAUS`, `DEXSZUS`. History runs to 1999
for the majors.

**Pair selection, and this is a design decision not a convenience:**

- **Primary: JPY (`DEXJPUS`).** Carry unwinds are JPY-centric — JPY is the canonical funding
  currency, so crowded-carry cascades show up here most strongly. This is the pair where PBMRS's
  mechanism should be visible if it is visible anywhere.
- **Control: EUR (`DEXUSEU`).** Deep, liquid, but carry crowding is far weaker. **If the model finds
  the same fragility structure in EUR as in JPY, it is picking up noise, not carry dynamics.** Build
  this in as a falsification test, not a robustness afterthought.

**Sign convention is a real trap.** `DEXJPUS` is JPY *per USD*, so a carry unwind — JPY strengthening
— is a **fall** in the series. `DEXUSEU` is USD *per EUR*, the opposite orientation. Fix one
convention, assert it in code, and state it in prose. A silent inversion here flips every
positioning interpretation and will not announce itself.

### CFTC FX positioning

COT covers EUR, JPY and GBP futures, so `fetch_cftc_positioning` transfers. **Discover the contract
codes from the Socrata catalogue — there are no verified values in this brief, and a wrong code
returns a plausible-looking wrong series.** The same four definition variants
(legacy/disaggregated × futures-only/combined) apply; run the same reconciliation loop.

Positioning is **context and event-dating input only**. It is not a regressor in J inference. Keep
that boundary exactly as the commodities notebook did.

### Window

2010 → present at minimum (~4,000 sessions). Longer is better for power and worse for
stationarity — see §6.

---

## 5. Phase 2 — recalibration

1. **Set σ_ε per pair from realized volatility, and solve rather than assign.** The model's total
   return SD exceeds σ_ε because the flow term adds variance (at σ_ε = 0.0045 the measured SD was
   0.0052–0.0059). Target the *total* SD to the pair's empirical SD and solve for σ_ε numerically.
   Record the solved value in the manifest.
2. **Re-run the full stability screen** across the J grid at the solved σ_ε, 500 paths, T=2000.
   Report pathological fraction with binomial intervals. Apply the same 1% eligibility rule.
3. **Re-verify test calibration at the new scale.** Positive control: ≥300 true-model pseudo-samples
   at two J values, rejection rate against nominal α with intervals. **Do not assume the commodities
   calibration carries over** — it was measured at a different σ_ε.
4. **Re-measure excess kurtosis** and report the gap. Expect failure; report it.

---

## 6. Phase 3 — the analysis

This is the part that could not be done on commodities, and it is where a **positive** result
becomes possible for the first time. That cuts both ways — see the multiple-comparisons discipline
below, which is non-negotiable.

### 6a. Rolling-J estimation — the centerpiece

**Estimator discipline, read this carefully.** The commodities notebook correctly insisted that a
larger p-value does **not** mean better fit. `argmax_J p(J)` is therefore **not** a point estimator
and must not be used as one. Use two distinct objects:

- **Point estimate: minimum-distance.** `Ĵ = argmin_J D(a_emp; μ_J, σ_J)` — the J whose sampling
  distribution the empirical profile sits closest to, under the same standardised distance already
  implemented. This is standard minimum-distance estimation and is a legitimate estimator.
- **Uncertainty: the Neyman-inversion confidence set** already implemented — the non-rejected grid
  points at α.

Plot `Ĵ` as a line with the confidence set as a band. They are coherent: the point estimate is the
closest grid point, the band is everything not rejected.

**Window length is a bias–variance tradeoff and must be shown, not chosen silently.** From the power
curve, 250 sessions gives ~47% power and 500 gives ~69% for the reference contrast. Run both:
250 is responsive to regime change and noisy, 500 has power and lags. Step monthly (21 sessions).

### 6b. Carry-unwind event study

**Define episodes from data, not from a hardcoded list of remembered dates.** An episode is a window
where CFTC net positioning in the funding currency falls sharply while the currency moves adversely
by more than a threshold in SD terms. Pre-declare the thresholds before looking at the J path.

The test: **is `Ĵ` elevated in the K sessions before an episode, relative to matched non-episode
windows?** Report the effect with a confidence interval and a permutation null (shuffle episode
dates, recompute). Pre-declare K.

### 6c. Out-of-sample validation

Split at a fixed date. Estimate on the earlier segment, validate on the later. With this much
history an expanding-window walk-forward is feasible and stronger than a single split.

### 6d. Multiple-comparisons discipline — non-negotiable

The commodities notebook was credible because a positive result was never available to be
manufactured. Here it is. Therefore:

- **Pre-declare the primary test before running it** and record it in the manifest: one pair (JPY),
  one window length, one K, one α. Everything else is explicitly secondary.
- The EUR control is a falsification test, not a second chance at a result.
- Report every specification run, including the ones that produced nothing.
- If the primary test is null, **report it as null.** Do not promote a secondary specification into
  the headline. The single fastest way to destroy the credibility this project has built is to go
  hunting after an unfavourable primary result.

---

## 7. Phase 4 — notebook

`Public Repo/notebooks/03_fx_regime_calibration.ipynb`. Target ~9–11 main exhibits, appendices for
diagnostics, same visual language and provenance discipline as notebook 02.

Suggested spine:

1. **Why FX** — the σ_ε scale table from §3. This is the strongest opening available: the pivot was
   measured, not guessed.
2. **Stability at FX scale** — screen across J, eligibility rule, the boundary's new location.
3. **Data and positioning context** — pair, window, CFTC definition reconciliation, sign convention
   stated explicitly.
4. **Empirical fingerprint** — ACF(r²) profile with bootstrap bands, SD, kurtosis, JPY vs EUR.
5. **Rolling Ĵ with confidence band** — both window lengths.
6. **Carry-unwind event study** — the primary test, with its permutation null.
7. **EUR falsification control** — same pipeline, weaker carry, does the structure vanish?
8. **Out-of-sample walk-forward.**
9. **Power and limitations** — including the kurtosis defect and the intervention/peg-break analogue
   of the commodities exogenous-shock problem.

**Carry the five fixes from the notebook-02 review into this build** (they were identified and apply
generally): report the recovery/horizon contrast rather than MDD alone; put the Gaussian negative
control's explanation beside the main result rather than in an appendix; state plainly which claim
is and is not being tested; never describe a grid as "stable through J=X" when X fails the
eligibility rule; note in Limitations that the selected statistic excludes tail behaviour.

---

## 8. Traps

- **Do not reuse the commodities σ_ε, calibration, or stability results.** Everything is
  scale-dependent and must be re-measured.
- **Do not hardcode CFTC contract codes or trust the FRED IDs in §4 without checking.**
- **Do not let `argmax p(J)` become the point estimate.**
- **Do not duplicate `acf_r2_profile`.** One shared entry point for empirical and simulated paths —
  this is what guarantees they cannot diverge.
- **Do not forward-fill FX holidays.** Drop them, as the oil parser does; a forward-filled zero
  return contaminates ACF(r²).
- **Watch the JPY sign.** Assert it.
- **Do not bury the commodities failure.** It is the setup for this notebook's argument.

---

## 9. Acceptance criteria

- [ ] Notebook executes top to bottom from a clean kernel, no errors, execution counts strictly
      increasing, no `--allow-errors`
- [ ] σ_ε solved from realized volatility, value recorded in the manifest
- [ ] Stability screen re-run at FX scale with binomial intervals; eligibility rule applied
- [ ] Test calibration re-verified at FX scale (≥300 pseudo-samples, ≥2 J values)
- [ ] Point estimate is minimum-distance; no use of `argmax p(J)` as an estimator anywhere
- [ ] Confidence band is the Neyman-inversion set
- [ ] Both rolling-window lengths shown
- [ ] Primary test pre-declared in the manifest **before** results; every specification run reported
- [ ] EUR control present and interpreted as falsification
- [ ] Sign convention asserted in code and stated in prose
- [ ] Kurtosis gap reported, not omitted
- [ ] Raw FRED/CFTC payloads committed with URL, UTC timestamp, byte count, SHA-256
- [ ] Seed ledger, code hash, git hash, environment versions in the appendix
- [ ] Commodities result referenced, not hidden
- [ ] `git diff --cached --stat` shows only intended new files; stage explicit paths, never
      `git add -A`

---

## 10. Open dependency

**GMSG's latest FX report is not yet in hand.** Notebook 02's strongest feature was testing a claim
from the group's own published deck. Everything in Phases 1–3 proceeds without it — it is needed
only for framing in Phase 4. Sequence accordingly, and if it arrives, add a section testing one of
its stated claims the way notebook 02 did.
