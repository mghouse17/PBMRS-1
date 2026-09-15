# Review — 02_wti_regime_calibration.ipynb

Executed notebook read in full (40 cells, 17 code cells, execution counts 1–17 strictly increasing,
zero error outputs, 9 exhibits, kernel `pbmrs-1-venv`). Cross-read against the GMSG deck
("Hormuz Reignites, Warsh Holds Firm", Daksh Singh, 31 Jul 2026).

---

## What it found

**Empirical (FRED DCOILWTICO, 2 Jan – 31 Jul 2026, 145 closes / 144 returns)**
- 8 April return −17.52%, at grid position 64.
- Daily SD 4.18% (95% CI 3.27–5.07%); excess kurtosis 2.63 (0.59–4.83).
- Brent: SD 4.53%, excess kurtosis 1.39 (0.07–3.23).
- Lag-7 ACF(r²) = 0.240.

**Adequacy (D-sum, 8 lags, 1,000 calibration / 4,999 null, α = 0.10)**
- WTI full sample: rejected at J = 0.30 … 0.80 (p 0.025–0.084); non-rejected at J = 0.82, 1/β, 0.85.
- Those three non-rejections sit at 28.0%, 50.8% and 78.8% pathological path rates. The 1%
  eligibility rule admits only J ≤ 0.78.
- WTI with 8 April masked in position: **all 15 points non-rejected** (p 0.205–0.576).
- Brent reproduces both patterns. `D_max` and median/IQR agree.

**Power** — 29.6% at T = 144 (25.8–33.7%); 47.4% / 69.2% / 88.0% / 99.0% at 250 / 500 / 1,000 /
2,000. The 80% lower-bound crossing is bracketed at 500–1,000 sessions.

**Controls** — positive: 9.4% and 11.4% rejection at nominal 10%, both intervals covering nominal.
Negative: Gaussian noise non-rejected at every J.

**21-session regime contrast** — J = 0.78 vs J = 0.40: MDD p95 27.3% vs 15.6%, mean terminal log
return −5.15% vs −0.69%, mean ending drawdown 9.31% vs 5.28%, recovery-after-5%-drawdown 5.88% vs
7.47%.

---

## What works

**The high-J instability trap was caught and handled correctly.** This is the best thing in the
notebook. The full-sample table non-rejects exactly at J = 0.82, 1/β and 0.85 — which reads, on its
face, as *the confidence set for J is the near-critical regime, exactly as the deck claimed*. It
isn't. Those points are 28–79% pathological; blown-up paths widen the null and manufacture
admissibility. The notebook screens stability **before** inference, shades the ineligible region in
Exhibit 5, and states in the title cell that the higher non-rejected points are pathological and
support no near-critical estimate. A version that missed this would have shipped a fabricated
confirmation of the hypothesis under test.

**Provenance is genuinely institutional.** SHA-256 and retrieval timestamps on every raw payload,
cache-key validation that fails loudly, a seed ledger with disjoint blocks, code hash, git hash,
pinned environment. Assertions on `alpha_r`, `__version__`, positions, calibration/null counts and
seed ranges execute visibly in cell 2. This is the part most undergraduate submissions do not have
at all.

**The fixed-grid masking correction mattered more than I estimated.** On real data, lag 7 goes
0.240 → 0.071 under positional masking versus −0.011 under compressed deletion — a gap of 0.082,
roughly six times what my synthetic test suggested. Under the correct treatment the April pair's
influence is *reduced by 70%, not eliminated*. Getting this wrong would have overstated the
collapse.

**Test calibration is verified, not asserted** (9.4% / 11.4% against nominal 10%), and the power
curve is the honest core of the notebook.

---

## What doesn't work

### 1. The regime contrast cuts against the deck, and the notebook doesn't say so

This is the substantive gap. The deck's forward call is specific:

> "The more likely path into Q3 is another fast round-trip, not a durable break to a higher
> plateau."

The notebook's own horizon results, at the two regimes:

| | J = 0.40 | J = 0.78 |
|---|---|---|
| MDD p95 | 15.6% | 27.3% |
| mean terminal log return | −0.69% | −5.15% |
| mean ending drawdown | 5.28% | 9.31% |
| recovery after ≥5% DD | 7.47% (23/308) | 5.88% (20/340) |

Two things follow, and neither appears in the text:

- **Recovery rates are statistically indistinguishable** (z = 0.81) **and both are under 8%.**
  PBMRS essentially never produces a round-trip within 21 sessions, at either regime.
- **Higher J produces deeper drawdowns that stay down** — ending drawdown and terminal return both
  worsen. In model terms the near-critical regime looks less like "fast round-trip" and more like
  "durable break," in the *opposite* direction from the deck's call.

The build plan explicitly anticipated this case — "if recovery results do not differ materially,
explicitly state that PBMRS does not reproduce that portion of the deck's language" — and the
notebook then omits it. Cell 30 ("What the evidence licenses") discusses adequacy and power and
never mentions recovery. Exhibit 9's title quotes only the MDD p95 change, which is the one metric
that *looks* supportive.

This is a one-markdown-cell fix and it makes the notebook stronger, not weaker: a result that
declines to flatter the hypothesis is the most credible thing you can show a research group.

### 2. The masked non-rejection is indistinguishable from white noise

Masked WTI is non-rejected at every J (p 0.205–0.576). Gaussian noise at matched SD is also
non-rejected at every J (p 0.148–0.249). The appendix reports the negative control but never
connects it to the main result. The masked non-rejection therefore carries no information about
mechanism — it is what this test does to any series without a large influential pair. That sentence
belongs next to Exhibit 5, not in the appendix.

### 3. The test is blind to the model's clearest failure

WTI excess kurtosis is 2.63 (CI 0.59–4.83). PBMRS with Gaussian innovations produces approximately
zero. The 8-lag ACF statistic never looks at tail thickness, so the dimension where model and data
most obviously diverge is excluded by construction. Scoping to one predeclared statistic is correct
practice, but it means "non-rejected" is considerably weaker than it sounds. Limitations mentions
Gaussian innovations; it does not make this point.

### 4. The deck's actual mechanism is never tested

The deck's claim is that *thin positioning* implies a *particular forward path*. CFTC positioning
is explicitly excluded from inference ("contextual only and never an input to J inference"), and
correctly so. But that means the notebook tests whether an agent model's ACF profile can match
WTI's — a different question from the deck's. The notebook implies this; it should say it in one
sentence, because a reader will otherwise assume the deck was tested and found wanting.

### 5. Smaller items

- **Headline wording.** The title cell says the test "rejects the stable grid through J=0.80", but
  J=0.80 is 8.2% pathological and fails the notebook's own 1% rule. The stable grid ends at J=0.78.
- **Exhibit 5** shades from 0.78 upward as "above 1% stability eligibility" while Exhibit 1's title
  calls J=0.78 "just eligible" at exactly 1.0%. The two exhibits disagree about whether 0.78 is in
  or out.
- **Brent is not the same window.** `brent.log_returns[-144:]` takes Brent's final 144 returns;
  different holiday calendars mean it does not start on WTI's start date. Disclosed, but
  "replication" is doing slightly loose work.

---

## Is it a good indicator?

**As a market indicator, no — and the notebook is right to say so.** Three independent reasons, all
established inside the notebook itself:

1. Power at the actual sample length is 29.6%. The test fails to discriminate the declared regime
   contrast roughly seven times in ten.
2. The verdict is decided by a single trading session. Include 8 April and every eligible point is
   rejected; mask it and every point is non-rejected.
3. Once that session is masked, the test cannot distinguish WTI from Gaussian noise.

There is no configuration of this analysis that produces a tradeable or forecastable signal, and
the notebook makes no such claim.

**As a work sample, it is strong** — for the reasons in "What works," and particularly because the
instability screen caught a result that would have falsely confirmed the hypothesis.

---

## Do the findings support GMSG's research?

**No — and it is important to be precise about the three different senses in which that is true.**

1. **Not supported.** Nothing in the notebook bears on the deck's mechanism. The deck argues from
   positioning to a forward path; the notebook asks whether a fixed-parameter agent model can
   reproduce WTI's squared-return autocorrelation. These are different objects, and the positioning
   data is deliberately excluded from the inference.

2. **Not refuted either.** The adequacy rejection is a statement about PBMRS at `base.yaml`, one J
   at a time, under one selected statistic. It says nothing about whether Daksh's positioning read
   was correct. The deck could be entirely right and this result would be unchanged.

3. **One result points the other way.** The 21-session contrast is the only part of the notebook
   touching the deck's substance, and it does not cooperate: recovery is rare and regime-invariant
   (5.9% vs 7.5%, indistinguishable), while ending drawdown and terminal return both worsen at
   higher J. If PBMRS's near-critical regime resembles anything in the deck's vocabulary, it is a
   durable break lower — not the fast round-trip that was called.

**How to carry this into the application.** The honest framing is that this is a *methodology* work
sample, not a *findings* work sample. Its value is that you built a test, discovered it had no
power at the available sample length, found an instability artifact that would have manufactured a
confirmation of the thesis, and reported all of it without spin. Research groups select for exactly
that. But deliver it as the point rather than as an apology — and add the recovery finding, because
"my model disagrees with the deck's round-trip language, here is the number" is a far more
interesting thing to discuss in an interview than another non-rejection.

---

## Suggested edits, in priority order

1. Add a markdown cell after Exhibit 9 reporting the recovery contrast (5.88% vs 7.47%, z = 0.81,
   not distinguishable) and stating plainly that PBMRS does not reproduce the deck's round-trip
   language at either regime. Update Exhibit 9's title so it does not quote only MDD.
2. Move one sentence of the Gaussian negative control next to Exhibit 5: masked non-rejection is
   what this test does to any series without an influential pair.
3. Add one sentence to "What the evidence licenses" stating that the deck's positioning mechanism
   is not what was tested.
4. Fix "the stable grid through J=0.80" → J=0.78, and reconcile the Exhibit 1 / Exhibit 5
   disagreement about whether 0.78 is inside the eligible region.
5. Add one line to Limitations noting that the selected statistic excludes tail behaviour, where
   the measured mismatch (2.63 vs ≈0 excess kurtosis) is largest.
