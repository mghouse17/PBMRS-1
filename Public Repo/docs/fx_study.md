# FX companion study

The companion notebook retains the commodity failure as the motivation for testing
the same PBMRS engine at FX volatility. Stability at a lower scale is a feasibility
finding; empirical carry-specific explanatory power remains a separate hypothesis.

## Frozen design

`configs/fx_study.json` was copied into a timestamped, hash-identified registration
before data analysis. Primary: JPY, 500 observed sessions per estimate, monthly
steps, 21 prior sessions per event score, one-sided alpha 0.10, evaluation from
2018 onward. JPY 250-session results, EUR, training contrasts, power and horizons
are secondary. Every specification and excluded event appears in results.json.
This is a local timestamped declaration, not an independently registered protocol.

The registered event study produced five held-out JPY labels but only two matched
episodes, so its test did not execute. Its design, registration and `results.json`
remain unchanged. A separately frozen continuous design uses 500-session rolling
minimum-distance estimates, 21-session steps, and the full 2003-2026 sample. It
regresses next-21-session stress on J_hat and `log(current realized volatility)`;
the volatility control is required because otherwise the association would merely
restate volatility clustering. The primary outcome is forward realized volatility;
maximum drawdown and absolute terminal return are secondary. Inference uses 4,999
circular moving-block pairs resamples with a 24-step block.

## Data and quote convention

FRED DEXJPUS is JPY per USD and is inverted. DEXUSEU is already USD per EUR.
Positive returns therefore mean foreign currency appreciation for both pairs.
The original CSV observations and their hashes remain unchanged. H.10 attribution
is to the Federal Reserve Board via FRED; these are noon buying rates. Missing
observations are dropped without forward-filling.

The brief's commodity disaggregated categories do not apply to FX. CFTC's actual
four-way comparison is legacy non-commercial and TFF leveraged money, each in
futures-only and futures/options-combined form. Dataset metadata and exact CME
market names identify the contracts; codes are discovered rather than supplied
as constants. Contract-specific directories avoid the old fetcher's filename
collision across different contracts. No fetch, bootstrap or ACF code is copied.

Official sources:

- https://fred.stlouisfed.org/series/DEXJPUS
- https://fred.stlouisfed.org/series/DEXUSEU
- https://publicreporting.cftc.gov/stories/s/r4w3-av2u
- https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm

## Calibration and interpretation

The numerical solve matches mean simulated path SD to pre-2018 empirical SD at
J=0.40. Fixing this reference resolves the under-specified instruction to solve
one sigma per pair even though model SD depends on J. Common random numbers make
the solve repeatable. Tolerance is 0.5%, with 100 paths and 2,000 evaluation steps.
Sigma is frozen for later data; nuisance calibration uncertainty is not included
in the inversion sets.

The screen uses 500 paths, 500 burn-in steps and 2,000 evaluation returns at every
declared J. Pathology includes invariant failures, non-finite returns and SD above
5 times innovation sigma. Eligibility is the observed fraction at most 1%; Wilson
intervals remain visible. J=0.78 supplies the regime contrast if eligible, otherwise
the highest eligible tested J is substituted. Finite pathological paths are not
removed from any eligible adequacy bank.

Each bank contains 500 calibration and 999 disjoint null paths. D-sum sums squared
lag deviations standardized by calibration mean and SD, floored at 1e-12. The
minimum D selects the point estimate; a larger p-value never selects it. Plus-one
p-values form pointwise 90% inversion sets over the eligible discrete grid.
Empty and disconnected sets are retained. These sets are conditional on fixed
nuisance parameters and exclude uncertainty in the model specification.

Positive controls use 500 independent pseudo-paths, at two J values, both windows
and both pairs. Gaussian controls run through the same bank. Power is remeasured
at FX scales for every declared horizon; prior commodity power is not reused.
Seeds repeat across J/pairs/horizons for common random numbers, so Monte Carlo
errors are correlated. Calibration/null/pseudo ranges do not overlap.

This power calculation concerns a fixed-J model contrast, not the power of the
matched-event test. Squared-return ACF is invariant to a global sign reversal;
quote orientation matters for event labels and horizon interpretation, even
though it cannot change that ACF statistic.

## Events and validation

Short covering increases (long-short)/OI. The study defines a decline in net SHORT
exposure of at least 5 percentage points, from at least 5% prior net short exposure,
plus foreign-currency appreciation above 1.5 prior daily SD times the square root
of interval sessions. Volatility uses 250 observations before that interval.
Events are at least 42 sessions apart; report gaps above 10 calendar days are skipped.

The score averages the most recently available monthly J estimate at each of the
21 sessions before the event interval begins. It never includes event-interval
returns. Matching uses five non-event dates in the same calendar year with
log-volatility distance at most 0.25, outside a 42-session exclusion radius.
Sparse unmatched episodes are disclosed; fewer than three matched episodes yields
no test. The label-permutation null randomizes the episode within each matched
set and recomputes event-minus-control scores. Confidence intervals resample year
clusters. This is an observational, approximate exchangeability test: serially
overlapping windows and reused controls prevent a claim of randomized inference.

The primary comparison uses held-out dates since 2018. EUR on those same JPY event
dates is the direct falsification; EUR's own labels are a reported secondary check.
Annual post-split summaries show sequential evaluation with a fixed initial nuisance
calibration. An expanding-window nuisance refit was not implemented or claimed.
COT labels refer to Tuesday observations generally released on Friday, with holiday
delays. Because matching is retrospective and data are a current vintage, this is
not an implementable live trading strategy or a historical release-vintage backtest.

The detector independently identifies the deck's 28 July-4 August 2026 intervention
window, with prior net short exposure 0.2359, a 0.0909 reduction and appreciation
z=3.44. The registered matcher excludes it for insufficient controls. The notebook
surfaces this beside GMSG's "market caught offside" discussion while distinguishing
the fragile setup from the external policy trigger, which PBMRS does not model.

A post-hoc descriptive arm widens the pool to plus/minus one calendar year, retains
prior 250-session volatility, and chooses up to five nearest controls inside a 0.50
log-volatility caliper. It matches all five episodes. Its interval and permutation
p-value are explicitly exploratory and cannot replace the registered result.

## Continuous result

The continuous design was timestamped and hashed before coefficient estimation.
JPY, EUR and a JPY-SD-matched independent Gaussian series each yield 259 stepped
estimates. The primary JPY slope is 0.000806 with a 95% block-bootstrap interval
[-0.001393, 0.003311]. The interval includes zero: at this scale the statistic adds
no statistically resolved information beyond volatility clustering. EUR and
Gaussian primary intervals also include zero, as do every secondary-outcome
interval. Wide intervals are expected because the overlapping predictor windows
represent only about twelve independent 500-session spans.

## Reproduction and limitations

The raw, simulation and result payloads have byte hashes. Source hashes normalize
LF/CRLF only for code portability; raw hashes always use exact bytes. The notebook
checks code, design, data and simulation manifests before loading results. Archived
reconnaissance files and unrelated root research notes are not inputs or staged.

Large per-dataset CFTC schema dumps and the failed federated-catalogue response are
discovery-only artifacts and are ignored. Their original retrieval hashes remain in
the historical manifest and cache key. Offline discovery uses the compact CFTC views
snapshot; parsing each cached positioning payload validates the required fields.

Continuous FRED files live under `notebooks/fx_continuous_cache/raw`, separate from
the registered event-study cache. This prevents the longer 2003 history from
changing the event study's exact raw-input hash set. The continuous manifest checks
its design, sources, exact raw bytes, original event-design/result dependencies,
and output bytes.

The historical scale/flow-share opening table is explicitly attributed to the user
brief, which supplied no underlying paths/seeds. It is not presented as a fresh
verified run. Every FX finding is newly computed. Gaussian tails, omitted policy
and peg-break shocks, daily-step assumptions, finite-grid limits, weak carry labels,
and exclusion of tail behavior from the selected statistic remain explicit.
The GMSG FX report is now used for claim-specific framing. Its intervention account
is not treated as a model mechanism or a uniquely translated PBMRS hypothesis.
