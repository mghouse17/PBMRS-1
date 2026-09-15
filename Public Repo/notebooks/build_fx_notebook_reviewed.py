"""Build reviewed notebook 03 without changing registered event-study artifacts."""

from pathlib import Path
from textwrap import dedent

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def build():
    cells = []

    def md(text):
        cells.append(nbf.v4.new_markdown_cell(dedent(text).strip()))

    def code(text, exhibit=None):
        cell = nbf.v4.new_code_cell(dedent(text).strip())
        if exhibit is not None:
            cell.metadata.update(tags=["main-exhibit"], exhibit=exhibit)
        cells.append(cell)

    md("""
    # PBMRS on FX: does the fragility statistic add information beyond volatility?
    **GMSG application work sample - FX coverage | Salman**

    This notebook was reviewed against GMSG's available FX deck, *FX - Asia Becomes Dollar
    Pressure Point* (31 July 2026). The deck's "market caught offside" language is being
    operationalized as a crowded-positioning/fragility claim; it is not a unique translation
    into PBMRS parameters.

    The original registered event study remains unchanged and is reported exactly as registered.
    It did not execute because only two of five held-out episodes found matched controls. A new,
    separately registered continuous test therefore asks whether rolling minimum-distance J
    predicts forward stress after controlling for current realized volatility. The estimand
    changed because the old design had no testable sample, not because its results were unfavourable.
    """)
    code("""
    import hashlib, json
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import display, Markdown
    from pbmrs_core import __version__, SimConfig, load_config
    from pbmrs_core.calibration import load_npz_cache, wilson_interval
    from pbmrs_core.fx import fetch_fx, discover_definitions, fetch_fx_positioning, validate_raw_tree
    from pbmrs_core.fx_analysis import load_study
    from pbmrs_core.fx_continuous import load_continuous_study

    candidates = [Path.cwd(), Path.cwd().parent, Path.cwd() / "Public Repo"]
    ROOT = next(p for p in candidates if (p / "configs/fx_study.json").exists())
    CACHE = ROOT / "notebooks/fx_cache"
    CONTINUOUS_RAW = ROOT / "notebooks/fx_continuous_cache/raw"
    DECK = ROOT / "reference/FX.pdf"
    frozen, sim, manifest, result = load_study(ROOT)
    continuous_frozen, continuous_manifest, continuous = load_continuous_study(ROOT)
    design, continuous_design = frozen["design"], continuous_frozen["design"]
    cfg = SimConfig(**load_config(ROOT / "configs/base.yaml")["simulation"])
    assert __version__ == "0.2.2" and cfg.alpha_r == 12.0
    assert frozen["registered_at_utc"] < sim["started_at_utc"] < manifest["computed_at_utc"]
    assert continuous_frozen["registered_at_utc"] < continuous_manifest["started_at_utc"] < continuous_manifest["computed_at_utc"]
    assert continuous["registered_event_design_sha256"] == frozen["design_sha256"]
    series = {pair: fetch_fx(pair, cache_dir=CACHE / "raw", start=design["start"], end=design["end"])
              for pair in ("JPY", "EUR")}
    primary = next(row for row in result["specifications"] if row["primary"])
    primary_continuous = next(row for row in continuous["regressions"]
                              if row["series"] == "JPY" and row["outcome"] == continuous_design["primary"]["outcome"])
    plt.rcParams.update({"figure.figsize": (11, 4), "axes.spines.top": False,
                        "axes.spines.right": False, "axes.grid": True, "grid.alpha": .18,
                        "font.size": 10, "axes.titlesize": 12})
    colors = {"JPY": "#147d92", "EUR": "#c56a2d", "Gaussian": "#777777"}
    display(pd.DataFrame([
        {"design": "registered event study", "registered_at_utc": frozen["registered_at_utc"],
         "sha256": frozen["design_sha256"], "verified": True},
        {"design": "registered continuous test", "registered_at_utc": continuous_frozen["registered_at_utc"],
         "sha256": continuous_frozen["design_sha256"], "verified": True},
    ]))
    print("Deck reference:", DECK, "| present:", DECK.exists())
    display(Markdown(f"**Registered event study:** {primary['status']} - {primary['n_labelled']} labelled, "
                     f"{primary['n_matched']} matched. The test did not execute; this is not a tested null.**"))
    """)
    md("""
    ## Designs and interpretation

    The event design uses a held-out 2018 split and remains the primary result under its original
    question. Its configuration, hash, timestamp, twelve specifications, exclusions, and result
    bytes are unchanged. The continuous design is a new question frozen before its coefficients
    were computed: 500-session rolling estimates, 21-session steps, the full 2003-2026 sample,
    and the next 21 sessions as a non-overlapping outcome window.

    The continuous primary estimand is `b` in
    `forward realized volatility = a + b*J_hat + c*log(current realized volatility) + error`.
    Maximum drawdown and absolute terminal return are secondary. Circular moving-block pairs
    bootstrap intervals use 4,999 resamples and 24-step blocks. Roughly twelve independent
    500-session spans make wide intervals expected, despite 259 reported estimates per series.
    """)
    md("""
    ## Exhibit 1 - Why move from crude to FX?

    These prior scale measurements came from the supplied build brief; they are motivation rather
    than newly replicated results. Stability at FX scale makes inference feasible but does not
    establish that J measures carry fragility.
    """)
    code("""
    prior = pd.DataFrame(design["historical_scale_evidence"]["rows"])
    display(prior)
    ax = prior.set_index("analogue")[["pathological_J040", "pathological_J078"]].plot.bar(
        color=["#8ba7b3", "#c56a2d"], rot=0)
    ax.set(ylabel="Prior pathological fraction", title="Prior scale experiment: FX regimes were numerically usable")
    ax.legend(["J = 0.40", "J = 0.78"]); plt.show()
    """, 1)
    md("""
    ## Exhibit 2 - New stability screen at solved FX scales

    Each point uses 500 paths and 2,000 post-burn returns. Eligibility requires an observed
    pathological fraction no greater than 1%; Wilson intervals retain uncertainty. `J*beta=1`
    is an agent-layer reference, not a proven boundary for the coupled system.
    """)
    code("""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, pair in zip(axes, ("JPY", "EUR")):
        table = pd.DataFrame(sim["stability"][pair])
        ax.errorbar(table.J, table.fraction,
                    yerr=np.maximum(0, [table.fraction-table.lo, table.hi-table.fraction]),
                    fmt="o-", color=colors[pair], capsize=3)
        ax.axhline(.01, color="grey", ls="--", label="1% cutoff")
        ax.axvline(1/cfg.beta, color="black", ls=":", label="Agent-layer reference")
        ax.set(title=f"{pair}: sigma_eps={sim['scale'][pair]['sigma_eps']:.5f}",
               xlabel="Tested J", ylabel="Pathological fraction")
        ax.legend(fontsize=8)
    plt.tight_layout(); plt.show()
    display(pd.DataFrame({"pair": ["JPY", "EUR"],
                          "eligible_tested_J": [sim["eligible"]["JPY"], sim["eligible"]["EUR"]]}))
    """, 2)
    md("""
    ## Exhibit 3 - Market context and the deck's independently detected intervention window

    Both return series are USD per foreign-currency unit. JPY is inverted from
    [FRED DEXJPUS](https://fred.stlouisfed.org/series/DEXJPUS); EUR uses
    [DEXUSEU](https://fred.stlouisfed.org/series/DEXUSEU). CFTC TFF leveraged-money positioning
    supplies retrospective labels but never enters J inference.

    [The GMSG deck's intervention discussion](../reference/FX.pdf#page=23) describes the
    30-31 July joint US-Japan intervention, USDJPY moving from about 163 to 158 in under an hour,
    and volatility consistent with "a market caught offside." Without using that narrative, the
    detector identified the corresponding 28 July-4 August interval: prior net short exposure
    was 0.2359 of open interest, fell 0.0909, and appreciation was 3.44 prior-volatility standard
    units. PBMRS has no policy-shock mechanism, so this corroborates the fragile setup?not the
    intervention trigger. The registered matcher then excluded it for insufficient controls.
    """)
    code("""
    deck_event = next(event for event in primary["excluded"] if event["date"] == "2026-08-04")
    assert deck_event["interval_start"] == "2026-07-28"
    assert np.isclose(deck_event["net_short_before"], .2359, atol=5e-5)
    assert np.isclose(deck_event["net_short_fall"], .0909, atol=5e-5)
    assert np.isclose(deck_event["appreciation_z"], 3.44, atol=.005)
    display(pd.DataFrame([{k: deck_event[k] for k in
                          ["interval_start", "date", "net_short_before", "net_short_fall",
                           "appreciation_z", "reason"]}]))
    definitions = discover_definitions(CACHE / "raw")
    definition = next(d for d in definitions if d.key == design["positioning_primary"])
    cot, _ = fetch_fx_positioning(definition, "JPY", cache_dir=CACHE / "raw",
                                  start=design["start"], end=design["end"])
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(series["JPY"].dates, series["JPY"].close/series["JPY"].close[0], color=colors["JPY"])
    axes[1].plot(cot.dates, -cot.net_over_oi, color=colors["JPY"], lw=.8)
    for ax in axes: ax.axvspan(pd.Timestamp("2026-07-28"), pd.Timestamp("2026-08-04"), color="#d7a928", alpha=.25)
    axes[0].set(ylabel="USD value / first observation", title="FRED / Federal Reserve Board H.10 JPY orientation")
    axes[1].set(ylabel="Net short / OI", title="CFTC TFF leveraged money; detected deck window shaded")
    plt.tight_layout(); plt.show()
    """, 3)
    md("""
    The policy action is external to the simulator, just as a physical supply shock is external
    in the WTI application. The useful model question is narrower: did the pre-existing return
    dynamics carry stress information beyond observable volatility?
    """)
    md("""
    ## Exhibit 4 - Empirical fingerprint and tail gap

    The inference statistic is the ACF of demeaned squared returns at lags 1-8. Pointwise 95%
    stationary-bootstrap intervals use 1,000 draws and mean block length 21. SD and excess
    kurtosis are separate diagnostics; ACF adequacy does not test the tails.
    """)
    code("""
    fig, ax = plt.subplots()
    rows = []
    for pair in ("JPY", "EUR"):
        f = result["pairs"][pair]["fingerprint"]; x = np.arange(1, 9)
        ax.plot(x, f["point"][:8], "o-", label=pair, color=colors[pair])
        ax.fill_between(x, f["lo"][:8], f["hi"][:8], alpha=.15, color=colors[pair])
        rows.append({"pair": pair, "n_returns": result["pairs"][pair]["n_returns"],
                     "daily_SD": f["point"][8], "excess_kurtosis": f["point"][9],
                     "kurtosis_lo": f["lo"][9], "kurtosis_hi": f["hi"][9]})
    ax.set(xlabel="Lag in observed sessions", ylabel="ACF(r^2)", title="Empirical volatility persistence with pointwise bootstrap bands")
    ax.legend(); plt.show(); display(pd.DataFrame(rows))
    display(pd.DataFrame([{"pair": pair, **row} for pair in ("JPY", "EUR")
                          for row in result["pairs"][pair]["kurtosis"]]))
    """, 4)
    md("""
    ## Exhibit 5 - Rolling JPY estimates and why T=250 is nearly uninformative

    J_hat minimizes standardized D-sum; it is never selected by the largest p-value. Shaded points
    are exact non-rejected grid values, not a continuous-J interval. Empty and disconnected sets
    remain visible. Monthly steps make predictor windows overlap.

    The matched Gaussian control is non-rejected somewhere in **80.2% of T=250 windows** versus
    **48.4% of T=500 windows**. At T=250, independent noise passes roughly four times in five, so
    those broad inversion sets carry little discriminatory information. T=250 remains visible only
    because it was declared; the continuous test uses T=500.
    """)
    code("""
    def draw_rolling(ax, pair, T):
        rows = result["pairs"][pair]["rolling"][str(T)]
        dates = pd.to_datetime([row["date"] for row in rows])
        for day, row in zip(dates, rows):
            ax.scatter([day]*len(row["confidence_set"]), row["confidence_set"], marker="s",
                       s=13, alpha=.17, color=colors[pair])
        ax.plot(dates, [row["J_hat"] for row in rows], color=colors[pair], lw=1)
        empty = np.array([row["empty_set"] for row in rows])
        ax.scatter(dates[empty], [sim["eligible"][pair][0]]*empty.sum(), marker="x", color="black", s=20)
        ax.axvline(pd.Timestamp(design["split"]), color="grey", ls="--")
        ax.set(ylabel="Tested J", title=f"{pair}, T={T}: {empty.sum()}/{len(rows)} windows reject every eligible point")
    gaussian_rates = pd.DataFrame(result["gaussian"])
    check = gaussian_rates[gaussian_rates.pair == "JPY"].set_index("window").any_non_rejection_rate
    assert np.isclose(check.loc[250], .802) and np.isclose(check.loc[500], .484)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for ax, T in zip(axes, design["windows"]): draw_rolling(ax, "JPY", T)
    plt.tight_layout(); plt.show()
    display(gaussian_rates[["pair", "window", "n", "any_non_rejection_rate"]])
    """, 5)
    md("""
    ## Exhibit 6 - New continuous test: no resolved JPY information beyond volatility

    Every 500-session window contributes an observation, stepped by 21 sessions. The next 21
    returns never overlap across adjacent rows. The regression controls for `log(current realized
    volatility)` because J is estimated from squared-return dependence; without that control,
    any association would merely restate volatility clustering.

    The primary JPY coefficient is reported without re-specification. EUR is run identically as a
    falsification, and one JPY-SD-matched independent Gaussian series traverses the same rolling
    estimator, outcomes, regression, and bootstrap. Intervals are block-bootstrap percentile
    intervals, not naive iid intervals.
    """)
    code("""
    outcome = continuous_design["primary"]["outcome"]
    slopes = pd.DataFrame([row for row in continuous["regressions"] if row["outcome"] == outcome])
    slopes["b_per_0.1J_daily_vol_bp"] = slopes.b_J_hat * 1000
    slopes["lo_per_0.1J_bp"] = slopes.b_lo * 1000
    slopes["hi_per_0.1J_bp"] = slopes.b_hi * 1000
    slopes["resolved_beyond_zero"] = ~((slopes.b_lo <= 0) & (slopes.b_hi >= 0))
    display(slopes[["series", "n", "b_J_hat", "b_lo", "b_hi", "b_per_0.1J_daily_vol_bp",
                    "lo_per_0.1J_bp", "hi_per_0.1J_bp", "block_length_steps",
                    "rank_deficient_resamples_redrawn", "resolved_beyond_zero"]])
    x = np.arange(len(slopes)); fig, ax = plt.subplots()
    ax.errorbar(x, slopes.b_J_hat, yerr=[slopes.b_J_hat-slopes.b_lo, slopes.b_hi-slopes.b_J_hat],
                fmt="none", ecolor="#555555", capsize=5)
    ax.scatter(x, slopes.b_J_hat, c=[colors[s] for s in slopes.series], s=55)
    ax.axhline(0, color="black", lw=1); ax.set_xticks(x, slopes.series)
    ax.set(ylabel="b: daily forward-volatility change per 1.0 J",
           title=f"JPY b={primary_continuous['b_J_hat']:.6f}, 95% block interval [{primary_continuous['b_lo']:.6f}, {primary_continuous['b_hi']:.6f}]")
    plt.show()
    display(Markdown("**Finding:** at this scale, the JPY statistic adds no statistically resolved "
                     "information beyond volatility clustering. EUR and Gaussian intervals also include zero. "))
    """, 6)
    md("""
    ## Exhibit 7 - Secondary stress outcomes and volatility-stratified presentation

    Maximum drawdown and absolute terminal return were frozen as secondary outcomes. They cannot
    replace the primary forward-volatility result. The descriptive table sorts J_hat into quintiles
    within current-volatility terciles; it is a legibility aid, not a second inferential test.
    """)
    code("""
    secondary = pd.DataFrame([row for row in continuous["regressions"]
                              if row["outcome"] != continuous_design["primary"]["outcome"]])
    display(secondary[["series", "outcome", "n", "b_J_hat", "b_lo", "b_hi",
                       "c_log_current_volatility", "block_length_steps"]])
    quintiles = pd.DataFrame(continuous["quintiles"]["JPY"])
    display(quintiles.pivot(index="volatility_tercile", columns="J_hat_quintile",
                            values="mean_forward_stress"))
    panel = pd.DataFrame(continuous["series"]["JPY"]["panel"])
    fig, ax = plt.subplots()
    points = ax.scatter(panel.current_realized_volatility, panel.forward_realized_volatility,
                        c=panel.J_hat, cmap="viridis", alpha=.75)
    fig.colorbar(points, ax=ax, label="J_hat")
    ax.set(xlabel="Current 500-session realized volatility", ylabel="Next-21-session realized volatility",
           title="JPY: forward stress clusters with volatility; J colour adds no resolved slope")
    plt.show()
    """, 7)
    md("""
    ## Exhibit 8 - Registered event study, demoted to a descriptive result

    The registered JPY test remains `insufficient_events`: five held-out labels, two matches, and
    therefore no effect interval or permutation p-value. It did not execute; there is no null result.

    A clearly post-hoc arm widens the calendar pool to +/-1 year, uses prior 250-session volatility,
    and selects up to five nearest controls inside a 0.50 log-volatility caliper. It matches all five
    labels, including the deck event. Because these rules were chosen after observing sample failure,
    its interval and p-value are exploratory and have no pre-registered interpretation.
    """)
    code("""
    fields = ["pair", "window", "segment", "n_labelled", "n_matched", "effect", "lo", "hi", "p_value", "status"]
    display(pd.DataFrame([{key: primary.get(key) for key in fields}]))
    if primary["matched"]:
        display(pd.DataFrame(primary["matched"])[["date", "score", "difference", "pre_event_empty_set_fraction"]])
    exploratory = continuous["post_hoc_exploratory_event"]
    display(pd.DataFrame([{k: exploratory.get(k) for k in
                          ["status", "n_labelled", "n_matched", "effect_descriptive", "lo", "hi", "p_value",
                           "n_permutation", "n_bootstrap"]}]))
    explore_rows = pd.DataFrame(exploratory["matched"])
    display(explore_rows[["interval_start", "date", "n_controls", "max_selected_log_vol_distance", "difference"]])
    deck_match = explore_rows.loc[explore_rows.date == "2026-08-04"].iloc[0]
    assert deck_match.n_controls == 5
    fig, ax = plt.subplots()
    ax.bar(explore_rows.date, explore_rows.difference, color=colors["JPY"])
    ax.axhline(0, color="grey", lw=1); ax.tick_params(axis="x", rotation=35)
    ax.set(ylabel="Event minus matched-control J",
           title=f"Post-hoc only: mean={exploratory['effect_descriptive']:.3f}, p={exploratory['p_value']:.3f}")
    plt.show()
    """, 8)
    md("""
    ## Exhibit 9 - Fixed-split registered evaluation remains visible

    The event study's nuisance calibration and grid were frozen before 2018. Later rolling estimates
    use information available by each endpoint, but current-vintage data and retrospective matching
    make this neither a live signal nor an archived-vintage backtest.
    """)
    code("""
    walk = pd.DataFrame(result["walk_forward"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, pair in zip(axes, ("JPY", "EUR")):
        for T in design["windows"]:
            sub = walk[(walk.pair == pair) & (walk.window == T)]
            ax.plot(sub.year, sub.empty_set_fraction, "o-", label=f"T={T}")
        ax.tick_params(axis="x", rotation=45)
        ax.set(title=pair + ": fraction rejecting every eligible J", ylabel="Empty-set fraction")
        ax.legend()
    plt.tight_layout(); plt.show()
    display(pd.DataFrame([{k: row.get(k) for k in fields} for row in result["specifications"]
                          if row["pair"] == "JPY" and row["window"] == 500]))
    """, 9)
    md("""
    ## Exhibit 10 - FX-scale calibration and fixed-J power

    Eight true-model controls assess nominal 10% size; all Wilson intervals cover nominal. Power
    concerns a fixed-J contrast, not either association test. Common random numbers induce correlated
    Monte Carlo errors and are not independent replications.
    """)
    code("""
    controls = pd.DataFrame(result["controls"]); power = pd.DataFrame(result["power"])
    display(controls)
    fig, ax = plt.subplots()
    for pair in ("JPY", "EUR"):
        sub = power[power.pair == pair]
        ax.errorbar(sub["T"], sub.power, yerr=[sub.power-sub.lo, sub.hi-sub.power],
                    fmt="o-", color=colors[pair], label=pair, capsize=3)
    ax.axhline(.8, ls="--", color="grey")
    ax.set(xlabel="Observed sessions", ylabel="Rejection probability",
           title="FX-scale point-contrast power; 500 pseudo-samples")
    ax.legend(); plt.show()
    display(Markdown(f"Controls whose 95% interval misses nominal size: **{int((~controls.covers_nominal).sum())}/{len(controls)}**."))
    """, 10)
    md("""
    ## Exhibit 11 - Twenty-one-session regime contrast with a volatility-scaled drawdown threshold

    The old 5% recovery threshold was inherited from WTI and is too severe at FX scale. This main
    presentation uses one pair-specific 21-session volatility unit:
    `training daily SD * sqrt(21)`. It reclassifies the already cached horizon summaries and does
    not change registered event results or rerun the simulator. Maximum drawdown, ending drawdown,
    recovery, and terminal returns describe unconditional simulated regimes?not an FX forecast.
    """)
    code("""
    def cached(name):
        path = CACHE / name
        payload = load_npz_cache(path, sim["cache_keys"][name])
        assert payload is not None and hashlib.sha256(path.read_bytes()).hexdigest() == sim["cache_sha256"][name]
        return payload
    horizon_rows = []
    for pair in ("JPY", "EUR"):
        threshold = sim["scale"][pair]["target_sd"] * np.sqrt(21)
        for J in (0.4, sim["comparison_J"][pair]):
            h = cached(f"{pair}_horizon_J{J:.8f}.npz")["horizon"]
            qualifying = h[:, 0] >= threshold
            recovered = qualifying & np.isfinite(h[:, 3])
            lo, hi = wilson_interval(int(recovered.sum()), int(qualifying.sum()))
            horizon_rows.append({"pair": pair, "J": J, "threshold_1sigma": threshold,
                                 "n": len(h), "mdd_mean": h[:,0].mean(), "mdd_p95": np.percentile(h[:,0],95),
                                 "ending_drawdown_mean": h[:,2].mean(), "terminal_log_return_mean": h[:,1].mean(),
                                 "qualifying": int(qualifying.sum()), "recovered": int(recovered.sum()),
                                 "recovery_rate": recovered.sum()/qualifying.sum(), "recovery_lo": lo, "recovery_hi": hi})
    horizons = pd.DataFrame(horizon_rows); display(horizons)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    labels = [f"{row['pair']} J={row['J']:.2f}" for row in horizon_rows]
    for ax, metric, title in zip(axes, ["mdd_p95", "ending_drawdown_mean", "terminal_log_return_mean"],
                                  ["95th-percentile MDD", "Mean ending drawdown", "Mean terminal log return"]):
        ax.bar(labels, horizons[metric], color=[colors[row["pair"]] for row in horizon_rows])
        ax.tick_params(axis="x", rotation=35); ax.set_title(title)
    plt.tight_layout(); plt.show()
    """, 11)
    md("""
    ## Conclusion and limits

    The registered event test **did not execute**. Its hypothesis is neither supported nor refuted
    under that design; the post-hoc widened match cannot replace it. The detector nevertheless found
    the deck's intervention window independently, corroborating a crowded setup while saying nothing
    about the policy trigger.

    The separately registered continuous test does execute. For JPY, `b=0.000806` with a 95%
    24-step block-bootstrap interval `[-0.001393, 0.003311]`. At this scale and with this statistic,
    J adds no statistically resolved forward-volatility information beyond volatility clustering.
    EUR and Gaussian intervals also cross zero; secondary stress outcomes do not overturn the primary.

    Wide intervals are part of the finding: 259 stepped estimates arise from only about twelve
    independent 500-session spans. Gaussian innovations omit intervention and jump mechanisms; the
    empirical kurtosis gap remains. Fixed nuisance scale, finite J grid, current data vintage, daily
    timing assumptions, and CFTC's incomplete view of OTC carry remain material limitations.
    """)
    code("""
    summary = pd.DataFrame([row for row in continuous["regressions"]
                            if row["outcome"] == continuous_design["primary"]["outcome"]])
    summary["interval_contains_zero"] = (summary.b_lo <= 0) & (summary.b_hi >= 0)
    display(summary[["series", "n", "b_J_hat", "b_lo", "b_hi", "c_log_current_volatility",
                     "interval_contains_zero"]])
    assert summary.interval_contains_zero.all()
    """)
    md("""
    ## Appendix A - All registered event specifications and exclusions

    Nothing below was altered by the continuous test. All twelve specifications, null fields,
    matched episodes, and exclusions come directly from the original hashed `results.json`.
    """)
    code("""
    display(pd.DataFrame([{k: row.get(k) for k in ["primary", "label_pair", *fields]}
                          for row in result["specifications"]]))
    excluded = [{"pair": row["pair"], "window": row["window"], "segment": row["segment"],
                 "label_pair": row["label_pair"], **event}
                for row in result["specifications"] for event in row["excluded"]]
    display(pd.DataFrame(excluded))
    """)
    md("""
    ## Appendix B - Complete checks and legacy 5% horizon classification

    Stability, scale-solve traces, power, walk-forward summaries, and the original 5% recovery
    classification remain visible for full disclosure. The latter is not used in the main exhibit.
    """)
    code("""
    display(pd.concat([pd.DataFrame(sim["stability"][pair]).assign(pair=pair)
                       for pair in ("JPY", "EUR")], ignore_index=True))
    display(pd.concat([pd.DataFrame(sim["scale"][pair]["trace"]).assign(pair=pair)
                       for pair in ("JPY", "EUR")], ignore_index=True))
    display(pd.DataFrame(result["power"])); display(pd.DataFrame(result["horizons"]))
    """)
    md("""
    ## Appendix C - Reproduction, hashes, seeds and provenance

    The event registration/results and continuous registration/results have independent hashes and
    ordered timestamps. The continuous raw cache is separate, so adding its 2003-2009 observations
    cannot alter event-study input validation. Raw hashes use exact bytes; source hashes normalize
    only line endings. Existing caches support offline execution.

    Rebuild the continuous artifacts with `register_fx_continuous.py` once, then
    `build_fx_continuous_results.py`, followed by `build_fx_notebook_reviewed.py`. Re-running the
    registration script verifies the frozen design and refuses a mismatch.
    """)
    code("""
    display(pd.DataFrame([{"design": "event", "registered": frozen["registered_at_utc"],
                           "design_sha256": frozen["design_sha256"], "results_sha256": manifest["results_sha256"]},
                          {"design": "continuous", "registered": continuous_frozen["registered_at_utc"],
                           "design_sha256": continuous_frozen["design_sha256"],
                           "results_sha256": continuous_manifest["results_sha256"]}]))
    display(pd.DataFrame([{"file": name, **entry} for name, entry in validate_raw_tree(CONTINUOUS_RAW).items()]))
    display(pd.DataFrame([{"series": name, **{k:v for k,v in values.items() if k != "panel"}}
                          for name, values in continuous["series"].items()]))
    display(pd.DataFrame([{"name": key, "value": value}
                          for key, value in continuous_design["bootstrap"].items()]))
    print("Event result bytes unchanged:", hashlib.sha256((CACHE / "results.json").read_bytes()).hexdigest())
    print("Continuous result:", continuous_manifest["results_sha256"])
    print("All design, source, raw-data, simulation-dependency and result hashes validated at start.")
    """)

    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {
        "display_name": "PBMRS (.venv)", "language": "python", "name": "pbmrs-1-venv"
    }
    notebook.metadata.language_info = {"name": "python"}
    target = ROOT / "notebooks/03_fx_regime_calibration.ipynb"
    nbf.write(notebook, target)
    print(f"Built {len(cells)} cells, 11 main exhibits: {target}")


if __name__ == "__main__":
    build()
