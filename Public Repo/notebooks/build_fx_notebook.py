"""Build the ten-exhibit FX work sample; execute separately with nbconvert."""

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
    # PBMRS on FX: can a fragility statistic precede short-covering episodes?
    **GMSG application work sample - FX coverage | Salman**

    The companion [WTI study](02_wti_regime_calibration.ipynb) exposed a scale mismatch and
    unstable high-J non-rejections. Its April sensitivity weakened the original inference.
    That failure motivates this experiment. A lower FX volatility scale makes a new test
    feasible; feasibility alone does not establish that PBMRS measures carry fragility.

    This notebook tests one predeclared JPY association, displays EUR as a falsification,
    and keeps every secondary result visible. GMSG's FX report was unavailable: no claim
    from an unseen report is attributed to the group.
    """)
    code("""
    import json
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import display, Markdown
    from pbmrs_core import __version__, SimConfig, load_config
    from pbmrs_core.calibration import load_npz_cache
    from pbmrs_core.fx import fetch_fx, discover_definitions, fetch_fx_positioning, validate_raw_tree
    from pbmrs_core.fx_analysis import load_study

    candidates = [Path.cwd(), Path.cwd().parent, Path.cwd() / "Public Repo"]
    ROOT = next(p for p in candidates if (p / "configs/fx_study.json").exists())
    CACHE = ROOT / "notebooks/fx_cache"
    frozen, sim, manifest, result = load_study(ROOT)
    design = frozen["design"]
    cfg = SimConfig(**load_config(ROOT / "configs/base.yaml")["simulation"])
    assert __version__ == "0.2.2" and cfg.alpha_r == 12.0
    assert design["primary"] == frozen["design"]["primary"]
    assert design["n_stability"] == 500 and design["n_pseudo"] >= 300
    assert design["seeds"]["calibration"][1] < design["seeds"]["null"][0]
    assert frozen["registered_at_utc"] < sim["started_at_utc"] < manifest["computed_at_utc"]
    series = {pair: fetch_fx(pair, cache_dir=CACHE / "raw", start=design["start"], end=design["end"])
              for pair in ("JPY", "EUR")}
    assert series["JPY"].source == "FRED / Federal Reserve Board, H.10"
    plt.rcParams.update({"figure.figsize": (11, 4), "axes.spines.top": False,
                        "axes.spines.right": False, "axes.grid": True, "grid.alpha": .18,
                        "font.size": 10, "axes.titlesize": 12})
    colors = {"JPY": "#147d92", "EUR": "#c56a2d"}
    display(pd.DataFrame([design["primary"]]))
    display(pd.DataFrame(sim["scale"]).T.drop(columns="trace"))
    print("Registered:", frozen["registered_at_utc"], "| design SHA-256:", frozen["design_sha256"])
    print("Calibration/null:", design["n_cal"], design["n_null"], "| burn:", design["burn"])
    primary_summary = next(row for row in result["specifications"] if row["primary"])
    display(Markdown(f"**Primary outcome: {primary_summary['status']}.** "
                     f"{primary_summary['n_labelled']} held-out episodes; {primary_summary['n_matched']} matched. "
                     "Insufficient events is not a statistically tested null."))
    """)
    md("""
    ## What is being tested

    The primary outcome is the difference in **minimum-distance J** before retrospectively labelled
    JPY short-covering episodes versus volatility-matched non-episode dates: 500-session estimation,
    K=21 prior sessions, one-sided alpha=0.10, validation from 2018 onward. The 250-session specification
    is secondary. CFTC enters episode labels and context only; J inference uses returns alone.

    Total model volatility is solved against pre-2018 realized SD at reference J=0.40, with a 0.5%
    numerical tolerance, then frozen. Matching volatility separately at every J would define a
    different model family. J is therefore conditional on this fixed nuisance calibration.
    """)
    md("""
    ## Exhibit 1 - Why move from crude to FX?

    The table below reproduces the **prior measurements supplied in the build brief** (250 paths,
    2,000 evaluation sessions). These are motivation, not newly replicated results; the brief did
    not supply their raw paths or seed ledger. The following exhibit is the new 500-path screen.
    Prior flow-share numbers describe the measured structural channel and are not a causal
    variance decomposition of observed FX returns.
    """)
    code(
        """
    prior = pd.DataFrame(design["historical_scale_evidence"]["rows"])
    display(prior)
    ax = prior.set_index("analogue")[["pathological_J040", "pathological_J078"]].plot.bar(
        color=["#8ba7b3", "#c56a2d"], rot=0)
    ax.set(ylabel="Prior pathological fraction", title="Prior scale experiment: lower FX noise made the tested regimes usable")
    ax.legend(["J = 0.40", "J = 0.78"])
    plt.show()
    """,
        1,
    )
    md("""
    ## Exhibit 2 - New stability screen at each solved FX scale

    Each tested J uses 500 independent paths and 2,000 post-burn returns. Pathology means
    non-finite values, invariant failure, or return SD above 5 times innovation sigma. The eligibility
    rule is an observed rate at most 1%; Wilson intervals show sampling uncertainty, so zero
    observed failures does not prove zero risk. J*beta=1 is an agent-layer reference, not an
    established boundary of the coupled market system.
    """)
    code(
        """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, pair in zip(axes, ("JPY", "EUR")):
        table = pd.DataFrame(sim["stability"][pair])
        ax.errorbar(table.J, table.fraction,
                    yerr=np.maximum(0, [table.fraction-table.lo, table.hi-table.fraction]), fmt="o-", color=colors[pair], capsize=3)
        ax.axhline(.01, color="grey", ls="--", label="1% eligibility cutoff")
        ax.axvline(1/cfg.beta, color="black", ls=":", label="Agent-layer reference")
        ax.set(title=f"{pair}: sigma = {sim['scale'][pair]['sigma_eps']:.5f}", xlabel="Tested J", ylabel="Pathological fraction")
        ax.legend(fontsize=8)
        display(Markdown(f"**{pair} eligible tested points:** {sim['eligible'][pair]}"))
    plt.tight_layout(); plt.show()
    """,
        2,
    )
    md("""
    ## Exhibit 3 - Quote direction and positioning context

    Both series are **USD per foreign-currency unit**: JPY appreciation is positive after
    inverting [FRED DEXJPUS](https://fred.stlouisfed.org/series/DEXJPUS); EUR uses
    [DEXUSEU](https://fred.stlouisfed.org/series/DEXUSEU) directly. Raw FX observations are
    Federal Reserve Board H.10 noon buying rates, not exchange closing prices. Missing
    observations are dropped, with no holiday forward-fill.

    [CFTC financial contracts](https://publicreporting.cftc.gov/stories/s/r4w3-av2u) use TFF
    leveraged money, not disaggregated managed money. Contract codes were discovered by exact
    CME contract name from official rows. Net short exposure = (short-long)/OI; short covering
    reduces this measure and increases conventional (long-short)/OI. COT covers futures and
    cannot measure the entire OTC carry market or identify why traders held positions.
    """)
    code(
        """
    definitions = discover_definitions(CACHE / "raw")
    context = []
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for pair in ("JPY", "EUR"):
        fx = series[pair]
        axes[0].plot(fx.dates, fx.close/fx.close[0], label=pair, color=colors[pair])
        definition = next(d for d in definitions if d.key == design["positioning_primary"])
        cot, contract = fetch_fx_positioning(definition, pair, cache_dir=CACHE / "raw", start=design["start"], end=design["end"])
        axes[1].plot(cot.dates, -cot.net_over_oi, label=pair, color=colors[pair], lw=.8)
        for key, values in result["pairs"][pair]["positioning"].items():
            context.append({"pair": pair, "definition": key, **{k:v for k,v in values.items() if k != "contract"}})
    axes[0].set(ylabel="USD value / first observation", title="FRED / Federal Reserve Board H.10; positive means currency appreciation")
    axes[1].set(ylabel="Net short / OI", title="CFTC TFF leveraged money, futures only")
    for ax in axes: ax.legend()
    plt.tight_layout(); plt.show()
    display(pd.DataFrame(context))
    """,
        3,
    )
    md("""
    ## Exhibit 4 - Empirical fingerprint and the tail gap

    The shared statistic is ACF of demeaned squared returns, lags 1-8. Pointwise 95% stationary
    bootstrap intervals use 1,000 draws and mean block length 21. These describe the entire
    historical mixture; they do not establish constant J over the sample. SD and excess
    kurtosis are separate diagnostics: the selected ACF distance does not test tail adequacy.
    """)
    code(
        """
    fig, ax = plt.subplots()
    fingerprint_rows = []
    for pair in ("JPY", "EUR"):
        f = result["pairs"][pair]["fingerprint"]
        x = np.arange(1, 9)
        ax.plot(x, f["point"][:8], "o-", label=pair, color=colors[pair])
        ax.fill_between(x, f["lo"][:8], f["hi"][:8], alpha=.15, color=colors[pair])
        fingerprint_rows.append({"pair": pair, "n_returns": result["pairs"][pair]["n_returns"],
                                 "daily_SD": f["point"][8], "excess_kurtosis": f["point"][9],
                                 "kurtosis_lo": f["lo"][9], "kurtosis_hi": f["hi"][9]})
    ax.set(xlabel="Lag in observed sessions", ylabel="ACF(r^2)", title="Empirical persistence: pointwise bootstrap bands")
    ax.legend(); plt.show()
    display(pd.DataFrame(fingerprint_rows))
    display(pd.DataFrame([{ "pair": pair, **row } for pair in ("JPY", "EUR") for row in result["pairs"][pair]["kurtosis"]]))
    """,
        4,
    )
    md("""
    ## Exhibit 5 - Rolling JPY minimum-distance estimates

    J_hat minimizes the same standardized D-sum used for adequacy. A higher p-value is not a better
    fit. Shaded squares are the **exact non-rejected grid points**, a pointwise Neyman-inversion
    set conditional on the fixed scale; gaps and empty sets remain visible. No continuous-J
    interval or simultaneous coverage across dates is claimed. A closest grid point still
    exists when the entire grid is rejected, but then it is an inadequate approximation.

    Monthly steps reuse overlapping returns. The 250-session estimate responds faster and has
    greater uncertainty; the 500-session estimate has a longer memory. Confidence sets may be
    too broad to distinguish either. Both specifications were declared before results.
    """)
    code(
        """
    def draw_rolling(ax, pair, T):
        rows = result["pairs"][pair]["rolling"][str(T)]
        dates = pd.to_datetime([r["date"] for r in rows])
        for day, row in zip(dates, rows):
            ax.scatter([day]*len(row["confidence_set"]), row["confidence_set"], marker="s", s=13,
                       alpha=.17, color=colors[pair])
        ax.plot(dates, [r["J_hat"] for r in rows], color=colors[pair], lw=1, label="Minimum-distance J")
        empty = np.array([r["empty_set"] for r in rows])
        ax.scatter(dates[empty], [sim["eligible"][pair][0]]*empty.sum(), marker="x", color="black", s=20, label="Empty set")
        ax.axvline(pd.Timestamp(design["split"]), color="grey", ls="--")
        ax.set(ylabel="Tested J", title=f"{pair}, T={T}: {empty.sum()}/{len(rows)} windows reject every eligible point")
        ax.legend(fontsize=8)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for ax, T in zip(axes, design["windows"]): draw_rolling(ax, "JPY", T)
    plt.tight_layout(); plt.show()
    """,
        5,
    )
    md("""
    ### A non-rejection is not evidence of carry dynamics

    The Gaussian negative control below is evaluated using the identical FX banks. Frequent
    non-rejection of independent Gaussian returns would show that this statistic admits a
    process with no carry channel. This limits the meaning of broad confidence sets and
    belongs beside the rolling result, even if an event association is positive.
    """)
    code("""
    display(pd.DataFrame([{k:v for k,v in row.items() if k not in ("J", "per_J_non_rejection")}
                          for row in result["gaussian"]]))
    """)
    md("""
    ## Exhibit 6 - The predeclared carry-unwind proxy test

    An episode needs prior net short exposure >=5% of OI, a weekly reduction >=5 percentage
    points, and currency appreciation >=1.5 pre-interval SD times sqrt(sessions). SD uses the prior 250
    observations. Events are separated by 42 sessions; report gaps above 10 calendar days
    are excluded. This is a short-covering proxy, not direct identification of carry trades.

    Scores use the last available monthly J_hat over 21 sessions **before the price/positioning
    interval starts**. Match five non-episode dates in the same year with log-volatility
    difference <=0.25. Dates within 42 sessions of an episode are excluded from controls.
    Unmatched episodes are reported. Label permutations select pseudo-episode dates within
    matched sets and recompute the contrast; 95% effect intervals bootstrap event-year clusters.

    COT Tuesday observation labels are retrospective; publication is generally Friday and
    holidays can delay it. This is historical association evaluation, not a real-time alert
    backtest. The matched-label permutation assumes exchangeability, which overlapping
    estimates and reused controls only approximate. Small numbers of years weaken the interval.
    """)
    code(
        """
    primary = next(row for row in result["specifications"] if row["primary"])
    fields = ["pair", "window", "segment", "n_labelled", "n_matched", "effect", "lo", "hi", "p_value", "mc_lo", "mc_hi", "status"]
    display(pd.DataFrame([{key: primary.get(key) for key in fields}]))
    if primary["p_value"] is None:
        headline = "Primary JPY test is not estimable with the matched episodes available"
    else:
        headline = f"Primary JPY result: {primary['status']}; effect={primary['effect']:.3f}, permutation p={primary['p_value']:.4f}"
    display(Markdown("**" + headline + "**"))
    fig, ax = plt.subplots()
    ax.set(title=headline, xlabel="Event minus matched-control J", ylabel="Permutation draws")
    if primary["permutation"]:
        ax.hist(primary["permutation"], bins=40, color="#8ba7b3")
        ax.axvline(primary["effect"], color=colors["JPY"], lw=2, label="Observed")
        ax.legend()
    elif primary["matched"]:
        ax.bar([m["date"] for m in primary["matched"]], [m["difference"] for m in primary["matched"]], color=colors["JPY"])
        ax.axhline(0, color="grey", lw=1)
        ax.set(xlabel="Matched episode (descriptive only)", ylabel="Pre-event minus matched-control J")
        display(Markdown(f"Descriptive mean of available matched differences: "
                         f"{np.mean([m['difference'] for m in primary['matched']]):.3f}. "
                         "No confidence interval or permutation p-value is reported with fewer than three matched episodes."))
    else:
        ax.text(.5, .5, "No matched episodes: no test", ha="center", transform=ax.transAxes)
    plt.show()
    display(pd.DataFrame(primary["matched"]).drop(columns=["control_positions", "control_scores"], errors="ignore"))
    """,
        6,
    )
    md("""
    ## Exhibit 7 - EUR falsification

    EUR is a comparison market, not a proven carry-free negative control. The direct
    falsification scores EUR on the same JPY episode dates; EUR's own short-covering labels
    are also reported as secondary. Similar patterns would weaken carry specificity, but
    would not by themselves prove the estimator is measuring noise. No EUR result can
    replace an unfavourable primary JPY result.
    """)
    code(
        """
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for ax, T in zip(axes, design["windows"]): draw_rolling(ax, "EUR", T)
    plt.tight_layout(); plt.show()
    control_specs = [r for r in result["specifications"] if r["pair"] == "EUR" and r["segment"] == "validation"]
    display(pd.DataFrame([{k:r.get(k) for k in ["label_pair", *fields]} for r in control_specs]))
    """,
        7,
    )
    md("""
    ## Exhibit 8 - Fixed-split, sequential held-out evaluation

    The split is 1 January 2018. Innovation scales, J banks, thresholds and window choices
    were frozen using the earlier segment/design. Each later rolling estimate uses only
    observations available by its endpoint. Annual summaries show subsequent performance.
    This implements a fixed-split validation evaluated sequentially, **not** an expanding
    nuisance-parameter refit. Data are the current historical vintage, not archived real-time
    releases. Event/control matching remains retrospective and must not be sold as forecasting.
    """)
    code(
        """
    walk = pd.DataFrame(result["walk_forward"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, pair in zip(axes, ("JPY", "EUR")):
        for T in design["windows"]:
            sub = walk[(walk.pair == pair) & (walk.window == T)]
            ax.plot(sub.year, sub.empty_set_fraction, "o-", label=f"T={T}")
        ax.tick_params(axis="x", rotation=45)
        ax.set(title=pair + ": later windows rejecting all eligible J", ylabel="Empty confidence-set fraction")
        ax.legend()
    plt.tight_layout(); plt.show()
    display(pd.DataFrame([{k:r.get(k) for k in fields} for r in result["specifications"]
                          if r["pair"] == "JPY" and r["window"] == 500]))
    """,
        8,
    )
    md("""
    ## Exhibit 9 - FX-scale calibration and power

    Positive controls use 500 independent true-model pseudo-samples at two eligible J values,
    both window lengths and both solved scales. Wilson intervals are compared with nominal
    10% size. Any failures remain in the table; the test's finite-sample calibration cannot
    be assumed from the WTI work. Power contrasts J=0.40 with the declared eligible comparison.
    Common seeds across J/pairs/horizons induce correlated Monte Carlo errors and cannot be
    counted as independent replications. The 500 calibration/999 null paths are new FX runs.
    """)
    code(
        """
    controls = pd.DataFrame(result["controls"])
    display(controls)
    power = pd.DataFrame(result["power"])
    fig, ax = plt.subplots()
    for pair in ("JPY", "EUR"):
        sub = power[power.pair == pair]
        ax.errorbar(sub["T"], sub.power,
                    yerr=[sub.power-sub.lo, sub.hi-sub.power], fmt="o-", color=colors[pair], label=pair, capsize=3)
        crossing = sub.loc[sub.lo > .8, "T"]
        if len(crossing):
            first = int(crossing.iloc[0]); previous = max([t for t in design["power_lengths"] if t < first], default=0)
            display(Markdown(f"{pair}: first tested lower-bound crossing of 80% is bracketed by {previous}-{first} sessions."))
        else:
            display(Markdown(f"{pair}: no tested lower confidence bound exceeds 80% power."))
    ax.axhline(.8, ls="--", color="grey")
    ax.set(xlabel="Observed sessions", ylabel="Rejection probability", title="New FX-scale point-contrast power; 500 pseudo-samples")
    ax.legend(); plt.show()
    display(Markdown(f"Controls whose 95% interval misses nominal size: **{int((~controls.covers_nominal).sum())}/{len(controls)}**."))
    """,
        9,
    )
    md("""
    ## Exhibit 10 - What the simulated regime contrast means over 21 sessions

    These are unconditional regime contrasts at the solved scales. Each path discards 500
    burn-in returns, rebuilds prices from 1, and evaluates exactly 21 returns. Round trips
    require recovery evidence and ending drawdown, not maximum drawdown alone. Recovery
    refers to regaining the pre-trough peak after the maximum drawdown; unrecovered paths
    are censored at 21. Recovery after >=5% drawdown can have very few qualifying paths at FX
    scale; a zero denominator is undefined and displayed as missing.
    """)
    code(
        """
    horizons = pd.DataFrame(result["horizons"])
    display(horizons)
    display(Markdown("Recovery estimates have few observed recoveries; overlapping intervals do not establish a regime difference. "
                     "Ending drawdowns and terminal returns describe persistence separately from maximum drawdown."))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    labels = [f"{r['pair']} J={r['J']:.2f}" for r in result["horizons"]]
    for ax, metric, title in zip(axes, ["mdd_p95", "ending_drawdown_mean", "terminal_mean"],
                                  ["95th-percentile MDD", "Mean ending drawdown", "Mean terminal log return"]):
        ax.bar(labels, horizons[metric], color=[colors[r["pair"]] for r in result["horizons"]])
        ax.tick_params(axis="x", rotation=35)
        ax.set_title(title)
    plt.tight_layout(); plt.show()
    """,
        10,
    )
    md("""
    ## Conclusion and limits

    The primary result above controls the headline. Secondary specifications cannot rescue it.
    An association would support further investigation of this statistic; it would not identify
    the carry mechanism. A null leaves the hypothesis unsupported under this design.

    - Gaussian innovations and the measured kurtosis gap limit tail realism. ACF adequacy
      deliberately excludes tail behaviour; passing it is not a distributional validation.
    - Central-bank intervention, policy announcements and peg breaks are external mechanisms
      absent from PBMRS, analogous to the supply-shock limitation in the WTI work.
    - One simulator step equals one observed trading session by assumption. Holidays leave
      varying calendar-time gaps. FX H.10 rates and COT observation times do not coincide exactly.
    - Fixed sigma at a reference J and fixed other base.yaml parameters make J conditional, not a
      uniquely identified structural measure. Drifts, nuisance estimation uncertainty and
      nonstationarity are not included in the pointwise inversion sets.
    - Carry labels use one futures category and thresholds chosen before results, but they
      remain imperfect proxies. Matched permutations require approximate exchangeability;
      overlapping windows and reused controls limit precision. Historical labels are not live signals.
    - The finite grid bounds any conclusion. A final eligible point does not establish a
      continuous stable interval. J*beta=1 is only a theoretical agent-layer reference.

    Next work: obtain the GMSG FX report, preregister a claim from it, test independent vintages
    and markets, and revisit innovations or policy-jump mechanisms if the tail gap remains.
    """)
    md("""
    ## Appendix A - Every specification, including nulls and exclusions
    """)
    code("""
    display(pd.DataFrame([{k:r.get(k) for k in ["primary", "label_pair", *fields]}
                          for r in result["specifications"]]))
    excluded = [{"pair": r["pair"], "window": r["window"], "segment": r["segment"],
                 "label_pair": r["label_pair"], **e} for r in result["specifications"] for e in r["excluded"]]
    display(pd.DataFrame(excluded))
    """)
    md("""
    ## Appendix B - Complete stability, numerical solve and Monte Carlo checks
    Per-window rows, distances, p-values, Wilson bounds and borderline flags are retained in
    `fx_cache/results.json`; no point is selected by p-value rank.
    """)
    code("""
    display(pd.concat([pd.DataFrame(sim["stability"][p]).assign(pair=p) for p in ("JPY", "EUR")], ignore_index=True))
    display(pd.concat([pd.DataFrame(sim["scale"][p]["trace"]).assign(pair=p) for p in ("JPY", "EUR")], ignore_index=True))
    display(pd.DataFrame(result["power"]))
    display(pd.DataFrame(result["walk_forward"]))
    """)
    md("""
    ## Appendix C - Reproduction, hashes and provenance

    Raw payloads are committed byte-for-byte with URL, UTC retrieval timestamp, byte count and
    SHA-256. Git attributes prevent Windows newline conversion. Simulation blobs also carry
    content hashes and input-dependent keys. Source fingerprints normalize source line endings
    only; raw response hashes never do. No Yahoo dependency is used.

    From the project environment, run `build_fx_data.py`, `build_fx_simulations.py`,
    `build_fx_results.py`, then `build_fx_notebook.py` in this notebook directory. Only the
    data builder needs a network on an uncached run. Existing verified caches support offline
    execution; simulation generation is deliberately separate from presentation.
    """)
    code("""
    display(pd.DataFrame([{"file": name, **entry} for name, entry in
                         validate_raw_tree(CACHE / "raw").items()]))
    display(pd.DataFrame([{"name": k, "value": v} for k,v in design["seeds"].items()]))
    display(pd.DataFrame([{"package": k, "version": v} for k,v in manifest["versions"].items()]))
    print("Git HEAD at computation:", manifest["git_head"])
    print("Python:", manifest["python"])
    print("Simulation input hash:", sim["root_key"])
    print("Result payload SHA-256:", manifest["results_sha256"])
    display(pd.DataFrame([{"path": k, "source_sha256": v} for k,v in manifest["sources"].items()]))
    print("All raw, source and simulation manifests validated at notebook start.")
    """)
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {
        "display_name": "PBMRS (.venv)",
        "language": "python",
        "name": "pbmrs-1-venv",
    }
    notebook.metadata.language_info = {"name": "python"}
    target = ROOT / "notebooks/03_fx_regime_calibration.ipynb"
    nbf.write(notebook, target)
    print(f"Built {len(cells)} cells, 10 main exhibits: {target}")


if __name__ == "__main__":
    build()
