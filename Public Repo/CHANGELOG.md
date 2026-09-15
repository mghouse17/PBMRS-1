# Changelog

All notable changes to PBMRS are documented here.

## [Unreleased]
### Added
- Documentation foundation (model spec, architecture, configs, test plan)

### Changed
- N/A

### Fixed
- N/A

## [0.2.2] - 2026-09-14

### Changed
- Consolidated the simulator on the modular pbmrs_core implementation.
- Exported phase_map and the near-critical diagnostic heuristic.
- Retained pbmrs_core.sim and sim_v2 as one-release compatibility shims.
- Migrated every maintained notebook to the canonical package API.
- Moved pytest to the development extra and added a notebook extra.

### Fixed
- Repaired package-relative imports in analysis.py.
- Aligned package metadata and displayed model version.
- Preserved the specification's v[t] liquidity update order and corrected stale commentary.

### Added
- Immutable FRED, CFTC, and optional Yahoo data records with hash-verified raw caching.
- Fixed-grid ACF adequacy, Wilson intervals, stability, power, and exact-horizon utilities.
- A 40-cell, nine-exhibit FRED-primary GMSG application notebook and reproducible builders.
- An exact tested-environment lock and offline notebook/cache integration tests.
