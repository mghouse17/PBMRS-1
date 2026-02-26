# PBMRS Test Plan

## Test Types

### Unit Tests
- Magnetization range: m_t ∈ [-1, 1]
- Volatility update produces v_{t+1} >= 0
- Liquidity update produces ℓ_{t+1} > 0
- Drawdown + recovery correctness on toy series

### Property Tests (invariants)
- No NaNs/inf after long runs under baseline config
- Same seed + same config → same outputs (hash or summary stats)

### Regression Tests
- Freeze a baseline run summary (mean vol, max DD, tail prob)
- Future changes must match within tolerance (unless bumping major version)

## CI Expectations
- pytest must pass
- formatting/lint checks (optional early; recommended later)