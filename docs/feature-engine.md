# Feature Engine

The Feature Engine is OpenAlgo's unified feature calculation library for
algorithmic trading. It provides technical indicators, statistical measures,
domain-specific composite features, ML label generation, and preprocessing
pipelines in a single import.

```
from feature_engine import technical, statistical, custom, labels, preprocessing
```

Both OpenAlgo (strategies, scanners, tools) and Skopaq AI (dataset building,
model training) consume this same package. **Do not duplicate indicator
implementations** outside `feature_engine/`.

---

## Modules

### `technical` -- Technical Indicators

Wraps the `openalgo.indicators` library (100+ Numba-JIT-accelerated
indicators) into a clean, categorized API. Requires the `openalgo` pip
package.

| Category | Functions |
|----------|-----------|
| **Trend** | `sma`, `ema`, `wma`, `dema`, `tema`, `hma`, `vwma`, `alma`, `kama`, `zlema`, `supertrend`, `ichimoku`, `donchian` |
| **Momentum** | `rsi`, `macd`, `stochastic`, `cci`, `williams_r` |
| **Volatility** | `atr`, `bbands`, `keltner` |
| **Volume** | `obv`, `vwap`, `mfi`, `adl`, `cmf`, `rvol` |
| **Hybrid** | `adx`, `aroon`, `sar` |
| **Signals** | `crossover`, `crossunder`, `exrem`, `flip` |
| **Batch** | `compute_all` -- compute a standard set on any OHLCV DataFrame |

**Examples:**

```python
from feature_engine import technical as ta

# Single indicators
ema_20 = ta.ema(close, period=20)
rsi_14 = ta.rsi(close, period=14)
atr_14 = ta.atr(high, low, close, period=14)

# Multi-output indicators
macd_df = ta.macd(close)          # DataFrame: macd, signal, histogram
stoch_df = ta.stochastic(high, low, close)  # DataFrame: stoch_k, stoch_d
bbands_df = ta.bbands(close)      # DataFrame: bb_upper, bb_middle, bb_lower
ichimoku_df = ta.ichimoku(high, low, close)  # DataFrame: tenkan, kijun, senkou_a, senkou_b, chikou

# Tuples
st_line, st_dir = ta.supertrend(high, low, close, period=10, multiplier=3.0)
upper, mid, lower = ta.donchian(high, low, period=20)
aroon_up, aroon_down = ta.aroon(high, low, period=25)

# Signal detection
buy_signal = ta.crossover(ema_fast, ema_slow)
filtered = ta.exrem(buy_signal, rsi > 70)

# Compute everything at once
features = ta.compute_all(ohlcv_df)
```

All functions accept `pd.Series` and return `pd.Series` or `pd.DataFrame`.
Column names are set automatically (e.g. `EMA20`, `RSI14`, `supertrend`).

If the `openalgo` package is not installed, `import feature_engine.technical`
succeeds but individual functions raise `ImportError` with install instructions.

---

### `statistical` -- Statistical Features

Pure NumPy/Pandas implementations. No external dependencies beyond numpy and
pandas. Works anywhere Python runs.

| Category | Functions |
|----------|-----------|
| **Returns** | `simple_returns`, `log_returns`, `cumulative_returns`, `forward_returns`, `overnight_returns`, `intraday_returns` |
| **Rolling** | `rolling_mean`, `rolling_std`, `rolling_skew`, `rolling_kurtosis`, `rolling_min`, `rolling_max`, `rolling_quantile` |
| **Volatility** | `realized_volatility`, `parkinson_volatility`, `garman_klass_volatility`, `yang_zhang_volatility` |
| **Z-Score** | `zscore`, `modified_zscore` |
| **Cross-Asset** | `price_volume_correlation`, `volume_zscore`, `volume_ratio`, `relative_strength`, `beta`, `correlation` |

**Examples:**

```python
from feature_engine import statistical as stat

# Returns
ret = stat.simple_returns(close)
log_ret = stat.log_returns(close)
fwd_ret = stat.forward_returns(close, periods=5)
overnight = stat.overnight_returns(open_series, close_series)

# Volatility estimators (annualized by default)
rv = stat.realized_volatility(close, window=20, annualize=True)
pv = stat.parkinson_volatility(high, low, window=20)
gk = stat.garman_klass_volatility(open_series, high, low, close, window=20)
yz = stat.yang_zhang_volatility(open_series, high, low, close, window=20)

# Z-scores
z = stat.zscore(close, window=20)
mz = stat.modified_zscore(close, window=20)

# Cross-asset
vol_ratio = stat.volume_ratio(volume, short_window=5, long_window=20)
b = stat.beta(asset_returns, benchmark_returns, window=60)
```

---

### `custom` -- Domain-Specific Features

Composite features designed for Indian equity markets. Combine multiple raw
inputs into interpretable 0-100 or -100 to +100 scores.

| Function | Inputs | Output Range | Description |
|----------|--------|--------------|-------------|
| `smart_money` | close, volume, high, low | 0-100 | Smart Money Index proxy -- institutional accumulation vs. distribution |
| `fii_strength` | fii_buy, fii_sell | -100 to +100 | Net FII flow as percentage of total institutional flow |
| `breakout_score` | high, low, close, volume | 0-100 | Price breakout + volume surge composite |
| `institutional_score` | fii_buy, fii_sell, dii_buy, dii_sell | -100 to +100 | Weighted FII+DII net flow indicator |
| `earnings_quality` | revenue, net_income, operating_cf | 0-100 | Earnings consistency across revenue, income, and cash flow |
| `sector_rotation` | sector_returns_df | z-score | Cross-sector relative strength for rotation signals |
| `orderflow_score` | bid_volume, ask_volume | -100 to +100 | Bid/ask volume imbalance |
| `news_sentiment` | sentiment_series | -100 to +100 | Exponentially decayed news sentiment aggregation |

**Examples:**

```python
from feature_engine import custom

sm = custom.smart_money(close, volume, high, low)
fii = custom.fii_strength(fii_buy_vol, fii_sell_vol)
bo = custom.breakout_score(high, low, close, volume)
inst = custom.institutional_score(fii_buy, fii_sell, dii_buy, dii_sell)
```

---

### `labels` -- ML Label Generation

Create target variables for supervised learning from price data.

| Function | Type | Description |
|----------|------|-------------|
| `trend_target` | Binary (0/1) | 1 if price rises above threshold within horizon |
| `breakout_target` | Binary (0/1) | 1 if price breaks above N-period high within horizon |
| `trend_target_multiclass` | Multi-class (0-4) | 5-class trend direction label |
| `classification_labels` | Multi-class | Generic bin-based classification from forward returns |
| `regression_target` | Continuous | Forward N-period return |
| `volatility_target` | Continuous | Forward realized volatility over horizon |
| `max_drawdown_target` | Continuous (negative) | Maximum drawdown over forward horizon |

**Examples:**

```python
from feature_engine import labels

# Binary classification
buy_target = labels.trend_target(close, horizon=5, threshold=0.02)
breakout = labels.breakout_target(close, high, window=20, horizon=5, threshold=0.03)

# Regression
fwd_return = labels.regression_target(close, horizon=5)
fwd_vol = labels.volatility_target(close, horizon=20)

# Multiclass
cls = labels.classification_labels(close, horizon=5, thresholds=[-0.01, 0.01])
```

---

### `preprocessing` -- Scaling and Pipelines

Feature normalization, outlier handling, missing data treatment, and a
composable pipeline class.

| Category | Functions |
|----------|-----------|
| **Scaling** | `min_max_scale`, `standard_scale`, `robust_scale` |
| **Normalization** | `log_normalize`, `rank_normalize` |
| **Outliers** | `winsorize` (Series), `winsorize_df` (DataFrame) |
| **Missing Data** | `fill_missing` (forward_fill, backward_fill, mean, median, zero) |
| **Pipeline** | `FeaturePipeline` -- sequential transform chain |

**Examples:**

```python
from feature_engine import preprocessing as pp

# Individual transforms
scaled = pp.min_max_scale(features_df)
std = pp.standard_scale(features_df)
winsorized = pp.winsorize(price_series, lower=0.01, upper=0.99)

# Pipeline
pipeline = pp.FeaturePipeline()
pipeline.add_step("winsorize", lower=0.01, upper=0.99)
pipeline.add_step("standard_scale")
clean_features = pipeline.fit_transform(raw_features)
print(pipeline.describe())  # [{'step': 'winsorize', 'params': {...}}, ...]
```

---

## Architecture Principles

1. **Single source of truth** -- All indicator logic lives here. OpenAlgo
   strategies, scanners, and tools import from `feature_engine`. Skopaq AI
   imports from `feature_engine`. No duplicates.

2. **Layered dependencies** -- `statistical`, `custom`, `labels`, and
   `preprocessing` depend only on numpy/pandas. `technical` optionally depends
   on the `openalgo` pip package for its Numba-JIT backend.

3. **Graceful degradation** -- If `openalgo` is not installed, importing
   `feature_engine.technical` works. Individual functions raise `ImportError`
   with install instructions when called.

4. **Consistent API** -- All functions accept and return `pd.Series` or
   `pd.DataFrame`. Index alignment is preserved. Column names are set.

5. **No state** -- All functions are pure transformations. No global state,
   no side effects, no database connections.

---

## Testing

```bash
# Run all feature engine tests
uv run pytest test/test_feature_engine.py -v -o "addopts="

# 42 tests covering statistical, custom, labels, and preprocessing modules
```

Technical indicator tests are excluded from the default suite because they
require the `openalgo` pip package. The statistical/custom/labels/preprocessing
tests use only numpy and pandas.

---

## Dependency Map

```
feature_engine/
  technical/      --> openalgo.indicators (optional, Numba JIT)
  statistical/    --> numpy, pandas (only)
  custom/         --> numpy, pandas (only)
  labels/         --> numpy, pandas (only)
  preprocessing/  --> numpy, pandas (only)
```
