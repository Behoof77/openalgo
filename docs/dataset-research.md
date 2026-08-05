# Indian Market Open-Source Dataset Research

> Status: V1 Research Complete (July 2026)
> Goal: Find the best open-source datasets for ML training on Indian markets (NSE/BSE/MCX)
> Context: Skopaq AI platform - feeds Neon PostgreSQL -> Feature Engine -> Qlib/FinRL training

---

## 1. Executive Summary

We surveyed HuggingFace, Kaggle, GitHub, PyPI, and public APIs for open-source datasets covering four categories: OHLCV, Fundamentals, Macro/Market Context, and News/Sentiment.

### V1 Recommended Stack

| Category | Primary Source | Backup | Format |
|---|---|---|---|
| OHLCV (Equity) | vishnun0027/indian-market-historical-ohlcv (HF) | tejhq/indian-markets (HF) | Parquet |
| OHLCV (Minute) | Saintforest/indian-stock-market-minute-data (HF) | yfinance via OpenAlgo | Parquet |
| Fundamentals | sameerprogrammer/detailed-financials-4456-nse-bse (Kaggle) | bharat-sm-data (PyPI) | JSON/CSV |
| Macro/Market | yfinance (pull on demand) | bguzzo2k/ohlc_1d_mixture (HF) | CSV/Parquet |
| India VIX | alangeorgev/india-vix-historical-2025 (Kaggle) | NSE website CSV | CSV |
| FII/DII | MrChartist/fii-dii-data (GitHub JSON API) | Kaggle datasets | JSON |
| News/Sentiment | dixitdharmansh07/indic-finance (HF) | kdave/Indian_Financial_News (HF) | Parquet/JSON |
| Sentiment Model | Vansh180/FinBERT-India-v1 (HF) | tahp0604/finbert-sentfin (HF) | Model weights |

---

## 2. OHLCV Datasets

### 2A. Daily Equity Data

#### vishnun0027/indian-market-historical-ohlcv (PRIMARY)

- Source: HuggingFace vishnun0027/indian-market-historical-ohlcv
- License: MIT
- Format: Parquet (5 subsets)
- Size: Stocks: 7.56M rows / 2,394+ stocks. Indices: 70.8k. ETFs: 51.3k. Commodities: 48.6k. Forex: 46.4k
- Date Range: 2014+
- Update: Static snapshot
- Columns: date, open, high, low, close, adj_close, volume, dividends, stock_splits, symbol
- Subsets: stocks, indices, etfs, commodities, forex

Strengths: 10-year history, adj_close for splits/dividends, 2394+ stocks (full NSE coverage), includes indices/ETFs/commodities/forex. MIT license. Clean Parquet.
Weaknesses: Static (no daily updates). No OI data. No BSE-only stocks.
ML Readiness: HIGH - standard OHLCV, compatible with Qlib Alpha158/Alpha360 and FinRL.

#### tejhq/indian-markets

- Source: HuggingFace tejhq/indian-markets
- License: MIT
- Format: Hive-partitioned Parquet
- Size: ~2,300 NSE + ~2,200 BSE instruments/day
- Date Range: 2024-01-01+
- Update: Daily (~6:30 PM IST via GitHub Actions)
- Columns: OHLCV + corporate_actions + back_adjusted_prices + symbol_history + derived_metrics

Strengths: Official Bhavcopy data (exchange-sourced), updated daily, NSE+BSE, corporate actions, derived metrics.
Weaknesses: Only 2 years of data. Insufficient for robust ML training.
ML Readiness: MEDIUM - excellent quality/freshness but limited history. Best as DAILY UPDATE source.

#### Saintforest/indian-stock-market-minute-data

- Source: HuggingFace Saintforest/indian-stock-market-minute-data
- Format: Parquet shards (~1.5GB each)
- Size: 2,500+ NSE stocks
- Date Range: Minute: 2022-2026. Daily: 2000-2026
- Columns: OHLCV + OI (open interest)

Strengths: Minute-level data for 2500+ stocks. Has OI column (critical for options). 25+ years daily.
Weaknesses: Very large shards. License unclear.
ML Readiness: HIGH for intraday strategies.

#### Kaggle Datasets

| Dataset | Stocks | Period | Format | Size |
|---|---|---|---|---|
| Nifty500 5yr daily | 500 | 2021-2026 | CSV | 54.68MB |
| Nifty500 Multi-Timeframe | 501 | 1999-2026 | CSV | ~100MB |
| Sensex/Nifty all-time | 80 | 1999-2026 | CSV | ~50MB |
| Daily BSE SENSEX | 1 index | 2000-2024 | CSV | 396KB |

### 2B. OHLCV Comparison Matrix

| Dataset | Stocks | History | Freshness | OI | Format | License | V1 Pick |
|---|---|---|---|---|---|---|---|
| vishnun0027 (HF) | 2394+ | 10yr | Static | No | Parquet | MIT | PRIMARY |
| tejhq (HF) | 4500 | 2yr | Daily | No | Parquet | MIT | UPDATE |
| Saintforest (HF) | 2500+ | 25yr | Static | Yes | Parquet | TBD | MINUTE |
| Nifty500 5yr (KG) | 500 | 5yr | Static | No | CSV | CC | Backup |
| Nifty500 MT (KG) | 501 | 27yr | Static | No | CSV | CC | Backup |

---

## 3. Fundamental Datasets

### sameerprogrammer/detailed-financials-4456-nse-bse (PRIMARY)

- Source: Kaggle sameerprogrammer/detailed-financials-4456-nse-bse
- License: CC (check)
- Format: CSV/JSON
- Size: 4,456 companies
- Content: Yearly P&L, Balance Sheet, Cash Flow, Quarterly + Yearly Shareholding Pattern

Strengths: Comprehensive (P&L, BS, CF, Shareholding). 4,456 companies (essentially all listed). Annual + quarterly.
Weaknesses: Update cadence unclear. May be a snapshot.
ML Readiness: HIGH - standard financial statement format. Maps to FundamentalRow schema.

### bharat-sm-data (Python Package)

- Source: PyPI bharat-sm-data
- Format: Python API to DataFrames
- Content: Screener.in + MoneyControl + Tickertape scrape

Strengths: Live scraping (always up-to-date). Complete P&L, BS, CF, Ratios, Shareholding, Peer Comparison.
Weaknesses: Depends on scraping targets (may break). Rate-limited.
ML Readiness: MEDIUM - good for on-demand, not bulk historical. Best as BACKUP for freshness.

### Other Fundamental Sources

| Source | Content | Coverage | Notes |
|---|---|---|---|
| Kaggle bse-nse-fundamentals-ratios | Revenue, PE, EPS, PEG, EBITDA, Debt | ~500 stocks | Computed ratios, not raw |
| Kaggle Nifty 500 fundamentals | Market cap, P/E, 52w hi/lo, sector | 500 stocks | Summary stats |
| NiftyLens | P&L, BS, CF, ratios | ~100 NSE, 10-12yr | Excel format |
| BlueStock-Nifty100 | Full financials to star-schema DB | Nifty 100 | ETL pipeline |
| Screener.in Apify scraper | All Screener data | Unlimited | Paid (/1K rows) |

---

## 4. Macro/Market Context Datasets

### yfinance (PRIMARY)

- Source: Yahoo Finance via yfinance Python package
- License: Apache 2.0 (yfinance) / Yahoo TOS (data)
- Indian Tickers: ^INDIAVIX, ^NSEI (NIFTY 50), ^NSEBANK (BANKNIFTY), ^BSESN (SENSEX)
- Global Tickers: GC=F (Gold), CL=F (Crude), DX-Y.NYB (DXY), ^GSPC (S&P 500), ^VIX, EURUSD=X
- Date Range: 10+ years typically
- Update: On-demand (real-time pull)

Strengths: Universal coverage. Any ticker/timeframe/history. Python-native. Already used by OpenAlgo.
Weaknesses: Yahoo TOS (not for redistribution). May have gaps for Indian tickers.
ML Readiness: HIGH - standard OHLCV format.

### Static Macro Datasets

| Dataset | Content | Period | Format | Source |
|---|---|---|---|---|
| gold price 2015-2025 | SPX, GLD, USO, SLV, EUR/USD | 2015-2025 | CSV | Kaggle |
| bguzzo2k/ohlc_1d_mixture | 936 instruments (global) | Varies | Parquet | HuggingFace |
| India VIX historical | India VIX daily | 2010-2025 | CSV | Kaggle |
| NIFTY 500 and VIX 2010-2026 | NIFTY 500 + India VIX | 2010-2026 | CSV | Kaggle |
| Stooq commodities | Gold, silver, copper, platinum, WTI | 1985-2022 | CSV | Codeberg |

### FII/DII Data

| Source | Content | Format | Update |
|---|---|---|---|
| MrChartist/fii-dii-data (RECOMMENDED) | Daily FII/DII cash + F&O, NSDL sector data | JSON API | Daily |
| Kaggle arunkumar237 | FII/DII flows | CSV | Snapshot |
| Kaggle pravinpari | FII/DII investments | CSV | Snapshot |
| Kaggle luckyazure | FII/DII + NIFTY historical | CSV | Snapshot |

MrChartist: Free JSON API with full daily history. No API key needed.

---

## 5. News/Sentiment Datasets

### dixitdharmansh07/indic-finance (PRIMARY)

- Source: HuggingFace dixitdharmansh07/indic-finance
- Format: Parquet/JSON
- Size: 150+ Indian companies
- Date Range: January 2024+
- Sources: Google News, Economic Times, MoneyControl, LiveMint, Reddit, StockTwits
- Columns: headline, source, date, sentiment_label, sentiment_scores, forward_return_pct, return_direction

Strengths: Multi-source (6 outlets). Includes forward returns (usable as labels!). Most comprehensive Indian news dataset.
Weaknesses: Only from Jan 2024 (limited history).
ML Readiness: VERY HIGH - has forward_return_pct and return_direction as built-in labels.

### Other News/Sentiment Sources

| Dataset | Content | Size | Format | Notes |
|---|---|---|---|---|
| harixn/indian_news_sentiment | Sentiment labels (POS/NEU/NEG) | Varies | CSV | For FinBERT fine-tuning |
| kdave/Indian_Financial_News | 26K articles + T5 summaries + GPT sentiment | 112KB | JSON | LLM experiments |
| raeidsaqur/NIFTY | 2.1K examples paired with SPY | Small | JSON | LLM forecasting |
| SEntFiN 1.0 | 10.7K human-annotated headlines | 2002-2017 | CSV | Gold-standard annotations |
| AION/aion-sentiment-in-v3 | 1M+ headlines, 136 event taxonomy | Large | Parquet | Largest, Apache 2.0 |

### Pre-trained Sentiment Models

| Model | Base | Training Data | F1 | Use Case |
|---|---|---|---|---|
| Vansh180/FinBERT-India-v1 | FinBERT | 7,451 Indian headlines | 0.873 | Inference on new headlines |
| tahp0604/finbert-sentfin | FinBERT | SEntFiN 1.0 (10.7K) | 0.873 | Alternative model |

Recommendation: Use Vansh180/FinBERT-India-v1 for real-time sentiment scoring.

---

## 6. Qlib Compatibility Assessment

### Data Format Requirements

Qlib uses a specific directory structure:

    qlib_data/
      calendars/day.txt         -- trading calendar
      instruments/all.txt       -- stock list with date ranges
      features/SH600000/
        open.bin, close.bin, high.bin, low.bin, volume.bin, factor.bin (float32)

Key requirements:
- Float32 binary per feature per stock
- Trading calendar file
- Instrument list with start/end dates
- Supports custom features (Alpha158, Alpha360)

### Qlib Alpha158 Features (158 features per stock)

KMID, KLEN, KMID2, KUP, KUP2, KLOW, KLOW2, KSFT, KSFT2 (9)
ROC x5, MA x5, STD x5, BETA x5, RSQR x5, RESI x5 (25)
MAX x5, MIN x5, QTLU x5, QTLD x5, RANK x5, RSV x5 (30)
IMAX x5, IMIN x5, IMXD x5, CORR x5, CORD x5 (25)
CNTP x5, CNTN x5, CNTD x5, SUMP x5, SUMN x5, SUMD x5 (30)
VMA x5, VSTD x5, WVMA x5, VSUMP x5, VSUMN x5, VSUMD x5 (30)

### Compatibility with Our Datasets

| Dataset | Qlib Compatible? | Notes |
|---|---|---|
| vishnun0027 OHLCV | YES (with conversion) | Standard OHLCV maps directly |
| tejhq OHLCV | YES (with conversion) | Standard OHLCV maps directly |
| Saintforest minute | YES (with conversion) | Need timeframe normalization |
| Kaggle Nifty500 | YES (with conversion) | CSV to binary conversion needed |

Conversion: Parquet/CSV -> pandas DataFrame -> Qlib bin format. Straightforward.

---

## 7. FinRL Compatibility Assessment

### State Space Requirements

FinRL expects:
1. Stock prices: open, high, low, close, volume (5 dims per stock)
2. Technical indicators: MACD, RSI, CCI, ADX (4+ dims per stock)
3. Turbulence index (1 dim): market regime indicator
4. Account state: cash, stock holdings

### Compatibility with Our Datasets

| Dataset | FinRL Compatible? | Notes |
|---|---|---|
| vishnun0027 OHLCV | YES | Standard OHLCV feeds directly |
| tejhq OHLCV | YES | Same as above |
| feature_engine (our package) | YES | Compute indicators, extend state space |
| indic-finance sentiment | YES | Add as additional feature dimension |

Our feature_engine computes all standard indicators (MACD, RSI, CCI, ADX, Bollinger, etc.).
Pipeline: OHLCV -> feature_engine -> FinRL state vector.

---

## 8. V1 Recommended Dataset Stack

### Primary Stack (for immediate ML training)

| # | Dataset | Source | Why |
|---|---|---|---|
| 1 | OHLCV Daily | vishnun0027/indian-market-historical-ohlcv | 10yr, 2394+ stocks, MIT, Parquet |
| 2 | OHLCV Minute | Saintforest/indian-stock-market-minute-data | Minute data + OI, 2500+ stocks |
| 3 | OHLCV Daily Updates | tejhq/indian-markets | Daily fresh Bhavcopy, NSE+BSE |
| 4 | Fundamentals | sameerprogrammer/detailed-financials-4456-nse-bse | 4456 companies, full financials |
| 5 | Macro (on-demand) | yfinance | Gold, Crude, DXY, S&P, VIX, India indices |
| 6 | India VIX | alangeorgev/india-vix-historical-2025 | 2010-2025 daily VIX |
| 7 | FII/DII | MrChartist/fii-dii-data | Daily flows, free JSON API |
| 8 | News + Returns | dixitdharmansh07/indic-finance | 150+ companies, sentiment + forward returns |
| 9 | Sentiment Model | Vansh180/FinBERT-India-v1 | Fine-tuned for Indian financial headlines |

### Data Flow

    Static Datasets (HF/Kaggle) --> Neon PostgreSQL (initial load)
    Daily Updates (tejhq, MrChartist) --> Neon PostgreSQL (incremental)
    On-Demand (yfinance) --> Feature Engine (real-time)
    News (indic-finance) --> FinBERT-India-v1 --> Sentiment scores --> Neon

---

## 9. Risk Analysis

### Licensing Risks

| Dataset | License | Risk | Mitigation |
|---|---|---|---|
| vishnun0027 OHLCV | MIT | None | Commercial use OK |
| tejhq OHLCV | MIT | None | Commercial use OK |
| Saintforest minute | TBD | Medium | Check before production use |
| Kaggle datasets | CC-BY/CC-BY-SA | Low | Attribution required |
| sameerprogrammer fundamentals | CC | Low | Check specific CC variant |
| yfinance | Yahoo TOS | Medium | Not for redistribution, internal use OK |
| indic-finance | TBD | Low-Medium | Check dataset card |
| FinBERT-India-v1 | HuggingFace | Low | Check model card |

### Update Cadence Risks

| Dataset | Update | Risk | Mitigation |
|---|---|---|---|
| vishnun0027 | Static | HIGH - data goes stale | Supplement with tejhq daily updates |
| sameerprogrammer | Unknown | MEDIUM | Use bharat-sm-data for fresh data |
| yfinance | On-demand | LOW | Always current |
| MrChartist | Daily | LOW | Automated daily pull |
| indic-finance | Unknown | MEDIUM | Supplement with FinBERT inference |

### Completeness Risks

| Gap | Impact | Mitigation |
|---|---|---|
| No minute-level primary source | Medium | Saintforest dataset for intraday |
| Fundamentals may miss recent quarters | High | bharat-sm-data for current data |
| News only from Jan 2024 | Medium | SEntFiN 1.0 for historical sentiment |
| No options chain historical data | Medium | OpenAlgo Historify for live collection |

---

## 10. PoC Implementation Plan

### Phase 1: Data Collection Script

1. Download vishnun0027 OHLCV (stocks subset) via huggingface_hub
2. Download sameerprogrammer fundamentals via kaggle API
3. Pull yfinance macro data (Gold, Crude, DXY, VIX, NIFTY)
4. Download indic-finance news dataset
5. All save to local Parquet/CSV staging

### Phase 2: Neon PostgreSQL Import

1. Create tables matching Skopaq AI warehouse/models.py schema:
   - ohlcv_daily (symbol, date, open, high, low, close, adj_close, volume, source)
   - fundamentals (symbol, report_date, pe_ratio, pb_ratio, roe, roce, revenue, net_income, ...)
   - market_context (date, index_name, open, high, low, close, volume, source)
   - news_sentiment (date, headline, source, sentiment_score, sentiment_label, symbols, ...)
   - fii_dii (date, fii_buy, fii_sell, dii_buy, dii_sell, net_direction, ...)
   - india_vix (date, open, high, low, close, prev_close, change, ...)
2. Bulk insert using COPY or executemany
3. Validate row counts, null checks, date range checks

### Phase 3: Feature Engine Integration

1. Read from Neon -> pandas DataFrame
2. Run through feature_engine (technical, statistical, custom)
3. Generate ML-ready feature matrices
4. Export as Qlib bin format + FinRL state format
5. Validate: check feature shapes, no NaN in critical columns

---

## Appendix: Dataset URLs

| Dataset | URL |
|---|---|
| vishnun0027 OHLCV | https://huggingface.co/datasets/vishnun0027/indian-market-historical-ohlcv |
| tejhq OHLCV | https://huggingface.co/datasets/tejhq/indian-markets |
| Saintforest minute | https://huggingface.co/datasets/Saintforest/indian-stock-market-minute-data |
| sameerprogrammer fundamentals | https://www.kaggle.com/datasets/sameerprogrammer/detailed-financials-4456-nse-bse |
| bharat-sm-data | https://pypi.org/project/bharat-sm-data/ |
| indic-finance news | https://huggingface.co/datasets/dixitdharmansh07/indic-finance |
| FinBERT-India-v1 | https://huggingface.co/Vansh180/FinBERT-India-v1 |
| MrChartist FII/DII | https://github.com/MrChartist/fii-dii-data |
| India VIX Kaggle | https://www.kaggle.com/datasets/alangeorgev/india-vix-historical-2025 |
| bguzzo2k macro | https://huggingface.co/datasets/bguzzo2k/ohlc_1d_mixture |
| SEntFiN 1.0 | https://www.kaggle.com/datasets/ankurzing/sentfin-financial-sentiment-dataset |
| AION sentiment | https://huggingface.co/datasets/AION-Analytics/aion-sentiment-in-v3 |
