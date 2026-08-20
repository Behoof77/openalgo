# -*- coding: utf-8 -*-
"""
Comprehensive tests for the feature_engine package.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def ohlcv_df():
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_price = close + np.random.randn(n) * 0.2
    volume = np.random.randint(1000000, 5000000, n).astype(float)
    return pd.DataFrame({
        "open": open_price, "high": high, "low": low,
        "close": close, "volume": volume,
    }, index=dates)


@pytest.fixture
def simple_series():
    np.random.seed(42)
    return pd.Series(100 + np.cumsum(np.random.randn(100) * 0.5), name="price")


@pytest.fixture
def returns_series():
    np.random.seed(42)
    return pd.Series(np.random.randn(100) * 0.01, name="returns")


class TestStatistical:
    def test_simple_returns(self, simple_series):
        from feature_engine.statistical import simple_returns
        result = simple_returns(simple_series)
        assert len(result) == len(simple_series)
        assert result.iloc[0] != result.iloc[0]
        assert not np.isnan(result.iloc[1])

    def test_log_returns(self, simple_series):
        from feature_engine.statistical import log_returns
        result = log_returns(simple_series)
        assert len(result) == len(simple_series)
        assert not np.isnan(result.iloc[1])

    def test_cumulative_returns(self, returns_series):
        from feature_engine.statistical import cumulative_returns
        result = cumulative_returns(returns_series)
        assert len(result) == len(returns_series)
        assert abs(result.iloc[0]) < 0.1

    def test_forward_returns(self, simple_series):
        from feature_engine.statistical import forward_returns
        result = forward_returns(simple_series, periods=5)
        assert len(result) == len(simple_series)
        assert result.iloc[-1] != result.iloc[-1]

    def test_overnight_returns(self, ohlcv_df):
        from feature_engine.statistical import overnight_returns
        result = overnight_returns(ohlcv_df["open"], ohlcv_df["close"])
        assert len(result) == len(ohlcv_df)

    def test_intraday_returns(self, ohlcv_df):
        from feature_engine.statistical import intraday_returns
        result = intraday_returns(ohlcv_df["open"], ohlcv_df["close"])
        assert len(result) == len(ohlcv_df)

    def test_rolling_mean(self, simple_series):
        from feature_engine.statistical import rolling_mean
        result = rolling_mean(simple_series, window=20)
        assert np.isnan(result.iloc[18])
        assert not np.isnan(result.iloc[19])

    def test_rolling_std(self, simple_series):
        from feature_engine.statistical import rolling_std
        result = rolling_std(simple_series, window=20)
        assert not np.isnan(result.iloc[19])
        assert result.iloc[19] >= 0

    def test_realized_volatility(self, simple_series):
        from feature_engine.statistical import realized_volatility
        result = realized_volatility(simple_series, window=20, annualize=True)
        assert not np.isnan(result.iloc[20])
        assert result.iloc[20] >= 0

    def test_parkinson_volatility(self, ohlcv_df):
        from feature_engine.statistical import parkinson_volatility
        result = parkinson_volatility(ohlcv_df["high"], ohlcv_df["low"], window=20)
        assert not np.isnan(result.iloc[20])
        assert result.iloc[20] >= 0

    def test_zscore(self, simple_series):
        from feature_engine.statistical import zscore
        result = zscore(simple_series, window=20)
        assert not np.isnan(result.iloc[19])

    def test_volume_zscore(self, ohlcv_df):
        from feature_engine.statistical import volume_zscore
        result = volume_zscore(ohlcv_df["volume"], window=20)
        assert not np.isnan(result.iloc[19])

    def test_volume_ratio(self, ohlcv_df):
        from feature_engine.statistical import volume_ratio
        result = volume_ratio(ohlcv_df["volume"], short_window=5, long_window=20)
        assert not np.isnan(result.iloc[19])
        assert result.iloc[19] > 0

    def test_beta(self, ohlcv_df):
        from feature_engine.statistical import beta
        asset_ret = ohlcv_df["close"].pct_change()
        bench_ret = ohlcv_df["close"].pct_change() * 0.8 + np.random.randn(len(ohlcv_df)) * 0.001
        result = beta(asset_ret, bench_ret, window=60)
        assert not np.isnan(result.iloc[60])

    def test_correlation(self, ohlcv_df):
        from feature_engine.statistical import correlation
        result = correlation(ohlcv_df["close"], ohlcv_df["volume"], window=20)
        assert not np.isnan(result.iloc[20])


class TestCustomFeatures:
    def test_smart_money(self, ohlcv_df):
        from feature_engine.custom import smart_money
        result = smart_money(ohlcv_df["close"], ohlcv_df["volume"], ohlcv_df["high"], ohlcv_df["low"])
        assert len(result) == len(ohlcv_df)
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_fii_strength(self):
        from feature_engine.custom import fii_strength
        np.random.seed(42)
        n = 100
        buy = pd.Series(np.random.uniform(100, 500, n))
        sell = pd.Series(np.random.uniform(100, 500, n))
        result = fii_strength(buy, sell, period=5)
        valid = result.dropna()
        assert (valid >= -100).all()
        assert (valid <= 100).all()

    def test_breakout_score(self, ohlcv_df):
        from feature_engine.custom import breakout_score
        result = breakout_score(ohlcv_df["high"], ohlcv_df["low"], ohlcv_df["close"], ohlcv_df["volume"])
        assert len(result) == len(ohlcv_df)
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_institutional_score(self):
        from feature_engine.custom import institutional_score
        np.random.seed(42)
        n = 100
        fii_buy = pd.Series(np.random.uniform(100, 500, n))
        fii_sell = pd.Series(np.random.uniform(100, 500, n))
        dii_buy = pd.Series(np.random.uniform(100, 500, n))
        dii_sell = pd.Series(np.random.uniform(100, 500, n))
        result = institutional_score(fii_buy, fii_sell, dii_buy, dii_sell)
        valid = result.dropna()
        assert (valid >= -100).all()
        assert (valid <= 100).all()

    def test_earnings_quality(self):
        from feature_engine.custom import earnings_quality
        n = 100
        revenue = pd.Series(np.linspace(100, 200, n))
        net_income = revenue * 0.1
        operating_cf = net_income * 0.9
        result = earnings_quality(revenue, net_income, operating_cf)
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_sector_rotation(self):
        from feature_engine.custom import sector_rotation
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            "IT": np.random.randn(n) * 0.01,
            "Bank": np.random.randn(n) * 0.01,
            "Pharma": np.random.randn(n) * 0.01,
        })
        result = sector_rotation(df, window=20)
        assert result.shape == df.shape

    def test_orderflow_score(self):
        from feature_engine.custom import orderflow_score
        np.random.seed(42)
        n = 100
        bid = pd.Series(np.random.randint(1000, 10000, n).astype(float))
        ask = pd.Series(np.random.randint(1000, 10000, n).astype(float))
        result = orderflow_score(bid, ask, period=10)
        valid = result.dropna()
        assert (valid >= -100).all()
        assert (valid <= 100).all()

    def test_news_sentiment(self):
        from feature_engine.custom import news_sentiment
        np.random.seed(42)
        sentiment = pd.Series(np.random.uniform(-1, 1, 100))
        result = news_sentiment(sentiment, period=5)
        valid = result.dropna()
        assert (valid >= -100).all()
        assert (valid <= 100).all()


class TestLabels:
    def test_trend_target(self, simple_series):
        from feature_engine.labels import trend_target
        result = trend_target(simple_series, horizon=5, threshold=0.02)
        assert len(result) == len(simple_series)
        valid = result.dropna()
        assert set(valid.unique()).issubset({0, 1})

    def test_breakout_target(self, ohlcv_df):
        from feature_engine.labels import breakout_target
        result = breakout_target(ohlcv_df["close"], ohlcv_df["high"], window=20, horizon=5, threshold=0.03)
        assert len(result) == len(ohlcv_df)
        valid = result.dropna()
        assert set(valid.unique()).issubset({0, 1})

    def test_classification_labels(self, simple_series):
        from feature_engine.labels import classification_labels
        result = classification_labels(simple_series, horizon=5)
        assert len(result) == len(simple_series)
        valid = result.dropna()
        assert valid.min() >= 0
        assert valid.max() <= 3

    def test_regression_target(self, simple_series):
        from feature_engine.labels import regression_target
        result = regression_target(simple_series, horizon=5)
        assert len(result) == len(simple_series)
        assert result.iloc[-1] != result.iloc[-1]

    def test_volatility_target(self, simple_series):
        from feature_engine.labels import volatility_target
        result = volatility_target(simple_series, horizon=20)
        assert len(result) == len(simple_series)
        valid = result.dropna()
        assert (valid >= 0).all()


class TestPreprocessing:
    def test_min_max_scale(self, ohlcv_df):
        from feature_engine.preprocessing import min_max_scale
        result = min_max_scale(ohlcv_df)
        for col in result.columns:
            assert result[col].min() >= 0
            assert result[col].max() <= 1

    def test_standard_scale(self, ohlcv_df):
        from feature_engine.preprocessing import standard_scale
        result = standard_scale(ohlcv_df)
        for col in result.columns:
            assert abs(result[col].mean()) < 0.1

    def test_robust_scale(self, ohlcv_df):
        from feature_engine.preprocessing import robust_scale
        result = robust_scale(ohlcv_df)
        assert result.shape == ohlcv_df.shape

    def test_log_normalize(self, ohlcv_df):
        from feature_engine.preprocessing import log_normalize
        result = log_normalize(ohlcv_df)
        assert result.shape == ohlcv_df.shape

    def test_rank_normalize(self, ohlcv_df):
        from feature_engine.preprocessing import rank_normalize
        result = rank_normalize(ohlcv_df)
        for col in result.columns:
            assert result[col].max() <= 1

    def test_winsorize(self):
        from feature_engine.preprocessing import winsorize
        s = pd.Series([1, 2, 3, 4, 5, 100])
        result = winsorize(s, lower=0.1, upper=0.9)
        assert result.iloc[-1] < 100

    def test_fill_missing_forward(self, ohlcv_df):
        from feature_engine.preprocessing import fill_missing
        df = ohlcv_df.copy()
        df.iloc[5, 0] = np.nan
        result = fill_missing(df, strategy="forward_fill")
        assert not result.isna().iloc[5, 0]

    def test_fill_missing_zero(self, ohlcv_df):
        from feature_engine.preprocessing import fill_missing
        df = ohlcv_df.copy()
        df.iloc[5, 0] = np.nan
        result = fill_missing(df, strategy="zero")
        assert result.iloc[5, 0] == 0

    def test_feature_pipeline(self, ohlcv_df):
        from feature_engine.preprocessing import FeaturePipeline
        pipeline = FeaturePipeline()
        pipeline.add_step("winsorize", lower=0.01, upper=0.99)
        pipeline.add_step("standard_scale")
        result = pipeline.fit_transform(ohlcv_df)
        assert result.shape == ohlcv_df.shape
        desc = pipeline.describe()
        assert len(desc) == 2

    def test_feature_pipeline_invalid_step(self):
        from feature_engine.preprocessing import FeaturePipeline
        pipeline = FeaturePipeline()
        with pytest.raises(ValueError):
            pipeline.add_step("nonexistent_step")


class TestFeatureEngineImports:
    def test_import_statistical(self):
        from feature_engine import statistical
        assert hasattr(statistical, "simple_returns")
        assert hasattr(statistical, "realized_volatility")

    def test_import_custom(self):
        from feature_engine import custom
        assert hasattr(custom, "breakout_score")
        assert hasattr(custom, "smart_money")

    def test_import_labels(self):
        from feature_engine import labels
        assert hasattr(labels, "trend_target")
        assert hasattr(labels, "regression_target")

    def test_import_preprocessing(self):
        from feature_engine import preprocessing
        assert hasattr(preprocessing, "min_max_scale")
        assert hasattr(preprocessing, "FeaturePipeline")
