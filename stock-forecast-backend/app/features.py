"""Feature engineering — MUST match training pipeline in sonnet_v3.

Pure-numpy implementation (no pandas at runtime) to keep the container small.
"""
from __future__ import annotations

from typing import List

import numpy as np

FEATURE_COLS: List[str] = [
    "ret_1d",
    "ret_5d",
    "ret_10d",
    "ret_20d",
    "log_ret",
    "price_vs_sma20",
    "price_vs_sma50",
    "trend_up",
    "rsi_14",
    "macd_histogram",
    "volatility_20",
    "volume_ratio",
    "usd_ret_5d",
    "price_position",
    "bb_position",
]


def _pct_change(x: np.ndarray, n: int) -> np.ndarray:
    """Matches pandas Series.pct_change(n)."""
    shifted = np.concatenate([np.full(n, np.nan), x[:-n]]) if n > 0 else x.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        return (x - shifted) / shifted


def _rolling_mean(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=np.float64)
    if n <= 0 or len(x) < n:
        return out
    xf = np.where(np.isnan(x), 0.0, x).astype(np.float64)
    mask = (~np.isnan(x)).astype(np.float64)
    csum = np.concatenate([[0.0], np.cumsum(xf)])
    cmask = np.concatenate([[0.0], np.cumsum(mask)])
    win_sum = csum[n:] - csum[:-n]
    win_count = cmask[n:] - cmask[:-n]
    valid = win_count >= n
    mean = np.full(len(x) - n + 1, np.nan, dtype=np.float64)
    mean[valid] = win_sum[valid] / n
    out[n - 1:] = mean
    return out


def _rolling_std(x: np.ndarray, n: int, ddof: int = 1) -> np.ndarray:
    """pandas default: ddof=1 (sample std)."""
    out = np.full_like(x, np.nan, dtype=np.float64)
    if n <= 0 or n - ddof <= 0:
        return out
    xd = x.astype(np.float64)
    for i in range(n - 1, len(x)):
        window = xd[i - n + 1:i + 1]
        if np.isnan(window).any():
            continue
        out[i] = np.std(window, ddof=ddof)
    return out


def _rolling_max(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=np.float64)
    xd = x.astype(np.float64)
    for i in range(n - 1, len(x)):
        window = xd[i - n + 1:i + 1]
        if np.isnan(window).any():
            continue
        out[i] = np.max(window)
    return out


def _rolling_min(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=np.float64)
    xd = x.astype(np.float64)
    for i in range(n - 1, len(x)):
        window = xd[i - n + 1:i + 1]
        if np.isnan(window).any():
            continue
        out[i] = np.min(window)
    return out


def _ewm(x: np.ndarray, span: int) -> np.ndarray:
    """Matches pandas ewm(span=span, adjust=False).mean()."""
    alpha = 2.0 / (span + 1.0)
    out = np.full_like(x, np.nan, dtype=np.float64)
    prev = np.nan
    for i, val in enumerate(x):
        if np.isnan(val):
            out[i] = prev
            continue
        if np.isnan(prev):
            prev = float(val)
        else:
            prev = alpha * float(val) + (1 - alpha) * prev
        out[i] = prev
    return out


def _rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    delta = np.concatenate([[np.nan], np.diff(close)])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain[np.isnan(delta)] = np.nan
    loss[np.isnan(delta)] = np.nan
    avg_gain = _rolling_mean(gain, period)
    avg_loss = _rolling_mean(loss, period)
    rs = avg_gain / (avg_loss + 1e-12)
    return 100 - (100 / (1 + rs))


def _merge_usd_backward(dates: np.ndarray, usd_dates: np.ndarray, usd_values: np.ndarray) -> np.ndarray:
    """merge_asof(direction='backward'): for each date d pick latest usd where usd_dates <= d."""
    if len(usd_dates) == 0:
        return np.full(len(dates), np.nan, dtype=np.float64)
    out = np.full(len(dates), np.nan, dtype=np.float64)
    # Assume usd_dates is sorted ascending
    order = np.argsort(usd_dates)
    usd_dates_s = usd_dates[order]
    usd_vals_s = usd_values[order]
    # searchsorted: indices into usd_dates_s where each date would be inserted to keep sorted (right)
    # we want the largest index i such that usd_dates_s[i] <= date → idx = searchsorted(..., side='right') - 1
    idx = np.searchsorted(usd_dates_s, dates, side="right") - 1
    valid = idx >= 0
    out[valid] = usd_vals_s[idx[valid]]
    return out


def build_features(
    dates: np.ndarray,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    usd_dates: np.ndarray | None,
    usd_values: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Replicates add_stable_features() from training.

    Returns (feature_matrix, kept_dates), where feature_matrix has shape
    (N_remaining, 15) after dropping rows with NaN.
    """
    close = close.astype(np.float64)
    high = high.astype(np.float64)
    low = low.astype(np.float64)
    volume = volume.astype(np.float64)

    ret_1d = _pct_change(close, 1)
    ret_5d = _pct_change(close, 5)
    ret_10d = _pct_change(close, 10)
    ret_20d = _pct_change(close, 20)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret = np.log(close / np.concatenate([[np.nan], close[:-1]]))

    sma20 = _rolling_mean(close, 20)
    sma50 = _rolling_mean(close, 50)
    sma200 = _rolling_mean(close, 200)

    price_vs_sma20 = close / (sma20 + 1e-12) - 1.0
    price_vs_sma50 = close / (sma50 + 1e-12) - 1.0
    # For tickers with < 200 bars of history, fall back to sma50 so we can
    # still produce a feature vector.
    sma_for_trend = np.where(np.isnan(sma200), sma50, sma200)
    trend_up = (close > sma_for_trend).astype(np.float64)
    trend_up[np.isnan(sma_for_trend)] = np.nan

    rsi_14 = _rsi(close, 14)

    ema12 = _ewm(close, 12)
    ema26 = _ewm(close, 26)
    macd = ema12 - ema26
    macd_signal = _ewm(macd, 9)
    macd_histogram = macd - macd_signal

    volatility_20 = _rolling_std(ret_1d, 20, ddof=1)

    vol_mean20 = _rolling_mean(volume, 20)
    volume_ratio = volume / (vol_mean20 + 1e-8)

    if usd_dates is not None and usd_values is not None and len(usd_dates) > 0:
        usd_rub = _merge_usd_backward(dates, usd_dates, usd_values.astype(np.float64))
        usd_ret_5d = _pct_change(usd_rub, 5)
    else:
        usd_ret_5d = np.zeros_like(close)

    high20 = _rolling_max(high, 20)
    low20 = _rolling_min(low, 20)
    price_position = (close - low20) / ((high20 - low20) + 1e-8)

    bb_mid = _rolling_mean(close, 20)
    bb_std = _rolling_std(close, 20, ddof=1)
    bb_position = (close - bb_mid) / ((2 * bb_std) + 1e-8)

    cols = [
        ret_1d,
        ret_5d,
        ret_10d,
        ret_20d,
        log_ret,
        price_vs_sma20,
        price_vs_sma50,
        trend_up,
        rsi_14,
        macd_histogram,
        volatility_20,
        volume_ratio,
        usd_ret_5d,
        price_position,
        bb_position,
    ]
    # When close has a NaN due to upstream gap, drop that row as well.
    matrix = np.stack(cols, axis=1)  # (N, 15)
    valid = ~np.isnan(matrix).any(axis=1)
    valid &= ~np.isnan(close)
    return matrix[valid].astype(np.float32), dates[valid]
