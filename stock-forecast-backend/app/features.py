"""Feature engineering — replicates the new TCN multi-horizon training pipeline.

Order MUST match the RobustScaler fitted during training. Total = 28 features:

    Technical (14):
        0  logret_1
        1  logret_2
        2  logret_3
        3  logret_5
        4  logret_10
        5  trend_up_20
        6  trend_up_200
        7  vol_rel
        8  vol_spike
        9  rsi_14
        10 rsi_oversold
        11 rsi_overbought
        12 price_pos_20
        13 volatility_20

    Fundamental — dividends (3):
        14 div_yield_ttm
        15 days_since_last_div
        16 div_yield_is_missing

    Macro — USD/RUB (3):
        17 usdrub_logret_1
        18 usdrub_logret_5
        19 usdrub_volatility_20

    Macro — Brent proxy (2): zeros at runtime
        20 brent_logret_1
        21 brent_logret_5

    Macro — IMOEX (3):
        22 imoex_logret_1
        23 imoex_logret_5
        24 imoex_logret_20

    Yahoo fundamentals (3): zeros at runtime
        25..27

The Brent and Yahoo features were near-zero / missing during training as well
(scaler center=0, scale=1 → identity transform), so feeding zeros at inference
matches the training distribution.
"""
from __future__ import annotations

from typing import List

import numpy as np

FEATURE_COLS: List[str] = [
    "logret_1",
    "logret_2",
    "logret_3",
    "logret_5",
    "logret_10",
    "trend_up_20",
    "trend_up_200",
    "vol_rel",
    "vol_spike",
    "rsi_14",
    "rsi_oversold",
    "rsi_overbought",
    "price_pos_20",
    "volatility_20",
    "div_yield_ttm",
    "days_since_last_div",
    "div_yield_is_missing",
    "usdrub_logret_1",
    "usdrub_logret_5",
    "usdrub_volatility_20",
    "brent_logret_1",
    "brent_logret_5",
    "imoex_logret_1",
    "imoex_logret_5",
    "imoex_logret_20",
    "yahoo_f0",
    "yahoo_f1",
    "yahoo_f2",
]
N_FEATURES = len(FEATURE_COLS)
assert N_FEATURES == 28


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


def _ffill(x: np.ndarray) -> np.ndarray:
    """Forward-fill NaNs in 1D array."""
    out = x.astype(np.float64).copy()
    last = np.nan
    for i in range(len(out)):
        if np.isnan(out[i]):
            out[i] = last
        else:
            last = out[i]
    return out


def _logret_clipped(close: np.ndarray, lag: int) -> np.ndarray:
    """log(close / close.shift(lag)) with ratio clipped to [0.5, 2.0]."""
    if lag <= 0 or lag >= len(close):
        return np.full_like(close, np.nan, dtype=np.float64)
    shifted = np.concatenate([np.full(lag, np.nan), close[:-lag]])
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = close / shifted
    ratio = np.clip(ratio, 0.5, 2.0)
    out = np.log(ratio)
    out[~np.isfinite(out)] = np.nan
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
    return 100.0 - (100.0 / (1.0 + rs))


def _merge_backward(
    target_dates: np.ndarray,
    src_dates: np.ndarray,
    src_values: np.ndarray,
) -> np.ndarray:
    """For each target date d pick latest src where src_dates <= d."""
    if len(src_dates) == 0:
        return np.full(len(target_dates), np.nan, dtype=np.float64)
    order = np.argsort(src_dates)
    sd = src_dates[order]
    sv = src_values.astype(np.float64)[order]
    idx = np.searchsorted(sd, target_dates, side="right") - 1
    out = np.full(len(target_dates), np.nan, dtype=np.float64)
    valid = idx >= 0
    out[valid] = sv[idx[valid]]
    return out


def _ttm_div_yield(
    target_dates: np.ndarray,
    target_close: np.ndarray,
    div_dates: np.ndarray,
    div_values: np.ndarray,
    lag_days: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute TTM div yield, days_since_last_div (0..1), and is_missing flag."""
    n = len(target_dates)
    div_yield = np.zeros(n, dtype=np.float64)
    days_since = np.full(n, 1.0, dtype=np.float64)
    is_missing = np.ones(n, dtype=np.float64)

    if len(div_dates) == 0:
        return div_yield, days_since, is_missing

    eff = div_dates + np.timedelta64(int(lag_days), "D")
    one_year = np.timedelta64(365, "D")

    last_idx_arr = np.searchsorted(eff, target_dates, side="right") - 1
    for i in range(n):
        d = target_dates[i]
        # TTM: sum of dividends in (d - 365, d]
        lo = d - one_year
        mask = (eff > lo) & (eff <= d)
        ttm = float(div_values[mask].sum()) if mask.any() else 0.0
        close = float(target_close[i])
        if close > 0 and ttm > 0:
            div_yield[i] = float(np.clip(ttm / close, 0.0, 0.30))
        # days since last dividend
        li = int(last_idx_arr[i])
        if li >= 0:
            delta_days = (d - eff[li]).astype("timedelta64[D]").astype(int)
            delta_days = max(0, min(365, delta_days))
            days_since[i] = delta_days / 365.0
            is_missing[i] = 0.0
    return div_yield, days_since, is_missing


def build_features(
    *,
    dates: np.ndarray,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    usd_dates: np.ndarray | None = None,
    usd_values: np.ndarray | None = None,
    imoex_dates: np.ndarray | None = None,
    imoex_values: np.ndarray | None = None,
    div_dates: np.ndarray | None = None,
    div_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a (N_kept, 28) feature matrix and the date vector for kept rows.

    Replicates `build_features_one()` + macro merge + dividend merge from the
    training pipeline.
    """
    close = close.astype(np.float64)
    high = high.astype(np.float64)
    low = low.astype(np.float64)
    volume = volume.astype(np.float64)

    # --- Technical features (14) ------------------------------------------
    logret_1 = _logret_clipped(close, 1)
    logret_2 = _logret_clipped(close, 2)
    logret_3 = _logret_clipped(close, 3)
    logret_5 = _logret_clipped(close, 5)
    logret_10 = _logret_clipped(close, 10)

    sma_20 = _rolling_mean(close, 20)
    sma_200 = _rolling_mean(close, 200)
    trend_up_20 = (close > sma_20).astype(np.float64)
    trend_up_20[np.isnan(sma_20)] = np.nan
    sma_200_ff = _ffill(sma_200)
    trend_up_200 = (close > sma_200_ff).astype(np.float64)
    # Training fills trend_up_200 NaNs with 0; mirror that.
    trend_up_200[np.isnan(sma_200_ff)] = 0.0

    vol_ma_20 = _rolling_mean(volume, 20)
    vol_rel = volume / (vol_ma_20 + 1e-9)
    vol_rel = np.clip(vol_rel, 0.1, 3.0)
    vol_spike = (vol_rel > 2.0).astype(np.float64)
    vol_spike[np.isnan(vol_ma_20)] = np.nan

    rsi_14 = _rsi(close, 14)
    rsi_14 = np.clip(rsi_14, 0.0, 100.0)
    rsi_oversold = (rsi_14 < 30).astype(np.float64)
    rsi_overbought = (rsi_14 > 70).astype(np.float64)
    rsi_oversold[np.isnan(rsi_14)] = np.nan
    rsi_overbought[np.isnan(rsi_14)] = np.nan

    high_20 = _rolling_max(high, 20)
    low_20 = _rolling_min(low, 20)
    price_pos_20 = (close - low_20) / ((high_20 - low_20) + 1e-9)
    price_pos_20 = np.clip(price_pos_20, 0.0, 1.0)

    volatility_20 = _rolling_std(logret_1, 20, ddof=1)
    volatility_20 = np.clip(volatility_20, 0.0, 0.1)

    # --- Macro: USD/RUB (3) -----------------------------------------------
    if usd_dates is not None and usd_values is not None and len(usd_dates) > 0:
        usd_close = _merge_backward(dates, usd_dates, usd_values)
        usd_close = _ffill(usd_close)
        usdrub_logret_1 = _logret_clipped(usd_close, 1)
        usdrub_logret_5 = _logret_clipped(usd_close, 5)
        usdrub_vol_20 = _rolling_std(usdrub_logret_1, 20, ddof=1)
        usdrub_vol_20 = np.clip(usdrub_vol_20, 0.0, 0.1)
    else:
        usdrub_logret_1 = np.zeros_like(close)
        usdrub_logret_5 = np.zeros_like(close)
        usdrub_vol_20 = np.zeros_like(close)

    # --- Macro: IMOEX (3) -------------------------------------------------
    if imoex_dates is not None and imoex_values is not None and len(imoex_dates) > 0:
        imoex_close = _merge_backward(dates, imoex_dates, imoex_values)
        imoex_close = _ffill(imoex_close)
        imoex_logret_1 = _logret_clipped(imoex_close, 1)
        imoex_logret_5 = _logret_clipped(imoex_close, 5)
        imoex_logret_20 = _logret_clipped(imoex_close, 20)
    else:
        imoex_logret_1 = np.zeros_like(close)
        imoex_logret_5 = np.zeros_like(close)
        imoex_logret_20 = np.zeros_like(close)

    # --- Brent proxy (2): zeros (matches training: scaler center=0, scale=1)
    zeros = np.zeros_like(close)

    # --- Dividends (3) ----------------------------------------------------
    if div_dates is not None and div_values is not None and len(div_dates) > 0:
        div_y, days_since, div_missing = _ttm_div_yield(dates, close, div_dates, div_values)
    else:
        div_y = np.zeros_like(close)
        days_since = np.ones_like(close)
        div_missing = np.ones_like(close)

    # --- Yahoo (3): zeros (matches training: scaler center=0, scale=1) ----

    cols = [
        logret_1, logret_2, logret_3, logret_5, logret_10,
        trend_up_20, trend_up_200,
        vol_rel, vol_spike,
        rsi_14, rsi_oversold, rsi_overbought,
        price_pos_20, volatility_20,
        div_y, days_since, div_missing,
        usdrub_logret_1, usdrub_logret_5, usdrub_vol_20,
        zeros, zeros,                                    # brent_logret_1/5
        imoex_logret_1, imoex_logret_5, imoex_logret_20,
        zeros, zeros, zeros,                             # yahoo_f0..2
    ]
    matrix = np.stack(cols, axis=1)  # (N, 28)

    # Mirror training: drop rows with NaN in critical columns, then ffill/fillna.
    critical = [logret_1, logret_10, sma_20, vol_ma_20, rsi_14, price_pos_20]
    crit_mat = np.stack(critical, axis=1)
    keep = ~np.isnan(crit_mat).any(axis=1)
    matrix = matrix[keep]
    kept_dates = dates[keep]

    # Replace any remaining NaN/inf with 0 (training: fillna(0.0)).
    matrix = np.where(np.isfinite(matrix), matrix, 0.0)
    return matrix.astype(np.float32), kept_dates
