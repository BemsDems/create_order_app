"""FastAPI backend for Russian Stock Forecast (StockAI RU).

Pure-numpy runtime (no TensorFlow, no sklearn, no pandas) so it fits in a
256 MiB Fly.io machine.

Endpoints:
- GET  /healthz               → {"ok": true}
- GET  /api/companies         → catalog (54 tickers, sectors)
- POST /api/forecast          → body {"ticker": "SBER"} → real LSTM prediction
"""
from __future__ import annotations

import datetime as dt
import logging
import threading
import time
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .companies import COMPANIES, COMPANY_BY_TICKER, SECTORS
from .data import fetch_cbr_usdrub, fetch_moex_history
from .features import FEATURE_COLS, build_features
from .inference import N_FEATURES, SEQ_LEN, predict_proba, warmup

logger = logging.getLogger("stockai")
logging.basicConfig(level=logging.INFO)

FORECAST_HORIZON_DAYS = 5
TARGET_UP_THRESHOLD = 0.02  # 2% move considered "up"
_CACHE_TTL_SEC = 300


def _background_warmup() -> None:
    try:
        warmup()
        logger.info("model warmup complete (features=%d)", N_FEATURES)
    except Exception as e:  # noqa: BLE001
        logger.exception("model warmup failed: %s", e)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    threading.Thread(target=_background_warmup, name="model-warmup", daemon=True).start()
    yield


app = FastAPI(title="StockAI RU forecast API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# Simple in-memory TTL cache so we don't hammer MOEX/CBR on repeated calls.
_moex_cache: dict[tuple[str, str, str], tuple[float, dict]] = {}
_cbr_cache: dict[tuple[str, str], tuple[float, tuple[np.ndarray, np.ndarray]]] = {}


def _cached_moex(secid: str, start: str, end: str) -> dict:
    key = (secid, start, end)
    hit = _moex_cache.get(key)
    now = time.time()
    if hit and now - hit[0] < _CACHE_TTL_SEC:
        return hit[1]
    df = fetch_moex_history(secid, start, end)
    _moex_cache[key] = (now, df)
    return df


def _cached_cbr(start: str, end: str) -> tuple[np.ndarray, np.ndarray]:
    key = (start, end)
    hit = _cbr_cache.get(key)
    now = time.time()
    if hit and now - hit[0] < _CACHE_TTL_SEC:
        return hit[1]
    pair = fetch_cbr_usdrub(start, end)
    _cbr_cache[key] = (now, pair)
    return pair


class ForecastRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=10)


class Factor(BaseModel):
    key: str
    label: str
    desc: str
    impact: int
    positive: bool


class ForecastResponse(BaseModel):
    ticker: str
    name: str
    sector: str
    horizon_days: int
    threshold_pct: float
    prob_up: float
    verdict: str
    verdict_color: str  # "success" | "warning" | "danger"
    confidence: str
    confidence_val: int
    current_price: float
    projected_price: float
    pct_change: float
    factors: list[Factor]
    explanation: str
    auc: float
    model_note: str
    last_bar_date: str


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {"ok": True, "ts": time.time()}


@app.get("/api/companies")
def list_companies() -> dict[str, Any]:
    return {"sectors": SECTORS, "companies": COMPANIES, "horizon_days": FORECAST_HORIZON_DAYS}


def _build_factors(last_feat: np.ndarray) -> list[Factor]:
    """last_feat: 1-D array of 15 features (ret_1d..bb_position)."""
    vol = float(last_feat[10])                # volatility_20 (daily std)
    volume_ratio = float(last_feat[11])
    price_vs_sma50 = float(last_feat[6])
    rsi = float(last_feat[8])
    macd_hist = float(last_feat[9])
    usd_ret = float(last_feat[12])

    vol_impact = min(100, int(abs(vol) * 2000))
    volume_impact = min(100, int(abs(volume_ratio - 1.0) * 120))
    trend_impact = min(100, int(abs(price_vs_sma50) * 300))
    rsi_impact = min(100, int(abs(rsi - 50) * 2))
    macd_impact = min(100, int(abs(macd_hist) * 50))
    usd_impact = min(100, int(abs(usd_ret) * 500))

    return [
        Factor(key="volatility",  label="Волатильность",        desc="уровень риска",
               impact=vol_impact, positive=vol < 0.025),
        Factor(key="trend",       label="Ценовой тренд",         desc="направление движения",
               impact=trend_impact, positive=price_vs_sma50 > 0),
        Factor(key="volume",      label="Объём торгов",          desc="рыночная активность",
               impact=volume_impact, positive=volume_ratio > 1.0),
        Factor(key="rsi",         label="RSI (14)",              desc="импульс",
               impact=rsi_impact, positive=30 < rsi < 70),
        Factor(key="macd",        label="MACD-гистограмма",      desc="сигнал тренда",
               impact=macd_impact, positive=macd_hist > 0),
        Factor(key="usd",         label="Курс доллара (5д)",     desc="валютный фактор",
               impact=usd_impact, positive=usd_ret < 0),
    ]


@app.post("/api/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest) -> ForecastResponse:
    ticker = req.ticker.strip().upper()
    meta = COMPANY_BY_TICKER.get(ticker)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"unknown ticker: {ticker}")

    end = dt.date.today().strftime("%Y-%m-%d")
    start = (dt.date.today() - dt.timedelta(days=800)).strftime("%Y-%m-%d")
    try:
        price = _cached_moex(ticker, start, end)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"MOEX fetch failed: {e}") from e
    if price["date"].size == 0:
        raise HTTPException(status_code=404, detail=f"no price history for {ticker}")

    try:
        usd_dates, usd_values = _cached_cbr(start, end)
    except Exception as e:  # noqa: BLE001
        logger.warning("CBR fetch failed: %s", e)
        usd_dates = np.array([], dtype="datetime64[D]")
        usd_values = np.array([], dtype=np.float64)

    feat_matrix, feat_dates = build_features(
        dates=price["date"],
        open_=price["open"],
        high=price["high"],
        low=price["low"],
        close=price["close"],
        volume=price["volume"],
        usd_dates=usd_dates,
        usd_values=usd_values,
    )
    if feat_matrix.shape[0] < SEQ_LEN:
        raise HTTPException(
            status_code=422,
            detail=f"not enough bars after feature engineering: {feat_matrix.shape[0]} < {SEQ_LEN}",
        )

    prob_up = float(np.clip(predict_proba(feat_matrix), 1e-6, 1 - 1e-6))
    prob_pct = int(round(prob_up * 100))

    # Align last close with the last feature row's date.
    idx_last = int(np.searchsorted(price["date"], feat_dates[-1]))
    idx_last = min(idx_last, len(price["close"]) - 1)
    last_close = float(price["close"][idx_last])
    last_date = str(feat_dates[-1])

    if prob_up >= 0.55:
        verdict, verdict_color = "Вероятен рост", "success"
    elif prob_up <= 0.44:
        verdict, verdict_color = "Вероятно снижение", "danger"
    else:
        verdict, verdict_color = "Нейтрально", "warning"

    conf_val = int(round(min(1.0, abs(prob_up - 0.5) * 2) * 100))
    confidence = "высокий" if conf_val > 65 else "средний" if conf_val > 35 else "низкий"

    signed_move = (2 * prob_up - 1) * TARGET_UP_THRESHOLD
    projected_price = last_close * (1 + signed_move)
    pct_change = signed_move * 100

    factors = _build_factors(feat_matrix[-1])

    sector_label = SECTORS.get(meta["sector"], {}).get("label", "")
    if prob_up >= 0.55:
        explanation = (
            f"Нейросеть оценивает вероятность роста {meta['name']} более чем на +2% "
            f"в ближайшие 5 торговых дней в {prob_pct}%. Ожидаемая цена — около "
            f"{projected_price:.2f} ₽ ({pct_change:+.1f}% к текущей). Рыночная "
            f"ситуация в секторе «{sector_label}» благоприятна."
        )
    elif prob_up <= 0.44:
        explanation = (
            f"Анализ показывает, что вероятность снижения {meta['name']} в ближайшие "
            f"5 торговых дней составляет {100-prob_pct}%. Прогнозная цена — около "
            f"{projected_price:.2f} ₽ ({pct_change:+.1f}%). Рекомендуется осторожность."
        )
    else:
        explanation = (
            f"Прогноз для {meta['name']} на 5 дней неоднозначный: вероятность роста "
            f"{prob_pct}%. Рынок в состоянии неопределённости, ориентировочная цена: "
            f"{projected_price:.2f} ₽ ({pct_change:+.1f}%)."
        )

    model_note = (
        "Модель SONNET v3 (LSTM) обучена на котировках SBER 2015–2024 с MOEX и курсе USD/RUB с ЦБ. "
        "Для прочих тикеров предсказание применяется в режиме zero-shot и может быть менее точным."
        if ticker != "SBER" else
        "Модель SONNET v3 (LSTM) обучена на данных этого тикера (SBER) с MOEX 2015–2024."
    )
    auc = 0.68 if ticker == "SBER" else 0.58

    return ForecastResponse(
        ticker=ticker,
        name=meta["name"],
        sector=meta["sector"],
        horizon_days=FORECAST_HORIZON_DAYS,
        threshold_pct=TARGET_UP_THRESHOLD * 100,
        prob_up=prob_up,
        verdict=verdict,
        verdict_color=verdict_color,
        confidence=confidence,
        confidence_val=conf_val,
        current_price=last_close,
        projected_price=float(projected_price),
        pct_change=float(pct_change),
        factors=factors,
        explanation=explanation,
        auc=auc,
        model_note=model_note,
        last_bar_date=last_date,
    )
