"""FastAPI backend for Russian Stock Forecast (StockAI RU).

Pure-numpy runtime (no TensorFlow, no sklearn, no pandas) so it fits in a
shared-cpu-1x Fly.io machine (≤ 256 MiB).

Three pre-trained TCN models are bundled and routed by horizon:
    - short  (trained on 5d):   horizons 5, 10
    - medium (trained on 30d):  horizons 30, 60
    - long   (trained on 120d): horizons 120, 240, 365

Endpoints:
- GET  /healthz               → {"ok": true}
- GET  /api/companies         → catalog (54 tickers, sectors, allowed horizons)
- POST /api/forecast          → body {"ticker": "SBER", "horizon": 30}
                                 → real TCN prediction
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
from pydantic import BaseModel, Field, field_validator

from .companies import COMPANIES, COMPANY_BY_TICKER, SECTORS
from .data import (
    fetch_cbr_usdrub,
    fetch_imoex_history,
    fetch_moex_dividends,
    fetch_moex_history,
    fetch_usd000_history,
)
from .features import FEATURE_COLS, build_features
from .inference import (
    ALLOWED_HORIZONS,
    HORIZON_GROUPS,
    N_FEATURES,
    SEQ_LEN,
    horizon_to_group,
    predict_proba,
    warmup,
)

logger = logging.getLogger("stockai")
logging.basicConfig(level=logging.INFO)

DEFAULT_HORIZON_DAYS = 5
_CACHE_TTL_SEC = 300


def _background_warmup() -> None:
    try:
        warmup()
        logger.info("model warmup complete (features=%d, horizons=%s)", N_FEATURES, list(ALLOWED_HORIZONS))
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


_moex_cache: dict[tuple[str, str, str], tuple[float, dict]] = {}
_macro_cache: dict[str, tuple[float, Any]] = {}
_div_cache: dict[str, tuple[float, tuple[np.ndarray, np.ndarray]]] = {}


def _cache_get(d: dict, key, ttl: float = _CACHE_TTL_SEC):
    hit = d.get(key)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    return None


def _cache_set(d: dict, key, value) -> None:
    d[key] = (time.time(), value)


def _cached_moex(secid: str, start: str, end: str) -> dict:
    key = (secid, start, end)
    cached = _cache_get(_moex_cache, key)
    if cached is not None:
        return cached
    df = fetch_moex_history(secid, start, end)
    _cache_set(_moex_cache, key, df)
    return df


def _cached_usd(start: str, end: str) -> tuple[np.ndarray, np.ndarray]:
    """USD/RUB time series. CBR is the primary source (still updated daily);
    fall back to MOEX USD000UTSTOM for older history if CBR fails."""
    key = f"usd:{start}:{end}"
    cached = _cache_get(_macro_cache, key)
    if cached is not None:
        return cached
    try:
        d, v = fetch_cbr_usdrub(start, end)
        if d.size == 0:
            raise RuntimeError("empty CBR USD")
    except Exception as e:  # noqa: BLE001
        logger.warning("CBR USD fetch failed (%s); falling back to MOEX", e)
        try:
            d, v = fetch_usd000_history(start, end)
        except Exception as e2:  # noqa: BLE001
            logger.warning("MOEX USD fetch failed: %s", e2)
            d = np.array([], dtype="datetime64[D]")
            v = np.array([], dtype=np.float64)
    _cache_set(_macro_cache, key, (d, v))
    return d, v


def _cached_imoex(start: str, end: str) -> tuple[np.ndarray, np.ndarray]:
    key = f"imoex:{start}:{end}"
    cached = _cache_get(_macro_cache, key)
    if cached is not None:
        return cached
    try:
        d, v = fetch_imoex_history(start, end)
    except Exception as e:  # noqa: BLE001
        logger.warning("IMOEX fetch failed: %s", e)
        d = np.array([], dtype="datetime64[D]")
        v = np.array([], dtype=np.float64)
    _cache_set(_macro_cache, key, (d, v))
    return d, v


def _cached_dividends(secid: str) -> tuple[np.ndarray, np.ndarray]:
    cached = _cache_get(_div_cache, secid, ttl=24 * 3600)
    if cached is not None:
        return cached
    pair = fetch_moex_dividends(secid)
    _cache_set(_div_cache, secid, pair)
    return pair


class ForecastRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=10)
    horizon: int = Field(default=DEFAULT_HORIZON_DAYS)

    @field_validator("horizon")
    @classmethod
    def _check_horizon(cls, v: int) -> int:
        if v not in ALLOWED_HORIZONS:
            raise ValueError(f"horizon must be one of {sorted(ALLOWED_HORIZONS)}")
        return v


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
    model_group: str  # "short" | "medium" | "long"


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {"ok": True, "ts": time.time()}


@app.get("/api/companies")
def list_companies() -> dict[str, Any]:
    horizon_groups = [
        {
            "name": name,
            "horizons": list(info["horizons"]),
            "trained_on": info["trained_on"],
            "threshold_pct": info["thr"] * 100,
        }
        for name, info in HORIZON_GROUPS.items()
    ]
    return {
        "sectors": SECTORS,
        "companies": COMPANIES,
        "horizon_groups": horizon_groups,
        "default_horizon": DEFAULT_HORIZON_DAYS,
    }


def _build_factors(last_feat: np.ndarray) -> list[Factor]:
    """last_feat: 1-D array of 28 features (see FEATURE_COLS for order)."""
    # Indices into FEATURE_COLS:
    # 0..4 logret_{1,2,3,5,10}; 7 vol_rel; 8 vol_spike; 9 rsi_14;
    # 13 volatility_20; 14 div_yield_ttm; 17 usdrub_logret_1;
    # 22 imoex_logret_1.
    logret_5 = float(last_feat[3])
    vol_rel = float(last_feat[7])
    rsi = float(last_feat[9])
    volatility = float(last_feat[13])
    div_yield = float(last_feat[14])
    usd_lr1 = float(last_feat[17])
    imoex_lr1 = float(last_feat[22])

    vol_impact = min(100, int(abs(volatility) * 2000))
    volume_impact = min(100, int(abs(vol_rel - 1.0) * 120))
    trend_impact = min(100, int(abs(logret_5) * 1000))
    rsi_impact = min(100, int(abs(rsi - 50) * 2))
    div_impact = min(100, int(div_yield * 600))
    usd_impact = min(100, int(abs(usd_lr1) * 5000))
    imoex_impact = min(100, int(abs(imoex_lr1) * 5000))

    return [
        Factor(key="volatility", label="Волатильность",        desc="уровень риска",
               impact=vol_impact, positive=volatility < 0.025),
        Factor(key="trend",      label="Тренд цены (5д)",       desc="направление движения",
               impact=trend_impact, positive=logret_5 > 0),
        Factor(key="volume",     label="Объём торгов",          desc="рыночная активность",
               impact=volume_impact, positive=vol_rel > 1.0),
        Factor(key="dividend",   label="Дивидендная доходность", desc="TTM",
               impact=div_impact, positive=div_yield > 0.04),
        Factor(key="imoex",      label="Рынок (IMOEX, 1д)",      desc="фон рынка",
               impact=imoex_impact, positive=imoex_lr1 > 0),
        Factor(key="usd",        label="Курс USD/RUB (1д)",      desc="валютный фактор",
               impact=usd_impact, positive=usd_lr1 < 0),
        Factor(key="rsi",        label="RSI (14)",               desc="импульс",
               impact=rsi_impact, positive=30 < rsi < 70),
    ]


@app.post("/api/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest) -> ForecastResponse:
    ticker = req.ticker.strip().upper()
    horizon = int(req.horizon)
    meta = COMPANY_BY_TICKER.get(ticker)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"unknown ticker: {ticker}")

    end = dt.date.today().strftime("%Y-%m-%d")
    start = (dt.date.today() - dt.timedelta(days=900)).strftime("%Y-%m-%d")
    try:
        price = _cached_moex(ticker, start, end)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"MOEX fetch failed: {e}") from e
    if price["date"].size == 0:
        raise HTTPException(status_code=404, detail=f"no price history for {ticker}")

    usd_dates, usd_values = _cached_usd(start, end)
    imoex_dates, imoex_values = _cached_imoex(start, end)
    div_dates, div_values = _cached_dividends(ticker)

    feat_matrix, feat_dates = build_features(
        dates=price["date"],
        close=price["close"],
        high=price["high"],
        low=price["low"],
        volume=price["volume"],
        usd_dates=usd_dates,
        usd_values=usd_values,
        imoex_dates=imoex_dates,
        imoex_values=imoex_values,
        div_dates=div_dates,
        div_values=div_values,
    )
    if feat_matrix.shape[0] < SEQ_LEN:
        raise HTTPException(
            status_code=422,
            detail=f"not enough bars after feature engineering: {feat_matrix.shape[0]} < {SEQ_LEN}",
        )

    prob_up_raw, group, threshold = predict_proba(feat_matrix, horizon=horizon)
    prob_up = float(np.clip(prob_up_raw, 1e-6, 1 - 1e-6))
    prob_pct = int(round(prob_up * 100))

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

    # Projected price uses signed move scaled by the model's training threshold.
    signed_move = (2 * prob_up - 1) * threshold
    projected_price = last_close * (1 + signed_move)
    pct_change = signed_move * 100

    factors = _build_factors(feat_matrix[-1])

    sector_label = SECTORS.get(meta["sector"], {}).get("label", "")
    horizon_word = f"{horizon} торговых дней" if horizon != 1 else "1 торгового дня"
    if prob_up >= 0.55:
        explanation = (
            f"Нейросеть оценивает вероятность роста {meta['name']} более чем на "
            f"+{int(threshold*100)}% в ближайшие {horizon_word} в {prob_pct}%. "
            f"Ожидаемая цена — около {projected_price:.2f} ₽ ({pct_change:+.1f}% к текущей). "
            f"Рыночная ситуация в секторе «{sector_label}» благоприятна."
        )
    elif prob_up <= 0.44:
        explanation = (
            f"Анализ показывает, что вероятность снижения {meta['name']} в ближайшие "
            f"{horizon_word} составляет {100-prob_pct}%. Прогнозная цена — около "
            f"{projected_price:.2f} ₽ ({pct_change:+.1f}%). Рекомендуется осторожность."
        )
    else:
        explanation = (
            f"Прогноз для {meta['name']} на {horizon_word} неоднозначный: "
            f"вероятность роста {prob_pct}%. Рынок в состоянии неопределённости, "
            f"ориентировочная цена: {projected_price:.2f} ₽ ({pct_change:+.1f}%)."
        )

    trained_on = HORIZON_GROUPS[group]["trained_on"]
    model_note = (
        f"Модель TCN ({group}, ансамбль из 5 моделей с сидов 42–46) обучена на 24 ликвидных "
        f"тикерах MOEX 2015–сегодня с горизонтом {trained_on} торговых дней (порог "
        f"+{int(threshold*100)}%). Запрошенный горизонт ({horizon}д) обслуживает та же группа моделей."
    )
    auc = 0.62

    return ForecastResponse(
        ticker=ticker,
        name=meta["name"],
        sector=meta["sector"],
        horizon_days=horizon,
        threshold_pct=threshold * 100,
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
        model_group=group,
    )
