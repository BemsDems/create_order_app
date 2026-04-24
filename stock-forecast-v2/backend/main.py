"""
Stock Forecast Backend v2
Real MOEX data + Keras LSTM model (Sonnet v3)
"""
from __future__ import annotations

import os
import time
import threading
import uuid
import pickle
import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import requests
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sklearn.preprocessing import RobustScaler

# ─── Config ───────────────────────────────────────────────────────

SEQ_LEN = 30
FEATURE_COLS = [
    "ret_1d", "ret_5d", "ret_10d", "ret_20d", "log_ret",
    "price_vs_sma20", "price_vs_sma50", "trend_up",
    "rsi_14", "macd_histogram",
    "volatility_20", "volume_ratio",
    "usd_ret_5d", "price_position", "bb_position",
]

SECTORS = {
    "oil":       {"label": "Нефтегаз",     "color": "#f97316"},
    "finance":   {"label": "Финансы",      "color": "#3b82f6"},
    "tech":      {"label": "Технологии",   "color": "#a855f7"},
    "metal":     {"label": "Металлургия",  "color": "#94a3b8"},
    "retail":    {"label": "Ритейл",       "color": "#ec4899"},
    "telecom":   {"label": "Телеком",      "color": "#06b6d4"},
    "energy":    {"label": "Энергетика",   "color": "#eab308"},
    "transport": {"label": "Транспорт",    "color": "#22c55e"},
    "realty":    {"label": "Недвижимость", "color": "#14b8a6"},
    "chem":      {"label": "Химия",        "color": "#84cc16"},
    "holding":   {"label": "Холдинги",     "color": "#6366f1"},
    "timber":    {"label": "Лесопром",     "color": "#10b981"},
    "agro":      {"label": "Агро",         "color": "#16a34a"},
    "machine":   {"label": "Машиностр.",   "color": "#64748b"},
}

COMPANIES = [
    {"ticker": "SBER",  "name": "Сбербанк",             "sector": "finance"},
    {"ticker": "GAZP",  "name": "Газпром",               "sector": "oil"},
    {"ticker": "LKOH",  "name": "Лукойл",                "sector": "oil"},
    {"ticker": "ROSN",  "name": "Роснефть",              "sector": "oil"},
    {"ticker": "GMKN",  "name": "Норильский никель",     "sector": "metal"},
    {"ticker": "NVTK",  "name": "Новатэк",               "sector": "oil"},
    {"ticker": "YNDX",  "name": "Яндекс",                "sector": "tech"},
    {"ticker": "MTSS",  "name": "МТС",                   "sector": "telecom"},
    {"ticker": "MGNT",  "name": "Магнит",                "sector": "retail"},
    {"ticker": "FIVE",  "name": "X5 Retail Group",        "sector": "retail"},
    {"ticker": "PLZL",  "name": "Полюс Золото",           "sector": "metal"},
    {"ticker": "POLY",  "name": "Polymetal",              "sector": "metal"},
    {"ticker": "ALRS",  "name": "АЛРОСА",                "sector": "metal"},
    {"ticker": "CHMF",  "name": "Северсталь",            "sector": "metal"},
    {"ticker": "NLMK",  "name": "НЛМК",                  "sector": "metal"},
    {"ticker": "MAGN",  "name": "ММК",                   "sector": "metal"},
    {"ticker": "VTBR",  "name": "ВТБ",                   "sector": "finance"},
    {"ticker": "MOEX",  "name": "Московская биржа",      "sector": "finance"},
    {"ticker": "PHOR",  "name": "ФосАгро",               "sector": "chem"},
    {"ticker": "RUAL",  "name": "РУСАЛ",                 "sector": "metal"},
    {"ticker": "OZON",  "name": "Ozon",                  "sector": "tech"},
    {"ticker": "VKCO",  "name": "ВКонтакте",             "sector": "tech"},
    {"ticker": "IRAO",  "name": "Интер РАО",             "sector": "energy"},
    {"ticker": "FEES",  "name": "ФСК ЕЭС",              "sector": "energy"},
    {"ticker": "HYDR",  "name": "РусГидро",              "sector": "energy"},
    {"ticker": "RTKM",  "name": "Ростелеком",            "sector": "telecom"},
    {"ticker": "AFLT",  "name": "Аэрофлот",              "sector": "transport"},
    {"ticker": "PIKK",  "name": "ПИК",                   "sector": "realty"},
    {"ticker": "SMLT",  "name": "Самолёт",               "sector": "realty"},
    {"ticker": "SGZH",  "name": "Сегежа",                "sector": "timber"},
    {"ticker": "MTLR",  "name": "Мечел",                 "sector": "metal"},
    {"ticker": "AFKS",  "name": "АФК Система",           "sector": "holding"},
    {"ticker": "CBOM",  "name": "МКБ",                   "sector": "finance"},
    {"ticker": "TATN",  "name": "Татнефть",              "sector": "oil"},
    {"ticker": "SNGS",  "name": "Сургутнефтегаз",       "sector": "oil"},
    {"ticker": "BANE",  "name": "Башнефть",              "sector": "oil"},
    {"ticker": "TRNFP", "name": "Транснефть",            "sector": "oil"},
    {"ticker": "NMTP",  "name": "НМТП",                  "sector": "transport"},
    {"ticker": "FLOT",  "name": "Совкомфлот",            "sector": "transport"},
    {"ticker": "UWGN",  "name": "Уралвагонзавод",        "sector": "machine"},
    {"ticker": "TCSG",  "name": "Т-Банк",               "sector": "finance"},
    {"ticker": "FIXP",  "name": "Fix Price",             "sector": "retail"},
    {"ticker": "ENPG",  "name": "ЭН+",                  "sector": "energy"},
    {"ticker": "DSKY",  "name": "Детский мир",           "sector": "retail"},
    {"ticker": "LENT",  "name": "Лента",                "sector": "retail"},
    {"ticker": "RASP",  "name": "Распадская",            "sector": "metal"},
    {"ticker": "SELG",  "name": "Селигдар",              "sector": "metal"},
    {"ticker": "POGR",  "name": "ПОЭЗ",                 "sector": "energy"},
    {"ticker": "BSPB",  "name": "Банк Санкт-Петербург",  "sector": "finance"},
    {"ticker": "AQUA",  "name": "Инарктика",             "sector": "agro"},
    {"ticker": "RNFT",  "name": "РуссНефть",             "sector": "oil"},
    {"ticker": "MSNG",  "name": "Мосэнерго",             "sector": "energy"},
    {"ticker": "LSRG",  "name": "ЛСР",                  "sector": "realty"},
    {"ticker": "RENI",  "name": "Ренессанс Страхование", "sector": "finance"},
]

TICKERS_SET = {c["ticker"] for c in COMPANIES}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "sonnet_v3_final.keras")
SCALER_PATH = os.path.join(BASE_DIR, "model", "sonnet_v3_scaler.pkl")
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")

# ─── App ──────────────────────────────────────────────────────────

app = FastAPI(title="Stock Forecast v2 — Real Data + Keras Model")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── HTTP helpers ─────────────────────────────────────────────────

def _get_json(url: str, *, params: dict | None = None, timeout: int = 30) -> dict:
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


# ─── MOEX data ────────────────────────────────────────────────────

def fetch_moex_history(secid: str, start: str, end: Optional[str] = None) -> pd.DataFrame:
    url = f"https://iss.moex.com/iss/history/engines/stock/markets/shares/securities/{secid}.json"
    all_rows: list = []
    start_pos = 0
    cols: list = []

    while True:
        params = {
            "from": start,
            "till": end,
            "iss.meta": "off",
            "iss.only": "history",
            "history.columns": "TRADEDATE,OPEN,HIGH,LOW,CLOSE,VOLUME,VALUE",
            "start": start_pos,
        }
        j = _get_json(url, params=params, timeout=30)
        block = j.get("history", {})
        cols = block.get("columns", cols)
        data = block.get("data", [])
        if not data:
            break
        all_rows.extend(data)
        start_pos += len(data)
        if len(data) < 100:
            break

    df = pd.DataFrame(all_rows, columns=cols)
    if df.empty:
        return df

    df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"], errors="coerce")
    df = df.dropna(subset=["TRADEDATE"]).sort_values("TRADEDATE").reset_index(drop=True)
    df = df.rename(columns={"TRADEDATE": "date"})

    for c in ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "VALUE"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["CLOSE"])
    df = df[df["CLOSE"] > 0].reset_index(drop=True)
    return df


def fetch_cbr_usdrub(start: str, end: Optional[str] = None) -> pd.Series:
    if end is None:
        end = dt.date.today().strftime("%Y-%m-%d")

    start_dt = pd.to_datetime(start).strftime("%d/%m/%Y")
    end_dt = pd.to_datetime(end).strftime("%d/%m/%Y")

    url = "https://www.cbr.ru/scripts/XML_dynamic.asp"
    params = {"date_req1": start_dt, "date_req2": end_dt, "VAL_NM_RQ": "R01235"}
    xml = requests.get(url, params=params, timeout=30).text

    root = ET.fromstring(xml)
    rows: list = []
    for rec in root.findall("Record"):
        d = rec.attrib.get("Date")
        v = rec.findtext("Value")
        if d and v:
            rows.append((pd.to_datetime(d, dayfirst=True), float(v.replace(",", "."))))

    s = pd.Series(dict(rows)).sort_index()
    s.name = "CBR_USD_RUB"
    return s


# ─── Feature engineering ──────────────────────────────────────────

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / (loss + 1e-12)
    return 100 - (100 / (1 + rs))


def add_stable_features(df: pd.DataFrame, usd: pd.Series) -> pd.DataFrame:
    out = df.copy()

    out["ret_1d"] = out["CLOSE"].pct_change(1)
    out["ret_5d"] = out["CLOSE"].pct_change(5)
    out["ret_10d"] = out["CLOSE"].pct_change(10)
    out["ret_20d"] = out["CLOSE"].pct_change(20)
    out["log_ret"] = np.log(out["CLOSE"] / out["CLOSE"].shift(1))

    sma20 = out["CLOSE"].rolling(20).mean()
    sma50 = out["CLOSE"].rolling(50).mean()
    sma200 = out["CLOSE"].rolling(200).mean()
    out["price_vs_sma20"] = out["CLOSE"] / (sma20 + 1e-12) - 1
    out["price_vs_sma50"] = out["CLOSE"] / (sma50 + 1e-12) - 1
    out["trend_up"] = (out["CLOSE"] > sma200).astype(int)

    out["rsi_14"] = _rsi(out["CLOSE"], 14)

    ema12 = out["CLOSE"].ewm(span=12, adjust=False).mean()
    ema26 = out["CLOSE"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    out["macd_histogram"] = macd - macd_signal

    out["volatility_20"] = out["ret_1d"].rolling(20).std()
    out["volume_ratio"] = out["VOLUME"] / (out["VOLUME"].rolling(20).mean() + 1e-8)

    if isinstance(usd, pd.Series) and not usd.empty:
        usd_df = usd.reset_index()
        usd_df.columns = ["date", "usd_rub"]
        usd_df["date"] = pd.to_datetime(usd_df["date"])
        out = pd.merge_asof(
            out.sort_values("date"),
            usd_df.sort_values("date"),
            on="date",
            direction="backward",
        )
        out["usd_ret_5d"] = out["usd_rub"].pct_change(5)
    else:
        out["usd_ret_5d"] = 0.0

    high_20 = out["HIGH"].rolling(20).max()
    low_20 = out["LOW"].rolling(20).min()
    out["price_position"] = (out["CLOSE"] - low_20) / ((high_20 - low_20) + 1e-8)

    bb_mid = out["CLOSE"].rolling(20).mean()
    bb_std = out["CLOSE"].rolling(20).std()
    out["bb_position"] = (out["CLOSE"] - bb_mid) / ((2 * bb_std) + 1e-8)

    out = out.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return out


# ─── Model bundle ─────────────────────────────────────────────────

@dataclass
class ModelBundle:
    model: Any
    scaler: Any

_BUNDLE: ModelBundle | None = None

def get_bundle() -> ModelBundle:
    global _BUNDLE
    if _BUNDLE is not None:
        return _BUNDLE
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found: {MODEL_PATH}")
    if not os.path.exists(SCALER_PATH):
        raise RuntimeError(f"Scaler not found: {SCALER_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    _BUNDLE = ModelBundle(model=model, scaler=scaler)
    return _BUNDLE


# ─── Cache ────────────────────────────────────────────────────────

_cache: Dict[str, tuple] = {}

def _cache_get(key: str, ttl_s: int) -> Any:
    if key not in _cache:
        return None
    ts, val = _cache[key]
    if time.time() - ts > ttl_s:
        return None
    return val

def _cache_set(key: str, val: Any) -> None:
    _cache[key] = (time.time(), val)


# ─── Prediction jobs ──────────────────────────────────────────────

PRED_JOBS: Dict[str, Dict[str, Any]] = {}
PRED_JOBS_LOCK = threading.Lock()

def _job_set(job_id: str, **fields: Any) -> None:
    with PRED_JOBS_LOCK:
        job = PRED_JOBS.get(job_id) or {"job_id": job_id, "status": "queued"}
        job.update(fields)
        PRED_JOBS[job_id] = job

def _job_get(job_id: str) -> Dict[str, Any] | None:
    with PRED_JOBS_LOCK:
        j = PRED_JOBS.get(job_id)
        return dict(j) if j else None


# ─── Core prediction ──────────────────────────────────────────────

def _compute_prediction(ticker: str, *, job_id: str | None = None) -> Dict[str, Any]:
    t = ticker.upper().strip()

    def step(name: str) -> None:
        if job_id:
            _job_set(job_id, status="running", step=name)

    step("fetch_moex_history")
    df_price = fetch_moex_history(t, start="2015-01-01", end=None)
    if df_price.empty:
        raise HTTPException(status_code=404, detail=f"Нет данных MOEX для тикера {t}")

    step("fetch_cbr_usdrub")
    usd = fetch_cbr_usdrub("2015-01-01", None)

    step("feature_engineering")
    df_feat = add_stable_features(df_price, usd)

    missing = [c for c in FEATURE_COLS if c not in df_feat.columns]
    if missing:
        raise HTTPException(status_code=500, detail=f"Отсутствуют фичи: {missing}")

    if len(df_feat) < SEQ_LEN:
        raise HTTPException(status_code=400, detail=f"Недостаточно данных: нужно >= {SEQ_LEN} строк")

    X = df_feat[FEATURE_COLS].values.astype(float)

    step("load_model")
    bundle = get_bundle()

    step("scaling")
    X_scaled = bundle.scaler.transform(X)
    window = X_scaled[-SEQ_LEN:, :].reshape(1, SEQ_LEN, len(FEATURE_COLS))

    step("model_predict")
    prob = float(bundle.model.predict(window, verbose=0).reshape(-1)[0])

    if prob >= 0.55:
        verdict = "Вероятен рост"
        verdict_color = "success"
    elif prob <= 0.45:
        verdict = "Вероятно снижение"
        verdict_color = "danger"
    else:
        verdict = "Нейтрально"
        verdict_color = "warning"

    last = df_feat.iloc[-1]
    prev = df_feat.iloc[-2] if len(df_feat) >= 2 else last
    close_price = float(last["CLOSE"])
    prev_price = float(prev["CLOSE"])
    day_change = (close_price - prev_price) / (prev_price + 1e-12)

    features_last = {}
    for c in FEATURE_COLS:
        if c in df_feat.columns:
            features_last[c] = float(last[c])

    factor_labels = {
        "volatility_20": {"label": "Волатильность", "desc": "уровень риска"},
        "ret_5d": {"label": "Ценовой тренд", "desc": "направление движения"},
        "volume_ratio": {"label": "Объём торгов", "desc": "рыночная активность"},
        "price_position": {"label": "Позиция цены", "desc": "в диапазоне 20 дней"},
        "rsi_14": {"label": "RSI (14)", "desc": "индикатор перекупленности"},
        "usd_ret_5d": {"label": "Курс доллара", "desc": "валютный фактор"},
    }
    factors = []
    for key, meta in factor_labels.items():
        val = features_last.get(key, 0)
        if key == "rsi_14":
            impact = min(100, max(0, int(val)))
            positive = val < 70
        elif key == "volatility_20":
            impact = min(100, max(0, int(val * 2000)))
            positive = val < 0.02
        elif key == "volume_ratio":
            impact = min(100, max(0, int(val * 50)))
            positive = val > 1.0
        elif key == "ret_5d":
            impact = min(100, max(0, int(abs(val) * 500)))
            positive = val > 0
        elif key == "price_position":
            impact = min(100, max(0, int(val * 100)))
            positive = 0.3 < val < 0.8
        elif key == "usd_ret_5d":
            impact = min(100, max(0, int(abs(val) * 1000)))
            positive = val <= 0
        else:
            impact = 50
            positive = True
        factors.append({
            "key": key,
            "label": meta["label"],
            "desc": meta["desc"],
            "impact": impact,
            "positive": positive,
        })

    chart_pts = df_feat["CLOSE"].iloc[-30:].tolist() if len(df_feat) >= 30 else df_feat["CLOSE"].tolist()

    company_info = next((c for c in COMPANIES if c["ticker"] == t), None)

    confidence_val = int(min(95, max(15, abs(prob - 0.5) * 200)))
    if confidence_val > 65:
        confidence_label = "высокий"
    elif confidence_val > 40:
        confidence_label = "средний"
    else:
        confidence_label = "низкий"

    out = {
        "ticker": t,
        "company": company_info,
        "asof": str(pd.to_datetime(last["date"]).date()),
        "close": close_price,
        "day_change": day_change,
        "prob": prob,
        "prob_pct": int(round(prob * 100)),
        "verdict": verdict,
        "verdict_color": verdict_color,
        "confidence": confidence_label,
        "confidence_val": confidence_val,
        "factors": factors,
        "chart_pts": chart_pts,
        "features_last": features_last,
        "model_info": {
            "type": "LSTM (Sonnet v3)",
            "seq_len": SEQ_LEN,
            "features_count": len(FEATURE_COLS),
            "horizon_days": 5,
        },
    }

    step("done")
    return out


# ─── API endpoints ────────────────────────────────────────────────

@app.get("/api/companies")
def api_companies() -> Dict[str, Any]:
    return {"companies": COMPANIES, "sectors": SECTORS}


@app.get("/api/quote/{ticker}")
def api_quote(ticker: str) -> Dict[str, Any]:
    t = ticker.upper().strip()
    if not t:
        raise HTTPException(status_code=400, detail="ticker required")

    ck = f"quote:{t}"
    cached = _cache_get(ck, ttl_s=120)
    if cached is not None:
        return cached

    df = fetch_moex_history(t, start="2024-01-01", end=None)
    if df.empty or len(df) < 2:
        raise HTTPException(status_code=404, detail=f"Нет данных MOEX для {t}")

    last = df.iloc[-1]
    prev = df.iloc[-2]
    price = float(last["CLOSE"])
    prev_price = float(prev["CLOSE"])
    chg = (price - prev_price) / (prev_price + 1e-12)

    out = {
        "ticker": t,
        "date": str(pd.to_datetime(last["date"]).date()),
        "price": price,
        "change_1d": chg,
    }
    _cache_set(ck, out)
    return out


@app.post("/api/predict_job/{ticker}")
def api_predict_job_start(ticker: str) -> Dict[str, Any]:
    t = ticker.upper().strip()
    if not t:
        raise HTTPException(status_code=400, detail="ticker required")

    job_id = str(uuid.uuid4())
    _job_set(
        job_id,
        status="queued",
        step="queued",
        ticker=t,
        created_at=time.time(),
    )

    def runner() -> None:
        try:
            _job_set(job_id, status="running", step="starting")
            out = _compute_prediction(t, job_id=job_id)
            _job_set(job_id, status="done", step="done", result=out, finished_at=time.time())
        except HTTPException as e:
            _job_set(
                job_id,
                status="error",
                step="error",
                error={"status_code": e.status_code, "detail": e.detail},
                finished_at=time.time(),
            )
        except Exception as e:
            _job_set(
                job_id,
                status="error",
                step="error",
                error={"type": type(e).__name__, "message": str(e)},
                finished_at=time.time(),
            )

    threading.Thread(target=runner, daemon=True).start()
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/predict_job/{job_id}")
def api_predict_job_status(job_id: str) -> Dict[str, Any]:
    j = _job_get(str(job_id))
    if not j:
        raise HTTPException(status_code=404, detail="job not found")
    out = {
        k: j.get(k)
        for k in ("job_id", "status", "step", "ticker", "created_at", "finished_at")
        if k in j
    }
    if j.get("status") == "done":
        out["result"] = j.get("result")
    if j.get("status") == "error":
        out["error"] = j.get("error")
    return out


@app.get("/api/predict/{ticker}")
def api_predict(ticker: str) -> Dict[str, Any]:
    t = ticker.upper().strip()
    if not t:
        raise HTTPException(status_code=400, detail="ticker required")

    ck = f"pred:{t}"
    cached = _cache_get(ck, ttl_s=300)
    if cached is not None:
        return cached

    out = _compute_prediction(t)
    _cache_set(ck, out)
    return out


# Serve frontend
@app.get("/")
def serve_frontend():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Frontend not found. Place index.html in ../frontend/"}
