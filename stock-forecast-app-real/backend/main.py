from __future__ import annotations

import os
import time
import threading
import uuid
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# --- Config ---

SEQ_LEN = 30
PRED_START = "2018-01-01"  # enough for indicators, much faster than 2015
FEATURE_COLS = [
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

# If you want to extend to more tickers, keep this list in sync with frontend.
DEFAULT_TICKERS = [
    "SBER", "GAZP", "LKOH", "ROSN", "NVTK", "TATN", "NLMK", "GMKN", "ALRS",
    "MGNT", "CHMF", "MTSS", "SNGS", "SNGSP", "VTBR", "AFLT", "IRAO", "PHOR",
    "POLY", "PIKK", "SIBN", "MOEX", "RUAL", "PLZL", "MAGN", "FIVE", "OZON",
    "TCSG", "YNDX", "SELG", "AFKS", "FEES", "HYDR", "TRNFP", "RTKM",
    "RTKMP", "UPRO", "CBOM", "LSRG", "BSPB", "SMLT", "ENPG", "FIXP",
    "AGRO", "QIWI", "VKCO", "BELU", "MSNG", "NKNC", "NKNCP", "RASP",
    "MTLR", "MTLRP", "GCHE",
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "sonnet_v3_final.keras")
SCALER_PATH = os.path.join(BASE_DIR, "model", "sonnet_v3_scaler.pkl")

# --- App ---

app = FastAPI(title="Stock Forecast App (Real Data + Keras Model)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"] ,
    allow_headers=["*"],
)


def _get_json(url: str, *, params: dict | None = None, timeout: int = 30) -> dict:
    r = requests.get(url, params=params, timeout=timeout)
    ct = (r.headers.get("Content-Type") or "").lower()
    try:
        r.raise_for_status()
    except Exception as e:  # noqa: BLE001
        preview = (r.text or "")[:300].replace("\n", " ")
        raise RuntimeError(f"HTTP error for {url} status={r.status_code} ct={ct} preview={preview!r}") from e

    try:
        return r.json()
    except Exception as e:  # noqa: BLE001
        preview = (r.text or "")[:300].replace("\n", " ")
        raise RuntimeError(f"Non-JSON response for {url} status={r.status_code} ct={ct} preview={preview!r}") from e


def fetch_moex_history(secid: str, start: str, end: Optional[str]) -> pd.DataFrame:
    """Fetch OHLCV via MOEX ISS history."""
    url = f"https://iss.moex.com/iss/history/engines/stock/markets/shares/securities/{secid}.json"

    all_rows: list[list[Any]] = []
    start_pos = 0
    cols: list[str] = []

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

    return df


def fetch_cbr_usdrub(start: str, end: Optional[str]) -> pd.Series:
    """CBR XML dynamic for USD."""
    import datetime as dt
    from xml.etree import ElementTree as ET

    if end is None:
        end = dt.date.today().strftime("%Y-%m-%d")

    start_dt = pd.to_datetime(start).strftime("%d/%m/%Y")
    end_dt = pd.to_datetime(end).strftime("%d/%m/%Y")

    url = "https://www.cbr.ru/scripts/XML_dynamic.asp"
    params = {"date_req1": start_dt, "date_req2": end_dt, "VAL_NM_RQ": "R01235"}
    xml = requests.get(url, params=params, timeout=30).text

    root = ET.fromstring(xml)
    rows: list[tuple[pd.Timestamp, float]] = []
    for rec in root.findall("Record"):
        d = rec.attrib.get("Date")
        v = rec.findtext("Value")
        if d and v:
            rows.append((pd.to_datetime(d, dayfirst=True), float(v.replace(",", "."))))

    s = pd.Series(dict(rows)).sort_index()
    s.name = "CBR_USD_RUB"
    return s


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
        usd_df = usd.reset_index().rename(columns={"index": "date", "CBR_USD_RUB": "usd_rub"})
        usd_df["date"] = pd.to_datetime(usd_df["date"])
        out = pd.merge_asof(out.sort_values("date"), usd_df.sort_values("date"), on="date", direction="backward")
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


@dataclass
class ModelBundle:
    model: tf.keras.Model
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


# --- Simple in-process cache ---

_cache: Dict[str, tuple[float, Any]] = {}

# --- Prediction jobs (status tracking) ---
# In-memory job store: good enough for local/dev.
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


def _cache_get(key: str, ttl_s: int) -> Any | None:
    now = time.time()
    if key not in _cache:
        return None
    ts, val = _cache[key]
    if now - ts > ttl_s:
        return None
    return val


def _cache_set(key: str, val: Any) -> None:
    _cache[key] = (time.time(), val)


@app.get("/api/tickers")
def api_tickers() -> Dict[str, Any]:
    return {"tickers": DEFAULT_TICKERS}


@app.get("/api/quote/{ticker}")
def api_quote(ticker: str) -> Dict[str, Any]:
    t = str(ticker).upper().strip()
    if not t:
        raise HTTPException(status_code=400, detail="ticker required")

    ck = f"quote:{t}"
    cached = _cache_get(ck, ttl_s=60)
    if cached is not None:
        return cached

    df = fetch_moex_history(t, start=PRED_START, end=None)
    if df.empty or len(df) < 2:
        raise HTTPException(status_code=404, detail=f"No MOEX history for ticker={t}")

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




def _compute_prediction(ticker: str, *, horizon_days: int, job_id: str | None = None) -> Dict[str, Any]:
    """Compute prediction with optional job status updates."""
    t = str(ticker).upper().strip()

    def step(name: str) -> None:
        if job_id:
            _job_set(job_id, status="running", step=name)

    if int(horizon_days) != 5:
        raise HTTPException(status_code=400, detail="This model supports only horizon_days=5")

    step("fetch_moex_history")
    df_price = fetch_moex_history(t, start=PRED_START, end=None)
    if df_price.empty:
        raise HTTPException(status_code=404, detail=f"No MOEX history for ticker={t}")

    step("fetch_cbr_usdrub")
    usd = fetch_cbr_usdrub(PRED_START, None)

    step("feature_engineering")
    df_feat = add_stable_features(df_price, usd)

    missing = [c for c in FEATURE_COLS if c not in df_feat.columns]
    if missing:
        raise HTTPException(status_code=500, detail=f"Feature columns missing: {missing}")

    if len(df_feat) < SEQ_LEN:
        raise HTTPException(status_code=400, detail=f"Not enough data after features: need >= {SEQ_LEN} rows")

    X = df_feat[FEATURE_COLS].values.astype(float)

    step("load_model")
    bundle = get_bundle()

    step("scaling")
    X_scaled = bundle.scaler.transform(X)
    window = X_scaled[-SEQ_LEN:, :]
    window = window.reshape(1, SEQ_LEN, len(FEATURE_COLS))

    step("model_predict")
    prob = float(bundle.model.predict(window, verbose=0).reshape(-1)[0])

    verdict = "рост" if prob >= 0.55 else ("снижение" if prob <= 0.45 else "нейтрально")

    last = df_feat.iloc[-1]
    features_last = {c: float(last[c]) for c in FEATURE_COLS if c in df_feat.columns}

    out = {
        "ticker": t,
        "asof": str(pd.to_datetime(last["date"]).date()),
        "close": float(last["CLOSE"]),
        "prob_up_5d": prob,
        "verdict": verdict,
        "features_last": features_last,
        "model": {
            "type": "keras",
            "file": os.path.basename(MODEL_PATH),
            "seq_len": SEQ_LEN,
            "features": FEATURE_COLS,
            "horizon_days": 5,
        },
        "note": "Модель обучалась на MOEX/CBR фичах (horizon=5). Для других горизонтов нужна переобучение/отдельные веса.",
    }

    step("done")
    return out


@app.post("/api/predict_job/{ticker}")
def api_predict_job_start(ticker: str, horizon_days: int = 5) -> Dict[str, Any]:
    t = str(ticker).upper().strip()
    if not t:
        raise HTTPException(status_code=400, detail="ticker required")
    if int(horizon_days) != 5:
        raise HTTPException(status_code=400, detail="This model supports only horizon_days=5")

    job_id = str(uuid.uuid4())
    _job_set(job_id, status="queued", step="queued", ticker=t, horizon_days=int(horizon_days), created_at=time.time())

    def runner() -> None:
        try:
            _job_set(job_id, status="running", step="starting")
            out = _compute_prediction(t, horizon_days=int(horizon_days), job_id=job_id)
            _job_set(job_id, status="done", step="done", result=out, finished_at=time.time())
        except HTTPException as e:
            _job_set(job_id, status="error", step="error", error={"status_code": e.status_code, "detail": e.detail}, finished_at=time.time())
        except Exception as e:  # noqa: BLE001
            _job_set(job_id, status="error", step="error", error={"type": type(e).__name__, "message": str(e)}, finished_at=time.time())

    threading.Thread(target=runner, daemon=True).start()
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/predict_job/{job_id}")
def api_predict_job_status(job_id: str) -> Dict[str, Any]:
    j = _job_get(str(job_id))
    if not j:
        raise HTTPException(status_code=404, detail="job not found")
    out = {k: j.get(k) for k in ("job_id", "status", "step", "ticker", "horizon_days", "created_at", "finished_at") if k in j}
    if j.get("status") == "done":
        out["result"] = j.get("result")
    if j.get("status") == "error":
        out["error"] = j.get("error")
    return out

@app.get("/api/predict/{ticker}")
def api_predict(ticker: str, horizon_days: int = 5) -> Dict[str, Any]:
    t = str(ticker).upper().strip()
    if not t:
        raise HTTPException(status_code=400, detail="ticker required")

    if int(horizon_days) != 5:
        raise HTTPException(status_code=400, detail="This model supports only horizon_days=5")

    ck = f"pred:{t}"
    cached = _cache_get(ck, ttl_s=300)
    if cached is not None:
        return cached

    out = _compute_prediction(t, horizon_days=int(horizon_days))
    _cache_set(ck, out)
    return out
