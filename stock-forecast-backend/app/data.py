"""Fetch live market data from MOEX ISS and CBR XML — no pandas."""
from __future__ import annotations

import datetime as dt
from xml.etree import ElementTree as ET

import time

import numpy as np
import requests

ISO_DATE = "%Y-%m-%d"
DMY_DATE = "%d/%m/%Y"

_HEADERS = {
    "User-Agent": "StockAI-RU/0.1 (+https://github.com/BemsDems/create_order_app)",
    "Accept": "application/json, text/xml;q=0.9, */*;q=0.5",
    "Accept-Language": "en",
}


def _get_with_retry(url: str, params: dict, max_attempts: int = 3) -> requests.Response:
    last_err: Exception | None = None
    for attempt in range(max_attempts):
        try:
            r = requests.get(url, params=params, timeout=(10, 60), headers=_HEADERS)
            r.raise_for_status()
            return r
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(0.5 * (attempt + 1))
    assert last_err is not None
    raise last_err


def _parse_iso(s: str) -> np.datetime64:
    # Both ISO 'YYYY-MM-DD' and 'YYYY-MM-DDTHH:MM:SS' work for np.datetime64
    return np.datetime64(s[:10])


def _fetch_history_paged(url: str, start: str, end: str | None) -> tuple[list[str], list[list]]:
    rows: list[list] = []
    cols: list[str] = []
    start_pos = 0
    while True:
        params = {
            "from": start,
            "till": end,
            "iss.meta": "off",
            "iss.only": "history",
            "history.columns": "TRADEDATE,OPEN,HIGH,LOW,CLOSE,VOLUME,VALUE",
            "start": start_pos,
        }
        r = _get_with_retry(url, params)
        block = r.json().get("history", {})
        cols = block.get("columns", cols) or cols
        data = block.get("data", [])
        if not data:
            break
        rows.extend(data)
        start_pos += len(data)
        if len(data) < 100:
            break
    return cols, rows


def fetch_moex_history(secid: str, start: str, end: str | None = None) -> dict[str, np.ndarray]:
    """Fetch OHLCV daily bars from MOEX ISS history endpoint.

    Returns dict with keys: date (np.datetime64[D]), open, high, low, close, volume.
    """
    url = f"https://iss.moex.com/iss/history/engines/stock/markets/shares/securities/{secid}.json"
    rows: list[list] = []
    cols: list[str] = []
    start_pos = 0
    while True:
        params = {
            "from": start,
            "till": end,
            "iss.meta": "off",
            "iss.only": "history",
            "history.columns": "TRADEDATE,OPEN,HIGH,LOW,CLOSE,VOLUME,VALUE",
            "start": start_pos,
        }
        r = _get_with_retry(url, params)
        block = r.json().get("history", {})
        cols = block.get("columns", cols) or cols
        data = block.get("data", [])
        if not data:
            break
        rows.extend(data)
        start_pos += len(data)
        if len(data) < 100:
            break

    if not rows:
        return {
            "date": np.array([], dtype="datetime64[D]"),
            "open": np.array([]), "high": np.array([]), "low": np.array([]),
            "close": np.array([]), "volume": np.array([]),
        }

    idx = {c: i for i, c in enumerate(cols)}
    dates = np.array([_parse_iso(str(r[idx["TRADEDATE"]])) for r in rows], dtype="datetime64[D]")
    order = np.argsort(dates)
    dates = dates[order]

    def col(name: str) -> np.ndarray:
        arr = np.array([r[idx[name]] if r[idx[name]] is not None else np.nan for r in rows], dtype=np.float64)
        return arr[order]

    # Drop rows with missing CLOSE
    close = col("CLOSE")
    keep = ~np.isnan(close)
    return {
        "date": dates[keep],
        "open": col("OPEN")[keep],
        "high": col("HIGH")[keep],
        "low": col("LOW")[keep],
        "close": close[keep],
        "volume": col("VOLUME")[keep],
    }


def fetch_cbr_usdrub(start: str, end: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Fetch CBR USD/RUB dynamic (R01235). Returns (dates, values)."""
    if end is None:
        end = dt.date.today().strftime(ISO_DATE)
    date_req1 = dt.datetime.strptime(start, ISO_DATE).strftime(DMY_DATE)
    date_req2 = dt.datetime.strptime(end, ISO_DATE).strftime(DMY_DATE)
    r = _get_with_retry(
        "https://www.cbr.ru/scripts/XML_dynamic.asp",
        {"date_req1": date_req1, "date_req2": date_req2, "VAL_NM_RQ": "R01235"},
    )
    dates: list[np.datetime64] = []
    values: list[float] = []
    for rec in ET.fromstring(r.text).findall("Record"):
        d = rec.attrib.get("Date")
        v = rec.findtext("Value")
        if d and v:
            day, month, year = d.split(".")
            dates.append(np.datetime64(f"{year}-{month}-{day}"))
            values.append(float(v.replace(",", ".")))
    if not dates:
        return np.array([], dtype="datetime64[D]"), np.array([], dtype=np.float64)
    d = np.array(dates, dtype="datetime64[D]")
    v = np.array(values, dtype=np.float64)
    order = np.argsort(d)
    return d[order], v[order]


def fetch_imoex_history(start: str, end: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Fetch IMOEX daily close from MOEX ISS index history endpoint."""
    url = "https://iss.moex.com/iss/history/engines/stock/markets/index/securities/IMOEX.json"
    cols, rows = _fetch_history_paged(url, start, end)
    if not rows:
        return np.array([], dtype="datetime64[D]"), np.array([], dtype=np.float64)
    idx = {c: i for i, c in enumerate(cols)}
    dates = np.array([_parse_iso(str(r[idx["TRADEDATE"]])) for r in rows], dtype="datetime64[D]")
    closes = np.array(
        [r[idx["CLOSE"]] if r[idx["CLOSE"]] is not None else np.nan for r in rows],
        dtype=np.float64,
    )
    order = np.argsort(dates)
    dates = dates[order]
    closes = closes[order]
    keep = ~np.isnan(closes)
    return dates[keep], closes[keep]


def fetch_usd000_history(start: str, end: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Fetch USD000UTSTOM (MOEX TOM USD/RUB) daily close as the macro USD series.

    NOTE: After mid-2024 MOEX USD/RUB stopped trading on the public CETS board
    (sanctions). The endpoint still returns rows, but with zero CLOSE/HIGH/LOW
    fields for those dates, so we filter those out and only keep meaningful
    quotes. The caller is expected to fall back to CBR when the result is empty.
    """
    url = "https://iss.moex.com/iss/history/engines/currency/markets/selt/securities/USD000UTSTOM.json"
    cols, rows = _fetch_history_paged(url, start, end)
    if not rows:
        return np.array([], dtype="datetime64[D]"), np.array([], dtype=np.float64)
    idx = {c: i for i, c in enumerate(cols)}
    dates = np.array([_parse_iso(str(r[idx["TRADEDATE"]])) for r in rows], dtype="datetime64[D]")
    closes = np.array(
        [r[idx["CLOSE"]] if r[idx["CLOSE"]] is not None else np.nan for r in rows],
        dtype=np.float64,
    )
    order = np.argsort(dates)
    dates = dates[order]
    closes = closes[order]
    # Drop rows where the close is missing OR equal to zero (delisted board).
    keep = ~np.isnan(closes) & (closes > 1.0)
    return dates[keep], closes[keep]


def fetch_moex_dividends(secid: str) -> tuple[np.ndarray, np.ndarray]:
    """Fetch full dividend history (RUB only) from MOEX ISS.

    Returns (registry_close_dates, values_per_share). Sorted by date ascending.
    On any failure returns empty arrays.
    """
    url = f"https://iss.moex.com/iss/securities/{secid}/dividends.json"
    try:
        r = _get_with_retry(url, {"iss.meta": "off"})
        block = r.json().get("dividends", {})
        cols = block.get("columns", [])
        rows = block.get("data", [])
        if not rows:
            return np.array([], dtype="datetime64[D]"), np.array([], dtype=np.float64)
        idx = {c: i for i, c in enumerate(cols)}
        if "registryclosedate" not in idx or "value" not in idx:
            return np.array([], dtype="datetime64[D]"), np.array([], dtype=np.float64)
        dates: list[np.datetime64] = []
        values: list[float] = []
        for row in rows:
            cur = row[idx["currencyid"]] if "currencyid" in idx else "RUB"
            if str(cur).upper() != "RUB":
                continue
            d = row[idx["registryclosedate"]]
            v = row[idx["value"]]
            if d is None or v is None:
                continue
            try:
                dates.append(np.datetime64(str(d)[:10]))
                values.append(float(v))
            except (TypeError, ValueError):
                continue
        if not dates:
            return np.array([], dtype="datetime64[D]"), np.array([], dtype=np.float64)
        d = np.array(dates, dtype="datetime64[D]")
        v = np.array(values, dtype=np.float64)
        order = np.argsort(d)
        return d[order], v[order]
    except Exception:  # noqa: BLE001
        return np.array([], dtype="datetime64[D]"), np.array([], dtype=np.float64)
