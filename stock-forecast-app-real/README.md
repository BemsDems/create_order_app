# Stock Forecast App — Real Data + Keras

Это версия `stock-forecast-app`, переделанная так, чтобы:
- котировки брались из **MOEX ISS** (реальные данные)
- прогноз строился через **настоящую Keras-модель** (`sonnet_v3_final.keras`) и scaler (`sonnet_v3_scaler.pkl`)

## Структура

- `frontend/index.html` — SPA (React UMD + Tailwind CDN)
- `backend/main.py` — FastAPI API, который:
  - качает историю через MOEX ISS
  - качает USD/RUB через ЦБ (CBR XML_dynamic)
  - строит фичи (как в твоём prompt)
  - делает инференс Keras-модели
- `backend/model/*` — веса модели + scaler

## Запуск

### 1) Backend

```bash
cd stock-forecast-app-real/backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8001
```

### 2) Frontend

```bash
cd ../frontend
python3 -m http.server 8000
# открой http://localhost:8000
```

Frontend ожидает API по адресу `http://localhost:8001`.
Если нужно — можно переопределить в браузере:

```js
window.STOCK_API_BASE = 'http://<your-host>:8001'
```

## Ограничения

- В этой реализации **horizon зафиксирован = 5 торговых дней**, потому что текущие веса обучены именно под него.
  (Если нужны 10/30/… — добавим обучение/набор весов под каждый horizon.)
- API делает best-effort запросы к MOEX/ЦБ; при временных ошибках возможны 4xx/5xx.

