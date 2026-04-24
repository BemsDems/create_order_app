# Stock Forecast Backend (StockAI RU)

FastAPI-бэкенд, который обслуживает реальные LSTM-прогнозы для SPA `stock-forecast-app`.

## Что внутри

- **`app/inference.py`** — чистый numpy-форвард (LSTM → LayerNorm → LSTM → LayerNorm → Dense → Dense sigmoid). Веса модели экспортированы из Keras-чекпоинта `sonnet_v3_final.keras` и лежат в `app/models/sonnet_v3_weights.npz` (~38 КБ).
- **`app/features.py`** — feature engineering, воспроизводящий обучающий пайплайн (15 признаков: returns, RSI, MACD-гистограмма, SMA/Bollinger, volume ratio, USD-returns).
- **`app/data.py`** — живая подкачка котировок с MOEX ISS (`/iss/history/engines/stock/markets/shares/...`) и курса USD/RUB с ЦБ (`XML_dynamic.asp`).
- **`app/main.py`** — FastAPI-эндпоинты:
  - `GET  /healthz` — здоровье.
  - `GET  /api/companies` — каталог 54 тикеров + 14 секторов.
  - `POST /api/forecast` `{"ticker":"SBER"}` — вероятность роста ≥ +2% за 5 торговых дней, вердикт, 6 факторов влияния и объяснение.

## Почему без TensorFlow

Первая версия зависела от `tensorflow-cpu`, но на 256 МБ Fly.io-машине процесс OOM-убивался при загрузке модели. Мы экспортировали веса Keras в NumPy-массивы и написали форвард-пасс LSTM / LayerNorm вручную. На тестовом датасете (SBER 2022+) prediction совпадает с Keras до float32-точности:

```
SBER  keras=0.514816  numpy=0.514816  diff=0.00e+00
GAZP  keras=0.506998  numpy=0.506998  diff=5.96e-08
YNDX  keras=0.530449  numpy=0.530449  diff=0.00e+00
LKOH  keras=0.579728  numpy=0.579728  diff=0.00e+00
```

## Локальный запуск

```bash
cd stock-forecast-backend
pip install -e .
fastapi run app/main.py
# сервис на http://127.0.0.1:8000
```

Проверить:

```bash
curl http://127.0.0.1:8000/healthz
curl -s -X POST http://127.0.0.1:8000/api/forecast \
  -H 'Content-Type: application/json' \
  -d '{"ticker":"SBER"}' | jq
```

## Деплой

- Fly.io, запущен по адресу https://stock-forecast-backen-rmclwnun.fly.dev
- Конфиг в `fly.toml` (shared-cpu-1x / 256 MiB достаточно благодаря отсутствию TF).
