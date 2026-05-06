# Stock Forecast Backend (StockAI RU)

FastAPI-бэкенд, который обслуживает реальные TCN-прогнозы для SPA `stock-forecast-app`. Поддерживает 7 горизонтов через ансамбль из 3-х моделей.

## Что внутри

- **`app/inference.py`** — чистый numpy-форвард TCN: `LayerNorm → 3× (causal Conv1D + ReLU + BatchNorm) → GAP → Dense(64) → Dense(32) → Dense(1, sigmoid)`. Дилатации Conv1D: 1, 2, 4. Веса экспортированы из Keras (`model_short.keras`, `model_medium.keras`, `model_long.keras`) и лежат в `app/models/tcn_{short,medium,long}_weights.npz` (~50 КБ каждая). Per-model RobustScaler-параметры вшиты в код.
- **`app/features.py`** — feature engineering, воспроизводящий обучающий пайплайн (28 признаков): 14 технических, 3 дивидендных, 3 USD/RUB, 2 нефтяных (заглушка), 3 IMOEX, 3 yahoo (заглушка).
- **`app/data.py`** — живая подкачка с MOEX ISS (OHLCV, IMOEX-индекс, дивиденды, USD000UTSTOM) и курса USD/RUB с ЦБ.
- **`app/main.py`** — FastAPI-эндпоинты:
  - `GET  /healthz` — здоровье.
  - `GET  /api/companies` — каталог 54 тикеров + 14 секторов + список доступных горизонтов.
  - `POST /api/forecast` `{"ticker":"SBER","horizon":30}` — вероятность роста ≥ +N% за H торговых дней, вердикт, 7 факторов и объяснение. Поле `model_group` сообщает, какая из трёх TCN-моделей использовалась.

## Маршрутизация по горизонту

| Горизонт `horizon` | Группа    | Какая модель | Порог |
|--------------------|-----------|--------------|-------|
| 5, 10              | `short`   | `tcn_short`  | +3 %  |
| 30, 60             | `medium`  | `tcn_medium` | +5 %  |
| 120, 240, 365      | `long`    | `tcn_long`   | +12 % |

## Почему без TensorFlow

Первая версия зависела от `tensorflow-cpu`, но на 256 МБ Fly.io-машине процесс OOM-убивался при загрузке модели. Мы экспортировали веса Keras в NumPy-массивы и написали форвард-пасс TCN (causal Conv1D, BatchNorm, LayerNorm) вручную. Совпадение с Keras: diff ≤ 1.5 × 10⁻⁸ на случайных входах для всех трёх моделей.

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
  -d '{"ticker":"SBER","horizon":30}' | jq

# Все 7 горизонтов:
for H in 5 10 30 60 120 240 365; do
  curl -s -X POST http://127.0.0.1:8000/api/forecast \
    -H 'Content-Type: application/json' \
    -d "{\"ticker\":\"SBER\",\"horizon\":$H}" \
    | jq -r "\"h=\(.horizon_days) grp=\(.model_group) thr=\(.threshold_pct) p=\(.prob_up)\""
done
```

## Деплой

- Fly.io, запущен по адресу https://stock-forecast-backen-rmclwnun.fly.dev
- Конфиг в `fly.toml` (shared-cpu-1x / 1 GiB).
