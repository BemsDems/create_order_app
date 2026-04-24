# Stock Forecast App (StockAI RU)

Одностраничное веб-приложение (React JSX + inline styles) для прогнозирования акций российского фондового рынка. Использует **реальную нейронную сеть (LSTM, SONNET v3)**, обученную на котировках MOEX с 2015 года.

## Архитектура

```
┌─────────────────────┐          ┌───────────────────────────────┐
│ SPA (index.html)    │  POST    │ FastAPI backend                │
│ React + inline CSS  │ ──────▶  │   /api/forecast  /api/companies│
│ 54 тикера           │          │   pure-numpy LSTM inference    │
└─────────────────────┘          │   ↕                            │
                                 │   MOEX ISS (live OHLCV)        │
                                 │   CBR XML  (USD/RUB)           │
                                 └───────────────────────────────┘
```

- **Фронт** (`stock-forecast-app/index.html`) — полностью автономный HTML: React 18 UMD + Babel Standalone. Единственный внешний запрос — на бэкенд `stock-forecast-backend`.
- **Бэк** (`../stock-forecast-backend/`) — FastAPI + pure-numpy инференс LSTM-модели. Без TensorFlow, без sklearn, без pandas — весит ~70 МБ в проде.

## Секции

1. **Hero** — анимированные частицы на canvas, градиентный заголовок, кнопка «Начать анализ».
2. **Каталог** — 54 тикера Мосбиржи, поиск, фильтры по 14 секторам, glassmorphism-карточки.
3. **Прогноз** — горизонт **5 торговых дней** (как обучена модель), анимация «расчёта», вердикт, вероятность роста ≥ +2%, 6 факторов влияния (волатильность, тренд, объём, RSI, MACD, USD/RUB), AUC, дисклеймер.
4. **О проекте** — описание методологии.

## Как запустить локально

Без сборки, прямо в браузере:

```bash
cd stock-forecast-app
python3 -m http.server 8000
# откройте http://localhost:8000
```

По умолчанию фронт ходит на `https://stock-forecast-backen-rmclwnun.fly.dev`. Чтобы указать другой бэкенд:

```html
<script>window.__API_BASE__ = 'https://my-backend.example.com';</script>
```

## Модель

- Архитектура: `LSTM(32) → LayerNorm → LSTM(16) → LayerNorm → Dense(12, ReLU) → Dense(1, sigmoid)`
- Вход: окно в 30 торговых дней × 15 признаков
- Признаки: `ret_1d, ret_5d, ret_10d, ret_20d, log_ret, price_vs_sma20, price_vs_sma50, trend_up, rsi_14, macd_histogram, volatility_20, volume_ratio, usd_ret_5d, price_position, bb_position`
- Скейлинг: RobustScaler (медиана + IQR), обучен на train-сплите SBER 2015–2024
- Таргет: P(цена закрытия вырастет ≥ +2% в ближайшие 5 торговых дней)
- Исторически обучен на SBER; для остальных 53 тикеров применяется в режиме zero-shot (прогноз выдаётся, но точность может быть ниже, о чём предупреждает подпись к AUC)

## Деплой

- Фронт: https://stock-forecast-app-swdoofbq.devinapps.com
- Бэк:   https://stock-forecast-backen-rmclwnun.fly.dev (healthz)

## Дисклеймер

Прогнозы носят информационный характер и не являются инвестиционной рекомендацией.
