# Stock Forecast App (StockAI RU)

Одностраничное веб-приложение (React JSX + inline styles) для прогнозирования акций российского фондового рынка. **Инференс трёх TCN-сетей и feature engineering выполняются прямо в браузере** — никакого backend-сервиса.

## Архитектура

```
┌──────────────────────────────────────────────┐
│ SPA (index.html)                              │
│ React 18 + Babel Standalone (CDN)             │
│   ├── inference.js  (pure-JS TCN forward)     │
│   └── tcn_{short,medium,long}_weights.json    │
│       tcn_meta.json (RobustScaler, routing)   │
│                                               │
│  fetch ────▶ MOEX ISS  (OHLCV, IMOEX, divs)   │
│              CORS-OK, без посредников          │
└──────────────────────────────────────────────┘
```

- Frontend: `stock-forecast-app/index.html` + `assets/`. Грузится напрямую с любого статичного хостинга (devinapps, Netlify, GitHub Pages, локальный `python -m http.server`).
- Backend `stock-forecast-backend/` оставлен в репо как референс реализации — не требуется для работы фронта.

## Секции

1. **Hero** — анимированные частицы на canvas, градиентный заголовок, кнопка «Начать анализ».
2. **Каталог** — 54 тикера Мосбиржи, поиск, фильтры по 14 секторам, glassmorphism-карточки.
3. **Прогноз** — три группы горизонтов (короткий / средний / длинный), анимация «расчёта», вердикт, вероятность роста ≥ порога, 7 факторов влияния (волатильность, тренд, объём, дивиденды, IMOEX, USD/RUB, RSI), AUC, дисклеймер.
4. **О проекте** — описание методологии.

## Горизонты прогноза

| Группа    | Горизонты, торг. дн. | Порог | Какая модель |
|-----------|----------------------|-------|--------------|
| Короткий  | **5, 10**            | +3 %  | `tcn_short`  (обучена на 5-дн. таргете)   |
| Средний   | **30, 60**           | +5 %  | `tcn_medium` (обучена на 30-дн. таргете)  |
| Длинный   | **120, 240, 365**    | +12 % | `tcn_long`   (обучена на 120-дн. таргете) |

## Как запустить локально

```bash
cd stock-forecast-app
python3 -m http.server 8000
# откройте http://localhost:8000
```

Никаких сборщиков и токенов не нужно.

## Что внутри `assets/`

- `tcn_meta.json` (~5 KB) — RobustScaler центры/масштабы для всех трёх моделей, маршрутизация горизонтов, порядок 28 признаков.
- `tcn_short_weights.json`, `tcn_medium_weights.json`, `tcn_long_weights.json` (~225 KB каждый) — веса TCN-моделей, экспортированные из `.keras` в плоский JSON.
- `inference.js` (~26 KB) — чистый JS: feature engineering (28 признаков в том же порядке, что в `features.py`), causal Conv1D, BatchNorm, LayerNorm, GAP, Dense, sigmoid; MOEX ISS data fetch; high-level API: `TCN.fetchAllForTicker(ticker)`, `TCN.buildFeatures(...)`, `TCN.predict(matrix, horizon)`.

Весь pipeline проверен против reference numpy-инференса (`stock-forecast-backend/app/inference.py`): максимальная разница между JS и numpy на SBER — `7.3·10⁻⁹` для всех семи горизонтов.

## Модель

- Архитектура (каждая из трёх): `LayerNorm → Conv1D(32, k=3, causal, dil=1) → BN → Conv1D(32, k=3, causal, dil=2) → BN → Conv1D(16, k=3, causal, dil=4) → BN → GAP → Dense(64) → Dense(32) → Dense(1, sigmoid)`. Это TCN — temporal-convolutional network с экспоненциально растущей дилатацией.
- Вход: окно в 30 торговых дней × **28 признаков**: 14 технических (логдоходности 1/2/3/5/10, тренд относительно SMA20/SMA200, vol_rel, vol_spike, RSI14, oversold/overbought, price_pos_20, volatility_20), 3 дивидендных (`div_yield_ttm`, `days_since_last_div`, `div_yield_is_missing`), 3 валютных (USD/RUB log-return 1/5 + 20-дневная волатильность; в браузере подаются нули, т. к. ЦБ-API не CORS-friendly — соответствует zero-fallback из тренинг-кода), 2 нефтяных (Brent — заглушка), 3 IMOEX (log-return 1/5/20), 3 yahoo-фундамента (заглушки).
- Скейлинг: per-model RobustScaler (медиана + IQR), вшит в `tcn_meta.json`.
- Таргет: P(цена закрытия вырастет ≥ +N % в ближайшие H торговых дней), где (N, H) ∈ {(3, 5), (5, 30), (12, 120)}.

## Деплой

Live: https://stock-forecast-build-atlbkfsp.devinapps.com

## Дисклеймер

Прогнозы носят информационный характер и не являются инвестиционной рекомендацией.
