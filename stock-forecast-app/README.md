# Stock Forecast App (StockAI RU)

Одностраничное веб-приложение (React JSX + inline styles) для прогнозирования акций российского фондового рынка. Использует **реальный ансамбль из трёх TCN-сетей**, обученных на MOEX по разным горизонтам.

## Архитектура

```
┌─────────────────────┐          ┌──────────────────────────────────┐
│ SPA (index.html)    │  POST    │ FastAPI backend                   │
│ React + inline CSS  │ ──────▶  │   /api/forecast  /api/companies   │
│ 54 тикера           │          │   pure-numpy TCN inference        │
│ 7 горизонтов        │          │   ↕                               │
└─────────────────────┘          │   MOEX ISS (OHLCV, IMOEX, divs)   │
                                 │   CBR XML  (USD/RUB)              │
                                 └──────────────────────────────────┘
```

- **Фронт** (`stock-forecast-app/index.html`) — полностью автономный HTML: React 18 UMD + Babel Standalone. Единственный внешний запрос — на бэкенд `stock-forecast-backend`.
- **Бэк** (`../stock-forecast-backend/`) — FastAPI + pure-numpy инференс трёх TCN-моделей. Без TensorFlow, без sklearn, без pandas — весит ~70 МБ в проде.

## Секции

1. **Hero** — анимированные частицы на canvas, градиентный заголовок, кнопка «Начать анализ».
2. **Каталог** — 54 тикера Мосбиржи, поиск, фильтры по 14 секторам, glassmorphism-карточки.
3. **Прогноз** — три группы кнопок горизонта (короткий/средний/длинный), анимация «расчёта», вердикт, вероятность роста ≥ порога, 7 факторов влияния (волатильность, тренд, объём, дивиденды, IMOEX, USD/RUB, RSI), AUC, дисклеймер.
4. **О проекте** — описание методологии.

## Горизонты прогноза

| Группа    | Горизонты, торг. дн. | Порог | Какая модель |
|-----------|----------------------|-------|--------------|
| Короткий  | **5, 10**            | +3 %  | `tcn_short`  (обучена на 5-дн. таргете)   |
| Средний   | **30, 60**           | +5 %  | `tcn_medium` (обучена на 30-дн. таргете)  |
| Длинный   | **120, 240, 365**    | +12 % | `tcn_long`   (обучена на 120-дн. таргете) |

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

- Архитектура (каждая из трёх): `LayerNorm → Conv1D(32, k=3, causal, dil=1) → BN → Conv1D(32, k=3, causal, dil=2) → BN → Conv1D(16, k=3, causal, dil=4) → BN → GAP → Dense(64) → Dense(32) → Dense(1, sigmoid)`. Это TCN — temporal-convolutional network с экспоненциально растущей дилатацией.
- Вход: окно в 30 торговых дней × **28 признаков**: 14 технических (логдоходности 1/2/3/5/10, тренд относительно SMA20/SMA200, vol_rel, vol_spike, RSI14, oversold/overbought, price_pos_20, volatility_20), 3 дивидендных (`div_yield_ttm`, `days_since_last_div`, `div_yield_is_missing`), 3 валютных (USD/RUB log-return 1/5 + 20-дневная волатильность), 2 нефтяных (Brent — заглушка), 3 IMOEX (log-return 1/5/20), 3 yahoo-фундамента (заглушки).
- Скейлинг: per-model RobustScaler (медиана + IQR), вшит в код инференса (см. `app/inference.py`).
- Таргет: P(цена закрытия вырастет ≥ +N % в ближайшие H торговых дней), где (N, H) ∈ {(3, 5), (5, 30), (12, 120)}.

## Деплой

- Фронт: https://stock-forecast-app-swdoofbq.devinapps.com
- Бэк:   https://stock-forecast-backen-rmclwnun.fly.dev (healthz)

## Дисклеймер

Прогнозы носят информационный характер и не являются инвестиционной рекомендацией.
