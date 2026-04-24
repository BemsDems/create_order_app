# Stock Forecast v2 — Real Data + Keras LSTM

Real-time stock forecast for 54 Russian equities using a trained LSTM neural network (Sonnet v3) with live MOEX data.

## Architecture

```
stock-forecast-v2/
  backend/
    main.py              # FastAPI server
    requirements.txt
    model/
      sonnet_v3_final.keras   # Trained LSTM model
      best_sonnet_v3.keras    # Best checkpoint
      sonnet_v3_scaler.pkl    # RobustScaler for feature normalization
  frontend/
    index.html           # React SPA (single file)
```

## How it works

1. **Real data**: Backend fetches OHLCV from MOEX ISS API and USD/RUB from CBR XML API
2. **Feature engineering**: Computes 15 technical indicators (returns, RSI, MACD, Bollinger bands, volatility, volume ratio, USD impact, etc.)
3. **Model inference**: Feeds last 30 days of scaled features into the LSTM model
4. **Prediction**: Returns probability of 2%+ price increase within 5 trading days

## Quick Start

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Then open `frontend/index.html` in a browser, or access `http://localhost:8000/` (the backend serves the frontend too).

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/companies` | List all 54 companies with sectors |
| GET | `/api/quote/{ticker}` | Real-time price + daily change from MOEX |
| POST | `/api/predict_job/{ticker}` | Start async prediction job |
| GET | `/api/predict_job/{job_id}` | Poll prediction job status |
| GET | `/api/predict/{ticker}` | Synchronous prediction (blocking) |

## Model Details

- **Architecture**: LSTM (2 layers: 32 + 16 units) with LayerNormalization
- **Training data**: MOEX OHLCV from 2015 to present
- **Features**: 15 technical indicators
- **Target**: Binary classification — will price rise 2%+ in 5 days?
- **Scaler**: RobustScaler (fitted on training data)
