# Stock Price Prediction: PatchTST vs Baseline Models

Comparative study of deep learning architectures for S&P 500 stock price forecasting.

## Models

**Baseline** (`stock_prediction_baseline.ipynb`)  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IRedDragonICY/stock-prediction-patchtst/blob/main/stock_prediction_baseline.ipynb)
- MLP (2-layer, 256 hidden units)
- CNN (1D Conv with 64→128→256 channels)
- LSTM (2-layer, 256 hidden, bidirectional)

**PatchTST** (`stock_prediction_patchtst.ipynb`)  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IRedDragonICY/stock-prediction-patchtst/blob/main/stock_prediction_patchtst.ipynb)
- Transformer-based architecture with patch tokenization
- RevIN (Reversible Instance Normalization) for distribution shift
- Config: 64 lookback, patch_len=8, stride=4, d_model=384, 8 heads, 5 layers

## Features

Technical indicators derived from OHLCV data:
- Momentum: RSI, MACD, ROC
- Volatility: ATR, Bollinger Bands
- Volume: OBV, VWAP
- Trend: ADX, EMA

## Dataset

S&P 500 historical data (5 years) from [Kaggle](https://www.kaggle.com/datasets/camnugent/sandp500).

## Usage

```bash
pip install torch pandas pandas_ta matplotlib seaborn
```

Run notebooks in order:
1. `stock_prediction_baseline.ipynb` - Train baseline models
2. `stock_prediction_patchtst.ipynb` - Train PatchTST with weighted ensemble

## Metrics

- RMSE (Root Mean Square Error)
- MAPE (Mean Absolute Percentage Error)
- Directional Accuracy
- R² Score

## Authors

- Mohammad Farid Hendianto (2200018401)
- Fidyah Rahman (2200018185)

Kapita Selekta - Kelompok 3
