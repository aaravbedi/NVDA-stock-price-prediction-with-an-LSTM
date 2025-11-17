# NVDA Stock Price Prediction – LSTM Experiment

This is a small experiment where I used an LSTM model to predict the next-day closing price of NVIDIA (NVDA) stock based on historical daily data.

The goal here is not to build a production trading system, but to have a clean, readable example of:
- pulling stock price data into Python,
- turning that into supervised learning sequences,
- training a simple LSTM on closing prices,
- and plotting predicted vs. actual values on a test window.

The code is written so it can be reused for other tickers as well.

## Project Structure

```text
nvda-stock-lstm/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── data_utils.py     # data download + preprocessing + sequence generation
│   └── lstm_model.py     # PyTorch LSTM model definition
└── scripts/
    └── train_lstm.py     # training loop + evaluation + plots
