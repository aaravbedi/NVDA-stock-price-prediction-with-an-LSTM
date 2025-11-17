import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

def download_stock_data(ticker: str = "NVDA",
                        period: str = "5y",
                        interval: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval)
    df = df.dropna()
    return df


def prepare_close_price_series(df: pd.DataFrame) -> np.ndarray:
    if "Close" not in df.columns:
        raise ValueError("Input DataFrame must have a 'Close' column.")
    return df["Close"].values.reshape(-1, 1)


def create_sequences(data: np.ndarray,
                     seq_length: int = 60) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []

    for i in range(len(data) - seq_length):
        x = data[i : i + seq_length]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)


def train_test_split_sequences(X: np.ndarray,
                               y: np.ndarray,
                               test_ratio: float = 0.2) -> tuple:
    n_samples = len(X)
    n_test = int(n_samples * test_ratio)
    n_train = n_samples - n_test

    X_train = X[:n_train]
    y_train = y[:n_train]
    X_test = X[n_train:]
    y_test = y[n_train:]

    return X_train, y_train, X_test, y_test


def scale_series(values: np.ndarray) -> tuple[np.ndarray, MinMaxScaler]:
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    return scaled, scaler
