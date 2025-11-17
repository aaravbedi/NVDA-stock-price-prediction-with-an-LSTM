import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from src.data_utils import (
    download_stock_data,
    prepare_close_price_series,
    scale_series,
    create_sequences,
    train_test_split_sequences,
)
from src.lstm_model import StockLSTM


def create_dataloaders(seq_length=60,
                       batch_size=32,
                       test_ratio=0.2,
                       ticker="NVDA"):
    df = download_stock_data(ticker=ticker)
    close_series = prepare_close_price_series(df)
    scaled_close, scaler = scale_series(close_series)
    X, y = create_sequences(scaled_close, seq_length=seq_length)
    X_train, y_train, X_test, y_test = train_test_split_sequences(
        X, y, test_ratio=test_ratio
    )
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    train_ds = TensorDataset(X_train_t, y_train_t)
    test_ds = TensorDataset(X_test_t, y_test_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, scaler, df.index[-len(y):], y_test_t


def train_model(num_epochs=15,
                seq_length=60,
                batch_size=32,
                lr=1e-3,
                ticker="NVDA",
                device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, test_loader, scaler, date_index, y_test_t = create_dataloaders(
        seq_length=seq_length,
        batch_size=batch_size,
        ticker=ticker
    )

    model = StockLSTM(input_size=1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(X_batch)

        avg_loss = epoch_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs} - train MSE: {avg_loss:.6f}")

    model.eval()
    preds_list = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch)
            preds_list.append(preds.cpu().numpy())

    preds_array = np.vstack(preds_list)
    true_array = y_test_t.numpy()    
    preds_denorm = scaler.inverse_transform(preds_array)
    true_denorm = scaler.inverse_transform(true_array)

    return model, preds_denorm, true_denorm, date_index


def plot_results(dates, true_prices, pred_prices, ticker="NVDA"):
    plt.figure(figsize=(10, 5))
    plt.plot(dates, true_prices, label="True Close", linewidth=2)
    plt.plot(dates, pred_prices, label="Predicted Close", linewidth=2, linestyle="--")
    plt.title(f"{ticker} – LSTM next-day close prediction (test set)")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", f"{ticker}_lstm_test.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    model, preds, true_vals, date_index = train_model()
    plot_results(date_index, true_vals, preds, ticker="NVDA")
