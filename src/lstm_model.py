import torch
import torch.nn as nn
class StockLSTM(nn.Module):
    def __init__(self,
                 input_size: int = 1,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)    
        last_output = lstm_out[:, -1, :] 
        out = self.fc(last_output)        
        return out
