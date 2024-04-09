import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, data_len):
        super(LSTM, self).__init__()
        self.data_len = data_len
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    
    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, self.data_len, self.lstm.hidden_size)
        c0 = torch.zeros(self.lstm.num_layers, self.data_len, self.lstm.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        # print(out.shape)
        return out
