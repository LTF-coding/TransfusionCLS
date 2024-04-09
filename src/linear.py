import torch
import torch.nn as nn
import torch.optim as optim

class LinearNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 24),
            nn.ReLU(),
            nn.Linear(24, 48),
            nn.ReLU(),
            nn.Linear(48, 36),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(36, 12),
            nn.ReLU(),
            nn.Linear(12,output_size),
            )

    def forward(self, x):
        x = self.layers(x)
        return x
