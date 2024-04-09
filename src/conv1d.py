import torch.nn as nn

class ConvModel(nn.Module):
    def __init__(self, features_in, out, channels=32):
        super(ConvModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(channels, channels, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(channels, channels, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(channels, channels, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(channels, channels, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(channels, out, kernel_size=3),
            )

    def forward(self, x):
        x = x.view(x.shape[0], 1, x.shape[1])
        x = self.layers(x)
        x = x.squeeze()
        # print(x.shape)
        return x