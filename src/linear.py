import torch
import torch.nn as nn
import torch.optim as optim

# class LinearNet(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(LinearNet, self).__init__()
#         self.fc1 = nn.Linear(input_size, 24)
#         self.fc2 = nn.Linear(24, 48)
#         self.fc3 = nn.Linear(48, 36)
#         self.fc4 = nn.Linear(36, 12)
#         # self.fc5 = nn.Linear(32, 12)
#         self.fc5 = nn.Linear(12, output_size)
#         self.dropout = nn.Dropout(0.5)
#         self.relu = nn.ReLU()
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         x = self.relu(x)
#         x = self.fc3(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc4(x)
#         x = self.relu(x)
#         x = self.fc5(x)
#         return x
    

class LinearNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 24),
            nn.ReLU(),
            nn.Linear(24, 48),
            nn.ReLU(),
            # nn.Linear(48, 48),
            # nn.ReLU(),
            # nn.Dropout(0.5),
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



# class LinearNet(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(LinearNet, self).__init__()
#         self.fc1 = nn.Linear(input_size, 24)
#         self.fc2 = nn.Linear(24, 32)
#         self.fc3 = nn.Linear(32, 48)
#         self.fc4 = nn.Linear(48, 32)
#         self.fc5 = nn.Linear(32, 12)
#         self.fc6 = nn.Linear(12, output_size)
#         self.dropout = nn.Dropout(0.5)
#         self.relu = nn.ReLU()
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         x = self.relu(x)
#         x = self.fc3(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc4(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc5(x)
#         x = self.relu(x)
#         x = self.fc6(x)
#         # x = self.softmax(x)
#         return x