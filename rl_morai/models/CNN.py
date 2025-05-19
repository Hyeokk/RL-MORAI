import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNActionValue(nn.Module):
    def __init__(self, input_channels, action_dim):
        super(CNNActionValue, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1)

        # 💡 dummy 입력으로 in_features 자동 추정
        dummy_input = torch.zeros(1, input_channels, 120, 160)  # 실제 입력 크기 사용
        out = self.conv3(self.conv2(self.conv1(dummy_input)))
        self.in_features = out.view(1, -1).shape[1]  # flatten 후 사이즈 추정

        self.fc1 = nn.Linear(self.in_features, 256)
        self.fc2 = nn.Linear(256, action_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # 또는 torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
