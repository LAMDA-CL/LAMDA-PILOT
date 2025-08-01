import torch
from torch import nn
from torch.nn import functional as F
class sar_cnn(nn.Module):
    def __init__(self, in_features= 3, out_features = 10):
        super(sar_cnn, self).__init__()
        self.in_features = in_features
        self.out_features= out_features
        self.conv16 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7)
        self.conv32 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5)
        self.conv64 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.conv128 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.4)
        self.flatten=  nn.Flatten()

    def forward(self, x):
        x = F.relu(self.conv16(x))
        x = self.pool(x)
        x = F.relu(self.conv32(x))
        x = self.pool(x)
        x = F.relu(self.conv64(x))
        x = self.pool(x)
        x = F.relu(self.conv128(x))
        x = self.dropout(x)
        x= self.flatten(x)
        return x