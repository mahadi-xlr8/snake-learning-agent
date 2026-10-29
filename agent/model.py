import torch
from torch import nn


class SnakeCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(5, 16, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Flatten(),
        )

        self.head = nn.Sequential(
            nn.Linear(32 * 20 * 20 + 4, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )

    def forward(self, board, direction):
        features = self.features(board)

        combined = torch.cat(
            [features, direction], dim=1
        )

        return self.head(combined)