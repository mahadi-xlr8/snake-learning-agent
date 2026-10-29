import torch
from torch import nn

from agent.features import FEATURE_SIZE
from game.constants import GRID_SIZE


class SnakeCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(5, 16, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            # 20x20 -> 10x10: fewer numbers for the dense layer to learn
            nn.MaxPool2d(2),

            nn.Flatten(),
        )

        pooled_size = GRID_SIZE // 2

        self.head = nn.Sequential(
            nn.Linear(32 * pooled_size * pooled_size + 4, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )

    def forward(self, board, direction):
        features = self.features(board)

        combined = torch.cat(
            [features, direction], dim=1
        )

        return self.head(combined)


class SnakeMLP(nn.Module):
    def __init__(self, input_size=FEATURE_SIZE):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 3),
        )

    def forward(self, features):
        return self.layers(features)


MODELS = {
    "cnn": SnakeCNN,
    "mlp": SnakeMLP,
}


def build_model(model_type):
    return MODELS[model_type]()


def load_model(model_type, path):
    model = build_model(model_type)
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model
