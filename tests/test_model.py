import torch

from agent.features import FEATURE_SIZE
from agent.model import build_model


def test_cnn_output_shape():
    model = build_model("cnn")

    board = torch.zeros(1, 5, 20, 20)
    direction = torch.tensor(
        [[0, 1, 0, 0]],
        dtype=torch.float32,
    )

    output = model(board, direction)

    assert output.shape == (1, 3)


def test_mlp_output_shape():
    model = build_model("mlp")

    features = torch.zeros(1, FEATURE_SIZE)

    output = model(features)

    assert output.shape == (1, 3)
