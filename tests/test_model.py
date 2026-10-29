import torch

from agent.model import SnakeCNN


def test_model_output_shape():
    model = SnakeCNN()

    board = torch.zeros(1, 5, 20, 20)
    direction = torch.tensor(
        [[0, 1, 0, 0]],
        dtype=torch.float32,
    )

    output = model(board, direction)

    assert output.shape == (1, 3)