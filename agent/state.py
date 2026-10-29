import numpy as np
import torch

from agent.actions import DIRECTIONS
from agent.features import get_features
from agent.observation import encode_board


# How many 90-degree counter-clockwise turns make the heading point up
ROTATIONS = {direction: k for k, direction in enumerate(DIRECTIONS)}


def get_cnn_state(game):
    board = encode_board(game.get_board())

    # Rotate the board so the snake always faces up.
    # Then "turn right" always means the same thing on the image.
    k = ROTATIONS[game.direction]
    board = np.ascontiguousarray(np.rot90(board, k, axes=(1, 2)))

    direction = np.zeros(4, dtype=np.float32)
    direction[DIRECTIONS.index(game.direction)] = 1.0

    return board, direction


def get_mlp_state(game):
    return (get_features(game),)


STATE_BUILDERS = {
    "cnn": get_cnn_state,
    "mlp": get_mlp_state,
}


def get_state(game, model_type):
    return STATE_BUILDERS[model_type](game)


def states_to_tensors(states):
    """Stack a list of state tuples into one batched tensor per part."""
    return [
        torch.from_numpy(
            np.stack([state[i] for state in states]).astype(np.float32)
        )
        for i in range(len(states[0]))
    ]
