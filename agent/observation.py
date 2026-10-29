import numpy as np

from game.constants import (
    WALL,
    OBSTACLE,
    SNAKE_BODY,
    SNAKE_HEAD,
    FOOD,
)

CELL_TYPES = [
    WALL,
    OBSTACLE,
    SNAKE_BODY,
    SNAKE_HEAD,
    FOOD,
]


def encode_board(board):
    channels = []

    for cell_type in CELL_TYPES:
        channel = (board == cell_type).astype(np.float32)
        channels.append(channel)

    return np.stack(channels, axis=0)