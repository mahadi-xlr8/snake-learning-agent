import numpy as np

from agent.observation import encode_board
from game.constants import UP, RIGHT, DOWN, LEFT


DIRECTIONS = [UP, RIGHT, DOWN, LEFT]


def get_state(game):
    board = encode_board(game.get_board())

    direction = np.zeros(4, dtype=np.float32)
    direction[DIRECTIONS.index(game.direction)] = 1.0

    return board, direction