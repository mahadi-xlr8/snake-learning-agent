
from collections import deque

import numpy as np

from agent.actions import (
    DIRECTIONS,
    STRAIGHT,
    TURN_RIGHT,
    TURN_LEFT,
    action_to_direction,
)


# 3 danger + 4 heading + 4 food side + 3 free space + 1 length + 2 food offset
FEATURE_SIZE = 17

RELATIVE_ACTIONS = (STRAIGHT, TURN_RIGHT, TURN_LEFT)


def _add(cell, direction):
    return cell[0] + direction[0], cell[1] + direction[1]


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def _inside(game, cell):
    x, y = cell
    return 0 < x < game.size - 1 and 0 < y < game.size - 1


def _free_cell_count(game):
    interior = (game.size - 2) ** 2
    return interior - len(game.obstacles)


def _blocked_cells(game):
    # The tail moves away on a normal step, so it is not a danger.
    return set(list(game.snake)[:-1]) | game.obstacles


def _reachable_space(game, start, blocked):
    """Count the cells the snake could reach from `start` (flood fill)."""
    if not _inside(game, start) or start in blocked:
        return 0

    seen = {start}
    queue = deque([start])

    while queue:
        cell = queue.popleft()

        for direction in DIRECTIONS:
            neighbor = _add(cell, direction)
            if (
                neighbor not in seen
                and _inside(game, neighbor)
                and neighbor not in blocked
            ):
                seen.add(neighbor)
                queue.append(neighbor)

    return len(seen)


def get_features(game):
    head = game.snake[0]
    heading = game.direction
    right = action_to_direction(heading, TURN_RIGHT)

    blocked = _blocked_cells(game)
    free_cells = _free_cell_count(game)

    # 1. Danger one step away: straight, right, left
    danger = []
    free_space = []

    for action in RELATIVE_ACTIONS:
        next_cell = _add(head, action_to_direction(heading, action))
        is_blocked = not _inside(game, next_cell) or next_cell in blocked
        danger.append(float(is_blocked))

        # 2. How much room is left if we move there
        free_space.append(
            _reachable_space(game, next_cell, blocked) / free_cells
        )

    # 3. Heading as a one-hot vector
    heading_one_hot = [0.0] * 4
    heading_one_hot[DIRECTIONS.index(heading)] = 1.0

    # 4. Where the food is, relative to the heading
    food_ahead = food_behind = food_right = food_left = 0.0
    food_forward_offset = food_right_offset = 0.0

    if game.food is not None:
        offset = (game.food[0] - head[0], game.food[1] - head[1])
        forward = _dot(offset, heading)
        sideways = _dot(offset, right)

        food_ahead = float(forward > 0)
        food_behind = float(forward < 0)
        food_right = float(sideways > 0)
        food_left = float(sideways < 0)

        food_forward_offset = forward / game.size
        food_right_offset = sideways / game.size

    length = len(game.snake) / free_cells

    features = (
        danger
        + heading_one_hot
        + [food_ahead, food_behind, food_right, food_left]
        + free_space
        + [length, food_forward_offset, food_right_offset]
    )

    return np.array(features, dtype=np.float32)
