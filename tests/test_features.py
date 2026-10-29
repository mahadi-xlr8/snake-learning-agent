import numpy as np

from agent.features import FEATURE_SIZE, get_features
from agent.observation import CELL_TYPES
from agent.state import get_cnn_state
from game.constants import DOWN, FOOD, RIGHT, SNAKE_HEAD, UP
from game.snake_game import SnakeGame


def place_snake(game, cells, direction):
    game.snake.clear()
    game.snake.extend(cells)
    game.direction = direction
    game.next_direction = direction


def test_feature_size():
    assert get_features(SnakeGame(seed=1)).shape == (FEATURE_SIZE,)


def test_danger_straight_at_wall():
    game = SnakeGame(seed=1)
    place_snake(game, [(18, 10), (17, 10), (16, 10)], RIGHT)

    danger_straight, danger_right, danger_left = get_features(game)[:3]

    assert danger_straight == 1.0
    assert danger_right == 0.0
    assert danger_left == 0.0


def test_food_side_is_relative_to_heading():
    game = SnakeGame(seed=1)
    place_snake(game, [(10, 10), (10, 11), (10, 12)], UP)
    game.food = (14, 10)

    ahead, behind, right, left = get_features(game)[7:11]

    assert (ahead, behind, right, left) == (0.0, 0.0, 1.0, 0.0)


def test_free_space_is_zero_for_blocked_move():
    game = SnakeGame(seed=1)
    place_snake(game, [(18, 10), (17, 10), (16, 10)], RIGHT)

    free_straight = get_features(game)[11]

    assert free_straight == 0.0


def test_free_space_detects_dead_end():
    game = SnakeGame(seed=1)
    # Turning right leads into a single cell boxed in by the body.
    place_snake(
        game,
        [(2, 2), (2, 3), (3, 3), (4, 3), (4, 2), (4, 1), (3, 1), (2, 1), (1, 1)],
        UP,
    )
    free_cells = (game.size - 2) ** 2 - len(game.obstacles)

    free_straight, free_right, free_left = get_features(game)[11:14]

    assert free_straight == 0.0
    assert free_right == 1 / np.float32(free_cells)
    assert free_left > 0.5


def test_cnn_board_is_rotated_to_face_up():
    game = SnakeGame(seed=1)
    place_snake(game, [(10, 10), (9, 10), (8, 10)], RIGHT)
    game.food = (13, 10)  # 3 cells ahead of a right-facing snake

    board, _ = get_cnn_state(game)
    head_y, head_x = np.argwhere(board[CELL_TYPES.index(SNAKE_HEAD)] == 1)[0]
    food_y, food_x = np.argwhere(board[CELL_TYPES.index(FOOD)] == 1)[0]

    assert food_x == head_x
    assert food_y == head_y - 3


def test_cnn_rotation_for_down():
    game = SnakeGame(seed=1)
    place_snake(game, [(10, 10), (10, 9), (10, 8)], DOWN)
    game.food = (10, 12)

    board, _ = get_cnn_state(game)
    head_y, head_x = np.argwhere(board[CELL_TYPES.index(SNAKE_HEAD)] == 1)[0]
    food_y, food_x = np.argwhere(board[CELL_TYPES.index(FOOD)] == 1)[0]

    assert (food_y, food_x) == (head_y - 2, head_x)
