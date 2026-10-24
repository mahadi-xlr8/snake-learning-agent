import numpy as np
from game.snake_game import SnakeGame
from game.constants import RIGHT, UP, LEFT, SNAKE_HEAD, OBSTACLE


def test_board_shape_and_obstacles():
    game = SnakeGame(seed=7)
    board = game.get_board()
    assert board.shape == (20, 20)
    assert board[10, 10] == SNAKE_HEAD
    assert board[5, 5] == OBSTACLE


def test_auto_forward_and_no_reverse():
    game = SnakeGame(seed=7)
    game.change_direction(LEFT)
    game.step()
    assert game.snake[0] == (11, 10)
    game.change_direction(UP)
    game.step()
    assert game.snake[0] == (11, 9)


def test_wall_collision():
    game = SnakeGame(seed=7)
    for _ in range(10):
        _, _, done, _ = game.step()
        if done:
            break
    assert done


def test_food_growth():
    game = SnakeGame(seed=7)
    game.food = (11, 10)
    _, delta, done, info = game.step()
    assert delta == 1
    assert len(game.snake) == 4
    assert info["score"] == 1
    assert not done


def test_obstacle_collision():
    game = SnakeGame(seed=7)
    game.snake.clear()
    game.snake.extend([(4, 5), (3, 5), (2, 5)])
    game.direction = RIGHT
    game.next_direction = RIGHT
    _, _, done, _ = game.step()
    assert done


def test_board_is_a_copy():
    game = SnakeGame(seed=7)
    board = game.get_board()
    board[:] = 0
    assert np.any(game.get_board() != 0)
