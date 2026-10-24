from collections import deque
import random
import numpy as np

from game.constants import (
    GRID_SIZE, EMPTY, WALL, SNAKE_BODY, SNAKE_HEAD, FOOD, OBSTACLE,
    UP, DOWN, LEFT, RIGHT, OBSTACLES,
)


class SnakeGame:
    """Headless Snake environment. Coordinates are (x, y)."""

    def __init__(self, size=GRID_SIZE, obstacles=OBSTACLES, seed=None):
        if size < 6:
            raise ValueError("Board size must be at least 6")
        self.size = size
        self.obstacles = set(obstacles)
        if any(not (0 < x < size - 1 and 0 < y < size - 1) for x, y in self.obstacles):
            raise ValueError("Obstacles must be inside the walls")
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        mid = self.size // 2
        start = [(mid, mid), (mid - 1, mid), (mid - 2, mid)]
        if any(p in self.obstacles for p in start):
            raise ValueError("Obstacles overlap the snake's starting position")
        self.snake = deque(start)  # Head first
        self.direction = RIGHT
        self.next_direction = RIGHT
        self.score = 0
        self.done = False
        self.won = False
        self.food = None
        self._spawn_food()
        return self.get_board()

    def _spawn_food(self):
        occupied = set(self.snake) | self.obstacles
        free = [(x, y) for y in range(1, self.size - 1)
                for x in range(1, self.size - 1) if (x, y) not in occupied]
        self.food = self.rng.choice(free) if free else None
        if self.food is None:
            self.done = True
            self.won = True

    def change_direction(self, direction):
        if self.done:
            return
        if direction not in (UP, DOWN, LEFT, RIGHT):
            raise ValueError("Invalid direction")
        # Compare with current direction, not queued direction: no 180-degree turn.
        if (direction[0] + self.direction[0], direction[1] + self.direction[1]) != (0, 0):
            self.next_direction = direction

    def step(self, direction=None):
        """Advance exactly one cell. Returns (board, score_delta, done, info).

        score_delta is NOT the RL reward; a critic will be added later.
        """
        if self.done:
            return self.get_board(), 0, True, {"score": self.score, "won": self.won}
        if direction is not None:
            self.change_direction(direction)
        self.direction = self.next_direction
        x, y = self.snake[0]
        dx, dy = self.direction
        head = (x + dx, y + dy)
        growing = head == self.food
        # The tail vacates its cell on a normal move, so moving into it is legal.
        body_to_check = set(self.snake if growing else list(self.snake)[:-1])
        hit_wall = not (0 < head[0] < self.size - 1 and 0 < head[1] < self.size - 1)
        if hit_wall or head in self.obstacles or head in body_to_check:
            self.done = True
            return self.get_board(), 0, True, {"score": self.score, "won": False}
        self.snake.appendleft(head)
        score_delta = 0
        if growing:
            self.score += 1
            score_delta = 1
            self._spawn_food()
        else:
            self.snake.pop()
        return self.get_board(), score_delta, self.done, {"score": self.score, "won": self.won}

    def get_board(self):
        """Return a new (height, width) NumPy array of categorical cell IDs."""
        board = np.full((self.size, self.size), EMPTY, dtype=np.uint8)
        board[0, :] = WALL
        board[-1, :] = WALL
        board[:, 0] = WALL
        board[:, -1] = WALL
        for x, y in self.obstacles:
            board[y, x] = OBSTACLE
        if self.food is not None:
            board[self.food[1], self.food[0]] = FOOD
        for x, y in list(self.snake)[1:]:
            board[y, x] = SNAKE_BODY
        x, y = self.snake[0]
        board[y, x] = SNAKE_HEAD
        return board
