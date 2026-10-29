from game.snake_game import SnakeGame
from agent.observation import encode_board

game = SnakeGame(seed=42)

board = game.get_board()
observation = encode_board(board)

print("Board:", board.shape)
print("Observation:", observation)
print("Food cells:", observation[4].sum())