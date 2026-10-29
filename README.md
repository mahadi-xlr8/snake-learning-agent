# Snake Learning Agent

An AI agent that learns to play the classic Snake game through reinforcement learning. The game is rendered with Pygame, and the agent learns by trial and error: it gets rewarded for eating food and penalized for crashing into walls or itself.

## Requirements

- Python 3.12
- [pygame](https://www.pygame.org/) – game rendering and input
- [numpy](https://numpy.org/) – state representation and numerical work
- [PyTorch](https://pytorch.org/) – the neural networks (MLP and CNN)
- [matplotlib](https://matplotlib.org/) – training plots
- [python-dotenv](https://pypi.org/project/python-dotenv/) – settings from `.env`
- [pytest](https://pytest.org/) – tests

## Setup

```bash
# Create and activate a virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## How it works

1. **Environment** – a grid-based Snake game. Each step, the agent observes the game state (danger nearby, current direction, food location).
2. **Agent** – chooses an action: go straight, turn left, or turn right.
3. **Reward** – positive for eating food, negative for dying, small shaping signals otherwise.
4. **Learning** – the agent updates its policy from these rewards over many games, gradually improving its score.

## Two agents: MLP and CNN

| Model | Sees | Learns |
|---|---|---|
| `mlp` | 17 hand-made features (danger, food direction, free space) | Fast: hundreds of episodes |
| `cnn` | The whole board as an image, rotated so the snake faces up | Slow: thousands of episodes |

Both are trained with Double DQN (replay memory, target network, epsilon-greedy).

```bash
# Train (pick a model; MODEL_TYPE in .env sets the default)
python train.py --model mlp --episodes 500
python train.py --model cnn --episodes 3000

# Check a trained model (no random moves)
python evaluate.py --model mlp --episodes 50
python evaluate.py --model mlp --checkpoint snake_dqn_mlp_best.pth --render

# Plot one or compare both
python plot_training.py training_history_mlp.csv training_history_cnn.csv
```

Each model saves `snake_dqn_<model>.pth`, `snake_dqn_<model>_best.pth` and `training_history_<model>.csv`.

New to reinforcement learning? Read [documentations.md](documentations.md). It explains every idea in this project from the basics.

## Running tests

```bash
pytest
```

## License

This project is for personal learning and experimentation.
