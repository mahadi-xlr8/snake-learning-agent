# Snake Learning Agent

An AI agent that learns to play the classic Snake game through reinforcement learning. The game is rendered with Pygame, and the agent learns by trial and error: it gets rewarded for eating food and penalized for crashing into walls or itself.

## Requirements

- Python 3.12
- [pygame](https://www.pygame.org/) – game rendering and input
- [numpy](https://numpy.org/) – state representation and numerical work
- [pytest](https://pytest.org/) – tests

## Setup

```bash
# Create and activate a virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install pygame numpy pytest
```

## How it works

1. **Environment** – a grid-based Snake game. Each step, the agent observes the game state (danger nearby, current direction, food location).
2. **Agent** – chooses an action: go straight, turn left, or turn right.
3. **Reward** – positive for eating food, negative for dying, small shaping signals otherwise.
4. **Learning** – the agent updates its policy from these rewards over many games, gradually improving its score.

## Running tests

```bash
pytest
```

## License

This project is for personal learning and experimentation.
