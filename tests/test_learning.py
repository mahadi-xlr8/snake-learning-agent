import pytest

from agent.learning import DQNTrainer
from agent.state import get_state
from game.snake_game import SnakeGame


@pytest.mark.parametrize("model_type", ["cnn", "mlp"])
def test_training_step(model_type):
    trainer = DQNTrainer(model_type=model_type)

    state = get_state(SnakeGame(seed=1), model_type)

    batch = [
        (state, 0, 10.0, state, False),
        (state, 1, -10.0, state, True),
    ]

    loss = trainer.train_step(batch)

    assert isinstance(loss, float)
    assert loss >= 0
