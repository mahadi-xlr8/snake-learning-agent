import numpy as np

from agent.learning import DQNTrainer


def make_state():
    board = np.zeros((5, 20, 20), dtype=np.float32)
    direction = np.array([0, 1, 0, 0], dtype=np.float32)
    return board, direction


def test_training_step():
    trainer = DQNTrainer()

    state = make_state()

    batch = [
        (state, 0, 10.0, state, False),
        (state, 1, -10.0, state, True),
    ]

    loss = trainer.train_step(batch)

    assert isinstance(loss, float)
    assert loss >= 0