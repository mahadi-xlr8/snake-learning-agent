
from agent.rewards import calculate_reward


def test_safe_move():
    assert calculate_reward(0, False) == -0.01


def test_eating_food():
    assert calculate_reward(1, False) == 10.0


def test_collision():
    assert calculate_reward(0, True) == -10.0


def test_starvation():
    assert calculate_reward(0, True) == -10.0


def test_winning():
    assert calculate_reward(1, True, True) == 20.0
