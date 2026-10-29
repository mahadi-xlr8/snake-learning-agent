
from agent.rewards import calculate_reward


def test_eating_food():
    assert calculate_reward(1, False) == 10.0


def test_collision():
    assert calculate_reward(0, True) == -10.0


def test_winning():
    assert calculate_reward(1, True, True) == 20.0


def test_moving_closer_to_food():
    reward = calculate_reward(
        score_delta=0,
        done=False,
        old_distance=8,
        new_distance=7,
    )
    assert round(reward, 2) == 0.04


def test_moving_farther_from_food():
    reward = calculate_reward(
        score_delta=0,
        done=False,
        old_distance=7,
        new_distance=8,
    )
    assert round(reward, 2) == -0.06


def test_safe_move_without_distance():
    assert calculate_reward(0, False) == -0.01


def test_starvation():
    assert calculate_reward(0, True) == -10.0
