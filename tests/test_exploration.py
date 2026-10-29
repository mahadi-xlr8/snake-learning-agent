from agent.exploration import get_epsilon


def test_epsilon_starts_high():
    assert get_epsilon(0) == 1.0


def test_epsilon_decreases():
    assert get_epsilon(100) < get_epsilon(0)


def test_epsilon_has_minimum():
    assert get_epsilon(1000) == 0.05