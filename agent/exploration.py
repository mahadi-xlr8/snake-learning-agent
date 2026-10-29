
import random

import torch

from agent.state import states_to_tensors


def best_action(model, state):
    # Ask the network which action has the highest Q-value
    inputs = states_to_tensors([state])

    with torch.no_grad():
        q_values = model(*inputs)

    return q_values.argmax(dim=1).item()


def choose_action(model, state, epsilon):
    # Explore: choose a random action
    if random.random() < epsilon:
        return random.randint(0, 2)

    # Exploit: choose the best predicted action
    return best_action(model, state)


def get_epsilon(
    episode,
    start=1.0,
    end=0.01,
    decay=500,
):
    progress = min(episode / decay, 1.0)
    return end + (start - end) * (1 - progress)
