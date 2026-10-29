import random

import torch


def choose_action(model, state, epsilon):
    # Explore: choose a random action
    if random.random() < epsilon:
        return random.randint(0, 2)

    # Exploit: ask the CNN
    board, direction = state

    board_tensor = torch.from_numpy(board).unsqueeze(0)
    direction_tensor = torch.from_numpy(direction).unsqueeze(0)

    with torch.no_grad():
        q_values = model(board_tensor, direction_tensor)

    return q_values.argmax(dim=1).item()


def get_epsilon(
    episode,
    start=1.0,
    end=0.05,
    decay=500,
):
    return max(end, start - episode / decay)