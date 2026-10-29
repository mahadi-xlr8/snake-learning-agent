import random
from collections import deque


class ReplayMemory:
    def __init__(self, capacity=10_000):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        experience = (
            state,
            action,
            reward,
            next_state,
            done,
        )

        self.memory.append(experience)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)