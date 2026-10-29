
import copy

import numpy as np
import torch
from torch import nn

from agent.model import SnakeCNN


class DQNTrainer:
    def __init__(self, learning_rate=0.0001, gamma=0.99):
        self.gamma = gamma

        # The network we train
        self.policy_net = SnakeCNN()

        # A separate network for stable learning targets
        self.target_net = copy.deepcopy(self.policy_net)
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=learning_rate,
        )

        self.loss_fn = nn.SmoothL1Loss()

    def train_step(self, batch):
        # Each experience:
        # (state, action, reward, next_state, done)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert NumPy arrays into PyTorch tensors efficiently
        boards = torch.from_numpy(
            np.stack([s[0] for s in states]).astype(np.float32)
        )

        directions = torch.from_numpy(
            np.stack([s[1] for s in states]).astype(np.float32)
        )

        next_boards = torch.from_numpy(
            np.stack([s[0] for s in next_states]).astype(np.float32)
        )

        next_directions = torch.from_numpy(
            np.stack([s[1] for s in next_states]).astype(np.float32)
        )

        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # 1. Predict Q-values for all three actions
        q_values = self.policy_net(boards, directions)

        # 2. Select Q-values for actions actually taken
        predicted = q_values.gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)

        # 3. Calculate learning targets
        with torch.no_grad():
            next_q_values = self.target_net(
                next_boards,
                next_directions,
            )

            best_next_q = next_q_values.max(dim=1).values

            targets = (
                rewards
                + self.gamma * best_next_q * (1 - dones)
            )

        # 4. Calculate prediction error
        loss = self.loss_fn(predicted, targets)

        # 5. Update the policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        # Copy learned weights into the target network
        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )
