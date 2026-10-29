
import copy

import torch
from torch import nn

from agent.model import build_model
from agent.state import states_to_tensors


class DQNTrainer:
    def __init__(
        self,
        model_type="mlp",
        learning_rate=0.001,
        gamma=0.99,
        max_grad_norm=10.0,
    ):
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm

        # The network we train
        self.policy_net = build_model(model_type)

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

        # Each state is a tuple of arrays, e.g. (board, direction)
        # for the CNN or (features,) for the MLP.
        state_inputs = states_to_tensors(states)
        next_state_inputs = states_to_tensors(next_states)

        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # 1. Predict Q-values for all three actions
        q_values = self.policy_net(*state_inputs)

        # 2. Select Q-values for actions actually taken
        predicted = q_values.gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)

        # 3. Calculate learning targets (Double DQN):
        # the policy net picks the next action,
        # the target net says how good it is.
        with torch.no_grad():
            next_actions = self.policy_net(
                *next_state_inputs
            ).argmax(dim=1, keepdim=True)

            next_q = self.target_net(
                *next_state_inputs
            ).gather(1, next_actions).squeeze(1)

            targets = (
                rewards
                + self.gamma * next_q * (1 - dones)
            )

        # 4. Calculate prediction error
        loss = self.loss_fn(predicted, targets)

        # 5. Update the policy network
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.policy_net.parameters(),
            self.max_grad_norm,
        )
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        # Copy learned weights into the target network
        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)
