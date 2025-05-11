import torch
import torch.nn as nn
import torch.optim as optim
from config import Config
import random
import numpy as np


class PPONetwork(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=Config.ppo_hidden_dim):
        super(PPONetwork, self).__init__()

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1),
        )

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        state = torch.FloatTensor(np.array(state)).to(self.device)
        action_probs = self.policy_net(state)
        value = self.value_net(state)
        return action_probs, value

    def get_action(self, state, temperature=1.0):
        action_probs, _ = self.forward(state)
        # Apply temperature scaling to action probabilities
        action_probs = action_probs / temperature
        action_probs = torch.softmax(action_probs, dim=-1)  # Re-normalize after scaling
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob


class PPOAgent:
    def __init__(self, input_dim, action_dim):
        self.network = PPONetwork(input_dim, action_dim)
        self.optimizer = optim.Adam(
            self.network.parameters(), lr=Config.ppo_learning_rate
        )
        self.clip_epsilon = Config.ppo_clip_epsilon
        self.value_loss_coef = Config.ppo_value_loss_coef
        self.entropy_coef = Config.ppo_entropy_coef
        # Exploration parameters
        self.initial_temperature = Config.ppo_initial_temperature
        self.min_temperature = Config.ppo_min_temperature
        self.temperature_decay = Config.ppo_temperature_decay
        self.current_temperature = self.initial_temperature
        self.step_count = 0

    def compute_loss(self, trajectory):
        states, actions, log_probs_old, returns, advantages = trajectory

        states = torch.FloatTensor(np.array(states)).to(self.network.device)
        actions = torch.LongTensor(np.array(actions)).to(self.network.device)
        log_probs_old = torch.FloatTensor(np.array(log_probs_old)).to(
            self.network.device
        )
        returns = torch.FloatTensor(np.array(returns)).to(self.network.device)

        advantages = torch.FloatTensor(np.array(advantages)).to(self.network.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        action_probs, values = self.network(states)
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        # Policy loss
        ratio = torch.exp(log_probs - log_probs_old)
        surr1 = ratio * advantages
        surr2 = (
            torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
            * advantages
        )
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = ((returns - values.squeeze()) ** 2).mean()

        # Total loss
        loss = (
            policy_loss
            + self.value_loss_coef * value_loss
            - self.entropy_coef * entropy
        )

        return loss

    def update(self, trajectory):
        self.optimizer.zero_grad()
        loss = self.compute_loss(trajectory)
        loss.backward()
        self.optimizer.step()
        # Update temperature after each update
        self.step_count += 1
        self.current_temperature = max(
            self.min_temperature,
            self.initial_temperature * (self.temperature_decay**self.step_count),
        )
        return loss.item()

    def get_action(self, state, temperature=1.0, epsilon=0.1):
        if random.random() < epsilon:
            action = random.randint(0, self.network.action_dim - 1)
            return action, torch.tensor(0.0).to(self.network.device)  # Dummy log_prob
        return self.network.get_action(state, temperature)

    def save(self, path):
        torch.save(self.network.state_dict(), path)

    def load(self, path):
        self.network.load_state_dict(torch.load(path))
