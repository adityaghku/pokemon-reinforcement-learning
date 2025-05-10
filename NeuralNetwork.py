import torch
import torch.nn as nn
import torch.optim as optim
from config import Config


class PPONetwork(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=Config.ppo_hidden_dim):
        super(PPONetwork, self).__init__()

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
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
        state = torch.FloatTensor(state).to(self.device)
        action_probs = self.policy_net(state)
        value = self.value_net(state)
        return action_probs, value

    def get_action(self, state):
        action_probs, _ = self.forward(state)
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

    def compute_loss(self, trajectory):
        states, actions, log_probs_old, returns, advantages = trajectory

        states = torch.FloatTensor(states).to(self.network.device)
        actions = torch.LongTensor(actions).to(self.network.device)
        log_probs_old = torch.FloatTensor(log_probs_old).to(self.network.device)
        returns = torch.FloatTensor(returns).to(self.network.device)
        advantages = torch.FloatTensor(advantages).to(self.network.device)

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
        return loss.item()

    def save(self, path):
        torch.save(self.network.state_dict(), path)

    def load(self, path):
        self.network.load_state_dict(torch.load(path))
