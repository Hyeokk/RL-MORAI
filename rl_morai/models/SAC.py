# === SAC.py ===

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque

class GaussianPolicy(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, output_dim)
        self.log_std = nn.Linear(hidden_dim, output_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return mean, std

    def sample(self, x):
        mean, std = self.forward(x)
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
        return action, log_prob.sum(dim=1, keepdim=True)

class QNetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.buffer = deque(maxlen=max_size)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        s, a, r, s_prime, d = zip(*samples)
        return (np.array(s), np.array(a), np.array(r).reshape(-1, 1),
                np.array(s_prime), np.array(d).reshape(-1, 1))

    def size(self):
        return len(self.buffer)

class SACAgent:
    def __init__(self, state_dim, action_dim, action_bounds, gamma=0.99, tau=0.005, alpha=0.2, actor_lr=3e-4, critic_lr=3e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = GaussianPolicy(state_dim, action_dim).to(self.device)
        self.critic1 = QNetwork(state_dim, action_dim).to(self.device)
        self.critic2 = QNetwork(state_dim, action_dim).to(self.device)
        self.target_critic1 = QNetwork(state_dim, action_dim).to(self.device)
        self.target_critic2 = QNetwork(state_dim, action_dim).to(self.device)

        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_opt = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_opt = optim.Adam(self.critic2.parameters(), lr=critic_lr)

        self.replay_buffer = ReplayBuffer()
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.action_bounds = action_bounds

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action, _ = self.actor.sample(state)
        action = action.detach().cpu().numpy()[0]

        scaled = []
        for i in range(len(action)):
            low, high = self.action_bounds[i]
            scaled.append(np.clip((action[i] + 1) / 2 * (high - low) + low, low, high))
        return np.array(scaled, dtype=np.float32)

    def preprocess_image(self, image):
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # HWC → CHW
        return image

    def store(self, transition):
        self.replay_buffer.add(transition)

    def train(self, batch_size=64):
        if self.replay_buffer.size() < batch_size:
            return

        s, a, r, s_prime, d = self.replay_buffer.sample(batch_size)

        s = torch.FloatTensor(s).to(self.device)
        a = torch.FloatTensor(a).to(self.device)
        r = torch.FloatTensor(r).to(self.device)
        s_prime = torch.FloatTensor(s_prime).to(self.device)
        d = torch.FloatTensor(d).to(self.device)

        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(s_prime)
            target_q1 = self.target_critic1(s_prime, next_action)
            target_q2 = self.target_critic2(s_prime, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            y = r + self.gamma * (1 - d) * target_q

        q1 = self.critic1(s, a)
        q2 = self.critic2(s, a)
        critic1_loss = F.mse_loss(q1, y)
        critic2_loss = F.mse_loss(q2, y)

        self.critic1_opt.zero_grad()
        critic1_loss.backward()
        self.critic1_opt.step()

        self.critic2_opt.zero_grad()
        critic2_loss.backward()
        self.critic2_opt.step()

        new_action, log_prob = self.actor.sample(s)
        q1_new = self.critic1(s, new_action)
        q2_new = self.critic2(s, new_action)
        actor_loss = (self.alpha * log_prob - torch.min(q1_new, q2_new)).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        for param, target_param in zip(self.critic1.parameters(), self.target_critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

