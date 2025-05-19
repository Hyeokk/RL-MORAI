import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque


# Actor: state -> action
class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.tanh(self.fc3(x))


# Critic: (state, action) -> Q-value
class Critic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim + action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# Simple Replay Buffer
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


# DDPG Agent Class
class DDPGAgent:
    def __init__(self, 
                 state_dim, 
                 action_dim, 
                 action_bounds,
                 actor_lr=1e-4, 
                 critic_lr=1e-3, 
                 gamma=0.99, 
                 tau=0.005):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.actor_target = Actor(state_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.replay_buffer = ReplayBuffer()
        self.gamma = gamma
        self.tau = tau
        self.action_bounds = action_bounds  # e.g. [(-1, 1), (0, 1)] for steer/throttle

        self.total_steps = 0

    def preprocess_image(self, image):
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # HWC → CHW
        return image
    
    def get_action(self, state, noise=0.1):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state).detach().cpu().numpy()[0]
        action += noise * np.random.randn(len(action))
        action = np.clip(action, -1.0, 1.0)

        scaled = []
        for i in range(len(action)):
            low, high = self.action_bounds[i]
            scaled.append(np.clip((action[i] + 1) / 2 * (high - low) + low, low, high))
        return np.array(scaled, dtype=np.float32)

    def store(self, transition):
        self.replay_buffer.add(transition)

    def train(self, batch_size=64):
        if self.replay_buffer.size() < batch_size:
            return {}

        s, a, r, s_prime, d = self.replay_buffer.sample(batch_size)

        s = torch.FloatTensor(s).to(self.device)
        a = torch.FloatTensor(a).to(self.device)
        r = torch.FloatTensor(r).to(self.device)
        s_prime = torch.FloatTensor(s_prime).to(self.device)
        d = torch.FloatTensor(d).to(self.device)

        # Critic update
        with torch.no_grad():
            a_prime = self.actor_target(s_prime)
            q_target = self.critic_target(s_prime, a_prime)
            y = r + self.gamma * (1 - d) * q_target

        q_val = self.critic(s, a)
        critic_loss = F.mse_loss(q_val, y)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Actor update
        actor_loss = -self.critic(s, self.actor(s)).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Soft update
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.total_steps += 1

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "buffer_size": self.replay_buffer.size(),
            "total_steps": self.total_steps,
        }
