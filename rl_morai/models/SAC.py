import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from collections import deque

# === 네트워크 정의 ===
class CNNEncoder(nn.Module):
    def __init__(self, input_shape, output_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),  # 🔧 1채널로 고정
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.flatten_dim = self._get_flatten_dim(input_shape)
        self.fc = nn.Linear(self.flatten_dim, output_dim)

    def _get_flatten_dim(self, input_shape):
        with torch.no_grad():
            x = torch.zeros(1, 1, input_shape[0], input_shape[1])  # 🔧 (1, 1, H, W)
            x = self.conv(x)
            return int(np.prod(x.size()))

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)

class GaussianPolicy(nn.Module):
    def __init__(self, feature_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        nn.init.xavier_uniform_(self.mean.weight, gain=0.01)
        self.mean.bias.data.zero_()
        nn.init.xavier_uniform_(self.log_std.weight)
        self.log_std.bias.data.fill_(0.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), -20, 2)
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
    def __init__(self, feature_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight, gain=0.01)

    def forward(self, x, a):
        x = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# === 리플레이 버퍼 ===
class PrioritizedReplayBuffer:
    def __init__(self, max_size=200000, alpha=0.4, beta=0.6):
        self.buffer = deque(maxlen=max_size)
        self.priorities = deque(maxlen=max_size)
        self.alpha = alpha
        self.beta = beta
        self.max_priority = 1.0
        
    def add(self, transition, priority=None):
        if priority is None:
            priority = self.max_priority
        self.buffer.append(transition)
        self.priorities.append(priority ** self.alpha)

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
            
        priorities = np.array(self.priorities)
        probabilities = priorities / np.sum(priorities)
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities, replace=False)
        
        samples = [self.buffer[idx] for idx in indices]
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        
        s, a, r, s_prime, d = zip(*samples)
        return (np.array(s), np.array(a), np.array(r).reshape(-1, 1),
                np.array(s_prime), np.array(d).reshape(-1, 1)), indices, weights

    def update_priorities(self, indices, td_errors):
        for idx, error in zip(indices, td_errors):
            priority = min(abs(error) + 1e-5, 100.0)
            self.priorities[idx] = priority ** self.alpha
            self.max_priority = max(self.max_priority, priority)

    def size(self):
        return len(self.buffer)

# === SAC 에이전트 ===
class SACAgent:
    def __init__(self, input_shape, action_dim, action_bounds):
        # 기본 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = 0.99
        self.tau = 0.001
        self.batch_size = 128
        self.random_steps = 5000
        self.exploration_noise = 0.1
        self.noise_decay = 0.9999
        
        # 네트워크 초기화
        feature_dim = 128
        self.encoder = CNNEncoder(input_shape, feature_dim).to(self.device)
        self.actor = GaussianPolicy(feature_dim, action_dim).to(self.device)
        self.critic1 = QNetwork(feature_dim, action_dim).to(self.device)
        self.critic2 = QNetwork(feature_dim, action_dim).to(self.device)
        self.target_critic1 = QNetwork(feature_dim, action_dim).to(self.device)
        self.target_critic2 = QNetwork(feature_dim, action_dim).to(self.device)

        # 타겟 네트워크 초기화
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # 옵티마이저
        lr = 3e-4
        self.actor_lr = 1e-4
        self.critic_lr = 3e-4
        self.encoder_lr = 1e-4
        self.encoder_opt = optim.Adam(self.encoder.parameters(), lr=self.encoder_lr)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic1_opt = optim.Adam(self.critic1.parameters(), lr=self.critic_lr)
        self.critic2_opt = optim.Adam(self.critic2.parameters(), lr=self.critic_lr)  # 🔧 오타 수정

        # 자동 엔트로피 조정
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

        # 기본 변수
        self.replay_buffer = PrioritizedReplayBuffer()
        self.action_dim = action_dim
        self.action_bounds = action_bounds
        self.training_step = 0

    # 🔧 전처리 함수 추가
    def preprocess_obs(self, obs, is_batch=False):
        """MoraiSensor 출력 (120, 160, 1) → CNN 입력 tensor 변환"""
        if is_batch:
            # 배치: (B, 120, 160, 1) → (B, 1, 120, 160)
            return torch.FloatTensor(obs).permute(0, 3, 1, 2).to(self.device)
        else:
            # 단일: (120, 160, 1) → (1, 1, 120, 160)
            return torch.FloatTensor(obs).permute(2, 0, 1).unsqueeze(0).to(self.device)

    def get_action(self, obs, training=True):
        # 초기 랜덤 탐색
        if training and self.training_step < self.random_steps:
            action = np.random.uniform(-1, 1, size=self.action_dim)
            self.training_step += 1
            return self._scale_action(action)
        
        # 🔧 전처리 적용
        obs = self.preprocess_obs(obs)
        with torch.no_grad():
            feature = self.encoder(obs)
            if training:
                action, _ = self.actor.sample(feature)
                action = action.cpu().numpy()[0]
                
                # 탐색 노이즈 추가
                noise_scale = self.exploration_noise * (self.noise_decay ** self.training_step)
                noise = np.random.normal(0, noise_scale, size=self.action_dim)
                action = np.clip(action + noise, -1, 1)
                self.training_step += 1
            else:
                mean, _ = self.actor(feature)
                action = torch.tanh(mean).cpu().numpy()[0]
        
        return self._scale_action(action)

    def _scale_action(self, action):
        scaled = []
        for i in range(len(action)):
            low, high = self.action_bounds[i]
            scaled_val = (action[i] + 1) / 2 * (high - low) + low
            scaled.append(np.clip(scaled_val, low, high))
        return np.array(scaled, dtype=np.float32)

    def store(self, transition):
        self.replay_buffer.add(transition)

    def train(self):
        if self.replay_buffer.size() < 1000:
            return
            
        batch = self.replay_buffer.sample(self.batch_size)
        if batch is None:
            return
            
        (s, a, r, s_prime, d), indices, weights = batch
        weights = torch.FloatTensor(weights).to(self.device)

        # 🔧 배치 전처리 적용
        s = self.preprocess_obs(s, is_batch=True)
        s_prime = self.preprocess_obs(s_prime, is_batch=True)
        a = torch.FloatTensor(a).to(self.device)
        r = torch.FloatTensor(r).to(self.device)
        d = torch.FloatTensor(d).to(self.device)

        # Critic 업데이트
        z_critic = self.encoder(s).detach()
        z_prime = self.encoder(s_prime).detach()

        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(z_prime)
            target_q1 = self.target_critic1(z_prime, next_action)
            target_q2 = self.target_critic2(z_prime, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            y = r + self.gamma * (1 - d) * target_q

        q1 = self.critic1(z_critic, a)
        q2 = self.critic2(z_critic, a)
        
        # TD 오차 및 우선순위 업데이트
        td_errors = torch.abs(q1 - y).detach().cpu().numpy().flatten()
        self.replay_buffer.update_priorities(indices, td_errors)
        
        # Critic 손실
        critic1_loss = (F.mse_loss(q1, y, reduction='none') * weights.unsqueeze(1)).mean()
        critic2_loss = (F.mse_loss(q2, y, reduction='none') * weights.unsqueeze(1)).mean()

        self.critic1_opt.zero_grad()
        self.critic2_opt.zero_grad()
        critic1_loss.backward(retain_graph=True)
        critic2_loss.backward()
        self.critic1_opt.step()
        self.critic2_opt.step()

        # Actor 업데이트
        z_actor = self.encoder(s)
        new_action, log_prob = self.actor.sample(z_actor)
        q1_new = self.critic1(z_actor, new_action)
        q2_new = self.critic2(z_actor, new_action)
        
        actor_loss = (self.alpha * log_prob - torch.min(q1_new, q2_new)).mean()

        self.encoder_opt.zero_grad()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.encoder_opt.step()
        self.actor_opt.step()
        
        # 엔트로피 조정
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        # 타겟 네트워크 업데이트
        for param, target_param in zip(self.critic1.parameters(), self.target_critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save_model(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.encoder.state_dict(), os.path.join(save_path, "sac_encoder.pt"))
        torch.save(self.actor.state_dict(), os.path.join(save_path, "sac_actor.pt"))
        torch.save(self.critic1.state_dict(), os.path.join(save_path, "sac_critic1.pt"))
        torch.save(self.critic2.state_dict(), os.path.join(save_path, "sac_critic2.pt"))

    def load_model(self, load_path):
        self.encoder.load_state_dict(torch.load(os.path.join(load_path, "sac_encoder.pt")))
        self.actor.load_state_dict(torch.load(os.path.join(load_path, "sac_actor.pt")))
        self.critic1.load_state_dict(torch.load(os.path.join(load_path, "sac_critic1.pt")))
        self.critic2.load_state_dict(torch.load(os.path.join(load_path, "sac_critic2.pt")))