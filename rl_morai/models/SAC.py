import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque

# === Feature Extractor ===
class CNNEncoder(nn.Module):
    def __init__(self, input_shape, output_dim):
        super().__init__()
        # 더 큰 커널 사이즈로 시작하여 더 넓은 영역을 보도록 수정
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=5, stride=2, padding=2),  # 더 큰 커널
            nn.ReLU(),
            nn.BatchNorm2d(32),  # 배치 정규화 추가
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),  # 배치 정규화 추가
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.flatten_dim = self._get_flatten_dim(input_shape)
        self.fc = nn.Linear(self.flatten_dim, output_dim)

    def _get_flatten_dim(self, input_shape):
        with torch.no_grad():
            x = torch.zeros(1, *input_shape)
            x = self.conv(x)
            return int(np.prod(x.size()))

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)

# === Policy ===
class GaussianPolicy(nn.Module):
    def __init__(self, feature_dim, action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # 초기 파라미터 설정
        # 평균값 초기화: 조향은 0 근처, 속도는 중간값으로
        nn.init.xavier_uniform_(self.mean.weight, gain=0.01)
        self.mean.bias.data.zero_()  # 평균값을 0으로 초기화
        
        # 초기에 높은 표준편차 설정 (충분한 탐색 보장)
        nn.init.xavier_uniform_(self.log_std.weight)
        self.log_std.bias.data.fill_(0.0)  # log(1.0) = 0, 초기 std = 1.0

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

# === Critic ===
class QNetwork(nn.Module):
    def __init__(self, feature_dim, action_dim, hidden_dim=256):
        super().__init__()
        # 더 깊은 Q-네트워크
        self.fc1 = nn.Linear(feature_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        # 파라미터 초기화
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight, gain=0.01)

    def forward(self, x, a):
        x = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# === Replay Buffer ===
class PrioritizedReplayBuffer:
    def __init__(self, max_size=100000, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.buffer = deque(maxlen=max_size)
        self.priorities = deque(maxlen=max_size)
        self.alpha = alpha  # 우선순위 지수
        self.beta = beta    # 중요도 샘플링 지수
        self.beta_increment = beta_increment  # 베타 증가율
        self.max_priority = 1.0
        
    def add(self, transition, priority=None):
        if priority is None:
            priority = self.max_priority
        self.buffer.append(transition)
        self.priorities.append(priority ** self.alpha)

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
            
        # 중요도 가중치 계산
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # 우선순위에 따른 확률 계산
        priorities = np.array(self.priorities)
        probabilities = priorities / np.sum(priorities)
        
        # 인덱스 샘플링
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities, replace=False)
        
        # 샘플 및 가중치 계산
        samples = [self.buffer[idx] for idx in indices]
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # 정규화
        
        s, a, r, s_prime, d = zip(*samples)
        
        samples_tuple = (
            np.array(s), 
            np.array(a), 
            np.array(r).reshape(-1, 1),
            np.array(s_prime), 
            np.array(d).reshape(-1, 1)
        )
        
        return samples_tuple, indices, weights

    def update_priorities(self, indices, td_errors):
        for idx, error in zip(indices, td_errors):
            priority = min(abs(error) + 1e-5, 100.0)  # 오류가 큰 경험에 높은 우선순위
            self.priorities[idx] = priority ** self.alpha
            self.max_priority = max(self.max_priority, priority)

    def size(self):
        return len(self.buffer)

# === SAC Agent ===
class SACAgent:
    def __init__(self, input_shape, action_dim, action_bounds, 
                 feature_dim=128, gamma=0.99, tau=0.005, alpha=0.2, 
                 actor_lr=3e-4, critic_lr=3e-4, buffer_size=100000,
                 min_buffer_size=1000, batch_size=64,
                 auto_entropy_tuning=True):
                 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Device: {self.device}")

        # 네트워크 초기화
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
        self.encoder_opt = optim.Adam(self.encoder.parameters(), lr=actor_lr)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_opt = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_opt = optim.Adam(self.critic2.parameters(), lr=critic_lr)

        # 자동 엔트로피 조정
        self.auto_entropy_tuning = auto_entropy_tuning
        if auto_entropy_tuning:
            self.target_entropy = -action_dim  # 목표 엔트로피
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=actor_lr)
        else:
            self.alpha = alpha

        # 경험 리플레이 버퍼 및 하이퍼파라미터
        self.replay_buffer = PrioritizedReplayBuffer(max_size=buffer_size)
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim
        self.action_bounds = action_bounds
        self.min_buffer_size = min_buffer_size
        self.batch_size = batch_size
        
        # 학습 관련 변수
        self.current_episode = 0
        self.training_step = 0
        self.random_steps = 5000  # 처음 5000 스텝은 완전 랜덤 행동
        
        # 액션 노이즈 관련 파라미터
        self.exploration_noise = 0.4  # 탐색용 노이즈 스케일
        self.noise_decay = 0.9999    # 스텝마다 감소 비율

    def start_new_episode(self):
        """새 에피소드 시작 시 호출"""
        self.current_episode += 1
        print(f"[INFO] Starting episode {self.current_episode}")
        
    def get_action(self, obs, training=True):
        """현재 상태에서 액션 샘플링"""
        if training and (self.training_step < self.random_steps or 
                        self.replay_buffer.size() < self.min_buffer_size):
            # 초기 완전 랜덤 행동
            action = np.random.uniform(-1, 1, size=self.action_dim)
            self.training_step += 1
            return self._scale_action(action)
        
        # 정책에서 액션 샘플링
        obs = torch.FloatTensor(obs).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
        with torch.no_grad():
            feature = self.encoder(obs)
            if training:
                action, _ = self.actor.sample(feature)
                action = action.cpu().numpy()[0]
                
                # 탐색 노이즈 추가 (학습 중일 때만)
                noise_scale = self.exploration_noise * (self.noise_decay ** self.training_step)
                noise = np.random.normal(0, noise_scale, size=self.action_dim)
                action = np.clip(action + noise, -1, 1)
                
                self.training_step += 1
            else:
                # 평가 시에는 결정론적 액션
                mean, _ = self.actor(feature)
                action = torch.tanh(mean).cpu().numpy()[0]
        
        return self._scale_action(action)
    
    def get_deterministic_action(self, obs):
        """결정론적 액션 (평가용)"""
        return self.get_action(obs, training=False)

    def _scale_action(self, action):
        """액션 스케일링 (-1,1 범위에서 실제 범위로)"""
        scaled = []
        for i in range(len(action)):
            low, high = self.action_bounds[i]
            scaled_val = (action[i] + 1) / 2 * (high - low) + low
            scaled.append(np.clip(scaled_val, low, high))
        return np.array(scaled, dtype=np.float32)

    def preprocess_image(self, image):
        """이미지 전처리"""
        if image is None:
            return None
            
        image = image.astype(np.float32) / 255.0
        if image.ndim == 2:
            image = image[:, :, None]
        return image  # HWC 유지

    def store(self, transition, reward=None):
        """경험 저장"""
        priority = None
        if reward is not None:
            # 보상 기반 우선순위: 높은 절대값 보상에 높은 우선순위
            priority = abs(reward) + 0.01
            
        self.replay_buffer.add(transition, priority)

    def train(self, min_buffer_size=None):
        """SAC 알고리즘 학습 단계"""
        if min_buffer_size is None:
            min_buffer_size = self.min_buffer_size
            
        if self.replay_buffer.size() < min_buffer_size:
            return {'critic_loss': 0, 'actor_loss': 0, 'alpha': self.alpha.item() if self.auto_entropy_tuning else self.alpha}

        # 배치 샘플링
        batch = self.replay_buffer.sample(self.batch_size)
        if batch is None:
            return {'critic_loss': 0, 'actor_loss': 0, 'alpha': self.alpha.item() if self.auto_entropy_tuning else self.alpha}
            
        (s, a, r, s_prime, d), indices, weights = batch
        weights = torch.FloatTensor(weights).to(self.device)

        s = torch.FloatTensor(s).permute(0, 3, 1, 2).to(self.device)
        s_prime = torch.FloatTensor(s_prime).permute(0, 3, 1, 2).to(self.device)
        a = torch.FloatTensor(a).to(self.device)
        r = torch.FloatTensor(r).to(self.device)
        d = torch.FloatTensor(d).to(self.device)

        # Critic update
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
        
        # TD 오차 계산 및 우선순위 업데이트
        td_error1 = torch.abs(q1 - y).detach().cpu().numpy()
        td_error2 = torch.abs(q2 - y).detach().cpu().numpy()
        td_errors = np.mean([td_error1, td_error2], axis=0).flatten()
        self.replay_buffer.update_priorities(indices, td_errors)
        
        # Critic 손실 계산 (중요도 가중치 적용)
        critic1_loss = F.mse_loss(q1, y, reduction='none') * weights.unsqueeze(1)
        critic2_loss = F.mse_loss(q2, y, reduction='none') * weights.unsqueeze(1)
        critic1_loss = critic1_loss.mean()
        critic2_loss = critic2_loss.mean()

        self.critic1_opt.zero_grad()
        self.critic2_opt.zero_grad()
        critic1_loss.backward(retain_graph=True)
        critic2_loss.backward()
        self.critic1_opt.step()
        self.critic2_opt.step()

        # Actor update
        z_actor = self.encoder(s)
        new_action, log_prob = self.actor.sample(z_actor)
        q1_new = self.critic1(z_actor, new_action)
        q2_new = self.critic2(z_actor, new_action)
        
        # 액터 손실 계산
        actor_loss = (self.alpha * log_prob - torch.min(q1_new, q2_new)).mean()

        self.encoder_opt.zero_grad()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.encoder_opt.step()
        self.actor_opt.step()
        
        # 자동 엔트로피 조정
        if self.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()

        # 타겟 네트워크 소프트 업데이트
        for param, target_param in zip(self.critic1.parameters(), self.target_critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        return {
            'critic_loss': (critic1_loss.item() + critic2_loss.item()) / 2,
            'actor_loss': actor_loss.item(),
            'alpha': self.alpha.item() if self.auto_entropy_tuning else self.alpha,
            'avg_q_value': torch.min(q1, q2).mean().item()
        }