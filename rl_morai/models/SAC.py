import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from collections import deque

class JetsonNanoLaneCNN(nn.Module):
    """Jetson Nano 최적화 초경량 차선 검출 CNN"""
    def __init__(self, input_shape=(120, 160), output_dim=128, roi_crop_ratio=0.3):
        super().__init__()
        
        self.roi_crop_ratio = roi_crop_ratio
        self.roi_height = int(input_shape[0] * (1 - roi_crop_ratio))  # 84
        self.roi_width = input_shape[1]  # 160
        
        # 극도로 경량화된 백본 (MobileNet 스타일)
        self.efficient_backbone = nn.Sequential(
            # 1단계: 최소한의 특징 추출
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # 84x160 → 42x80
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            
            # 2단계: Depthwise Separable Conv (MobileNet 핵심)
            self._depthwise_separable_conv(16, 32, stride=2),  # 42x80 → 21x40
            self._depthwise_separable_conv(32, 32, stride=1),  # 크기 유지
            
            # 3단계: 최종 특징 (채널 수 최소화)
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),  # 21x40 → 11x20
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32)
        )
        
        # 단일 경량 어텐션 (가장 중요한 것만)
        self.lane_attention = nn.Sequential(
            # 수직 차선 검출에만 집중
            nn.Conv2d(32, 8, kernel_size=(5, 1), padding=(2, 0)),  # 세로 필터
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 글로벌 평균 풀링 (FC 레이어 최소화)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 최소한의 출력 레이어
        self.output_fc = nn.Sequential(
            nn.Linear(32, output_dim),
            nn.ReLU(inplace=True)
        )

    def _depthwise_separable_conv(self, in_channels, out_channels, stride=1):
        """MobileNet의 핵심: Depthwise Separable Convolution"""
        return nn.Sequential(
            # Depthwise: 각 채널별로 별도 컨볼루션
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, 
                     padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            
            # Pointwise: 1x1 컨볼루션으로 채널 믹싱
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def apply_roi(self, x):
        """ROI 적용 (상단 30% 제거)"""
        crop_height = int(x.size(2) * self.roi_crop_ratio)
        return x[:, :, crop_height:, :]

    def forward(self, x):
        # 1. ROI 적용
        roi_x = self.apply_roi(x)  # (B, 1, 84, 160)
        
        # 2. 경량 백본으로 특징 추출
        features = self.efficient_backbone(roi_x)  # (B, 32, 11, 20)
        
        # 3. 간단한 어텐션 적용
        attention = self.lane_attention(features)  # (B, 1, 11, 20)
        attended_features = features * attention
        
        # 4. 글로벌 풀링 및 출력
        pooled = self.global_pool(attended_features)  # (B, 32, 1, 1)
        flattened = pooled.view(pooled.size(0), -1)   # (B, 32)
        output = self.output_fc(flattened)            # (B, output_dim)
        
        return output

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
    def __init__(self, max_size=200000, alpha=0.6, beta=0.4):
        self.buffer = deque(maxlen=max_size)
        self.priorities = deque(maxlen=max_size)
        self.alpha = alpha
        self.beta = beta
        self.max_priority = 1.0
        self.episode_boundaries = [] #episode 길이 추적 (추가)
        
    def add(self, transition, priority=None):
        if priority is None:
            priority = self.max_priority

        self.buffer.append(transition)
        self.priorities.append(priority ** self.alpha) #짧은 에피소드도 같은 우선순위로 학습하게 됨


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

    def clear_bad_patterns(self, threshold_reward=-1.0): #낮은 보상 경험 제거 (추가)
        new_buffer = deque(maxlen=self.buffer.maxlen)
        new_priorities = deque(maxlen=self.priorities.maxlen)
        
        for i, transition in enumerate(self.buffer):
            if len(transition) >= 3 and transition[2] > threshold_reward:  # reward > threshold
                new_buffer.append(transition)
                new_priorities.append(self.priorities[i])
        
        self.buffer = new_buffer
        self.priorities = new_priorities

    def size(self):
        return len(self.buffer)

# === SAC 에이전트 ===
class SACAgent:
    def __init__(self, input_shape, action_dim, action_bounds):
        # 기본 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = 0.995
        self.tau = 0.01
        self.batch_size = 128
        self.random_steps = 5000
        
        self.current_episode_steps = 0
        self.total_episodes = 0  
        self.short_episode_count = 0

        # 네트워크 초기화
        feature_dim = 128
        self.encoder = JetsonNanoLaneCNN(input_shape, feature_dim).to(self.device)
        self.actor = GaussianPolicy(feature_dim + 2, action_dim).to(self.device)
        self.critic1 = QNetwork(feature_dim + 2, action_dim).to(self.device)
        self.critic2 = QNetwork(feature_dim + 2, action_dim).to(self.device)
        self.target_critic1 = QNetwork(feature_dim + 2, action_dim).to(self.device)
        self.target_critic2 = QNetwork(feature_dim + 2, action_dim).to(self.device)

        # 타겟 네트워크 초기화
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # 옵티마이저
        lr = 3e-4
        self.actor_lr = 1e-5
        self.critic_lr = 3e-5
        self.encoder_lr = 1e-5
        self.encoder_opt = optim.Adam(self.encoder.parameters(), lr=self.encoder_lr)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic1_opt = optim.Adam(self.critic1.parameters(), lr=self.critic_lr)
        self.critic2_opt = optim.Adam(self.critic2.parameters(), lr=self.critic_lr)  # 🔧 오타 수정

        # 자동 엔트로피 조정
        #self.target_entropy = -action_dim
        self.target_entropy = -np.prod(action_dim).item() #엔트로피를 높게하여 다양한 행동 시도
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

        # 기본 변수
        self.replay_buffer = PrioritizedReplayBuffer()
        self.action_dim = action_dim
        self.action_bounds = action_bounds
        self.training_step = 0

    # 전처리 함수 추가
    def preprocess_obs(self, obs, is_batch=False):
        """복합 observation 처리: 이미지 + velocity + steering"""
        if is_batch:
            # 배치 처리: obs는 obs_dict들의 리스트
            images = []
            vectors = []
            for obs_dict in obs:
                images.append(obs_dict['image'])
                vectors.append(np.concatenate([obs_dict['velocity'], obs_dict['steering']]))
            
            # 텐서 변환
            images_np = np.array(images)
            images = torch.FloatTensor(images_np).permute(0, 3, 1, 2).to(self.device)
            vectors = torch.FloatTensor(vectors).to(self.device)
            
            # CNN으로 이미지 특징 추출
            image_features = self.encoder(images)
            
            # 이미지 특징 + 벡터 정보 결합
            combined_features = torch.cat([image_features, vectors], dim=1)
            return combined_features
        else:
            # 단일 observation 처리
            if isinstance(obs, dict):
                # 복합 observation
                image = torch.FloatTensor(obs['image']).permute(2, 0, 1).unsqueeze(0).to(self.device)
                vector = torch.FloatTensor(np.concatenate([obs['velocity'], obs['steering']])).unsqueeze(0).to(self.device)
                
                # CNN으로 이미지 특징 추출
                image_features = self.encoder(image)
                
                # 이미지 특징 + 벡터 정보 결합
                combined_features = torch.cat([image_features, vector], dim=1)
                return combined_features
            else:
                # 단순 이미지 (하위 호환성)
                image = torch.FloatTensor(obs).permute(2, 0, 1).unsqueeze(0).to(self.device)
                return self.encoder(image)

    def get_action(self, obs, training=True):
        # 초기 랜덤 탐색
        if training and self.training_step < self.random_steps:
            action = np.random.uniform(-1, 1, size=self.action_dim)
            self.training_step += 1
            return self._scale_action(action)
        
        # 🔧 전처리 적용
        feature = self.preprocess_obs(obs)
        with torch.no_grad():
            if training:
                action, _ = self.actor.sample(feature)
                action = action.cpu().numpy()[0]
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
        #self.replay_buffer.add(transition)
        obs, action, reward, next_obs, done = transition
        
        self.current_episode_steps += 1

        if done:
            self.total_episodes += 1
            
            # 짧은 에피소드 감지
            if self.current_episode_steps < 25:
                reduced_priority = self.replay_buffer.max_priority * 0.3
                self.replay_buffer.add(transition, priority=reduced_priority)
                
                self.short_episode_count += 1
            else:
                self.replay_buffer.add(transition)
            
            self.current_episode_steps = 0
        else:
            self.replay_buffer.add(transition)

    def train(self):
        if self.replay_buffer.size() < 1000:
            return
            
        batch = self.replay_buffer.sample(self.batch_size)
        if batch is None:
            return
            
        (s, a, r, s_prime, d), indices, weights = batch
        weights = torch.FloatTensor(weights).to(self.device)

        # 배치 전처리 적용
        s_features = self.preprocess_obs(s, is_batch=True)
        s_prime_features = self.preprocess_obs(s_prime, is_batch=True)
        a = torch.FloatTensor(a).to(self.device)
        r = torch.FloatTensor(r).to(self.device)
        d = torch.FloatTensor(d).to(self.device)

        # Critic 업데이트
        z_critic = s_features.detach()
        z_prime = s_prime_features.detach()

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
        z_actor = s_features
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

        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0) #Gradient Clinpping 추가
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)

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
        #self.encoder.load_state_dict(torch.load(os.path.join(load_path, "sac_encoder.pt")))
        # self.actor.load_state_dict(torch.load(os.path.join(load_path, "sac_actor.pt")))
        # self.critic1.load_state_dict(torch.load(os.path.join(load_path, "sac_critic1.pt")))
        # self.critic2.load_state_dict(torch.load(os.path.join(load_path, "sac_critic2.pt")))
        self.encoder.load_state_dict(torch.load(os.path.join(load_path, "sac_encoder.pt"), weights_only=True))
        self.actor.load_state_dict(torch.load(os.path.join(load_path, "sac_actor.pt"), weights_only=True))
        self.critic1.load_state_dict(torch.load(os.path.join(load_path, "sac_critic1.pt"), weights_only=True))
        self.critic2.load_state_dict(torch.load(os.path.join(load_path, "sac_critic2.pt"), weights_only=True))