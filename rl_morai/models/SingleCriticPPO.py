import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from collections import deque
from torch.distributions import Normal

# TensorBoard 선택적 import
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    print("Warning: TensorBoard not available. Install with: pip install tensorboard")
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None

class RunningMeanStd:
    """보상 정규화를 위한 실행 평균/표준편차 계산"""
    def __init__(self, epsilon=1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = len(x)
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

class LaneFollowingCNN(nn.Module):
    """차선 검출을 위한 경량 CNN"""
    def __init__(self, input_shape=(120, 160), output_dim=256):
        super().__init__()
        
        # ROI 설정 (하단 70%만 사용)
        self.roi_crop_ratio = 0.3
        self.roi_height = int(input_shape[0] * (1 - self.roi_crop_ratio))
        
        # 백본 네트워크
        self.backbone = nn.Sequential(
            # 첫 번째 블록
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 두 번째 블록
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 세 번째 블록
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 글로벌 평균 풀링
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # 최종 특징 추출
        self.feature_extractor = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim),
            nn.ReLU(inplace=True)
        )

    def apply_roi(self, x):
        """ROI 적용 (상단 30% 제거)"""
        crop_height = int(x.size(2) * self.roi_crop_ratio)
        return x[:, :, crop_height:, :]

    def forward(self, x):
        # ROI 적용
        x = self.apply_roi(x)
        
        # 백본 통과
        x = self.backbone(x)
        
        # 평탄화 및 특징 추출 - reshape 사용으로 수정
        x = x.reshape(x.size(0), -1)  # reshape 대신 reshape 사용
        x = self.feature_extractor(x)
        
        return x

    """차선 검출을 위한 경량 CNN (출력 크기 자동 계산)"""
    def __init__(self, input_shape=(120, 160), output_dim=256):
        super().__init__()

        self.roi_crop_ratio = 0.3
        self.roi_height = int(input_shape[0] * (1 - self.roi_crop_ratio))

        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))  # 고정 출력 크기
        )

        # 실제 출력 크기 자동 계산
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_shape[0], input_shape[1])
            dummy = self.apply_roi(dummy)
            dummy = self.backbone(dummy)
            flattened_dim = dummy.reshape(1, -1).shape[1]

        self.feature_extractor = nn.Sequential(
            nn.Linear(flattened_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim),
            nn.ReLU(inplace=True)
        )

    def apply_roi(self, x):
        crop_height = int(x.size(2) * self.roi_crop_ratio)
        return x[:, :, crop_height:, :]

    def forward(self, x):
        x = self.apply_roi(x)
        x = self.backbone(x)
        x = x.reshape(x.size(0), -1)
        x = self.feature_extractor(x)
        return x

class PPOActor(nn.Module):
    """PPO Actor 네트워크"""
    def __init__(self, feature_dim, action_dim, hidden_dim=512):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # 초기화
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0)
        
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.constant_(self.mean_head.bias, 0)

    def forward(self, x):
        features = self.network(x)
        mean = self.mean_head(features)
        std = torch.exp(self.log_std.clamp(-20, 1))
        return mean, std

    def get_action_and_log_prob(self, x):
        mean, std = self.forward(x)
        dist = Normal(mean, std)
        raw_action = dist.sample()
        action = torch.tanh(raw_action)
        
        # log_prob 계산 (tanh 변환 고려)
        log_prob = dist.log_prob(raw_action).sum(dim=-1, keepdim=True)
        log_prob -= torch.log(1 - action.pow(2) + 1e-7).sum(dim=-1, keepdim=True)
        
        return action, log_prob

    def get_log_prob(self, x, action):
        mean, std = self.forward(x)
        dist = Normal(mean, std)
        
        # tanh 역변환
        raw_action = torch.atanh(torch.clamp(action, -0.999, 0.999))
        log_prob = dist.log_prob(raw_action).sum(dim=-1, keepdim=True)
        log_prob -= torch.log(1 - action.pow(2) + 1e-7).sum(dim=-1, keepdim=True)
        
        return log_prob

class PPOCritic(nn.Module):
    """PPO Critic 네트워크"""
    def __init__(self, feature_dim, hidden_dim=512):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 초기화
        for layer in self.network[:-1]:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0)
        
        # 마지막 레이어는 작은 가중치로 초기화
        nn.init.orthogonal_(self.network[-1].weight, gain=0.01)
        nn.init.constant_(self.network[-1].bias, 0)

    def forward(self, x):
        return self.network(x)

class PPOBuffer:
    """PPO 경험 버퍼"""
    def __init__(self, max_size=10000):
        self.max_size = max_size
        self.clear()

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.next_states = []
        self.current_episode_length = 0

    def add(self, state, action, reward, value, log_prob, done, next_state):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.next_states.append(next_state)
        self.current_episode_length += 1
        
        # 극단적으로 짧은 에피소드 감지 및 제거
        if done and self.current_episode_length <= 3:
            print(f"[BUFFER] 극단적으로 짧은 에피소드 ({self.current_episode_length}스텝) 제거")
            self.remove_last_episode()
            return True
        return False

    def remove_last_episode(self):
        episode_length = self.current_episode_length
        for _ in range(episode_length):
            if self.states:
                self.states.pop()
                self.actions.pop()
                self.rewards.pop()
                self.values.pop()
                self.log_probs.pop()
                self.dones.pop()
                self.next_states.pop()
        self.current_episode_length = 0

    def episode_reset(self):
        self.current_episode_length = 0

    def get(self):
        return (self.states, self.actions, self.rewards, 
                self.values, self.log_probs, self.dones, self.next_states)

    def size(self):
        return len(self.states)

class PPOAgent:
    """차선 주행을 위한 PPO 에이전트"""
    def __init__(self, input_shape, action_dim, action_bounds, log_dir=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        
        # 하이퍼파라미터 (차선 주행에 최적화)
        self.gamma = 0.995          # 할인율 (미래 보상 중시)
        self.lam = 0.95             # GAE lambda
        self.clip_epsilon = 0.1     # PPO 클리핑
        self.c1 = 0.4               # Value function 계수
        self.c2 = 0.05              # Entropy 계수
        self.ppo_epochs = 4         # PPO 업데이트 횟수
        self.mini_batch_size = 128  # 미니배치 크기
        self.max_grad_norm = 0.5    # Gradient clipping
        
        # 네트워크 초기화
        feature_dim = 256 + 2  # CNN features + velocity + steering
        self.encoder = LaneFollowingCNN(input_shape, 256).to(self.device)
        self.actor = PPOActor(feature_dim, action_dim).to(self.device)
        self.critic = PPOCritic(feature_dim).to(self.device)
        
        # 옵티마이저
        self.actor_lr = 5e-5
        self.critic_lr = 1e-4
        self.encoder_lr = 5e-5
        
        self.encoder_opt = optim.Adam(self.encoder.parameters(), lr=self.encoder_lr)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        
        # 기타 설정
        self.buffer = PPOBuffer()
        self.action_dim = action_dim
        self.action_bounds = action_bounds
        self.training_step = 0
        self.update_count = 0
        
        # 보상 정규화
        self.reward_rms = RunningMeanStd()
        self.normalize_rewards = True
        
        # TensorBoard
        if log_dir and TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(log_dir=log_dir)
            self.log_enabled = True
            print(f"TensorBoard logging: {log_dir}")
        else:
            self.writer = None
            self.log_enabled = False

    def preprocess_obs(self, obs, is_batch=False):
        """관측값 전처리"""
        if is_batch:
            images = np.array([obs_dict['image'] for obs_dict in obs])
            vectors = np.array([np.concatenate([obs_dict['velocity'], obs_dict['steering']]) 
                               for obs_dict in obs])
            
            images = torch.FloatTensor(images).permute(0, 3, 1, 2).to(self.device)
            vectors = torch.FloatTensor(vectors).to(self.device)
            
            image_features = self.encoder(images)
            combined_features = torch.cat([image_features, vectors], dim=1)
            return combined_features
        else:
            if isinstance(obs, dict):
                image = torch.FloatTensor(obs['image']).permute(2, 0, 1).unsqueeze(0).to(self.device)
                vector = torch.FloatTensor(np.concatenate([obs['velocity'], obs['steering']])).unsqueeze(0).to(self.device)
                
                image_features = self.encoder(image)
                combined_features = torch.cat([image_features, vector], dim=1)
                return combined_features

    def get_action(self, obs, training=True):
        feature = self.preprocess_obs(obs)
        
        with torch.no_grad():
            if training:
                action, log_prob = self.actor.get_action_and_log_prob(feature)
                value = self.critic(feature)
                action = action.cpu().numpy()[0]
                log_prob = log_prob.cpu().numpy()[0]
                value = value.cpu().numpy()[0]
            else:
                mean, _ = self.actor(feature)
                action = torch.tanh(mean).cpu().numpy()[0]
                log_prob = None
                value = None
        
        scaled_action = self._scale_action(action)
        self.training_step += 1
        
        return (scaled_action, log_prob, value) if training else scaled_action

    def _scale_action(self, action):
        """액션 스케일링"""
        scaled = []
        for i in range(len(action)):
            low, high = self.action_bounds[i]
            scaled_val = (action[i] + 1) / 2 * (high - low) + low
            scaled.append(np.clip(scaled_val, low, high))
        return np.array(scaled, dtype=np.float32)

    def _unscale_action(self, scaled_action):
        """액션 역스케일링"""
        unscaled = []
        for i in range(len(scaled_action)):
            low, high = self.action_bounds[i]
            unscaled_val = 2 * (scaled_action[i] - low) / (high - low) - 1
            unscaled.append(np.clip(unscaled_val, -1, 1))
        return np.array(unscaled, dtype=np.float32)

    def store(self, obs, action, reward, value, log_prob, done, next_obs):
        """경험 저장"""
        # 조기 종료 패널티
        if done and hasattr(self.buffer, 'current_episode_length'):
            episode_length = self.buffer.current_episode_length + 1
            if episode_length < 30:  # 30스텝 미만 조기 종료시 패널티
                penalty = -2.0 * (30 - episode_length) / 30
                reward += penalty
        
        # 보상 정규화
        if self.normalize_rewards:
            self.reward_rms.update([reward])
            normalized_reward = reward / max(np.sqrt(self.reward_rms.var + 1e-8), 1.0)
            normalized_reward = np.clip(normalized_reward, -10.0, 10.0)
        else:
            normalized_reward = reward
        
        unscaled_action = self._unscale_action(action)
        is_bad_episode = self.buffer.add(obs, unscaled_action, normalized_reward, 
                                       value, log_prob, done, next_obs)
        return is_bad_episode

    def compute_gae(self, rewards, values, dones, next_values):
        """GAE 계산"""
        advantages = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            next_value = next_values[i] if i < len(next_values) else 0
            if i == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[i]
                next_value = next_value
            else:
                next_non_terminal = 1.0 - dones[i]
                next_value = values[i + 1]
            
            delta = rewards[i] + self.gamma * next_value * next_non_terminal - values[i]
            gae = delta + self.gamma * self.lam * next_non_terminal * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages, returns

    def train(self):
        """PPO 학습"""
        if self.buffer.size() < self.mini_batch_size:
            return {}
        
        states, actions, rewards, values, old_log_probs, dones, next_states = self.buffer.get()
        
        # 다음 값 계산
        next_values = []
        for next_state in next_states:
            if next_state is not None:
                with torch.no_grad():
                    feature = self.preprocess_obs(next_state)
                    next_value = self.critic(feature).cpu().numpy()[0][0]
                next_values.append(next_value)
            else:
                next_values.append(0.0)
        
        # GAE 계산
        advantages, returns = self.compute_gae(rewards, values, dones, next_values)
        
        # 정규화
        advantages = np.array(advantages)
        returns = np.array(returns)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 텐서 변환
        with torch.no_grad():
            states_tensor = self.preprocess_obs(states, is_batch=True).detach()
        actions_tensor = torch.FloatTensor(actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device).unsqueeze(1)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device).unsqueeze(1)
        returns_tensor = torch.FloatTensor(returns).to(self.device).unsqueeze(1)
        
        # 학습 통계
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        batch_count = 0
        
        # PPO 업데이트
        for epoch in range(self.ppo_epochs):
            indices = torch.randperm(len(states))
            for start in range(0, len(states), self.mini_batch_size):
                end = min(start + self.mini_batch_size, len(states))
                mb_indices = indices[start:end]
                
                mb_states = states_tensor[mb_indices]
                mb_actions = actions_tensor[mb_indices]
                mb_old_log_probs = old_log_probs_tensor[mb_indices]
                mb_advantages = advantages_tensor[mb_indices]
                mb_returns = returns_tensor[mb_indices]
                
                # 현재 정책 평가
                new_log_probs = self.actor.get_log_prob(mb_states, mb_actions)
                new_values = self.critic(mb_states)
                
                # PPO 손실 계산
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Value 손실 (Huber Loss 사용)
                critic_loss = F.smooth_l1_loss(new_values, mb_returns)
                
                # Entropy 보너스
                mean, std = self.actor(mb_states)
                entropy = torch.distributions.Normal(mean, std).entropy().sum(dim=-1).mean()
                
                # 통계 수집
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()
                batch_count += 1
                
                # Actor 업데이트
                actor_total_loss = actor_loss - self.c2 * entropy
                self.encoder_opt.zero_grad()
                self.actor_opt.zero_grad()
                actor_total_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.encoder_opt.step()
                self.actor_opt.step()
                
                # Critic 업데이트
                critic_total_loss = self.c1 * critic_loss
                self.critic_opt.zero_grad()
                critic_total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_opt.step()
        
        # TensorBoard 로깅
        if self.log_enabled and batch_count > 0:
            metrics = {
                'actor_loss': total_actor_loss / batch_count,
                'critic_loss': total_critic_loss / batch_count,
                'entropy': total_entropy / batch_count
            }
            
            for key, value in metrics.items():
                self.writer.add_scalar(f'Loss/{key.title()}', value, self.update_count)
            
            self.writer.add_scalar('Policy/Advantages_Mean', advantages.mean(), self.update_count)
            self.writer.add_scalar('Policy/Returns_Mean', returns.mean(), self.update_count)
            self.writer.add_scalar('Training/Buffer_Size', self.buffer.size(), self.update_count)
        
        self.update_count += 1
        self.buffer.clear()
        
        return {'actor_loss': total_actor_loss / batch_count if batch_count > 0 else 0,
                'critic_loss': total_critic_loss / batch_count if batch_count > 0 else 0,
                'entropy': total_entropy / batch_count if batch_count > 0 else 0}

    def log_episode_metrics(self, episode, reward, length, total_steps):
        """에피소드 메트릭 로깅"""
        if self.log_enabled:
            self.writer.add_scalar('Episode/Reward', reward, episode)
            self.writer.add_scalar('Episode/Length', length, episode)
            self.writer.add_scalar('Episode/Total_Steps', total_steps, episode)

    def save_model(self, save_path):
        """모델 저장"""
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.encoder.state_dict(), os.path.join(save_path, "PPO_encoder.pt"))
        torch.save(self.actor.state_dict(), os.path.join(save_path, "PPO_actor.pt"))
        torch.save(self.critic.state_dict(), os.path.join(save_path, "PPO_critic.pt"))
        print(f"모델 저장 완료: {save_path}")

    def load_model(self, load_path):
        """모델 로드"""
        self.encoder.load_state_dict(torch.load(os.path.join(load_path, "PPO_encoder.pt"), weights_only=True))
        self.actor.load_state_dict(torch.load(os.path.join(load_path, "PPO_actor.pt"), weights_only=True))
        self.critic.load_state_dict(torch.load(os.path.join(load_path, "PPO_critic.pt"), weights_only=True))
        print(f"모델 로드 완료: {load_path}")

    def close(self):
        """리소스 정리"""
        if self.writer:
            self.writer.close()