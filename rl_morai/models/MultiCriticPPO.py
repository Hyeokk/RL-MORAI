import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import cv2
import os
from collections import deque

# TensorBoard 선택적 import
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    print("Warning: TensorBoard not available. Install with: pip install tensorboard")
    TENSORBOARD_AVAILABLE = False

class SpatialAttention(nn.Module):
    """Spatial Attention Module for vision tasks"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_attention = torch.cat([avg_out, max_out], dim=1)
        x_attention = self.conv1(x_attention)
        return self.sigmoid(x_attention)

class ChannelAttention(nn.Module):
    """Channel Attention Module"""
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class CNNEncoder(nn.Module):
    """Lane Following CNN with Vision Attention (기존 LaneFollowingCNN 장점 통합)"""
    def __init__(self, input_channels=1, feature_dim=512):
        super(CNNEncoder, self).__init__()
        
        # ROI 설정 (하단 70%만 사용 - 차선에 집중)
        self.roi_crop_ratio = 0.3
        
        # 백본 네트워크 (기존 LaneFollowingCNN 구조 + Attention)
        self.backbone = nn.Sequential(
            # 첫 번째 블록
            nn.Conv2d(input_channels, 32, kernel_size=5, stride=2, padding=2),
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
            
            # 네 번째 블록
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 글로벌 적응형 풀링
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Attention 모듈들
        self.channel_attention1 = ChannelAttention(64)
        self.spatial_attention1 = SpatialAttention()
        self.channel_attention2 = ChannelAttention(128)
        self.spatial_attention2 = SpatialAttention()
        self.channel_attention3 = ChannelAttention(256)
        self.spatial_attention3 = SpatialAttention()
        
        # 특징 추출기
        self.feature_extractor = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # 초기화
        self._initialize_weights()

    def _initialize_weights(self):
        """가중치 초기화"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)

    def apply_roi(self, x):
        """ROI 적용 (상단 30% 제거 - 차선에 집중)"""
        crop_height = int(x.size(2) * self.roi_crop_ratio)
        return x[:, :, crop_height:, :]

    def forward(self, x):
        # ROI 적용
        x = self.apply_roi(x)
        
        # 첫 번째 conv 블록
        x = self.backbone[0:3](x)  # conv1 + bn + relu
        
        # 두 번째 conv 블록 + attention
        x = self.backbone[3:6](x)  # conv2 + bn + relu
        x = self.channel_attention1(x) * x
        x = self.spatial_attention1(x) * x
        
        # 세 번째 conv 블록 + attention  
        x = self.backbone[6:9](x)  # conv3 + bn + relu
        x = self.channel_attention2(x) * x
        x = self.spatial_attention2(x) * x
        
        # 네 번째 conv 블록 + attention
        x = self.backbone[9:12](x)  # conv4 + bn + relu
        x = self.channel_attention3(x) * x
        x = self.spatial_attention3(x) * x
        
        # 글로벌 풀링
        x = self.backbone[12](x)  # AdaptiveAvgPool2d
        
        # 평탄화 및 특징 추출
        x = x.view(x.size(0), -1)
        x = self.feature_extractor(x)
        
        return x

class EnvironmentClassifier(nn.Module):
    """Environment classifier to distinguish between solid, dashed, and shadow lanes"""
    def __init__(self, feature_dim=512, num_environments=3):
        super(EnvironmentClassifier, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_environments),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, features):
        return self.classifier(features)

class Actor(nn.Module):
    """Shared Actor network for steering control (PPO 스타일 개선)"""
    def __init__(self, feature_dim=512, action_dim=1, max_action=1.0, hidden_dim=512):
        super(Actor, self).__init__()
        
        self.max_action = max_action
        
        # 기존 PPOActor의 안정성 향상된 구조 사용
        self.network = nn.Sequential(
            nn.Linear(feature_dim + 2, hidden_dim),  # velocity, steering 추가
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # 초기화 (PPO 스타일)
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0)
        
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.constant_(self.mean_head.bias, 0)
        
    def forward(self, features, velocity=None, steering=None):
        # 속도와 조향각 정보 추가
        if velocity is not None and steering is not None:
            additional_features = torch.cat([velocity, steering], dim=-1)
            combined_features = torch.cat([features, additional_features], dim=-1)
        else:
            # 기본값 사용
            batch_size = features.size(0)
            default_additional = torch.zeros(batch_size, 2, device=features.device)
            combined_features = torch.cat([features, default_additional], dim=-1)
        
        x = self.network(combined_features)
        mean = self.mean_head(x)
        std = torch.exp(self.log_std.clamp(-20, 2))
        
        return mean, std
    
    def get_action(self, features, velocity=None, steering=None):
        mean, std = self.forward(features, velocity, steering)
        normal = Normal(mean, std)
        raw_action = normal.sample()
        action = torch.tanh(raw_action) * self.max_action
        
        # log_prob 계산 (tanh 변환 고려)
        log_prob = normal.log_prob(raw_action) - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob
    
    def get_log_prob(self, features, action, velocity=None, steering=None):
        mean, std = self.forward(features, velocity, steering)
        normal = Normal(mean, std)
        
        # tanh 역변환
        raw_action = torch.atanh(torch.clamp(action / self.max_action, -0.999, 0.999))
        log_prob = normal.log_prob(raw_action) - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return log_prob

class Critic(nn.Module):
    """Environment-specific Critic network (PPO 스타일 개선)"""
    def __init__(self, feature_dim=512, hidden_dim=512):
        super(Critic, self).__init__()
        
        # 기존 PPOCritic의 안정성 향상된 구조 사용
        self.network = nn.Sequential(
            nn.Linear(feature_dim + 2, hidden_dim),  # velocity, steering 추가
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 초기화 (PPO 스타일)
        for layer in self.network[:-1]:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0)
        
        # 마지막 레이어는 작은 가중치로 초기화
        nn.init.orthogonal_(self.network[-1].weight, gain=0.01)
        nn.init.constant_(self.network[-1].bias, 0)
        
    def forward(self, features, velocity=None, steering=None):
        # 속도와 조향각 정보 추가
        if velocity is not None and steering is not None:
            additional_features = torch.cat([velocity, steering], dim=-1)
            combined_features = torch.cat([features, additional_features], dim=-1)
        else:
            # 기본값 사용
            batch_size = features.size(0)
            default_additional = torch.zeros(batch_size, 2, device=features.device)
            combined_features = torch.cat([features, default_additional], dim=-1)
            
        return self.network(combined_features)

class MultiCriticPPO(nn.Module):
    """Multi-Critic PPO with integrated buffer and environment detection"""
    def __init__(self, 
                 input_channels=1,
                 feature_dim=512, 
                 action_dim=1, 
                 num_environments=3,
                 max_action=0.7,  # 조향각 범위
                 action_bounds=[(-0.4, 0.4), (15.0, 25.0)],
                 # 하이퍼파라미터들을 클래스 내부에서 관리
                 lr=3e-4,
                 clip_ratio=0.2,
                 value_loss_coef=0.25,
                 entropy_coef=0.02,
                 classifier_loss_coef=0.3,
                 ppo_epochs=4,
                 mini_batch_size=128,
                 max_grad_norm=0.5,
                 update_interval=2048,
                 normalize_rewards=True,
                 # 학습 파라미터
                 max_steps_per_episode=2000,
                 min_episode_steps=10,
                 num_episodes=5000,
                 # 로깅 파라미터
                 log_interval=50,
                 save_interval=500,
                 eval_interval=200):
        super(MultiCriticPPO, self).__init__()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # =============================================================================
        # 하이퍼파라미터 설정 (모든 학습 관련 파라미터를 여기서 관리)
        # =============================================================================
        
        # 모델 파라미터
        self.num_environments = num_environments
        self.action_bounds = action_bounds
        self.max_action = max_action
        
        # PPO 하이퍼파라미터
        self.clip_ratio = clip_ratio
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.classifier_loss_coef = classifier_loss_coef
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.max_grad_norm = max_grad_norm
        self.normalize_rewards = normalize_rewards
        
        # 학습 스케줄 파라미터
        self.update_interval = update_interval
        self.max_steps_per_episode = max_steps_per_episode
        self.min_episode_steps = min_episode_steps
        self.num_episodes = num_episodes
        
        # 로깅 파라미터
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.eval_interval = eval_interval
        
        # 학습률 설정 (컴포넌트별 분리)
        self.encoder_lr = lr * 0.5
        self.actor_lr = lr
        self.critic_lr = lr * 0.5
        
        # =============================================================================
        # 네트워크 구성
        # =============================================================================
        
        # 환경 감지기
        self.env_detector = LaneEnvironmentDetector()
        
        # 네트워크 구성
        self.encoder = CNNEncoder(input_channels, feature_dim)
        self.classifier = EnvironmentClassifier(feature_dim, num_environments)
        self.actor = Actor(feature_dim, action_dim, max_action)
        
        # 환경별 Critic들
        self.critics = nn.ModuleList([
            Critic(feature_dim) for _ in range(num_environments)
        ])
        
        # 옵티마이저 (기존 PPO 스타일로 분리)
        self.encoder_opt = optim.Adam(self.encoder.parameters(), lr=self.encoder_lr)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_opt = optim.Adam(
            list(self.critics.parameters()) + list(self.classifier.parameters()), 
            lr=self.critic_lr
        )
        
        # 경험 버퍼
        self.buffer = MultiCriticBuffer()
        
        # 보상 정규화
        self.reward_rms = RunningMeanStd()
        
        # 환경 라벨 매핑
        self.env_labels = {
            'solid': 0,
            'dashed': 1,  
            'shadow': 2
        }
        
        # 학습 통계
        self.training_step = 0
        self.update_count = 0
        self.value_estimation_errors = {i: deque(maxlen=100) for i in range(num_environments)}
        
        print(f"MultiCriticPPO initialized on {self.device}")
        print(f"하이퍼파라미터:")
        print(f"  - Learning Rate: {lr}, Update Interval: {update_interval}")
        print(f"  - Episodes: {num_episodes}, Max Steps: {max_steps_per_episode}")
        print(f"  - PPO Epochs: {ppo_epochs}, Batch Size: {mini_batch_size}")
    
    def get_hyperparameters(self):
        """하이퍼파라미터 딕셔너리 반환 (TensorBoard 로깅용)"""
        return {
            'learning_rate': self.actor_lr,
            'feature_dim': self.encoder.feature_extractor[-2].in_features,
            'clip_ratio': self.clip_ratio,
            'value_loss_coef': self.value_loss_coef,
            'entropy_coef': self.entropy_coef,
            'classifier_loss_coef': self.classifier_loss_coef,
            'ppo_epochs': self.ppo_epochs,
            'mini_batch_size': self.mini_batch_size,
            'update_interval': self.update_interval,
            'max_steps_per_episode': self.max_steps_per_episode,
            'num_episodes': self.num_episodes
        }

    def preprocess_obs(self, obs_dict, is_batch=False):
        """MORAI 관측값 전처리"""
        if is_batch:
            images = np.array([obs['image'] for obs in obs_dict])
            velocities = np.array([obs['velocity'] for obs in obs_dict])
            steerings = np.array([obs['steering'] for obs in obs_dict])
            
            # 이미지 텐서 변환 (H,W,C) -> (C,H,W)
            images = torch.FloatTensor(images).permute(0, 3, 1, 2).to(self.device)
            velocities = torch.FloatTensor(velocities).to(self.device)
            steerings = torch.FloatTensor(steerings).to(self.device)
            
            image_features = self.encoder(images)
            return image_features, velocities, steerings
        else:
            image = torch.FloatTensor(obs_dict['image']).permute(2, 0, 1).unsqueeze(0).to(self.device)
            velocity = torch.FloatTensor(obs_dict['velocity']).unsqueeze(0).to(self.device)
            steering = torch.FloatTensor(obs_dict['steering']).unsqueeze(0).to(self.device)
            
            image_features = self.encoder(image)
            return image_features, velocity, steering

    def detect_environment(self, obs_dict, sensor=None):
        """환경 자동 감지"""
        return self.env_detector.detect_lane_environment(obs_dict, sensor)

    def forward(self, obs_dict, environment_label=None, use_classifier=False, training=True):
        """
        Forward pass with MORAI observation dict
        """
        # 관측값 전처리
        features, velocity, steering = self.preprocess_obs(obs_dict)
        
        # 환경 분류기 예측
        env_probs = self.classifier(features)
        
        # 환경 결정
        if environment_label is not None:
            env_idx = torch.LongTensor([environment_label]).to(self.device)
        elif use_classifier:
            env_idx = torch.argmax(env_probs, dim=-1)
        else:
            # 자동 감지 사용
            detected_env = self.detect_environment(obs_dict)
            env_idx = torch.LongTensor([detected_env]).to(self.device)
        
        # 액션 생성
        if training:
            action, log_prob = self.actor.get_action(features, velocity, steering)
        else:
            mean, _ = self.actor(features, velocity, steering)
            action = torch.tanh(mean)
            log_prob = None
        
        # 해당 환경의 Critic으로 가치 추정
        env_id = env_idx.item()
        value = self.critics[env_id](features, velocity, steering)
        
        return action, log_prob, value, env_probs, features, env_idx

    def get_action(self, obs_dict, training=True, sensor=None):
        """액션 선택 (기존 PPO 스타일)"""
        with torch.no_grad():
            if training:
                # 자동 환경 감지 사용
                action, log_prob, value, env_probs, features, env_idx = self.forward(
                    obs_dict, training=True
                )
                
                # 스케일링된 액션 반환
                scaled_action = self._scale_action(action.cpu().numpy()[0])
                
                return (scaled_action, 
                       log_prob.cpu().numpy()[0] if log_prob is not None else None,
                       value.cpu().numpy()[0] if value is not None else None,
                       env_idx.cpu().numpy()[0])
            else:
                action, _, _, _, _, env_idx = self.forward(obs_dict, training=False, use_classifier=True)
                scaled_action = self._scale_action(action.cpu().numpy()[0])
                return scaled_action, env_idx.cpu().numpy()[0]

    def _scale_action(self, action):
        """액션 스케일링"""
        steering = action[0]  # 조향각만 학습
        # 조향각을 action_bounds에 맞게 스케일링
        steering_low, steering_high = self.action_bounds[0]
        scaled_steering = (steering + 1) / 2 * (steering_high - steering_low) + steering_low
        scaled_steering = np.clip(scaled_steering, steering_low, steering_high)
        
        # 스로틀은 고정값 사용
        throttle_low, throttle_high = self.action_bounds[1]
        fixed_throttle = (throttle_low + throttle_high) / 2  # 중간값 사용
        
        return np.array([scaled_steering, fixed_throttle], dtype=np.float32)

    def _unscale_action(self, scaled_action):
        """액션 역스케일링"""
        steering = scaled_action[0]
        steering_low, steering_high = self.action_bounds[0]
        unscaled_steering = 2 * (steering - steering_low) / (steering_high - steering_low) - 1
        unscaled_steering = np.clip(unscaled_steering, -1, 1)
        return np.array([unscaled_steering], dtype=np.float32)

    def store_experience(self, obs_dict, action, reward, value, log_prob, done, next_obs_dict, env_label, env_prediction=None):
        """경험 저장"""
        # 조기 종료 패널티
        if done and self.buffer.current_episode_length < 30:
            penalty = -2.0 * (30 - self.buffer.current_episode_length) / 30
            reward += penalty
        
        # 보상 정규화
        if self.normalize_rewards:
            self.reward_rms.update([reward])
            normalized_reward = reward / max(np.sqrt(self.reward_rms.var + 1e-8), 1.0)
            normalized_reward = np.clip(normalized_reward, -10.0, 10.0)
        else:
            normalized_reward = reward
        
        # 조향각만 저장 (언스케일링)
        unscaled_action = self._unscale_action(action)
        
        is_short_episode = self.buffer.add(
            obs_dict, unscaled_action[0], normalized_reward, value, log_prob,
            done, next_obs_dict if not done else None, env_label, env_prediction
        )
        
        return is_short_episode
        
    def compute_gae(self, rewards, values, dones, next_values, gamma=0.995, lam=0.95):
        """GAE (Generalized Advantage Estimation) 계산"""
        advantages = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = next_values[-1] if len(next_values) > 0 else 0
            else:
                next_value = values[i + 1]
            
            next_non_terminal = 1.0 - dones[i]
            delta = rewards[i] + gamma * next_value * next_non_terminal - values[i]
            gae = delta + gamma * lam * next_non_terminal * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages, returns

    def train(self):
        """PPO 학습 실행"""
        if self.buffer.size() < self.mini_batch_size:
            return {}
        
        # 버퍼에서 데이터 가져오기
        states, actions, rewards, values, old_log_probs, dones, next_states, env_labels, env_predictions = self.buffer.get()
        
        # 다음 상태의 가치 계산
        next_values = []
        for i, next_state in enumerate(next_states):
            if next_state is not None:
                with torch.no_grad():
                    features, velocity, steering = self.preprocess_obs(next_state)
                    env_id = env_labels[i]  # 같은 환경으로 가정
                    next_value = self.critics[env_id](features, velocity, steering).cpu().numpy()[0][0]
                next_values.append(next_value)
            else:
                next_values.append(0.0)
        
        # GAE 계산
        advantages, returns = self.compute_gae(rewards, values, dones, next_values)
        
        # 정규화
        advantages = np.array(advantages)
        returns = np.array(returns)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 배치 데이터 준비
        features_batch, velocities_batch, steerings_batch = self.preprocess_obs(states, is_batch=True)
        
        actions_tensor = torch.FloatTensor(actions).unsqueeze(1).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).unsqueeze(1).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).unsqueeze(1).to(self.device)
        returns_tensor = torch.FloatTensor(returns).unsqueeze(1).to(self.device)
        env_labels_tensor = torch.LongTensor(env_labels).to(self.device)
        
        # 학습 통계
        total_actor_loss = 0
        total_critic_loss = 0
        total_classifier_loss = 0
        total_entropy = 0
        batch_count = 0
        
        # PPO 업데이트
        for epoch in range(self.ppo_epochs):
            indices = torch.randperm(len(states))
            
            for start in range(0, len(states), self.mini_batch_size):
                end = min(start + self.mini_batch_size, len(states))
                mb_indices = indices[start:end]
                
                mb_features = features_batch[mb_indices]
                mb_velocities = velocities_batch[mb_indices]
                mb_steerings = steerings_batch[mb_indices]
                mb_actions = actions_tensor[mb_indices]
                mb_old_log_probs = old_log_probs_tensor[mb_indices]
                mb_advantages = advantages_tensor[mb_indices]
                mb_returns = returns_tensor[mb_indices]
                mb_env_labels = env_labels_tensor[mb_indices]
                
                # 현재 정책 평가
                new_log_probs = self.actor.get_log_prob(mb_features, mb_actions, mb_velocities, mb_steerings)
                
                # 환경별 가치 계산
                mb_values = torch.zeros_like(mb_returns)
                for i in range(self.num_environments):
                    mask = (mb_env_labels == i)
                    if mask.any():
                        env_values = self.critics[i](mb_features[mask], mb_velocities[mask], mb_steerings[mask])
                        mb_values[mask] = env_values
                
                # 분류기 예측
                env_probs = self.classifier(mb_features)
                
                # PPO Actor 손실
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Critic 손실 (환경별)
                critic_loss = F.smooth_l1_loss(mb_values, mb_returns)
                
                # 분류기 손실
                classifier_loss = F.cross_entropy(env_probs, mb_env_labels)
                
                # Entropy 계산
                mean, std = self.actor(mb_features, mb_velocities, mb_steerings)
                entropy = torch.distributions.Normal(mean, std).entropy().sum(dim=-1).mean()
                
                # 통계 수집
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_classifier_loss += classifier_loss.item()
                total_entropy += entropy.item()
                batch_count += 1
                
                # Actor 업데이트
                actor_total_loss = actor_loss - self.entropy_coef * entropy
                self.encoder_opt.zero_grad()
                self.actor_opt.zero_grad()
                actor_total_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.encoder_opt.step()
                self.actor_opt.step()
                
                # Critic + Classifier 업데이트
                critic_total_loss = self.value_loss_coef * critic_loss + self.classifier_loss_coef * classifier_loss
                self.critic_opt.zero_grad()
                critic_total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critics.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), self.max_grad_norm)
                self.critic_opt.step()
        
        # 분류기 정확도 계산
        with torch.no_grad():
            all_env_probs = self.classifier(features_batch)
            predicted_envs = torch.argmax(all_env_probs, dim=-1)
            classifier_accuracy = (predicted_envs == env_labels_tensor).float().mean().item()
        
        # 버퍼 클리어
        self.buffer.clear()
        self.update_count += 1
        
        # 통계 반환
        return {
            'actor_loss': total_actor_loss / batch_count if batch_count > 0 else 0,
            'critic_loss': total_critic_loss / batch_count if batch_count > 0 else 0,
            'classifier_loss': total_classifier_loss / batch_count if batch_count > 0 else 0,
            'classifier_accuracy': classifier_accuracy,
            'entropy': total_entropy / batch_count if batch_count > 0 else 0,
            'total_loss': (total_actor_loss + total_critic_loss + total_classifier_loss) / batch_count if batch_count > 0 else 0
        }

    def save_model(self, save_path):
        """모델 저장"""
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.encoder.state_dict(), os.path.join(save_path, "encoder.pt"))
        torch.save(self.actor.state_dict(), os.path.join(save_path, "actor.pt"))
        torch.save(self.classifier.state_dict(), os.path.join(save_path, "classifier.pt"))
        for i, critic in enumerate(self.critics):
            torch.save(critic.state_dict(), os.path.join(save_path, f"critic_{i}.pt"))
        print(f"Multi-Critic PPO 모델 저장 완료: {save_path}")

    def load_model(self, load_path):
        """모델 로드"""
        self.encoder.load_state_dict(torch.load(os.path.join(load_path, "encoder.pt"), weights_only=True))
        self.actor.load_state_dict(torch.load(os.path.join(load_path, "actor.pt"), weights_only=True))
        self.classifier.load_state_dict(torch.load(os.path.join(load_path, "classifier.pt"), weights_only=True))
        for i, critic in enumerate(self.critics):
            critic.load_state_dict(torch.load(os.path.join(load_path, f"critic_{i}.pt"), weights_only=True))
        print(f"Multi-Critic PPO 모델 로드 완료: {load_path}")

    def episode_reset(self):
        """에피소드 리셋"""
        self.buffer.episode_reset()

    def get_buffer_size(self):
        """버퍼 크기 반환"""
        return self.buffer.size()