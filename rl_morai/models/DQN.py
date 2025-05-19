import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from CNN import CNNActionValue

class DonkeyDQN:
    def __init__(
            self,
            state_dim,                    # 입력 이미지 상태의 크기
            action_dim=2,                 # action_dim은 2로 설정 (steer, throttle)
            lr=0.01,                     # 학습률
            epsilon=1.0,                  # epsilon-greedy 1.0에서 0.1으로 감소
            epsilon_min=0.1,              
            gamma=0.99,                   # discount factor
            batch_size=32,                # sample 묶는 개수
            warmup_steps=5000,            # 훈련 시작 전 랜덤으로 행동하는 개수
            buffer_size=int(1e5),         # buffer의 최대 크기
            target_update_interval=10000, # target network 업데이트 주기
            action_limits=(-1.0, 1.0, 0.0, 1.0),  # (steer_min, steer_max, throttle_min, throttle_max)
    ):
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.target_update_interval = target_update_interval
        
        # Donkey 환경에 맞는 action limits 설정
        self.steer_min, self.steer_max, self.throttle_min, self.throttle_max = action_limits

        # CNN 네트워크 초기화 - donkey_env의 observation_space에 맞게 설정
        self.network = CNNActionValue(state_dim[2], action_dim)
        self.target_network = CNNActionValue(state_dim[2], action_dim)
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = torch.optim.RMSprop(self.network.parameters(), lr)

        # Replay Buffer 초기화 - donkey_env의 observation_space와 action_space에 맞게 설정
        #self.buffer = ReplayBuffer(state_dim, (action_dim,), buffer_size)
        state_dim = (state_dim[2], state_dim[0], state_dim[1])  # [C, H, W]
        self.buffer = ReplayBuffer(state_dim, (action_dim,), buffer_size)

        
        # 디바이스 설정
        self.device = torch.device("mps" if torch.backends.mps.is_available() 
                                  else "cuda" if torch.cuda.is_available() 
                                  else "cpu")

        self.network.to(self.device)
        self.target_network.to(self.device)

        self.total_steps = 0
        self.epsilon_decay = (epsilon - epsilon_min) / 20000  # epsilon을 1e5 step 동안 감소

    #이미지 전처리 함수
    def preprocess_image(self, image):
        image = image.astype(np.float32) / 255.0  # numpy 배열을 float32로 변환하고 0-1 범위로 정규화
        image = np.transpose(image, (2, 0, 1))    # HWC(Height, Width, Channel) -> CHW(Channel, Height, Width) 변환
    
        return image

    @torch.no_grad()
    def act(self, x, training=True):
        self.network.train(training)
        
        # Epsilon-greedy 정책으로 행동 선택
        # if training and ((np.random.rand() < self.epsilon) or (self.total_steps < self.warmup_steps)):
        #     # 랜덤 행동 선택 (Donkey 환경에 맞게 조정)
        #     steer = np.random.uniform(self.steer_min, self.steer_max)
        #     throttle = np.random.uniform(self.throttle_min, self.throttle_max)
        #     return np.array([steer, throttle], dtype=np.float32)
        if training and ((np.random.rand() < self.epsilon)):
            # 랜덤 행동 선택 (Donkey 환경에 맞게 조정)
            steer = np.random.uniform(self.steer_min, self.steer_max)
            throttle = np.random.uniform(self.throttle_min, self.throttle_max)
            return np.array([steer, throttle], dtype=np.float32)
        else:
            # 네트워크를 통한 행동 선택
            x = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
            q_values = self.network(x)
            
            # Q-value를 연속적인 액션으로 변환
            # 여기서는 간단한 예시로, Q-value의 출력을 normalize하여 적절한 범위로 변환
            # 실제 구현에서는 더 복잡한 매핑이 필요할 수 있음
            steer = self.steer_min + (self.steer_max - self.steer_min) * torch.sigmoid(q_values[0, 0]).item()
            throttle = self.throttle_min + (self.throttle_max - self.throttle_min) * torch.sigmoid(q_values[0, 1]).item()
            
            return np.array([steer, throttle], dtype=np.float32)

    def learn(self):
        s, a, r, s_prime, terminated = map(lambda x: x.to(self.device), self.buffer.sample(self.batch_size))

        # Target Q-value 계산
        with torch.no_grad():
            next_q = self.target_network(s_prime)
            td_target = r + (1. - terminated) * self.gamma * next_q.max(dim=1, keepdim=True).values
        
        # 현재 Q-value 계산
        current_q = self.network(s)
        
        # Donkey 환경에 맞게 action을 처리 (연속적인 action space에 맞게 조정)
        # 간단한 MSE loss 사용
        q_values = torch.gather(current_q, 1, a.long())
        loss = F.mse_loss(q_values, td_target)
        
        # 네트워크 업데이트
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        result = {
            'total_steps': self.total_steps,
            'value_loss': loss.item()
        }
        return result

    def process(self, transition):
        result = {}
        self.total_steps += 1
        self.buffer.update(*transition)  # (s, a, r, s', terminated) tuple을 buffer에 저장

        if self.total_steps > self.warmup_steps:  # warmup step이 지나면 학습 시작
            result = self.learn()

        if self.total_steps % self.target_update_interval == 0:  # target network 업데이트
            self.target_network.load_state_dict(self.network.state_dict())
        
        # epsilon 감소
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)
        
        return result

    def save(self, path):
        checkpoint = {
            'network': self.network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
            'epsilon': self.epsilon,
        }
        torch.save(checkpoint, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.total_steps = checkpoint['total_steps']
        self.epsilon = checkpoint['epsilon']


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e5)):
        self.s = np.zeros((max_size, *state_dim), dtype=np.float32)
        self.a = np.zeros((max_size, *action_dim), dtype=np.float32)  # 연속적인 action space에 맞게 float32로 변경
        self.r = np.zeros((max_size, 1), dtype=np.float32)
        self.s_prime = np.zeros((max_size, *state_dim), dtype=np.float32)
        self.terminated = np.zeros((max_size, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0
        self.max_size = max_size

    def update(self, s, a, r, s_prime, terminated):
        self.s[self.ptr] = s
        self.a[self.ptr] = a
        self.r[self.ptr] = r
        self.s_prime[self.ptr] = s_prime
        self.terminated[self.ptr] = terminated

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, batch_size)
        return (
            torch.FloatTensor(self.s[ind]),
            torch.FloatTensor(self.a[ind]),
            torch.FloatTensor(self.r[ind]),
            torch.FloatTensor(self.s_prime[ind]),
            torch.FloatTensor(self.terminated[ind]),
        )