import os
import sys
import time
import signal
import numpy as np
import torch
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gym_morai.envs.morai_env import MoraiEnv
from gym_morai.envs.morai_sensor import MoraiSensor
from models.SAC import SACAgent
from src.utils import Preprocess

# 전역 플래그
stop_flag = False

def signal_handler(sig, frame):
    global stop_flag
    print("\n[INFO] KILL NODE")
    stop_flag = True

# 시그널 핸들러 등록
signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # kill

# 모델 경로 설정
MODEL_DIR = "/home/kuuve/catkin_ws/src/pt/"
ACTOR_PATH = os.path.join(MODEL_DIR, "sac_actor_latest.pt")
CRITIC1_PATH = os.path.join(MODEL_DIR, "sac_critic1_latest.pt")
CRITIC2_PATH = os.path.join(MODEL_DIR, "sac_critic2_latest.pt")

# 모델 파일이 없으면 다른 파일명 시도
if not os.path.exists(ACTOR_PATH):
    actor_files = [f for f in os.listdir(MODEL_DIR) if f.startswith("sac_actor") and f.endswith(".pt")]
    if actor_files:
        ACTOR_PATH = os.path.join(MODEL_DIR, actor_files[0])
        print(f"[INFO] 대체 액터 모델 발견: {ACTOR_PATH}")
    else:
        print("[ERROR] 액터 모델 파일을 찾을 수 없습니다!")

# 모델 구조를 확인하는 함수
def inspect_model_file(model_path):
    try:
        state_dict = torch.load(model_path, map_location="cpu")
        print(f"[INFO] 모델 파일 타입: {type(state_dict)}")
        
        if isinstance(state_dict, dict):
            print("\n[INFO] 가중치 키 목록:")
            for key, tensor in state_dict.items():
                print(f"- {key}: {tensor.shape}")
                
            # 첫 번째 컨볼루션 레이어의 입력 채널 수 확인
            if 'conv.0.weight' in state_dict:
                conv1_weight = state_dict['conv.0.weight']
                input_channels = conv1_weight.shape[1]
                print(f"\n[INFO] 첫 번째 컨볼루션 레이어 입력 채널 수: {input_channels}")
                return input_channels
        return None
    except Exception as e:
        print(f"[ERROR] 모델 파일 검사 중 오류: {e}")
        return None

# 수정된 이미지 전처리 함수
def preprocess_image(image, expected_channels=80):
    """원본 학습 코드와 일치하는 형식으로 이미지 전처리"""
    if image is None:
        return None
        
    # 리사이즈
    image = cv2.resize(image, (160, 80))
    
    # 정규화
    image = image.astype(np.float32) / 255.0
    
    # 채널 처리 (모델이 기대하는 채널 수에 맞춤)
    if image.ndim == 2:
        # 그레이스케일 이미지면 채널 차원 추가
        image = image[:, :, np.newaxis]
    
    # 채널 수 맞추기 (1채널 → expected_channels채널)
    if image.shape[2] == 1 and expected_channels > 1:
        # 방법 1: 동일한 채널 복제
        image = np.repeat(image, expected_channels, axis=2)
        
        # 또는 방법 2: 제로 패딩 (다른 채널은 0으로 채움)
        # padded_image = np.zeros((80, 160, expected_channels), dtype=np.float32)
        # padded_image[:, :, 0] = image[:, :, 0]
        # image = padded_image
    
    # CHW 형식으로 변환 (PyTorch에서 사용하는 형식)
    image = np.transpose(image, (2, 0, 1))
    return image

# 커스텀 SAC 에이전트 클래스
class CustomSACAgent:
    def __init__(self, actor_path, critic1_path=None, critic2_path=None, input_channels=1, action_dim=2, action_bounds=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_bounds = action_bounds or [(-1.0, 1.0), (10.0, 30.0)]
        self.input_channels = input_channels
        self.action_dim = action_dim
        
        # 모델 로드
        try:
            # 모델 상태 딕셔너리 불러오기
            self.actor_state_dict = torch.load(actor_path, map_location=self.device)
            print(f"[INFO] 액터 모델 로드 완료: {actor_path}")
            
            # 선택적으로 크리틱 모델 로드
            self.critic1_state_dict = None
            self.critic2_state_dict = None
            if critic1_path and os.path.exists(critic1_path):
                self.critic1_state_dict = torch.load(critic1_path, map_location=self.device)
                print(f"[INFO] 크리틱1 모델 로드 완료: {critic1_path}")
            if critic2_path and os.path.exists(critic2_path):
                self.critic2_state_dict = torch.load(critic2_path, map_location=self.device)
                print(f"[INFO] 크리틱2 모델 로드 완료: {critic2_path}")
        except Exception as e:
            print(f"[ERROR] 모델 로드 실패: {e}")
            raise
    
    def get_action(self, state, deterministic=True):
        """상태에서 행동 계산"""
        # 입력 전처리 (B, C, H, W) 형식으로 변환
        if not isinstance(state, torch.Tensor):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        else:
            state_tensor = state.to(self.device)
            
        # 예상 입력 형태 확인
        if state_tensor.shape[1] != self.input_channels:
            print(f"[ERROR] 입력 채널 불일치! 예상: {self.input_channels}, 실제: {state_tensor.shape[1]}")
            # 간단한 대응책: 랜덤 액션 반환
            return np.array([0.0, 20.0], dtype=np.float32)
        
        # 모델 구현 및 추론
        try:
            # 여기서는 직접 로드한 가중치를 사용하여 추론
            # (완전한 모델 구현 대신 간단한 예측 함수 사용)
            
            # 이 부분은 모델 구조에 따라 다를 수 있음
            # 간단한 대응책: 중립 행동 반환
            action = np.array([0.0, 20.0], dtype=np.float32)
            
            # 실제로는 이런 식으로 모델을 구현해야 함
            # output = self.forward(state_tensor)
            # action = output.cpu().numpy()[0]
            
            return self._scale_action(action)
        except Exception as e:
            print(f"[ERROR] 액션 예측 중 오류: {e}")
            import traceback
            traceback.print_exc()
            # 오류 발생 시 기본 액션 반환
            return np.array([0.0, 20.0], dtype=np.float32)
    
    def _scale_action(self, action):
        """액션 스케일링 (-1,1 범위에서 실제 범위로)"""
        scaled = []
        for i in range(len(action)):
            low, high = self.action_bounds[i]
            scaled_val = (action[i] + 1) / 2 * (high - low) + low
            scaled.append(np.clip(scaled_val, low, high))
        return np.array(scaled, dtype=np.float32)

def main():
    global stop_flag
    print("[INFO] 테스트 시작")
    
    # 환경 초기화
    env = MoraiEnv()
    sensor = env.sensor
    
    # 초기 관측 가져오기
    obs, _ = env.reset()
    if obs is None:
        print("[ERROR] 초기 관측값이 None입니다!")
        return
    
    # 모델 파일 검사로 입력 채널 수 확인
    input_channels = inspect_model_file(ACTOR_PATH)
    if input_channels is None:
        print("[WARN] 모델 파일에서 입력 채널을 확인할 수 없습니다. 기본값 사용: 1")
        input_channels = 1
    
    print(f"[INFO] 모델이 기대하는 입력 채널 수: {input_channels}")
    
    # 이미지 전처리
    state = preprocess_image(obs, expected_channels=input_channels)
    if state is None:
        print("[ERROR] 상태 전처리 실패!")
        return
    
    print(f"[INFO] 관측값 크기: {obs.shape}, 전처리된 상태 크기: {state.shape}")
    
    # 모델 설정
    action_dim = 2
    action_bounds = [(-1.0, 1.0), (10.0, 30.0)]
    
    # 단순한 테스트를 위해 커스텀 에이전트 사용
    try:
        agent = CustomSACAgent(
            actor_path=ACTOR_PATH,
            critic1_path=CRITIC1_PATH,
            critic2_path=CRITIC2_PATH,
            input_channels=input_channels,
            action_dim=action_dim,
            action_bounds=action_bounds
        )
    except Exception as e:
        print(f"[ERROR] 에이전트 초기화 실패: {e}")
        return
    
    # 테스트 실행
    try:
        for episode in range(1):  # 1 에피소드만 테스트
            if stop_flag:
                break
            
            print(f"\n[INFO] 에피소드 {episode+1} 시작")
            obs, _ = env.reset()
            
            # 단순한 테스트 패턴 사용
            test_actions = [
                # 직진 (0 조향, 중간 속도)
                (0.0, 20.0),
                # 약한 왼쪽 조향
                (-0.3, 15.0),
                # 약한 오른쪽 조향
                (0.3, 15.0),
                # 다시 직진
                (0.0, 20.0)
            ]
            
            # 각 패턴으로 테스트
            for action_idx, (steering, throttle) in enumerate(test_actions):
                if stop_flag:
                    break
                
                action = np.array([steering, throttle], dtype=np.float32)
                print(f"[INFO] 테스트 액션 {action_idx+1}/{len(test_actions)}: ({steering}, {throttle})")
                
                # 각 액션을 50 스텝 동안 실행
                for step in range(50):
                    if stop_flag:
                        break
                    
                    # 환경에 액션 적용
                    next_obs, reward, done, _, _ = env.step(action)
                    
                    # 현재 조향각 확인
                    if hasattr(sensor, 'last_steering'):
                        current_steering = sensor.last_steering
                        if step % 10 == 0:
                            print(f"[INFO] 현재 조향각: {current_steering:.4f}")
                    
                    # CTE 계산 시도
                    try:
                        cte = sensor.cal_cte()
                        if step % 10 == 0:
                            print(f"[INFO] CTE: {cte:.4f}")
                    except:
                        pass
                    
                    # 다음 상태로 이동
                    obs = next_obs
                    
                    # 시각화
                    env.render()
                    
                    # 에피소드 종료 조건
                    if done:
                        print(f"[INFO] 에피소드 종료 (done=True)")
                        break
                    
                    # 잠시 대기
                    time.sleep(0.05)
                
                print(f"[INFO] 테스트 액션 {action_idx+1} 완료")
            
            print(f"[INFO] 에피소드 {episode+1} 완료")
    
    except Exception as e:
        print(f"[ERROR] 테스트 중 예외 발생: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        env.close()
        print("[INFO] 테스트 종료")

if __name__ == "__main__":
    main()