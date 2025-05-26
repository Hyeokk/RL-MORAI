import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rospy
import time
import subprocess
import cv2
import sys

from .morai_sensor import MoraiSensor

# 사용자 정의 예외 클래스
class SensorTimeoutError(Exception):
    """타임아웃 동안 유효한 센서 데이터를 받지 못한 경우 발생하는 예외"""
    pass

class MoraiEnv(gym.Env):
    def __init__(self, reward_fn=None, terminated_fn=None, action_bounds=None):
        super(MoraiEnv, self).__init__()
        rospy.init_node('morai_rl_env', anonymous=True)

        self.sensor = MoraiSensor()
        self._reward_fn = reward_fn
        self._terminated_fn = terminated_fn
        self._first_reset = True

        self.last_steering = 0.0

        # 액션 설정
        if action_bounds is None:
            action_bounds = [(-0.7, 0.7), (10.0, 30.0)]  # 기본: 조향 [-0.7, 0.7], 스로틀 [10, 30]

        low = np.array([bound[0] for bound in action_bounds], dtype=np.float32)
        high = np.array([bound[1] for bound in action_bounds], dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=0.0, high=1.0, shape=(120, 160, 1), dtype=np.float32),
            'velocity': spaces.Box(low=0.0, high=50.0, shape=(1,), dtype=np.float32),
            'steering': spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        })

        rospy.loginfo("Init Sensors...")
        while self.sensor.get_image() is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
        rospy.loginfo("Init Complete")

    def get_observation(self):
        """현재 관측값을 복합 딕셔너리 형태로 반환"""
        image = self.sensor.get_image()
        velocity = self.sensor.get_velocity()
        
        if image is None:
            return None
            
        obs = {
            'image': image,
            'velocity': np.array([velocity], dtype=np.float32),
            'steering': np.array([self.last_steering], dtype=np.float32)
        }
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.last_steering = 0.0

        # 키보드 명령으로 시뮬레이터 리셋
        try:
            if self._first_reset:
                # 최초 1회: i → q
                subprocess.run(['xdotool', 'search', '--name', 'Simulator', 
                               'windowactivate', '--sync', 'key', 'i'], check=True)
                time.sleep(0.1)
                subprocess.run(['xdotool', 'search', '--name', 'Simulator', 
                               'windowactivate', '--sync', 'key', 'q'], check=True)
                self._first_reset = False
            else:
                # 이후: q → i → q
                subprocess.run(['xdotool', 'search', '--name', 'Simulator', 
                               'windowactivate', '--sync', 'key', 'q'], check=True)
                time.sleep(0.1)
                subprocess.run(['xdotool', 'search', '--name', 'Simulator', 
                               'windowactivate', '--sync', 'key', 'i'], check=True)
                time.sleep(0.1)
                subprocess.run(['xdotool', 'search', '--name', 'Simulator', 
                               'windowactivate', '--sync', 'key', 'q'], check=True)
        except subprocess.CalledProcessError as e:
            rospy.logerr(f"Reset failed: {e}")

        # 유효한 관측값을 받을 때까지 기다림 (리셋 후에만 확인)
        obs = None
        timeout = 1.0  # 2초 타임아웃
        start_time = time.time()
        
        while obs is None and time.time() - start_time < timeout:
            obs = self.sensor.get_image()
            if obs is None or not np.any(obs):  # 이미지가 None이거나 모두 0인 경우
                obs = None
                rospy.sleep(0.1)
        
        # 타임아웃 후에도 유효한 관측값이 없으면 예외 발생
        if obs is None:
            rospy.logerr("CRITICAL ERROR: Failed to get valid observation after reset - terminating node")
            raise SensorTimeoutError("No valid sensor data received after timeout")
            
        return self.get_observation(), {}

    def step(self, action):
        steering, throttle = action
        self.last_steering = float(steering)

        self.sensor.send_control(steering, throttle)

        time.sleep(0.1)  # 물리 엔진 반영 시간 대기
        obs = self.sensor.get_image()

        # 보상 및 종료 계산
        reward = self._reward_fn(obs) if self._reward_fn else 0.0
        done = self._terminated_fn(obs) if self._terminated_fn else False

        info = {}
        return self.get_observation(), reward, done, False, info
    
    def render(self):
        image = self.sensor.get_image()
        if image is not None:
            # 정규화된 이미지를 시각화용으로 변환
            display_image = (image * 255).astype(np.uint8)
            display_image = cv2.cvtColor(display_image, cv2.COLOR_GRAY2BGR)
            show = cv2.resize(display_image, (320, 240))
            
            # 추가 정보 표시
            velocity = self.sensor.get_velocity()
            cv2.putText(show, f"Velocity: {velocity*3.6:.1f} km/h", (10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(show, f"Steering: {self.last_steering:.2f}", (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            cv2.imshow("Morai Camera", show)
            cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()

    def set_reward_fn(self, reward_fn):
        self._reward_fn = reward_fn

    def set_episode_over_fn(self, terminated_fn):
        self._terminated_fn = terminated_fn