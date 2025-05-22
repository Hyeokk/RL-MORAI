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

        # 액션 설정
        if action_bounds is None:
            action_bounds = [(-0.7, 0.7), (10.0, 30.0)]  # 기본: 조향 [-0.7, 0.7], 스로틀 [10, 30]

        low = np.array([bound[0] for bound in action_bounds], dtype=np.float32)
        high = np.array([bound[1] for bound in action_bounds], dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # 관측공간은 grayscale 이미지 (80x160x1)
        self.observation_space = spaces.Box(low=0, high=255,
                                           shape=(160, 240, 1),
                                           dtype=np.float32)

        rospy.loginfo("Init Sensors...")
        while self.sensor.get_image() is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
        rospy.loginfo("Init Complete")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

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
        timeout = 2.0  # 2초 타임아웃
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
            
        return obs, {}

    def step(self, action):
        steering, throttle = action
        self.sensor.send_control(steering, throttle)

        time.sleep(0.1)  # 물리 엔진 반영 시간 대기
        obs = self.sensor.get_image()

        # 보상 및 종료 계산
        reward = self._reward_fn(obs) if self._reward_fn else 0.0
        done = self._terminated_fn(obs) if self._terminated_fn else False

        info = {}
        return obs, reward, done, False, info

    def render(self):
        image = self.sensor.get_image()
        if image is not None:
            show = cv2.resize(image, (320, 160))
            cv2.imshow("Morai Camera", show)
            cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()

    def set_reward_fn(self, reward_fn):
        self._reward_fn = reward_fn

    def set_episode_over_fn(self, terminated_fn):
        self._terminated_fn = terminated_fn