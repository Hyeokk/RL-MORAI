import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rospy
import time
import subprocess
import cv2

from morai_sensor import MoraiSensor

# 메인 강화학습 환경 클래스
class MoraiEnv(gym.Env):
    def __init__(self, reward_fn=None, done_fn=None):
        super(MoraiEnv, self).__init__()
        rospy.init_node('morai_rl_env', anonymous=True)

        self.sensor = MoraiSensor()
        self._reward_fn = reward_fn
        self._done_fn = done_fn

        # 조향 [-1, 1], 스로틀 [0, 1]
        self.action_space = spaces.Box(low=np.array([-1.0, 0.0]),
                                       high=np.array([1.0, 1.0]),
                                       dtype=np.float32)

        # 관측공간은 grayscale 이미지 (80x160x1)
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(80, 160, 1),
                                            dtype=np.uint8)

        rospy.loginfo("센서 초기화 대기 중...")
        while self.sensor.get_image() is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
        rospy.loginfo("강화학습 환경 초기화 완료")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # MORAI 시뮬레이터 창에 'i' 키 입력으로 초기화
        try:
            subprocess.run([
                'xdotool', 'search', '--name', 'MORAI',
                'windowactivate', '--sync', 'key', 'i'
            ], check=True)
            print("[INFO] 'i' 키 입력으로 초기화 완료")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] 시뮬레이터 리셋 실패: {e}")

        time.sleep(0.5)
        obs = self.sensor.get_image()
        return obs, {}

    def step(self, action):
        steering, throttle = action
        self.sensor.send_control(steering, throttle)

        time.sleep(0.1)  # 물리 엔진 반영 시간 대기
        obs = self.sensor.get_image()

        reward = self._reward_fn(obs) if self._reward_fn else 0.0
        done = self._done_fn(obs) if self._done_fn else False

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

    def set_episode_over_fn(self, done_fn):
        self._done_fn = done_fn
