import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rospy
from .morai_sensor import MoraiSensor
import time
import pyautogui
from cv_bridge import CvBridge
import cv2

class MoraiEnv(gym.Env):
    def __init__(self):
        super(MoraiEnv, self).__init__()
        rospy.init_node('reinforcement_node', anonymous=True)

        self.sensor = MoraiSensor()
        self.action_space = spaces.Box(low=np.array([-1.0, 0.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255, shape=(80, 160, 1), dtype=np.uint8)

        rospy.loginfo("환경 초기화 중...")
        while self.sensor.get_image() is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
        rospy.loginfo("환경 초기화 완료")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 시뮬레이터 리셋 명령이 있다면 여기에 추가
        # 예: `/reset_sim` 서비스 호출
        pyautogui.press('i')
        obs = self.sensor.get_image()
        return obs, {}

    def step(self, action):
        steering, throttle = action
        image = self.sensor.get_image()
        gps = self.sensor.get_gps()

        done = False
        reward = self._compute_reward(image, gps)

        info = {}

        return image, reward, done, False, info

    def render(self):
        image = self.sensor.get_image()
        if image is not None:
            cv2.imshow("Camera", image)
            cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()

    def _compute_reward(self, image, gps):
        # 차후 목적에 맞게 보상 정의
        # 예: 흑백 이미지의 중심선을 기준으로 편차 측정
        return 1.0  # 임시 리워드
