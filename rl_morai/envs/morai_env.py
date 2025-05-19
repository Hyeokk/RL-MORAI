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
    def __init__(self,
                 reward_fn=None,
                 done_fn=None):
        super(MoraiEnv, self).__init__()
        rospy.init_node('reinforcement_node', anonymous=True)

        self.sensor = MoraiSensor()
        self.bridge = CvBridge()

        # 행동과 관측 공간 설정
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0]), 
            high=np.array([1.0, 1.0]),
            dtype=np.float32)
        
        self.observation_space = spaces.Box(low=0, high=255, 
                                            shape=(80, 160, 1), 
                                            dtype=np.uint8)

        # 외부로부터 보상 및 종료 함수 주입
        self._reward_fn = reward_fn
        self._done_fn = done_fn

        rospy.loginfo("환경 초기화 중...")
        while self.sensor.get_image() is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
        rospy.loginfo("환경 초기화 완료")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        pyautogui.press('i')  # 초기화용 키 입력
        time.sleep(0.1)
        obs = self.sensor.get_image()
        return obs, {}

    def step(self, action):
        steering, throttle = action
        self.sensor.send_control(steering, throttle)

        image = self.sensor.get_image()
        gps = self.sensor.get_position()

        # 보상 계산
        reward = self._reward_fn(image, gps) if self._reward_fn else 0.0
        done = self._done_fn(image, gps) if self._done_fn else False

        info = {}
        return image, reward, done, False, info

    def render(self):
        image = self.sensor.get_image()
        if image is not None:
            cv2.imshow("Camera", image)
            cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()
