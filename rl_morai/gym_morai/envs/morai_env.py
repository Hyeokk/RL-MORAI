import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rospy
import time
import subprocess
import cv2

from .morai_sensor import MoraiSensor

# 메인 강화학습 환경 클래스
class MoraiEnv(gym.Env):
    def __init__(self, reward_fn=None, done_fn=None):
        super(MoraiEnv, self).__init__()
        rospy.init_node('morai_rl_env', anonymous=True)

        self.sensor = MoraiSensor()
        self._reward_fn = reward_fn
        self._done_fn = done_fn
        self._first_reset = True

        # 조향 [-1, 1], 스로틀 [0, 1]
        self.action_space = spaces.Box(low=np.array([-1.0, 10.0]),
                                       high=np.array([1.0, 30.0]),
                                       dtype=np.float32)

        # 관측공간은 grayscale 이미지 (80x160x1)
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(80, 160, 1),
                                            dtype=np.uint8)

        rospy.loginfo("Init Sensors...")
        while self.sensor.get_image() is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
        rospy.loginfo("Init Complete")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        try:
            if self._first_reset:
                # 최초 1회: i → q
                subprocess.run(['xdotool', 'search', '--name', 'Simulator', 'windowactivate', '--sync', 'key', 'i'], check=True)
                time.sleep(0.5)
                subprocess.run(['xdotool', 'search', '--name', 'Simulator', 'windowactivate', '--sync', 'key', 'q'], check=True)
                #print("[INFO] 첫 reset: i → q 완료")
                self._first_reset = False  # 다음부터는 일반 reset
            else:
                # 이후: q → i → q
                subprocess.run(['xdotool', 'search', '--name', 'Simulator', 'windowactivate', '--sync', 'key', 'q'], check=True)
                time.sleep(0.5)
                subprocess.run(['xdotool', 'search', '--name', 'Simulator', 'windowactivate', '--sync', 'key', 'i'], check=True)
                time.sleep(0.5)
                subprocess.run(['xdotool', 'search', '--name', 'Simulator', 'windowactivate', '--sync', 'key', 'q'], check=True)
                #print("[INFO] 이후 reset: q → i → q 완료")


        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed Init: {e}")

        time.sleep(0.5)
        obs = self.sensor.get_image()
        return obs, {}

    def step(self, action):
        steering, throttle = action
        #print(f"[STEP] action received → steering: {steering:.2f}, throttle: {throttle:.2f}")
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
