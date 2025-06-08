import os
import sys
import signal
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gym_morai.envs.morai_env import MoraiEnv
from src.reward_fns import RewardFns
from src.terminated_fns import TerminatedFns
from models.MultiCriticPPO import MultiCriticPPOAgent
from models.SingleCriticPPO import PPOAgent as SingleCriticPPOAgent

# ================= 설정 =================
MODEL_TYPE = "multi"         # "multi", "no_transfer", "single"
ENV_ID = 0                   # 0: 실선, 1: 점선, 2: 야간
LANE_TYPE = "solid"         # "solid", "dashed", "night"
#MODEL_DIR = "/home/hyeokk/catkin_ws/src/pt/Single-Critic/2_dashed/pt/"
MODEL_DIR = "/home/hyeokk/catkin_ws/src/pt/Multi-Critic_noTransfer/1_solid/pt/"
NUM_TEST_EPISODES = 50
ACTION_BOUNDS = [(-0.4, 0.4), (15.0, 25.0)]

# 종료 플래그 설정 (Ctrl+C 대응)
stop_flag = False
def signal_handler(sig, frame):
    global stop_flag
    stop_flag = True
signal.signal(signal.SIGINT, signal_handler)

# ================= 환경 초기화 =================
env = MoraiEnv(action_bounds=ACTION_BOUNDS, csv_path=f"/home/hyeokk/catkin_ws/src/data/{LANE_TYPE}_lane.csv")
sensor = env.sensor
env.set_reward_fn(RewardFns.adaptive_speed_lanefollow_reward(sensor))
env.set_episode_over_fn(TerminatedFns.cte_done(sensor, max_cte=0.7))

# ================= 에이전트 로드 =================
if MODEL_TYPE in ["multi", "no_transfer"]:
    agent = MultiCriticPPOAgent((120, 160), 2, ACTION_BOUNDS, log_dir=None)
    agent.load_model(MODEL_DIR, env_id=ENV_ID)
else:
    agent = SingleCriticPPOAgent((120, 160), 2, ACTION_BOUNDS, log_dir=None)
    agent.load_model(MODEL_DIR)

# ================= 테스트 루프 =================
total_rewards = []
episode_lengths = []
success_count = 0

print("===== 테스트 시작 =====")
for ep in range(1, NUM_TEST_EPISODES + 1):
    if stop_flag:
        print("\n[INTERRUPTED] 테스트 중단됨 (Ctrl+C)")
        break

    obs_dict, _ = env.reset()
    total_reward = 0
    steps = 0

    while True:
        if stop_flag:
            print("\n[INTERRUPTED] 현재 에피소드 종료")
            break

        if MODEL_TYPE in ["multi", "no_transfer"]:
            action = agent.get_action(obs_dict, training=False, env_id=ENV_ID)
        else:
            action = agent.get_action(obs_dict, training=False)

        next_obs_dict, reward, done, _, _ = env.step(action)
        total_reward += reward
        steps += 1
        obs_dict = next_obs_dict

        if done:
            break

    total_rewards.append(total_reward)
    episode_lengths.append(steps)
    if steps >= 30:
        success_count += 1

    print(f"Episode {ep:3d} | Reward: {total_reward:7.2f} | Steps: {steps}")

# ================= 결과 출력 =================
print("\n===== 테스트 완료 =====")
print(f"총 에피소드 수 : {len(total_rewards)}")
print(f"평균 보상      : {np.mean(total_rewards):.2f}")
print(f"평균 길이      : {np.mean(episode_lengths):.1f}")
print(f"성공률         : {success_count / len(total_rewards) * 100:.1f}%")

agent.close()
env.close()
