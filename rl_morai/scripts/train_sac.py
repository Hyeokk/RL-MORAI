# train.py 파일 수정

import sys
import os
import time
import signal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gym_morai.envs.morai_env import MoraiEnv
from gym_morai.envs.morai_sensor import MoraiSensor
from src.utils import Preprocess
from src.reward_fns import RewardFns
from src.terminated_fns import TerminatedFns
from models.SAC import SACAgent

# 전역 플래그
stop_flag = False

def signal_handler(sig, frame):
    global stop_flag
    print("\n[INFO] KILL NODE")
    stop_flag = True

# 시그널 핸들러 등록
signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # kill

def main():
    global stop_flag
    
    # 절대 경로 설정 - 원하는 경로로 수정하세요
    SAVE_DIR = "/home/kuuve/catkin_ws/src/"  # 이 부분을 원하는 절대 경로로 변경하세요
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    print(f"[INFO] Model PT file root: {SAVE_DIR}")

    env = MoraiEnv()
    sensor = env.sensor
    env.set_reward_fn(RewardFns.lanefollow_cte_reward(sensor))
    env.set_episode_over_fn(TerminatedFns.cte_done(sensor, max_cte=0.8))

    obs, _ = env.reset()
    state = Preprocess.preprocess_image(obs)
    
    state_dim = np.prod(state.shape)
    action_dim = 2
    action_bounds = [(-1, 1), (10.0, 30.0)]

    agent = SACAgent(state_dim, action_dim, action_bounds)

    num_episodes = 1000
    max_steps_per_episode = 1000

    try:
        for episode in range(num_episodes):
            if stop_flag:
                break

            obs, _ = env.reset()
            state = agent.preprocess_image(obs).flatten()
            total_reward = 0

            for step in range(max_steps_per_episode):
                if stop_flag:
                    break

                action = agent.get_action(state)
                next_obs, reward, done, _, _ = env.step(action)
                next_state = agent.preprocess_image(next_obs).flatten()

                agent.store((state, action, reward, next_state, float(done)))
                agent.train()

                state = next_state
                total_reward += reward

                #env.render()

                if done:
                    break

            print(f"Episode {episode+1}, Total Reward: {total_reward:.2f}")

            # 모델 저장 - 지정된 절대 경로에 저장 (SAC 기준)
            if (episode + 1) % 100 == 0:
                actor_path = os.path.join(SAVE_DIR, "sac_actor.pt")
                critic1_path = os.path.join(SAVE_DIR, "sac_critic1.pt")
                critic2_path = os.path.join(SAVE_DIR, "sac_critic2.pt")

                try:
                    torch.save(agent.actor.state_dict(), actor_path)
                    torch.save(agent.critic1.state_dict(), critic1_path)
                    torch.save(agent.critic2.state_dict(), critic2_path)
                    print(f"[INFO] Save Model: {actor_path}, {critic1_path}, {critic2_path}")
                except Exception as e:
                    print(f"[ERROR] Fail Model: {e}")

    except Exception as e:
        print(f"[ERROR] : {e}")

    finally:
        # 마지막 모델 저장 시도
        try:
            final_actor_path = os.path.join(SAVE_DIR, "sac_actor.pt")
            final_critic1_path = os.path.join(SAVE_DIR, "sac_critic1.pt")
            final_critic2_path = os.path.join(SAVE_DIR, "sac_critic2.pt")
            torch.save(agent.actor.state_dict(), final_actor_path)
            torch.save(agent.critic1.state_dict(), final_critic1_path)
            torch.save(agent.critic2.state_dict(), final_critic2_path)
        except Exception as e:
            print(f"[ERROR] Fail Model: {e}")
            
        print("[INFO] RL EVN EXIT")
        env.close()
        print("[INFO] FINISHED")
        sys.exit(0)

if __name__ == "__main__":
    main()