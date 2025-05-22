import sys
import os
import signal
import numpy as np
import torch
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gym_morai.envs.morai_env import MoraiEnv
from src.reward_fns import RewardFns
from src.terminated_fns import TerminatedFns
from models.SAC import SACAgent

# 설정
SAVE_DIR = "/home/kuuve/catkin_ws/src/pt/"
os.makedirs(SAVE_DIR, exist_ok=True)

# 시드 설정
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# 종료 플래그
stop_flag = False
def signal_handler(sig, frame):
    global stop_flag
    stop_flag = True
signal.signal(signal.SIGINT, signal_handler)

def main():
    global stop_flag
    
    # 환경 설정
    action_bounds = [(-0.7, 0.7), (10.0, 30.0)]
    env = MoraiEnv(action_bounds=action_bounds)
    
    sensor = env.sensor
    env.set_reward_fn(RewardFns.lanefollow_cte_reward(sensor))
    env.set_episode_over_fn(TerminatedFns.cte_done(sensor, max_cte=0.8))

    # 초기 관측
    obs, _ = env.reset()
    if obs is None:
        print("[ERROR] 환경 초기화 실패")
        sys.exit(1)
    
    # SAC 에이전트 생성
    agent = SACAgent(obs.shape, 2, action_bounds)
    
    num_episodes = 2000
    max_steps_per_episode = 10000

    # 학습 루프
    for episode in range(num_episodes):
        if stop_flag:
            break
            
        obs, _ = env.reset()
        if obs is None:
            continue
            
        total_reward = 0
        episode_steps = 0
        
        for step in range(max_steps_per_episode):
            if stop_flag:
                break
                
            action = agent.get_action(obs, training=True)
            next_obs, reward, done, _, _ = env.step(action)
            
            if next_obs is None:
                break
                
            agent.store((obs, action, reward, next_obs, float(done)))
            agent.train()
            
            obs = next_obs
            total_reward += reward
            episode_steps += 1
            
            if done:
                break

        print(f"에피소드 {episode+1}: 스텝={episode_steps}, 보상={total_reward:.2f}")
        
        # 100 에피소드마다 저장
        if (episode + 1) % 100 == 0:
            agent.save_model(SAVE_DIR)
            print(f"[INFO] 모델 저장 완료")

    # 최종 저장
    agent.save_model(SAVE_DIR)
    env.close()

if __name__ == "__main__":
    main()