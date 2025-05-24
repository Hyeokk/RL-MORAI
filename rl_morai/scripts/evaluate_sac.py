import sys
import os
import numpy as np
import signal

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gym_morai.envs.morai_env import MoraiEnv
from src.reward_fns import RewardFns
from src.terminated_fns import TerminatedFns
from models.SAC import SACAgent

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
    env.set_reward_fn(RewardFns.adaptive_speed_lanefollow_reward(sensor))
    env.set_episode_over_fn(TerminatedFns.cte_done(sensor, max_cte=0.8))
    
    # 초기 관측
    obs, _ = env.reset()
    if obs is None:
        print("[ERROR] 환경 초기화 실패")
        return
    
    # 에이전트 생성 및 모델 로드
    agent = SACAgent(obs.shape, 2, action_bounds)
    model_path = "/home/kuuve/catkin_ws/src/pt"
    agent.load_model(model_path)
    
    # 평가 실행
    for episode in range(5):
        if stop_flag:
            break
        obs, _ = env.reset()
        if obs is None:
            continue
            
        total_reward = 0
        steps = 0
        
        for step in range(1000):
            if stop_flag:
                break
            
            action = agent.get_action(obs, training=False)
            obs, reward, done, _, _ = env.step(action)
            
            if obs is None:
                break
                
            total_reward += reward
            steps += 1
            env.render()
            
            if done:
                break
        
        print(f"에피소드 {episode+1}: 보상={total_reward:.2f}, 스텝={steps}")
    
    env.close()

if __name__ == "__main__":
    main()