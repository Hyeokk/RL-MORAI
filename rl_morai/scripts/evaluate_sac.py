import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gym_morai.envs.morai_env import MoraiEnv
from models.SAC import SACAgent

def main():
    # 환경 설정
    action_bounds = [(-0.7, 0.7), (10.0, 30.0)]
    env = MoraiEnv(action_bounds=action_bounds)
    
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
        obs, _ = env.reset()
        if obs is None:
            continue
            
        total_reward = 0
        steps = 0
        
        for step in range(1000):
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