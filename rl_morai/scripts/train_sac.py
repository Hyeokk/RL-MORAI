import sys
import os
import time
import signal
import numpy as np
import torch
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gym_morai.envs.morai_env import MoraiEnv
from src.reward_fns import RewardFns
from src.terminated_fns import TerminatedFns
from models.SAC import SACAgent
from src.reward_fns import CurriculumReward
from src.terminated_fns import CurriculumTermination

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

def force_reset_environment(env, action_bounds):
    """환경 강제 재초기화"""
    try:
        print("[RESET] 환경 강제 재초기화 중...")
        env.close()
        time.sleep(2)
        
        new_env = MoraiEnv(action_bounds=action_bounds)
        sensor = new_env.sensor
        new_env.set_reward_fn(RewardFns.adaptive_speed_lanefollow_reward(sensor))
        new_env.set_episode_over_fn(TerminatedFns.cte_done(sensor, max_cte=0.8))  # 좀 더 완화
        
        print("[RESET] 환경 재초기화 완료")
        return new_env
    except Exception as e:
        print(f"[ERROR] 환경 재초기화 실패: {e}")
        return env

def main():
    global stop_flag
    
    # 🔧 보수적 설정 (조향각 축소, CTE 완화)
    action_bounds = [(-0.4, 0.4), (15.0, 25.0)]  # 조향각 축소
    env = MoraiEnv(action_bounds=action_bounds)
    
    sensor = env.sensor
    env.set_reward_fn(RewardFns.adaptive_speed_lanefollow_reward(sensor))
    env.set_episode_over_fn(TerminatedFns.cte_done(sensor, max_cte=0.8))  # 0.8 → 1.5 완화

    # 초기 관측
    obs_dict, _ = env.reset()
    if obs_dict is None:
        print("[ERROR] 환경 초기화 실패")
        sys.exit(1)
    
    image_shape = obs_dict['image'].shape[:2]  # (120, 160)
    vector_dim = len(obs_dict['velocity']) + len(obs_dict['steering'])  # 2

    # SAC 에이전트 생성
    agent = SACAgent((120, 160), 2, action_bounds)
    
    num_episodes = 10000
    max_steps_per_episode = 10000
    
    # Step 1 방어 변수
    consecutive_step1 = 0
    total_step1_count = 0

    # 학습 루프
    for episode in range(num_episodes):
        if stop_flag:
            break

        obs_dict, _ = env.reset()
        if obs_dict is None:
            continue
            
        total_reward = 0
        episode_steps = 0
        episode_data = []  # 에피소드 데이터 임시 저장
        
        for step in range(max_steps_per_episode):
            if stop_flag:
                break
                
            action = agent.get_action(obs_dict, training=True)
            next_obs_dict, reward, done, _, _ = env.step(action)
            
            if next_obs_dict is None:
                break
                
            # 데이터를 임시로 저장 (바로 replay buffer에 넣지 않음)
            episode_data.append((obs_dict, action, reward, next_obs_dict, float(done)))
            
            obs_dict = next_obs_dict
            total_reward += reward
            episode_steps += 1
            #env.render()
            
            if done:
                break

        # Step 1 방어 로직
        if episode_steps == 1:
            consecutive_step1 += 1
            total_step1_count += 1
            
            print(f"Step 1 감지! Episode {episode+1} (연속 {consecutive_step1}회, 총 {total_step1_count}회)")
            print("   → 데이터 버림 (replay buffer 저장 안함)")
            
            # 연속 3회 이상이면 환경 강제 리셋
            if consecutive_step1 >= 3:
                print(f"   → 연속 {consecutive_step1}회 발생, 환경 강제 재초기화")
                env = force_reset_environment(env, action_bounds)
                consecutive_step1 = 0
                time.sleep(1)  # 안정화 대기
            
        else:
            consecutive_step1 = 0  # 성공하면 연속 카운터 리셋
            
            print(f"Episode {episode+1:4d}: Steps={episode_steps:3d}, Reward={total_reward:7.2f}")
            
            # 정상 데이터만 저장 및 학습
            for transition in episode_data:
                agent.store(transition)
                agent.train()

        # 짧은 에피소드 버퍼 정리 (정상 에피소드만 대상)
        if episode_steps > 1 and episode_steps < 10:
            if agent.replay_buffer.size() > 2000:
                agent.replay_buffer.clear_bad_patterns(threshold_reward=1.0)
                print(f"[INFO] 버퍼 정리 완료")
        
        # 100 에피소드마다 저장
        if (episode + 1) % 100 == 0:
            agent.save_model(SAVE_DIR)
            print(f"[INFO] 모델 저장 완료 (Episode {episode+1})")
            print(f"[STATS] 총 Step 1 발생: {total_step1_count}회")

    # 최종 저장
    agent.save_model(SAVE_DIR)
    print(f"\n[FINAL] 학습 완료! 총 Step 1 발생: {total_step1_count}회")
    env.close()

if __name__ == "__main__":
    main()