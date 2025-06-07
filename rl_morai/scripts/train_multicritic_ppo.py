import sys
import os
import time
import signal
import numpy as np
import torch
import random
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gym_morai.envs.morai_env import MoraiEnv, press_key
from src.reward_fns import RewardFns
from src.terminated_fns import TerminatedFns
from models.MultiCriticPPO import MultiCriticPPOAgent as PPOAgent

# =============================================================================
# 설정
# =============================================================================
SAVE_DIR = "/home/kuuve/catkin_ws/src/pt/"
ALGO_NAME = "MultiCriticPPO"
LANE_TYPE = "solid"  # "solid", "dashed", "night"
ENV_NUM = 0  # 환경 ID (0:실선, 1:점선, 2:야간)
ENV_ID = 2 # 환경 ID (0:실선, 1:점선, 2:야간)
LOG_DIR = f"/home/kuuve/catkin_ws/src/logs/{ALGO_NAME}_env{ENV_NUM}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
os.makedirs(LOG_DIR, exist_ok=True)

CENTERLINE_CSV_PATH = f"/home/kuuve/catkin_ws/src/data/{LANE_TYPE}_lane.csv"

# 시드 설정
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# 학습 파라미터
NUM_EPISODES = 3000
MAX_STEPS_PER_EPISODE = 500
UPDATE_INTERVAL = 2048
MIN_EPISODE_STEPS = 5
ACTION_BOUNDS = [(-0.4, 0.4), (15.0, 25.0)]
ENV_ID = 0 # 환경 ID (0:실선, 1:점선, 2:야간)

# 종료 플래그
stop_flag = False
def signal_handler(sig, frame):
    global stop_flag
    stop_flag = True
signal.signal(signal.SIGINT, signal_handler)

# =============================================================================
# 유틸리티 함수
# =============================================================================
def is_invalid_episode(episode_steps, min_steps=MIN_EPISODE_STEPS):
    """유효하지 않은 에피소드 판별"""
    return episode_steps <= min_steps

def print_stats(episode, episode_rewards, episode_lengths, total_short_episodes, total_step1_count):
    """통계 출력"""
    if len(episode_rewards) > 0:
        recent_rewards = episode_rewards[-50:] if len(episode_rewards) >= 50 else episode_rewards
        avg_reward = np.mean(recent_rewards)
        avg_length = np.mean(episode_lengths[-50:]) if len(episode_lengths) >= 50 else np.mean(episode_lengths)
        
        print(f"[STATS] Episode {episode}: 최근 평균 - Reward: {avg_reward:.2f}, Length: {avg_length:.1f}")
        print(f"[STATS] 실패 에피소드: Step1={total_step1_count}, 짧은 에피소드={total_short_episodes}")

# =============================================================================
# 메인 훈련 루프
# =============================================================================
def main():
    global stop_flag
    
    print("PPO 강화학습 시작")
    print(f"업데이트 간격: {UPDATE_INTERVAL} steps")
    print(f"최소 에피소드 길이: {MIN_EPISODE_STEPS}")
    print(f"TensorBoard: tensorboard --logdir={LOG_DIR}")
    
    # 환경 초기화
    env = MoraiEnv(action_bounds=ACTION_BOUNDS, csv_path=CENTERLINE_CSV_PATH)
    sensor = env.sensor
    env.set_reward_fn(RewardFns.adaptive_speed_lanefollow_reward(sensor))
    env.set_episode_over_fn(TerminatedFns.cte_done(sensor, max_cte=0.7))

    # 초기 관측 확인
    obs_dict, _ = env.reset()
    if obs_dict is None:
        print("[ERROR] 환경 초기화 실패")
        return

    # 에이전트 생성
    agent = PPOAgent((120, 160), 2, ACTION_BOUNDS, log_dir=LOG_DIR)

    if ENV_ID == 1:
        agent.transfer_critic(from_env_id=0, to_env_id=1)
    elif ENV_ID == 2:
        agent.transfer_critic(from_env_id=1, to_env_id=2)
    else :
        agent.create_critic(env_id=ENV_ID)
        
    # 학습 통계 변수
    episode_rewards = []
    episode_lengths = []
    total_steps = 0
    total_short_episodes = 0
    total_step1_count = 0
    consecutive_short_episodes = 0
    last_update_step = -1
    low_velocity_count = 0
    LOW_VELOCITY_THRESHOLD = 0.5   # 속도 임계값 (m/s)
    LOW_VELOCITY_LIMIT = 5
    
    # =============================================================================
    # 학습 루프
    # =============================================================================
    for episode in range(NUM_EPISODES):
        if stop_flag:
            break

        # 에피소드 초기화
        obs_dict, _ = env.reset()
        if obs_dict is None:
            continue
        
        agent.buffer.episode_reset()
        total_reward = 0
        episode_steps = 0
        
        # 에피소드 실행
        for step in range(MAX_STEPS_PER_EPISODE):
            if stop_flag:
                break
                
            # 액션 선택
            action, log_prob, value = agent.get_action(obs_dict, training=True, env_id=ENV_ID)
            next_obs_dict, reward, done, _, _ = env.step(action)
            
            if next_obs_dict is None:
                break
            
            current_velocity = env.sensor.get_velocity()
            #print(f"[DEBUG] 현재 속도: {current_velocity:.2f} m/s")

            if current_velocity < LOW_VELOCITY_THRESHOLD:
                low_velocity_count += 1
                if low_velocity_count >= LOW_VELOCITY_LIMIT:
                    print(f"[WARN] 후진 또는 저속 감지: {LOW_VELOCITY_LIMIT}회 이상 velocity < {LOW_VELOCITY_THRESHOLD} m/s")

                    press_key('q')  # 자율주행 모드 재시작

                    agent.buffer.remove_last_episode()
                    obs_dict, _ = env.reset()

                    low_velocity_count = 0
                    break
            else:
                low_velocity_count = 0

            if episode_steps > 1:
                is_step1 = agent.store(obs_dict, action, reward, value[0], log_prob[0],
                               done, next_obs_dict if not done else None)
            else:
                is_step1 = True
            
            # Step 1 에피소드 조기 감지
            if is_step1 and done:
                total_step1_count += 1
                break
            
            # 상태 업데이트
            obs_dict = next_obs_dict
            total_reward += reward
            episode_steps += 1
            total_steps += 1
            
            # PPO 업데이트
            if total_steps % UPDATE_INTERVAL ==0 and total_steps != last_update_step:
                print(f"[UPDATE] Step {total_steps}에서 PPO 업데이트 수행")
                train_metrics = agent.train(env_id=ENV_ID)
                last_update_step = total_steps
                if train_metrics:
                    print(f"  손실 - Actor: {train_metrics['actor_loss']:.4f}, "
                          f"Critic: {train_metrics['critic_loss']:.4f}, "
                          f"Entropy: {train_metrics['entropy']:.4f}")
                if obs_dict is None:
                    continue
                agent.buffer.clear()
                break
            
            if done:
                break

        if is_invalid_episode(episode_steps):
            consecutive_short_episodes += 1
            total_short_episodes += 1
            if episode_steps == 1:
                total_step1_count += 1
            
            print(f"무효 에피소드! Episode {episode+1}: {episode_steps}스텝 "
                  f"(연속 {consecutive_short_episodes}회)")

            agent.buffer.remove_last_episode()
            total_steps = max(0, total_steps - episode_steps)

            # 일정 횟수 이상 연속되었을 때만 환경 리셋
            if consecutive_short_episodes >= 3:
                print(f"[WARN] 연속 {consecutive_short_episodes}회 무효 에피소드 → 환경 리셋")
                press_key('q')
                obs_dict, _ = env.reset()
                consecutive_short_episodes = 0

            continue
        else:
            consecutive_short_episodes = 0
            
            # 정상 에피소드 기록
            episode_rewards.append(total_reward)
            episode_lengths.append(episode_steps)
            
            print(f"Episode {episode+1:4d}: Steps={episode_steps:3d}, "
                  f"Reward={total_reward:7.2f}, Total Steps={total_steps}")
            
            # TensorBoard 로깅
            agent.log_episode_metrics(episode+1, total_reward, episode_steps, total_steps)

        # 모델 저장
        if (episode + 1) % 100 == 0:
            agent.save_model(SAVE_DIR, env_id=ENV_ID)
            print(f"[SAVE] 모델 저장 완료 (Episode {episode+1})")

    # =============================================================================
    # 학습 완료
    # =============================================================================
    agent.save_model(SAVE_DIR, env_id=ENV_ID)
    
    print("PPO 학습 완료!")
    print(f"총 에피소드: {episode + 1}")
    print(f"정상 에피소드: {len(episode_rewards)}개")
    print(f"실패 에피소드: {total_short_episodes}개 (Step1: {total_step1_count})")
    if len(episode_rewards) > 0:
        print(f"최종 평균 보상: {np.mean(episode_rewards[-100:]):.2f}")
        print(f"성공률: {len(episode_rewards)/(episode+1)*100:.1f}%")
    print(f"TensorBoard: tensorboard --logdir={LOG_DIR}")
    
    # 정리
    agent.close()
    env.close()

if __name__ == "__main__":
    main()