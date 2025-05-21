import sys
import os
import time
import signal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gym_morai.envs.morai_env import MoraiEnv
from gym_morai.envs.morai_sensor import MoraiSensor
from src.utils import Preprocess
from src.utils import Plot
from src.reward_fns import RewardFns
from src.terminated_fns import TerminatedFns
from models.SAC import SACAgent

# ==== 설정 ====
SAVE_DIR = "/home/kuuve/catkin_ws/src/pt/"
LOG_DIR = "/home/kuuve/catkin_ws/src/rl_morai/log"
REWARD_CSV_PATH = os.path.join(LOG_DIR, "reward_log.csv")
REWARD_PNG_PATH = os.path.join(LOG_DIR, "reward_plot.png")
EVAL_CSV_PATH = os.path.join(LOG_DIR, "eval_log.csv")
EVAL_PNG_PATH = os.path.join(LOG_DIR, "eval_plot.png")

# 학습 하이퍼파라미터
RANDOM_STEPS = 5000       # 초기 랜덤 행동 스텝 수
MIN_BUFFER_SIZE = 1000    # 학습 시작 전 최소 버퍼 크기
BATCH_SIZE = 64           # 배치 크기
INITIAL_ALPHA = 0.2       # 초기 엔트로피 계수
EXPLORATION_NOISE = 0.4   # 탐색 노이즈 크기
AUTO_ENTROPY = True       # 자동 엔트로피 조정 사용
STEPS_PER_EPOCH = 1       # 매 스텝당 학습 횟수

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 시드 설정
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# ==== 종료 플래그 ====
stop_flag = False

def signal_handler(sig, frame):
    global stop_flag
    print("\n[INFO] KILL NODE")
    stop_flag = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ==== 기존 보상 로그 불러오기 ====
episode_rewards = []
eval_rewards = []
if os.path.exists(REWARD_CSV_PATH):
    with open(REWARD_CSV_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episode_rewards.append(float(row["Reward"]))

if os.path.exists(EVAL_CSV_PATH):
    with open(EVAL_CSV_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            eval_rewards.append((int(row["Episode"]), float(row["EvalReward"])))

# ==== 평가 함수 ====
def evaluate_policy(agent, env, episodes=3):
    total_reward = 0.0
    episode_lengths = []
    
    for ep in range(episodes):
        obs, _ = env.reset()
        state = agent.preprocess_image(obs)
        done = False
        ep_reward = 0
        step = 0
        
        while not done:
            action = agent.get_deterministic_action(state)
            obs, reward, done, _, _ = env.step(action)
            state = agent.preprocess_image(obs)
            ep_reward += reward
            step += 1
            
            if step >= 1000:  # 최대 스텝 수 제한
                break
                
        total_reward += ep_reward
        episode_lengths.append(step)
        print(f"[EVAL] Episode {ep+1}: Reward={ep_reward:.2f}, Steps={step}")
        
    avg_reward = total_reward / episodes
    avg_length = sum(episode_lengths) / len(episode_lengths)
    print(f"[EVAL] Average Reward: {avg_reward:.2f}, Average Steps: {avg_length:.1f}")
    return avg_reward

# ==== 메인 루프 ====
def main():
    global stop_flag
    print(f"[INFO] Model PT file root: {SAVE_DIR}")
    print(f"[INFO] Random Steps: {RANDOM_STEPS}, Min Buffer Size: {MIN_BUFFER_SIZE}")

    env = MoraiEnv()
    sensor = env.sensor
    env.set_reward_fn(RewardFns.lanefollow_cte_reward(sensor))
    env.set_episode_over_fn(TerminatedFns.cte_done(sensor, max_cte=0.8))

    obs, _ = env.reset()
    # 여기서 state를 먼저 계산하지 말고, agent를 먼저 초기화합니다

    action_dim = 2
    action_bounds = [(-1, 1), (10.0, 30.0)]

    # SAC 에이전트 초기화
    if obs is None:
        print("[ERROR] 초기 관측값이 None입니다. 기본 입력 형태를 사용합니다.")
        input_shape = (1, 80, 160)  # 기본 이미지 형태 설정
    else:
        state = Preprocess.preprocess_image(obs)
        input_shape = state.shape

    agent = SACAgent(
        input_shape=input_shape,
        action_dim=action_dim,
        action_bounds=action_bounds,
        feature_dim=128,
        gamma=0.99,
        tau=0.005,
        alpha=INITIAL_ALPHA,
        actor_lr=3e-4,
        critic_lr=3e-4,
        buffer_size=100000,
        min_buffer_size=MIN_BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        auto_entropy_tuning=AUTO_ENTROPY
    )

    # 이제 agent가 초기화되었으므로 state를 계산할 수 있습니다
    state = agent.preprocess_image(obs) if obs is not None else None
    if state is None:
        print("[WARN] 초기 상태가 None입니다. 센서 연결을 확인하세요.")
        # 에러 대신 경고만 표시하고 계속 진행

    num_episodes = 15000
    max_steps_per_episode = 10000
    start_episode = len(episode_rewards)
    total_steps = 0
    
    # 랜덤 행동으로 초기 버퍼 채우기
    print(f"[INFO] 초기 랜덤 데이터 수집 중 (목표: {MIN_BUFFER_SIZE}개 샘플)...")
    
    if agent.replay_buffer.size() < MIN_BUFFER_SIZE:
        obs, _ = env.reset()
        state = agent.preprocess_image(obs)
        random_step = 0
        
        while agent.replay_buffer.size() < MIN_BUFFER_SIZE and not stop_flag:
            # 완전 랜덤 행동
            steering = np.random.uniform(-1, 1)
            speed = np.random.uniform(action_bounds[1][0], action_bounds[1][1])
            action = np.array([steering, speed], dtype=np.float32)
            
            next_obs, reward, done, _, _ = env.step(action)
            if next_obs is None:
                print("[WARN] next_obs is None, skipping this step")
                continue
                
            next_state = agent.preprocess_image(next_obs)
            
            # 경험 저장
            agent.store((state, action, reward, next_state, float(done)), reward)
            
            random_step += 1
            state = next_state
            
            if random_step % 100 == 0:
                print(f"[INFO] 랜덤 데이터 수집: {agent.replay_buffer.size()}/{MIN_BUFFER_SIZE} 샘플")
                
            if done or random_step >= 2000:  # 한 에피소드 최대 2000 스텝
                obs, _ = env.reset()
                state = agent.preprocess_image(obs)
                random_step = 0
    
    print(f"[INFO] 초기 데이터 수집 완료. 버퍼 크기: {agent.replay_buffer.size()}")

    try:
        for episode in range(start_episode, num_episodes):
            if stop_flag:
                break
                
            # 에피소드 시작 알림
            agent.start_new_episode()
            
            obs, _ = env.reset()
            state = agent.preprocess_image(obs)
            total_reward = 0
            episode_steps = 0
            training_info = {'critic_loss': 0, 'actor_loss': 0, 'alpha': agent.alpha.item() if agent.auto_entropy_tuning else agent.alpha}
            
            ep_start_time = time.time()
            
            for step in range(max_steps_per_episode):
                if stop_flag:
                    break
                    
                # 액션 선택 (학습 모드)
                action = agent.get_action(state, training=True)
                
                # 환경에 적용
                next_obs, reward, done, _, _ = env.step(action)
                if next_obs is None:
                    print("[WARN] next_obs is None, using previous state")
                    next_state = state
                else:
                    next_state = agent.preprocess_image(next_obs)
                
                # 경험 저장
                agent.store((state, action, reward, next_state, float(done)), reward)
                
                # 학습 단계 (매 스텝마다 STEPS_PER_EPOCH번 학습)
                for _ in range(STEPS_PER_EPOCH):
                    training_info = agent.train()
                
                state = next_state
                total_reward += reward
                episode_steps += 1
                total_steps += 1
                
                if done:
                    break
                    
                # 장시간 에피소드 방지 (10분 이상)
                if time.time() - ep_start_time > 600:
                    print("[WARN] 에피소드 시간 제한 도달, 강제 종료")
                    break

            episode_rewards.append(total_reward)
            
            # 학습 정보 출력
            print(f"Episode {episode+1}/{num_episodes}, "
                  f"총 스텝: {total_steps}, "
                  f"에피소드 스텝: {episode_steps}, "
                  f"보상: {total_reward:.2f}, "
                  f"손실(크리틱/액터): {training_info.get('critic_loss', 0):.4f}/{training_info.get('actor_loss', 0):.4f}, "
                  f"알파: {training_info.get('alpha', 0):.4f}")

            # 모델 저장 (100 에피소드마다 또는 높은 성능 기록 시)
            should_save = (episode + 1) % 100 == 0
            high_score = len(episode_rewards) > 1 and total_reward > max(episode_rewards[:-1])
            
            if should_save or high_score:
                save_id = f"ep{episode+1}" if should_save else "best"
                actor_path = os.path.join(SAVE_DIR, f"sac_actor_{save_id}.pt")
                critic1_path = os.path.join(SAVE_DIR, f"sac_critic1_{save_id}.pt")
                critic2_path = os.path.join(SAVE_DIR, f"sac_critic2_{save_id}.pt")
                try:
                    torch.save(agent.actor.state_dict(), actor_path)
                    torch.save(agent.critic1.state_dict(), critic1_path)
                    torch.save(agent.critic2.state_dict(), critic2_path)
                    print(f"[INFO] 모델 저장 완료: {actor_path}")
                    
                    # 또한 최신 모델 저장
                    torch.save(agent.actor.state_dict(), os.path.join(SAVE_DIR, "sac_actor_latest.pt"))
                    torch.save(agent.critic1.state_dict(), os.path.join(SAVE_DIR, "sac_critic1_latest.pt"))
                    torch.save(agent.critic2.state_dict(), os.path.join(SAVE_DIR, "sac_critic2_latest.pt"))
                except Exception as e:
                    print(f"[ERROR] 모델 저장 실패: {e}")

                # 보상 로그 및 그래프 업데이트
                Plot.save_reward_csv(episode_rewards, REWARD_CSV_PATH)
                Plot.plot_rewards(episode_rewards, REWARD_PNG_PATH)
                
                # 평가 수행 (100 에피소드마다)
                if should_save:
                    print(f"[INFO] 평가 수행 중...")
                    avg_eval_reward = evaluate_policy(agent, env, episodes=3)
                    eval_rewards.append((episode + 1, avg_eval_reward))
                    Plot.save_eval_csv(eval_rewards, EVAL_CSV_PATH)
                    Plot.plot_eval_rewards(eval_rewards, EVAL_PNG_PATH)

    except Exception as e:
        print(f"[ERROR] 예외 발생: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 최종 모델 저장
        try:
            torch.save(agent.actor.state_dict(), os.path.join(SAVE_DIR, "sac_actor_final.pt"))
            torch.save(agent.critic1.state_dict(), os.path.join(SAVE_DIR, "sac_critic1_final.pt"))
            torch.save(agent.critic2.state_dict(), os.path.join(SAVE_DIR, "sac_critic2_final.pt"))
            print("[INFO] 최종 모델 저장 완료")
        except Exception as e:
            print(f"[ERROR] 최종 모델 저장 실패: {e}")

        # 로그 및 그래프 저장
        Plot.save_reward_csv(episode_rewards, REWARD_CSV_PATH)
        Plot.plot_rewards(episode_rewards, REWARD_PNG_PATH)
        Plot.save_eval_csv(eval_rewards, EVAL_CSV_PATH)
        Plot.plot_eval_rewards(eval_rewards, EVAL_PNG_PATH)

        print("[INFO] RL ENV EXIT")
        env.close()
        print("[INFO] FINISHED")
        sys.exit(0)

if __name__ == "__main__":
    main()