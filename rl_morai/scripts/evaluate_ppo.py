import sys
import os
import numpy as np
import signal
import time
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gym_morai.envs.morai_env import MoraiEnv
from src.reward_fns import RewardFns
from src.terminated_fns import TerminatedFns
from models.PPO import PPOAgent  # SAC에서 PPO로 변경

# 종료 플래그
stop_flag = False
def signal_handler(sig, frame):
    global stop_flag
    stop_flag = True
signal.signal(signal.SIGINT, signal_handler)

def main():
    global stop_flag
    
    # 환경 설정 (학습 시와 동일하게)
    action_bounds = [(-0.3, 0.3), (12.0, 25.0)]  # train_ppo.py와 동일한 설정으로 변경
    env = MoraiEnv(action_bounds=action_bounds)
    sensor = env.sensor
    #env.set_reward_fn(RewardFns.adaptive_speed_lanefollow_reward(sensor))
    #env.set_episode_over_fn(TerminatedFns.cte_done(sensor, max_cte=0.8))
    
    # 초기 관측
    obs_dict, _ = env.reset()
    if obs_dict is None:
        print("[ERROR] 환경 초기화 실패")
        return
    
    # PPO 에이전트 생성 및 모델 로드
    agent = PPOAgent((120, 160), 2, action_bounds)  # PPO 에이전트로 변경
    model_path = "/home/kuuve/catkin_ws/src/pt"
    
    try:
        agent.load_model(model_path)
        print(f"[INFO] PPO 모델 로드 완료: {model_path}")
    except Exception as e:
        print(f"[ERROR] 모델 로드 실패: {e}")
        return
    
    # 평가 설정
    num_episodes = 10  # 평가할 에피소드 수
    max_steps_per_episode = 2000  # 에피소드당 최대 스텝
    
    # 평가 통계
    episode_rewards = []
    episode_lengths = []
    episode_velocities = []
    
    print("="*50)
    print("PPO 모델 평가 시작")  # 메시지 변경
    print("="*50)
    
    # 평가 실행
    for episode in range(num_episodes):
        if stop_flag:
            break
            
        obs_dict, _ = env.reset()
        if obs_dict is None:
            print(f"[WARNING] 에피소드 {episode+1} 초기화 실패, 재시도...")
            time.sleep(1)
            continue
            
        total_reward = 0
        steps = 0
        velocities = []
        
        print(f"\n에피소드 {episode+1}/{num_episodes} 시작...")
        
        for step in range(max_steps_per_episode):
            if stop_flag:
                break
            
            # 평가 모드로 액션 선택 (training=False)
            action = agent.get_action(obs_dict, training=False)
            
            # 환경 스텝
            next_obs_dict, reward, done, _, _ = env.step(action)
            
            if next_obs_dict is None:
                print(f"[WARNING] 스텝 {step}에서 관측값 없음")
                break
                
            # 통계 수집
            total_reward += reward
            steps += 1
            velocities.append(sensor.get_velocity() * 3.6)  # km/h
            
            # 시각화
            env.render()
            
            # 주기적 상태 출력
            if step % 100 == 0:
                current_velocity = sensor.get_velocity() * 3.6
                print(f"  스텝 {step:4d}: 속도={current_velocity:5.1f}km/h, "
                      f"조향={action[0]:5.2f}, 누적보상={total_reward:7.2f}")
            
            obs_dict = next_obs_dict
            
            if done:
                break
        
        # 에피소드 통계
        avg_velocity = np.mean(velocities) if velocities else 0
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        episode_velocities.append(avg_velocity)
        
        print(f"\n에피소드 {episode+1} 완료:")
        print(f"  - 총 보상: {total_reward:.2f}")
        print(f"  - 스텝 수: {steps}")
        print(f"  - 평균 속도: {avg_velocity:.1f} km/h")
        
        # 짧은 에피소드 경고
        if steps < 50:
            print(f"  [WARNING] 짧은 에피소드 감지 ({steps} steps)")
    
    # 전체 평가 결과
    print("\n" + "="*50)
    print("평가 완료 - 전체 통계")
    print("="*50)
    
    if episode_rewards:
        print(f"총 {len(episode_rewards)}개 에피소드 완료")
        print(f"평균 보상: {np.mean(episode_rewards):.2f} (±{np.std(episode_rewards):.2f})")
        print(f"평균 에피소드 길이: {np.mean(episode_lengths):.1f} (±{np.std(episode_lengths):.1f})")
        print(f"평균 속도: {np.mean(episode_velocities):.1f} km/h (±{np.std(episode_velocities):.1f})")
        print(f"최고 보상: {np.max(episode_rewards):.2f}")
        print(f"최저 보상: {np.min(episode_rewards):.2f}")
        
        # 성공률 계산 (100스텝 이상을 성공으로 가정)
        success_rate = sum(1 for l in episode_lengths if l >= 100) / len(episode_lengths) * 100
        print(f"성공률 (100+ 스텝): {success_rate:.1f}%")
    else:
        print("[ERROR] 평가된 에피소드가 없습니다.")
    
    # 환경 종료
    env.close()
    cv2.destroyAllWindows()
    print("\n평가 종료")

if __name__ == "__main__":
    main()