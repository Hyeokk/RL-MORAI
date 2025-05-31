#!/usr/bin/env python3
"""
Multi-Critic PPO 학습 스크립트 (간결 버전)
"""

import sys
import os
import argparse

# 모듈 임포트
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Multi_PPO import MultiCriticPPO
from utils import TrainingSession, EnvironmentManager

def main():
    """메인 학습 함수"""
    
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description='Multi-Critic PPO Training')
    parser.add_argument('--environment', '-e', type=str, default='auto',
                       choices=['auto', 'solid', 'dashed', 'dash', 'shadow'],
                       help='학습할 환경 지정')
    parser.add_argument('--episodes', type=int, default=None,
                       help='학습 에피소드 수 (기본값: 모델 내 설정)')
    parser.add_argument('--save_dir', type=str, default="/home/kuuve/catkin_ws/src/pt/",
                       help='모델 저장 디렉토리')
    parser.add_argument('--log_dir', type=str, default=None,
                       help='로그 저장 디렉토리')
    
    args = parser.parse_args()
    
    # 환경 설정
    manual_env = args.environment if args.environment != 'auto' else None
    
    # 학습 세션 초기화
    session = TrainingSession(
        manual_env=manual_env,
        save_dir=args.save_dir,
        log_dir=args.log_dir
    )
    
    session.print_experiment_info()
    
    try:
        # MORAI 환경 초기화
        action_bounds = [(-0.4, 0.4), (15.0, 25.0)]
        env, sensor = EnvironmentManager.setup_environment(action_bounds)
        
        # Multi-Critic PPO 에이전트 생성
        agent = MultiCriticPPO()
        
        # 에피소드 수 오버라이드
        if args.episodes:
            agent.num_episodes = args.episodes
        
        # 로깅 시스템 초기화
        session.setup_logging(agent)
        session.env_manager.print_mode_info()
        
        # 초기 관측 확인
        obs_dict, _ = env.reset()
        if obs_dict is None:
            print("[ERROR] 환경 초기화 실패")
            return
        
        print(f"📋 학습 설정:")
        print(f"   - 총 에피소드: {agent.num_episodes}")
        print(f"   - 업데이트 간격: {agent.update_interval} steps")
        print(f"   - 최대 스텝/에피소드: {agent.max_steps_per_episode}")
        print(f"   - 최소 에피소드 길이: {agent.min_episode_steps}")
        print()
        
        # =================================================================
        # 메인 학습 루프
        # =================================================================
        for episode in range(agent.num_episodes):
            if session.should_stop():
                break
            
            # 에피소드 초기화
            obs_dict, _ = env.reset()
            if obs_dict is None:
                continue
            
            agent.episode_reset()
            total_reward = 0
            episode_steps = 0
            
            # 에피소드 실행
            for step in range(agent.max_steps_per_episode):
                if session.should_stop():
                    break
                
                # 환경 라벨 획득
                env_label = session.env_manager.get_environment_label(obs_dict, sensor)
                
                # 액션 선택
                action, log_prob, value, detected_env = agent.get_action(obs_dict, training=True)
                
                # 환경 스텝
                next_obs_dict, reward, done, _, _ = env.step(action)
                if next_obs_dict is None:
                    break
                
                # 경험 저장
                is_short_episode = agent.store_experience(
                    obs_dict, action, reward, 
                    value[0] if value is not None else 0.0, 
                    log_prob[0] if log_prob is not None else 0.0, 
                    done, next_obs_dict if not done else None, 
                    env_label, detected_env
                )
                
                # 짧은 에피소드 조기 종료
                if is_short_episode and done:
                    session.stats.add_short_episode(episode_steps + 1)
                    break
                
                # 상태 업데이트
                obs_dict = next_obs_dict
                total_reward += reward
                episode_steps += 1
                session.stats.total_steps += 1
                
                # PPO 업데이트
                if session.stats.total_steps % agent.update_interval == 0:
                    print(f"[UPDATE] Step {session.stats.total_steps}에서 PPO 업데이트 수행")
                    train_metrics = agent.train()
                    if train_metrics:
                        print(f"  손실 - Actor: {train_metrics['actor_loss']:.4f}, "
                              f"Critic: {train_metrics['critic_loss']:.4f}, "
                              f"Classifier: {train_metrics['classifier_loss']:.4f}")
                        
                        # 학습 지표 로깅
                        session.log_training_metrics(session.stats.total_steps, train_metrics, manual_env)
                
                if done:
                    break
            
            # 에피소드 후처리
            if session.stats.is_invalid_episode(episode_steps, agent.min_episode_steps):
                if episode_steps > 1:
                    session.stats.add_short_episode(episode_steps)
                
                print(f"무효 에피소드! Episode {episode+1}: {episode_steps}스텝 "
                      f"(연속 {session.stats.consecutive_short_episodes}회)")
                
                # 환경 강제 리셋 (연속 실패 시)
                if session.stats.consecutive_short_episodes >= 5:
                    env, sensor = EnvironmentManager.force_reset_environment(env, action_bounds)
                    agent.buffer.clear()
                    session.stats.total_steps = max(0, session.stats.total_steps - episode_steps)
                    session.stats.reset_consecutive_count()
                continue
            else:
                session.stats.reset_consecutive_count()
                
                # 정상 에피소드 기록
                final_env_label = session.env_manager.get_environment_label(obs_dict, sensor)
                session.stats.add_episode(total_reward, episode_steps, final_env_label)
                
                env_name = session.env_manager.get_environment_name(final_env_label)
                print(f"Episode {episode+1:4d}: Steps={episode_steps:3d}, "
                      f"Reward={total_reward:7.2f}, Env={env_name}")
                
                if manual_env:
                    env_label_num = session.env_manager.env_name_to_label.get(manual_env.lower(), 0)
                    print(f"  → 학습 환경: {manual_env} (Critic_{env_label_num})")
                
                # 에피소드 지표 로깅
                session.log_episode_metrics(episode+1, total_reward, episode_steps, final_env_label, manual_env)
            
            # 주기적 로깅
            if episode % agent.log_interval == 0:
                session.stats.print_stats(episode+1, manual_env)
            
            # 주기적 평가
            if episode % agent.eval_interval == 0 and episode > 0:
                session.analyze_performance()
            
            # 모델 저장
            if (episode + 1) % agent.save_interval == 0:
                save_path = session.get_save_path()
                agent.save_model(save_path)
                print(f"[SAVE] 모델 저장: {save_path}")
        
        # =================================================================
        # 학습 완료 처리
        # =================================================================
        
        # 최종 모델 저장
        final_save_path = session.get_save_path(final=True)
        agent.save_model(final_save_path)
        
        # 최종 요약 출력
        session.print_final_summary(episode + 1)
        
    except KeyboardInterrupt:
        print("\n학습이 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n학습 중 오류 발생: {e}")
        raise
    finally:
        # 리소스 정리
        session.cleanup(env)

if __name__ == "__main__":
    main()