import numpy as np
import gym_morai.envs.morai_sensor
import time
from collections import deque

class TerminatedFns:
    @staticmethod
    def cte_done(sensor, max_cte=0.8, stuck_frames=30, min_velocity=0.1, max_steps=2000):
        recent_velocities = deque(maxlen=stuck_frames)
        step_counter = [0]  # 스텝 카운터 (mutable closure)
        
        def done_fn(obs):
            step_counter[0] += 1
            
            # 1. 최대 스텝 수 체크
            if step_counter[0] >= max_steps:
                #print(f"[DONE] 최대 스텝 수 도달: {step_counter[0]} >= {max_steps}")
                step_counter[0] = 0  # 리셋
                return True
                
            # 2. 차선 이탈 체크
            cte = sensor.cal_cte()
            if cte is None:
                return False
                
            if cte > max_cte:
                #print(f"[DONE] 차선 이탈: CTE {cte:.2f} > {max_cte}")
                step_counter[0] = 0  # 리셋
                return True
                
            # 3. 정지 감지
            vel = sensor.get_velocity()
            if vel is not None:
                recent_velocities.append(vel)
                
                # 충분한 데이터가 모였을 때
                if len(recent_velocities) == stuck_frames:
                    avg_vel = sum(recent_velocities) / stuck_frames
                    if avg_vel < min_velocity:
                        #print(f"[DONE] 정지 감지: 평균 속도 {avg_vel:.3f} < {min_velocity}")
                        step_counter[0] = 0  # 리셋
                        return True
            
            return False
        
        return done_fn
    
class CurriculumTermination:
    """
    학습 진행도에 따라 점진적으로 엄격해지는 종료 조건
    
    단계별 커리큘럼:
    1. 초급 (0-200 에피소드): 매우 관대한 조건
    2. 중급 (200-500 에피소드): 표준 조건  
    3. 고급 (500+ 에피소드): 엄격한 조건
    """
    
    @staticmethod
    def progressive_lane_termination(sensor, episode_counter=[0]):
        # 각 단계별 설정
        curriculum_stages = {
            'beginner': {
                'episodes': (0, 100),
                'max_cte': 2.0,           # 매우 관대
                'grace_period': 20,       # 긴 유예 기간
                'violation_threshold': 0.6, # 60% 위반 시 종료
                'description': '초급: 탐험과 기본 주행 학습'
            },
            'intermediate': {
                'episodes': (100, 400),
                'max_cte': 1.5,           # 표준
                'grace_period': 15,       # 중간 유예 기간
                'violation_threshold': 0.7, # 70% 위반 시 종료
                'description': '중급: 안정적인 차선 유지'
            },
            'advanced': {
                'episodes': (400, 1000),
                'max_cte': 1.0,           # 엄격
                'grace_period': 10,       # 짧은 유예 기간
                'violation_threshold': 0.8, # 80% 위반 시 종료
                'description': '고급: 정밀한 차선 중심 주행'
            },
            'expert': {
                'episodes': (1000, float('inf')),
                'max_cte': 0.8,           # 매우 엄격
                'grace_period': 5,        # 매우 짧은 유예 기간
                'violation_threshold': 0.9, # 90% 위반 시 종료
                'description': '전문가: 실제 도로 수준'
            }
        }
        
        # 상태 변수들
        violation_history = deque()
        step_counter = [0]
        last_stage = ['beginner']
        
        def get_current_stage(episode_num):
            for stage_name, config in curriculum_stages.items():
                if config['episodes'][0] <= episode_num < config['episodes'][1]:
                    return stage_name, config
            return 'expert', curriculum_stages['expert']
        
        def done_fn(obs):
            step_counter[0] += 1
            
            # 현재 단계 확인
            current_episode = episode_counter[0]
            stage_name, stage_config = get_current_stage(current_episode)
            
            # 단계 변경 시 알림
            if stage_name != last_stage[0]:
                # print(f"\n🎓 커리큘럼 단계 변경: {last_stage[0]} → {stage_name}")
                # print(f"   {stage_config['description']}")
                # print(f"   Max CTE: {stage_config['max_cte']}, Grace Period: {stage_config['grace_period']}")
                last_stage[0] = stage_name
            
            # 1. 최대 스텝 수 체크 (단계별 조정)
            max_steps = 3000 if stage_name == 'beginner' else 2000
            if step_counter[0] >= max_steps:
                step_counter[0] = 0
                violation_history.clear()
                return True
            
            # 2. CTE 기반 종료 조건
            cte = sensor.cal_cte()
            if cte is None:
                return False
            
            max_cte = stage_config['max_cte']
            grace_period = stage_config['grace_period']
            violation_threshold = stage_config['violation_threshold']
            
            # 유예 기간 동안의 위반 기록
            violation_history.append(cte > max_cte)
            if len(violation_history) > grace_period:
                violation_history.popleft()
            
            # 유예 기간이 찼을 때 종료 조건 체크
            if len(violation_history) == grace_period:
                violation_rate = sum(violation_history) / grace_period
                if violation_rate >= violation_threshold:
                    print(f"[DONE] 차선 이탈 ({stage_name}): 위반율 {violation_rate:.1%} >= {violation_threshold:.1%}")
                    step_counter[0] = 0
                    violation_history.clear()
                    return True
            
            # 3. 속도 기반 종료 (모든 단계 공통)
            vel = sensor.get_velocity()
            if vel is not None and vel < 0.5 and step_counter[0] > 100:
                print(f"[DONE] 정지 감지")
                step_counter[0] = 0
                violation_history.clear()
                return True
            
            return False
        
        return done_fn