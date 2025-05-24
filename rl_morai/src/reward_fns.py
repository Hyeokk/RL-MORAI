import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gym_morai.envs.morai_sensor import MoraiSensor

class RewardFns:
    @staticmethod
    def adaptive_speed_lanefollow_reward(sensor, max_cte=0.8, max_speed=30.0):
        """
        곡선에서는 속도를 줄이고, 직선에서는 빠르게 가도록 유도하는 보상 함수
        """
        last_steering = [0.0]
        last_cte = [None]
        
        def reward_fn(obs):
            cte = sensor.cal_cte()
            if cte is None:
                return 0.0
                
            velocity = sensor.get_velocity() or 0.0
            current_steering = getattr(sensor, 'last_steering', 0.0)
            steering_change = abs(current_steering - last_steering[0])
            
            # 1. 기본 차선 유지 보상
            lane_reward = (max_cte - cte) / max_cte if cte <= max_cte else 0.0
            
            # 🔥 2. 적응적 속도 보상 (핵심!)
            
            # 방법 1: 조향각 기반 목표 속도 조정
            abs_steering = abs(current_steering)
            if abs_steering < 0.1:  # 거의 직선
                target_speed = max_speed * 0.9  # 90% 속도
            elif abs_steering < 0.3:  # 약간의 곡선
                target_speed = max_speed * 0.7  # 70% 속도
            else:  # 급격한 곡선
                target_speed = max_speed * 0.5  # 50% 속도
                
            # 목표 속도에 따른 보상 (빠를수록 좋되, 목표 초과 시 페널티)
            if velocity <= target_speed:
                speed_reward = velocity / target_speed  # 0~1.0
            else:
                speed_reward = max(0.0, 1.0 - (velocity - target_speed) / target_speed)
            
            # 방법 2: CTE 변화율 기반 속도 조정 (추가 보완)
            cte_change_penalty = 0.0
            if last_cte[0] is not None:
                cte_change = abs(cte - last_cte[0])
                if cte_change > 0.1 and velocity > max_speed * 0.6:  # CTE가 빠르게 변하는데 빠른 속도
                    cte_change_penalty = -0.5
            last_cte[0] = cte
            
            # 3. 조향 안정성 보상
            steering_reward = max(0.0, 1.0 - steering_change / 0.5)
            last_steering[0] = current_steering
            
            # 총 보상 계산
            total = (lane_reward * 1.5 + 
                    speed_reward * 1.0 + 
                    steering_reward * 0.5 + 
                    cte_change_penalty)
            
            return max(total, 0.0)
        
        return reward_fn

class CurriculumReward:
    @staticmethod
    def progressive_lane_reward(sensor, episode_counter=[0]):
        """
        학습 단계에 따라 보상 구조가 변화하는 함수
        """
        
        step_count = [0]
        
        def get_reward_weights(episode_num):
            """에피소드 수에 따른 보상 가중치 조정"""
            if episode_num < 200:
                # 초급: 생존과 기본 주행에 집중
                return {
                    'survival': 2.0,    # 생존 보상 높음
                    'cte': 1.0,         # CTE 보상 보통
                    'speed': 0.5,       # 속도 보상 낮음
                    'precision': 0.0    # 정밀도 보상 없음
                }
            elif episode_num < 500:
                # 중급: 안정적인 차선 유지
                return {
                    'survival': 1.5,    # 생존 보상 감소
                    'cte': 2.0,         # CTE 보상 증가
                    'speed': 1.0,       # 속도 보상 증가
                    'precision': 0.5    # 정밀도 보상 추가
                }
            else:
                # 고급: 정밀한 차선 중심 주행
                return {
                    'survival': 1.0,    # 생존 보상 기본
                    'cte': 3.0,         # CTE 보상 최대
                    'speed': 1.5,       # 속도 보상 높음
                    'precision': 2.0    # 정밀도 보상 높음
                }
        
        def reward_fn(obs):
            step_count[0] += 1
            current_episode = episode_counter[0]
            weights = get_reward_weights(current_episode)
            
            # 기본 센서 데이터
            cte = sensor.cal_cte()
            if cte is None:
                return -1.0
            
            velocity = sensor.get_velocity() or 0.0
            
            # 1. 생존 보상
            survival_reward = weights['survival']
            
            # 2. CTE 보상 (단계별 조정)
            if current_episode < 200:
                # 초급: 관대한 CTE 보상
                max_cte = 2.0
                cte_reward = max(0.0, (max_cte - cte) / max_cte) * weights['cte']
            elif current_episode < 500:
                # 중급: 표준 CTE 보상
                max_cte = 1.5
                cte_reward = max(0.0, (max_cte - cte) / max_cte) * weights['cte']
            else:
                # 고급: 엄격한 CTE 보상
                max_cte = 1.0
                cte_reward = max(0.0, (max_cte - cte) / max_cte) * weights['cte']
            
            # 3. 속도 보상
            target_speed = 20.0
            speed_diff = abs(velocity - target_speed)
            speed_reward = max(0.0, 1.0 - speed_diff / target_speed) * weights['speed']
            
            # 4. 정밀도 보상 (고급 단계에서만)
            precision_reward = 0.0
            if weights['precision'] > 0:
                # 매우 정확한 차선 중심 주행에 대한 보너스
                if cte < 0.3:  # 30cm 미만
                    precision_reward = weights['precision']
                elif cte < 0.5:  # 50cm 미만
                    precision_reward = weights['precision'] * 0.5
            
            # 5. 장기 생존 보너스
            longevity_bonus = min(1.0, step_count[0] / 200.0) * 0.5
            
            total_reward = (survival_reward + cte_reward + speed_reward + 
                           precision_reward + longevity_bonus)
            
            return total_reward
        
        return reward_fn