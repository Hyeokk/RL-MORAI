import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gym_morai.envs.morai_sensor import MoraiSensor

class RewardFns:
    @staticmethod
    def lanefollow_cte_reward(sensor, max_cte=1.5, max_speed=30.0):
        last_steer = [0.0]  # 조향 기록 (mutable closure)
        last_cte = [None]   # 이전 CTE 기록
        steps = [0]         # 스텝 카운터

        def reward_fn(obs):
            # 스텝 카운터 증가
            steps[0] += 1
            
            # 1. CTE 계산 (차선 중앙으로부터의 거리)
            cte = sensor.cal_cte()
            if cte is None:
                # CTE를 계산할 수 없는 경우 (차선 인식 실패 등)
                if last_cte[0] is not None:
                    cte = last_cte[0]  # 이전 값 재사용
                else:
                    cte = max_cte * 0.5  # 기본값 사용
            
            if cte > max_cte:
                return -2.0  # 차선 이탈 시 큰 페널티
                
            # CTE 변화량 계산 (안정성 평가)
            cte_change = 0.0
            if last_cte[0] is not None:
                cte_change = abs(cte - last_cte[0])
            last_cte[0] = cte

            # 2. 속도 계산
            velocity = sensor.get_velocity()
            if velocity is None:
                velocity = 0.0

            # 3. 조향 변화량 계산 (부드러운 주행 유도)
            current_steer = sensor.last_steering if hasattr(sensor, 'last_steering') else 0.0
            steer_change = abs(current_steer - last_steer[0])
            last_steer[0] = current_steer  # 업데이트

            # 보상 구성 요소 계산
            # 가. 차선 중앙 유지 보상 (가우시안 형태로 중앙에 가까울수록 높은 보상)
            cte_term = np.exp(-(cte**2) / (max_cte**2 * 0.25))  # 중앙에 가까울수록 1에 가까움
            
            # 나. 속도 보상 (적절한 속도 유지 장려)
            ideal_speed = max_speed * 0.7  # 적정 속도 (max의 70%)
            speed_term = 1.0 - min(1.0, abs(velocity - ideal_speed) / ideal_speed)
            
            # 다. 조향 안정성 보상 (급격한 조향 변화 방지)
            steer_penalty = max(0.0, 1.0 - steer_change / 0.2)  # 조향 변화 0~0.2 기준
            
            # 라. CTE 변화 안정성 (급격한 CTE 변화 방지)
            cte_stability = max(0.0, 1.0 - cte_change / 0.1)  # CTE 변화 0~0.1 기준
            
            # 마. 장기적 경로 유지 보상 (에피소드 길이에 비례)
            survival_bonus = min(2.0, steps[0] / 1000)  # 최대 2.0
            
            # 보상 가중치 조정
            w_cte = 0.5       # 차선 중앙 유지
            w_speed = 0.2     # 속도 유지
            w_steer = 0.1     # 조향 안정성
            w_stability = 0.1 # CTE 안정성
            w_survival = 0.1  # 생존 보너스
            
            # 총 보상 계산
            reward = (w_cte * cte_term + 
                     w_speed * speed_term + 
                     w_steer * steer_penalty + 
                     w_stability * cte_stability + 
                     w_survival * survival_bonus)
            
            return reward

        return reward_fn