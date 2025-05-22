import numpy as np
import gym_morai.envs.morai_sensor
import time
from collections import deque

class TerminatedFns:
    @staticmethod
    def cte_done(sensor, max_cte=0.8, stuck_frames=30, min_velocity=0.1, max_steps=1500):
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