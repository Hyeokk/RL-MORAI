import numpy as np
import gym_morai.envs.morai_sensor
import time
from collections import deque

class TerminatedFns:
    @staticmethod
    def cte_done(sensor, max_cte=0.8, stuck_frames=30, min_velocity=0.1):
        recent_velocities = deque(maxlen=stuck_frames)
        
        def done_fn(obs):
            # 1. CTE 체크
            cte = sensor.cal_cte()
            if cte is not None and cte > max_cte:
                print(f"[DONE] CTE 초과: {cte:.3f} > {max_cte}")
                return True
                
            # 2. 속도 체크 (정지 감지)
            vel = sensor.get_velocity()
            if vel is not None:
                recent_velocities.append(vel)
                
                # 충분한 데이터가 모였을 때
                if len(recent_velocities) == stuck_frames:
                    avg_vel = sum(recent_velocities) / stuck_frames
                    if avg_vel < min_velocity:
                        print(f"[DONE] 정지 감지: 평균 속도 {avg_vel:.3f} < {min_velocity}")
                        return True
            
            return False
        
        return done_fn