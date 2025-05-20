import numpy as np
import gym_morai.envs.morai_sensor
import time
from collections import deque

class TerminatedFns:
    @staticmethod
    def cte_done(sensor, max_cte=0.3):
        """
        CTE(Cross Track Error) 기반으로만 done을 판단하는 간단한 함수
        
        - sensor: MoraiSensor 객체
        - max_cte: 최대 허용 CTE
        """
        def done_fn(obs):
            # CTE 기준으로만 판단
            cte = sensor.cal_cte()
            if cte is not None and cte > max_cte:
                #print(f"[DONE] CTE 초과: {cte:.3f} > {max_cte}")
                return True
                
            return False
        
        return done_fn