import numpy as np

class RewardFns:
    @staticmethod
    def lanefollow_cte_reward(sensor, max_cte=1.5):
        def reward_fn(obs):
            cte = sensor.cal_cte()
            if cte is None:
                return 0.0
            return -100.0 if cte > max_cte else 1.0 - (cte / max_cte)
        return reward_fn