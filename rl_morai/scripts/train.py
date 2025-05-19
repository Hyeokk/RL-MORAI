import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from envs.morai_env import MoraiEnv

if __name__ == '__main__':
    env = MoraiEnv()
    obs, _ = env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, done, _, _ = env.step(action)
        env.render()
        if done:
            break
    env.close()
