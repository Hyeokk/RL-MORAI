import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.DDPG import DDPGAgent
from gym_morai.envs.morai_env import MoraiEnv
from utils.utils import preprocess_image

def main():
    env = MoraiEnv()
    obs, _ = env.reset()
    state = preprocess_image(obs)
    
    state_dim = np.prod(state.shape)
    
    action_dim = 2
    action_bounds = [(-1.0, 1.0), (0.0, 1.0)]

    agent = DDPGAgent(state_dim, action_dim, action_bounds)

    num_episodes = 1000
    max_steps_per_episode = 1000

    for episode in range(num_episodes):
        obs, _ = env.reset()
        state = agent.preprocess_image(obs).flatten()

        total_reward = 0

        for step in range(max_steps_per_episode):
            action = agent.get_action(state)  # 연속적인 액션 선택
            next_obs, reward, done, _, _ = env.step(action)
            next_state = agent.preprocess_image(next_obs).flatten()

            # 경험 저장 및 학습
            agent.store((state, action, reward, next_state, float(done)))
            agent.train()

            state = next_state
            total_reward += reward

            env.render()
            if done:
                break

        print(f"Episode {episode+1}, Total Reward: {total_reward:.2f}")

        # 모델 저장
        if (episode + 1) % 100 == 0:
            torch.save(agent.actor.state_dict(), "ddpg_actor.pt")
            torch.save(agent.critic.state_dict(), "ddpg_critic.pt")

    env.close()

if __name__ == "__main__":
    main()