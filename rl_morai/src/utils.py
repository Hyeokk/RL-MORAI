import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

class Cal_CTE:
    @staticmethod
    def load_centerline(csv_path):
        x_list = []
        y_list = []
        with open(csv_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # 헤더 스킵
            for row in reader:
                x = float(row[1])  # 2열: x
                y = float(row[2])  # 3열: y
                x_list.append(x)
                y_list.append(y)
        return np.array(list(zip(x_list, y_list)))  # (N, 2) array

    @staticmethod
    def calculate_cte(agent_pos, path_points):
        agent_pos = np.array(agent_pos)  # tuple -> ndarray
        distances = np.linalg.norm(path_points - agent_pos, axis=1)
        min_index = np.argmin(distances)
        closest_point = path_points[min_index]
        cte = np.linalg.norm(agent_pos - closest_point)
        return cte
    
class Plot:
    @staticmethod
    def save_reward_csv(reward_list, csv_path):
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Episode", "Reward"])
            for i, r in enumerate(reward_list, start=1):
                writer.writerow([i, r])

    @staticmethod
    def plot_rewards(reward_list, save_path=None):
        plt.figure(figsize=(10, 5))
        plt.plot(reward_list, label="Total Reward")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Training Reward Curve")
        plt.grid(True)
        plt.legend()
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    # ✅ 평가 보상 CSV 저장
    @staticmethod
    def save_eval_csv(eval_rewards, csv_path):
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Episode", "EvalReward"])
            for i, r in eval_rewards:
                writer.writerow([i, r])

    # ✅ 평가 보상 그래프
    @staticmethod
    def plot_eval_rewards(eval_rewards, save_path=None):
        episodes = [e for e, _ in eval_rewards]
        rewards = [r for _, r in eval_rewards]

        plt.figure(figsize=(10, 5))
        plt.plot(episodes, rewards, label="Eval Avg Reward", color='orange')
        plt.xlabel("Episode")
        plt.ylabel("Avg Eval Reward")
        plt.title("Evaluation Reward Curve")
        plt.grid(True)
        plt.legend()
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()