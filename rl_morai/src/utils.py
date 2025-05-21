import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

class Preprocess:
    @staticmethod
    def preprocess_image(image):
        """
        이미지를 CNN 입력에 적합한 (C x H x W) 형태로 전처리하며,
        크기를 최소 (80, 160)으로 리사이즈합니다.
        """
        # 리사이즈 (W, H)
        image = cv2.resize(image, (160, 80))  # 최소 안전 사이즈 보장

        # 정규화
        image = image.astype(np.float32) / 255.0

        # 채널 차원 처리
        if image.ndim == 2:
            image = image[:, :, None]  # H x W x 1

        # HWC → CHW
        image = np.transpose(image, (2, 0, 1))
        return image

class Cal_CTE:
    @staticmethod
    def load_centerline(csv_filename='data.csv'):
        """
        data.csv에서 x, y만 추출
        """
        data_dir = "/home/kuuve/catkin_ws/src/data"
        csv_path = os.path.join(data_dir, csv_filename)

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV 파일 없음: {csv_path}")

        # 전체 열을 불러오되, x: 1번 열, y: 2번 열만 슬라이싱
        data = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=(1, 2))
        return data  # (N, 2) 크기의 numpy 배열: [[x1, y1], [x2, y2], ...]

    @staticmethod
    def calculate_cte(agent_pos, path_points):
        """
        차량 위치와 경로 점들 간의 최소 거리 (CTE) 계산
        """
        if agent_pos is None or len(path_points) == 0:
            return None

        agent_pos = np.array(agent_pos)
        distances = np.linalg.norm(path_points - agent_pos, axis=1)
        return np.min(distances)
    
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