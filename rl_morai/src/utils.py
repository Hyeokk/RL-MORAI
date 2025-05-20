import os
import numpy as np

class Preprocess:
    @staticmethod
    def preprocess_image(image):
        """
        이미지를 신경망 입력에 적합한 형태로 전처리합니다.
        
        Parameters:
        - image: 입력 이미지 (HxWx1 또는 HxW)
        
        Returns:
        - 전처리된 이미지 (CxHxW 형태, 값 범위 0-1)
        """
        image = image.astype(np.float32) / 255.0
        if image.ndim == 2:  # grayscale, no channel dim
            image = image[:, :, None]
        image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
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