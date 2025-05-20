import os
import numpy as np

class Preprocess:
    def preprocess_image(image):
        image = image.astype(np.float32) / 255.0
        if image.ndim == 2:  # grayscale, no channel dim
            image = image[:, :, None]
        image = np.transpose(image, (2, 0, 1))
        return image

class Cal_CTE:
    @staticmethod
    def load_centerline(csv_filename='data.csv'):
        """
        중심선 데이터가 저장된 CSV 파일을 로드합니다.
        
        Parameters:
        - csv_filename: CSV 파일 이름
        
        Returns:
        - data: 전체 데이터 배열
        - xy_path: x, y 좌표만 추출된 배열
        """
        data_dir = "/home/kuuve/catkin_ws/src/data"
        csv_path = os.path.join(data_dir, csv_filename)

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"[Cal_CTE] NO CSV: {csv_path}")

        # 헤더 없이 데이터 로드 (첫 번째 행부터 데이터로 가정)
        data = np.loadtxt(csv_path, delimiter=',', skiprows=0)
        
        # x, y 좌표 추출 (1, 2번 열)
        xy_path = data[:, 1:3]
        
        #print(f"[Cal_CTE] Loaded centerline with {len(xy_path)} points from {csv_path}")
        return data, xy_path

    @staticmethod
    def calculate_cte(agent_pos, centerline, debug=False):
        """
        차량 위치와 경로 사이의 최소 거리(CTE)를 계산합니다.
        
        Parameters:
        - agent_pos: 차량의 현재 위치 (x, y)
        - centerline: 경로 점들의 배열 [(x1, y1), (x2, y2), ...]
        - debug: 디버깅 출력 활성화 여부
        
        Returns:
        - cte: 차량과 경로 사이의 최소 거리
        """
        if agent_pos is None or len(centerline) < 2:
            if debug:
                #print("[Cal_CTE] Invalid inputs: agent_pos or centerline")
                return None
            
        agent_pos = np.array(agent_pos)
        
        # 1. 가장 가까운 경로 점 찾기
        distances = np.sqrt(np.sum((centerline - agent_pos)**2, axis=1))
        closest_idx = np.argmin(distances)
        
        # 2. 검색 윈도우 설정 (가장 가까운 점 주변의 세그먼트만 검사)
        search_window = 10  # 앞뒤로 검사할 세그먼트 수
        start_idx = max(0, closest_idx - search_window)
        end_idx = min(len(centerline) - 2, closest_idx + search_window)
        
        # 3. 각 선분에 대해 차량-선분 거리 계산
        min_cte = float('inf')
        min_segment_idx = start_idx
        
        for i in range(start_idx, end_idx + 1):
            p1 = centerline[i]
            p2 = centerline[i + 1]
            cte = Cal_CTE.point_to_segment_distance(agent_pos, p1, p2)
            
            if cte < min_cte:
                min_cte = cte
                min_segment_idx = i
        
        if debug:
            # print(f"[Cal_CTE] Agent pos: {agent_pos}")
            # print(f"[Cal_CTE] Closest point idx: {closest_idx}")
            # print(f"[Cal_CTE] Min segment: {min_segment_idx}-{min_segment_idx+1}")
            # print(f"[Cal_CTE] CTE: {min_cte:.4f}")
            return min_cte

    @staticmethod
    def point_to_segment_distance(p, v, w):
        """
        점과 선분 사이의 최단 거리를 계산합니다.
        
        Parameters:
        - p: 점 좌표 (x, y)
        - v: 선분의 시작점 (x1, y1)
        - w: 선분의 끝점 (x2, y2)
        
        Returns:
        - distance: 점과 선분 사이의 최단 거리
        """
        # 선분의 길이 제곱
        l2 = np.sum((w - v) ** 2)
        
        # 선분이 사실상 점인 경우
        if l2 == 0.0:
            return np.linalg.norm(p - v)
            
        # 선분 위 투영점의 매개변수 t 계산 (0 <= t <= 1)
        t = np.clip(np.dot(p - v, w - v) / l2, 0.0, 1.0)
        
        # 투영점 계산
        projection = v + t * (w - v)
        
        # 점과 투영점 사이의 거리 계산
        return np.linalg.norm(p - projection)