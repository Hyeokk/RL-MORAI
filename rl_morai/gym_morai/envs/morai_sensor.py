import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage
from nav_msgs.msg import Odometry
from morai_msgs.msg import CtrlCmd
from cv_bridge import CvBridge
import cv2
import time
from collections import deque
from src.utils import Cal_CTE

class MoraiSensor:
    def __init__(self):
        self.bridge = CvBridge()
        self.image = None
        self.odom = None
        self.image_subscribed = False
        self.odom_subscribed = False
        self.cmd_vel_pub = rospy.Publisher('/ctrl_cmd', CtrlCmd, queue_size=1)

        # 속도 계산을 위한 변수 추가
        self.position_history = deque(maxlen=10)  # 최근 10개 위치 저장
        self.timestamp_history = deque(maxlen=10)  # 위치에 대응하는 타임스탬프
        self.last_velocity = None  # 마지막으로 계산된 속도

        rospy.Subscriber('/image_jpeg/compressed', CompressedImage, self.image_callback)
        rospy.Subscriber('/odom', Odometry, self.odom_callback)

    def image_callback(self, msg):
        if not self.image_subscribed:
            rospy.loginfo("CAMERA INPUT : /image_jpeg/compressed")
            self.image_subscribed = True

        np_arr = np.frombuffer(msg.data, np.uint8)
        image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.image = self.preprocess_image(image)

    def odom_callback(self, msg):
        if not self.odom_subscribed:
            rospy.loginfo("ODOM DATA: /odom")
            self.odom_subscribed = True

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.odom = (x, y)
        
        # 위치와 시간 기록
        current_time = rospy.Time.now().to_sec()
        self.position_history.append((x, y))
        self.timestamp_history.append(current_time)
        
        # 속도 계산 (최소 2개 이상의 데이터가 있을 때)
        if len(self.position_history) >= 2:
            self._calculate_velocity()

    def _calculate_velocity(self):
        """
        최근 위치 데이터를 사용하여 속도 계산
        """
        # 최신 위치와 시간
        latest_pos = self.position_history[-1]
        latest_time = self.timestamp_history[-1]
        
        # 이전 위치와 시간
        prev_pos = self.position_history[-2]
        prev_time = self.timestamp_history[-2]
        
        # 시간 간격
        dt = latest_time - prev_time
        
        # 시간 간격이 너무 작으면 오차가 커질 수 있음
        if dt < 0.01:  # 10ms 미만이면 계산 생략
            return
        
        # 거리 계산
        dx = latest_pos[0] - prev_pos[0]
        dy = latest_pos[1] - prev_pos[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        # 속도 계산 (m/s)
        self.last_velocity = distance / dt
        
        # 노이즈 필터링 (갑작스러운 변화 방지)
        if self.last_velocity > 50.0:  # 비현실적으로 큰 속도 제한
            self.last_velocity = 50.0
            
    def get_velocity(self):
        """
        계산된 속도 반환
        """
        return self.last_velocity

    def preprocess_image(self, image):
        image = cv2.resize(image, (160, 80))  # 환경과 맞춤
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray[:, :, None]  # (H, W, 1)

    def get_image(self):
        return self.image

    def get_position(self):
        return self.odom
    
    def wait_for_valid_position(self, timeout=2.0):
        """
        최신 position 수신 전까지 최대 timeout초 대기
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            pos = self.get_position()
            if pos is not None and isinstance(pos, (list, tuple, np.ndarray)):
                return pos
            time.sleep(0.05)
        #print("[WARN] 유효한 position 수신 실패")
        return None

    def cal_cte(self, csv_path='data.csv'):
        #agent_pos = self.get_position()  # UTM 좌표 [x, y]
        agent_pos = self.wait_for_valid_position()
        if agent_pos is None:
            return None

        xy_path = Cal_CTE.load_centerline(csv_path)
        cte = Cal_CTE.calculate_cte(agent_pos, xy_path)
        #print("CTE:", cte)
        return cte

    def send_control(self, steering, throttle):
        self.last_steering = steering
        cmd = CtrlCmd()
        cmd.longlCmdType= 2
        cmd.velocity= throttle
        cmd.steering = steering
        self.cmd_vel_pub.publish(cmd)