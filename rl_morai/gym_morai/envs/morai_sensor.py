import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage
from nav_msgs.msg import Odometry
from morai_msgs.msg import CtrlCmd
from cv_bridge import CvBridge
import cv2
import time
from src.utils import Cal_CTE

class MoraiSensor:
    def __init__(self):
        self.bridge = CvBridge()
        self.image = None
        self.odom = None
        self.image_subscribed = False
        self.odom_subscribed = False
        self.cmd_vel_pub = rospy.Publisher('/ctrl_cmd', CtrlCmd, queue_size=1)

        # 속도 계산을 위한 간단한 변수들 추가
        self.last_position = None
        self.last_time = None
        self.current_velocity = 0.0

        rospy.Subscriber('/image_jpeg/compressed', CompressedImage, self.image_callback)
        rospy.Subscriber('/odom', Odometry, self.odom_callback)

    def image_callback(self, msg):
        if not self.image_subscribed:
            rospy.loginfo("CAMERA INPUT : /image_jpeg/compressed")
            self.image_subscribed = True

        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.image = self.preprocess_image(image)
        except Exception as e:
            rospy.logerr(f"Image processing error: {e}")
            self.image = None

    def preprocess_image(self, image):
        if image is None:
            return None
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # RGB을 Grayscale로 변환
        resized = cv2.resize(gray, (160, 120), interpolation=cv2.INTER_AREA)
        
        normalized = resized.astype(np.float32) / 255.0 # CNN 입력 데이터로 변환
        
        return normalized[:, :, np.newaxis]

    def odom_callback(self, msg):
        if not self.odom_subscribed:
            rospy.loginfo("ODOM DATA: /odom")
            self.odom_subscribed = True

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.odom = (x, y)

        # 간단한 속도 계산
        current_time = rospy.Time.now().to_sec()
        current_position = (x, y)
        
        if self.last_position is not None and self.last_time is not None:
            dt = current_time - self.last_time
            if dt > 0.01:  # 10ms 이상일 때만 계산
                dx = current_position[0] - self.last_position[0]
                dy = current_position[1] - self.last_position[1]
                distance = np.sqrt(dx**2 + dy**2)
                self.current_velocity = distance / dt
                
                # 속도 제한 (비현실적인 값 방지)
                if self.current_velocity > 50.0:
                    self.current_velocity = 50.0
        
        self.last_position = current_position
        self.last_time = current_time

    def get_velocity(self):
        """현재 속도 반환 (m/s)"""
        return self.current_velocity if self.current_velocity is not None else 0.0

    def cal_cte(self, csv_path='/home/kuuve/catkin_ws/src/data/data.csv'):
        agent_pos = self.get_position()
        if agent_pos is None:
            rospy.logwarn("Agent position not available")
            return None

        # 경로 불러오기
        xy_path = Cal_CTE.load_centerline(csv_path)
        
        # CTE 계산
        cte = Cal_CTE.calculate_cte(agent_pos, xy_path)
        
        return cte

    def get_image(self):
        return self.image

    def get_position(self):
        return self.odom

    def send_control(self, steering, throttle):
        cmd = CtrlCmd()
        cmd.longlCmdType = 2
        cmd.velocity = throttle
        cmd.steering = steering
        self.cmd_vel_pub.publish(cmd)