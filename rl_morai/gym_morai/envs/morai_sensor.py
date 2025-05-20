import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage
from nav_msgs.msg import Odometry
from morai_msgs.msg import CtrlCmd
from cv_bridge import CvBridge
import cv2
from src.utils import Cal_CTE

class MoraiSensor:
    def __init__(self):
        self.bridge = CvBridge()
        self.image = None
        self.odom = None
        self.image_subscribed = False
        self.odom_subscribed = False
        self.cmd_vel_pub = rospy.Publisher('/ctrl_cmd', CtrlCmd, queue_size=1)

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

    def preprocess_image(self, image):
        image = cv2.resize(image, (160, 80))  # 환경과 맞춤
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray[:, :, None]  # (H, W, 1)

    def get_image(self):
        return self.image

    def get_position(self):
        return self.odom
    
    def cal_cte(self, csv_path='data.csv'):
        # 현재 위치 받기
        agent_pos = self.get_position()
        if agent_pos is None:
            #rospy.logwarn("현재 위치 정보를 아직 수신하지 못했습니다.")
            return None

        # 중심선 불러오기
        data, xy_path = Cal_CTE.load_centerline(csv_path)  # 또는 '/mnt/data/data.csv' (위치에 따라 조정)

        # CTE 계산
        cte = Cal_CTE.calculate_cte(agent_pos, xy_path)
        #rospy.loginfo(f"현재 CTE: {cte:.3f}")

        return cte

    def send_control(self, steering, throttle):
        cmd = CtrlCmd()
        cmd.longlCmdType= 2
        cmd.velocity= throttle
        cmd.steering = steering
        self.cmd_vel_pub.publish(cmd)
