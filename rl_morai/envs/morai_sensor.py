import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2

from nav_msgs.msg import Odometry  # ← 수정된 메시지 타입

class MoraiSensor:
    def __init__(self):
        self.bridge = CvBridge()
        self.image = None
        self.odom = None
        self.image_subscribed = False
        self.odom_subscribed = False
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        
        rospy.Subscriber('/image_jpeg/compressed', CompressedImage, self.image_callback)
        rospy.Subscriber('/odom', Odometry, self.odom_callback)  # ← 메시지 타입 수정

    def image_callback(self, msg):
        if not self.image_subscribed:
            rospy.loginfo("CAMERA INPUT : /image_jpeg/compressed")
            self.image_subscribed = True

        np_arr = np.frombuffer(msg.data, np.uint8)
        image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.image = self.preprocess_image(image)

    def odom_callback(self, msg):
        if not self.odom_subscribed:
            rospy.loginfo("ODO 데이터 구독 시작: /odom")
            self.odom_subscribed = True

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.odom = (x, y)

    def preprocess_image(self, image):
        image = cv2.resize(image, (320, 240))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray[:, :, None]

    def get_image(self):
        return self.image

    def get_position(self):  # ← 이름 변경 (GPS 아님)
        return self.odom

    def send_control(self, steering, throttle): 
        cmd = Twist()
        cmd.linear.x = throttle
        cmd.angular.z = steering
        self.cmd_vel_pub.publish(cmd)
