import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2

class MoraiSensor:
    def __init__(self):
        self.bridge = CvBridge()
        self.image = None
        self.odom = None
        self.image_subscribed = False
        self.odom_subscribed = False
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

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
            rospy.loginfo("ODO 데이터 구독 시작: /odom")
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

    def send_control(self, steering, throttle):
        cmd = Twist()
        cmd.linear.x = throttle
        cmd.angular.z = steering
        self.cmd_vel_pub.publish(cmd)
