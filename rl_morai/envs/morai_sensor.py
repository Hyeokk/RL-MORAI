import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage, NavSatFix
from cv_bridge import CvBridge
import cv2

class MoraiSensor:
    def __init__(self):
        self.bridge = CvBridge()
        self.image = None
        self.gps = None
        self.image_subscribed = False
        self.gps_subscribed = False

        rospy.Subscriber('/image_jpeg/compressed', CompressedImage, self.image_callback)
        rospy.Subscriber('/odom', NavSatFix, self.gps_callback)

    def image_callback(self, msg):
        if not self.image_subscribed:
            rospy.loginfo("CAMERA INPUT : /image_jpeg/compressed")
            self.image_subscribed = True

        np_arr = np.frombuffer(msg.data, np.uint8)
        image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.image = self.preprocess_image(image)

    def gps_callback(self, msg):
        if not self.gps_subscribed:
            rospy.loginfo("GPS 데이터 구독 시작: /odom")
            self.gps_subscribed = True

        self.gps = (msg.latitude, msg.longitude)

    def preprocess_image(self, image):
        image = cv2.resize(image, (320, 240))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray[:, :, None]

    def get_image(self):
        return self.image

    def get_gps(self):
        return self.gps
