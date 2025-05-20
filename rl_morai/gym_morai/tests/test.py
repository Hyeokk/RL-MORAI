import rospy
import cv2
from morai_sensor import MoraiSensor

def main():
    rospy.init_node('render_test_node')
    sensor = MoraiSensor()

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        img = sensor.get_image()
        if img is not None:
            cv2.imshow("Camera", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        rate.sleep()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
