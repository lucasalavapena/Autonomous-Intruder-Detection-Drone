#!/usr/bin/env python
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import cv2
import time

def callback(Image):
    try:
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(Image, "bgr8")

    except CvBridgeError as e:
        print(e)

    cv2.imwrite(str(int(round(time.time())))+".jpg", cv_image)
    rospy.signal_shutdown("Image saved")
def main():
    rospy.init_node('take_picture')
    print("running...")
    image_sub = rospy.Subscriber("/cf1/camera/image_raw", Image, callback)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':
    main()