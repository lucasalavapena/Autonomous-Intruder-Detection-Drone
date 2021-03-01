#!/usr/bin/env python

import rospy
import tf2_ros
import tf2_geometry_msgs
from std_msgs.msg import Bool


def is_localised_callback(msg):
    if msg.data is True:
        print('Drone is localised. Safe to fly.')


def main():
    sub = rospy.Subscriber('localisation/is_localised', Bool, is_localised_callback)
    rospy.spin()


if __name__ == "__main__":
    rospy.init_node('brain')
    main()
