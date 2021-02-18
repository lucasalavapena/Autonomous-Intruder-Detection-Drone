#!/usr/bin/env python

import rospy
import tf2_ros
from geometry_msgs.msg import Transform, PoseStamped, TransformStamped


def main():
    print("Starting...")
    rospy.init_node('broadcast_map_odom')

    tf_buf = tf2_ros.Buffer()
    tf_lstn = tf2_ros.TransformListener(tf_buf)
    br = tf2_ros.TransformBroadcaster()
    tf_timeout = rospy.get_param('~tf_timeout', 0.1)

    rate = rospy.Rate(10.0)
    print("Ready")
    while not rospy.is_shutdown():
        time = rospy.Time(0)
        try:
            t_detected = tf_buf.lookup_transform("aruco/detected2", 'cf1/odom', time, rospy.Duration(tf_timeout))
            print(t_detected)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rate.sleep()
            continue

        try:
            t_map = tf_buf.lookup_transform('map', "aruco/marker2", time, rospy.Duration(tf_timeout))
            #print(t_map)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rate.sleep()
            continue

        t_map_odom = TransformStamped()
        t_map_odom.header = t_map.header
        t_map_odom.child_frame_id = "cf1/odom"
        t_map_odom.transform.translation.x = t_map.transform.translation.x - t_detected.transform.translation.x
        t_map_odom.transform.translation.y = t_map.transform.translation.y - t_detected.transform.translation.y
        t_map_odom.transform.translation.z = t_map.transform.translation.z - t_detected.transform.translation.z
        t_map_odom.transform.rotation = t_map.transform.rotation
        #print(t_map_odom)
        br.sendTransform(t_map_odom)


if __name__ == '__main__':
    main()
