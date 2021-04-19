#!/usr/bin/env python

import rospy
import tf2_ros
import tf2_geometry_msgs
from aruco_msgs.msg import MarkerArray
from geometry_msgs.msg import Transform, PoseStamped, TransformStamped


def marker_callback(msg):
    for marker in msg.markers:
        broadcast_marker_transform(marker)


def broadcast_marker_transform(m):

    test = TransformStamped()
    test.header = m.header
    test.child_frame_id = 'aruco/detected' + str(m.id)
    test.transform.translation = m.pose.pose.position
    test.transform.rotation = m.pose.pose.orientation

    br.sendTransform(test)


print("Starting...")
rospy.init_node('detect_marker')
sub_marker = rospy.Subscriber('/aruco/markers', MarkerArray, marker_callback)
tf_buf = tf2_ros.Buffer()
tf_lstn = tf2_ros.TransformListener(tf_buf)
br = tf2_ros.TransformBroadcaster()
tf_timeout = rospy.get_param('~tf_timeout', 0.1)
frame_id = rospy.get_param('~frame_id', 'cf1/camera_link')
print("Ready")


def main():
    rospy.spin()


if __name__ == '__main__':
    main()
