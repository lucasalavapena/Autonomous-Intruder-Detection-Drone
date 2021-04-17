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
    #rospy.sleep(0.1)

    if not tf_buf.can_transform(frame_id, m.header.frame_id, m.header.stamp, rospy.Duration(tf_timeout)):
        rospy.logwarn_throttle(5.0, 'detect_marker: No transform from %s to %s', m.header.frame_id, frame_id)
        return

    marker = tf_buf.transform(PoseStamped(header=m.header, pose=m.pose.pose), frame_id)

    t = TransformStamped()
    t.header = marker.header
    t.child_frame_id = 'aruco/detected' + str(m.id)
    t.transform = Transform(translation=marker.pose.position, rotation=marker.pose.orientation)

    br.sendTransform(t)


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
