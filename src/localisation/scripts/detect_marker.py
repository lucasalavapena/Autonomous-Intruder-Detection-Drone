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
    """
    The detected marker m is a point expressed in cf1/camera_link, therefore no transformation of the point is needed
    and the transform to the detected marker from cf1/camera_link is simply the point, since cf1/camera_link is the
    origin.
    :param m:
    :broadcast t: cf1/camera_link -> aruco/detected[id]
    """

    t = TransformStamped()
    t.header = m.header
    t.child_frame_id = 'aruco/detected' + str(m.id)
    t.transform.translation = m.pose.pose.position
    t.transform.rotation = m.pose.pose.orientation

    br.sendTransform(t)


print("Starting...")
rospy.init_node('detect_marker')
sub_marker = rospy.Subscriber('/aruco/markers', MarkerArray, marker_callback, queue_size=1, buff_size=2**24)
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
