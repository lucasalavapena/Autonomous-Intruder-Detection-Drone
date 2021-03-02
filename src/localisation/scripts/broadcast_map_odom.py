#!/usr/bin/env python

import rospy
import tf2_ros
import tf2_geometry_msgs
import numpy as np
from tf.transformations import quaternion_multiply, euler_from_quaternion
from aruco_msgs.msg import MarkerArray
from geometry_msgs.msg import PoseStamped, TransformStamped, Quaternion


def marker_callback(msg):
    for marker in msg.markers:
        broadcast_marker(marker)


def broadcast_marker(m):
    # Find transform of pose of detected marker in odom
    if not tf_buf.can_transform(frame_id, m.header.frame_id, m.header.stamp, rospy.Duration(tf_timeout)):
        rospy.logwarn_throttle(5.0, 'No transform from %s to %s', m.header.frame_id, frame_id)
        return
    marker = tf_buf.transform(PoseStamped(header=m.header, pose=m.pose.pose), 'cf1/odom')

    # Find transform of pose of static marker in map
    try:
        t_map = tf_buf.lookup_transform('map', "aruco/marker" + str(m.id), m.header.stamp, rospy.Duration(tf_timeout))
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        return

    # Create new message with time stamps and frames
    t = TransformStamped()
    t.header = marker.header
    t.header.frame_id = 'map'
    t.child_frame_id = frame_id

    # Find rotation between map and odom
    # Inverse of marker orientation
    q_marker_inv = [0] * 4
    q_marker_inv[0] = marker.pose.orientation.x
    q_marker_inv[1] = marker.pose.orientation.y
    q_marker_inv[2] = marker.pose.orientation.z
    q_marker_inv[3] = -marker.pose.orientation.w

    # rotation of static marker in map
    q_trans = [0] * 4
    q_trans[0] = t_map.transform.rotation.x
    q_trans[1] = t_map.transform.rotation.y
    q_trans[2] = t_map.transform.rotation.z
    q_trans[3] = t_map.transform.rotation.w

    # Calculate resulting rotation between map and odom
    q_r = quaternion_multiply(q_trans, q_marker_inv)
    roll, pitch, yaw = euler_from_quaternion((q_r[0], q_r[1], q_r[2], q_r[3]))
    t.transform.rotation = Quaternion(q_r[0], q_r[1], q_r[2], q_r[3])

    #  Calculate the translation vector
    t1 = t_map.transform.translation.x - np.cos(yaw)*marker.pose.position.x + np.sin(yaw)*marker.pose.position.y
    t2 = t_map.transform.translation.y - np.sin(yaw) * marker.pose.position.x - np.cos(yaw) * marker.pose.position.y

    #  Add values to transform
    t.transform.translation.x = t1
    t.transform.translation.y = t2
    t.transform.translation.z = marker.pose.position.z
    br.sendTransform(t)


def main():
    rospy.spin()


if __name__ == '__main__':
    rospy.init_node('broadcast')

    tf_buf = tf2_ros.Buffer()
    tf_lstn = tf2_ros.TransformListener(tf_buf)
    br = tf2_ros.TransformBroadcaster()
    sub_marker = rospy.Subscriber('/aruco/markers', MarkerArray, marker_callback)
    tf_timeout = rospy.get_param('~tf_timeout', 0.1)
    frame_id = rospy.get_param('~frame_id', 'cf1/odom')
    main()
