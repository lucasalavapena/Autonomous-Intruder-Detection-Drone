#!/usr/bin/env python

import rospy
import tf2_ros
import tf2_geometry_msgs
import numpy as np
from tf.transformations import quaternion_multiply, quaternion_conjugate
from aruco_msgs.msg import MarkerArray
from geometry_msgs.msg import Transform, PoseStamped, TransformStamped, Quaternion, Vector3


def marker_callback(msg):
    for marker in msg.markers:
        broadcast_marker(marker)


def broadcast_marker(m):
    # Find transform of pose of detected marker in odom
    if not tf_buf.can_transform(frame_id, m.header.frame_id, m.header.stamp, rospy.Duration(tf_timeout)):
        rospy.logwarn_throttle(5.0, 'No transform from %s to %s', m.header.frame_id, frame_id)
        return
    marker = tf_buf.transform(PoseStamped(header=m.header, pose=m.pose.pose), frame_id)

    # find transform of pose of static marker in map
    try:
        t_map = tf_buf.lookup_transform('map', "aruco/marker" + str(m.id), m.header.stamp, rospy.Duration(tf_timeout))
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        return

    # Create new message with time stamps and frames
    t_map_odom = TransformStamped()
    t_map_odom.header = marker.header
    t_map_odom.header.frame_id = 'map'
    t_map_odom.child_frame_id = frame_id

    # Find rotation between map and odom
    # Inverse of marker orientation
    #q_marker_inv = [0] * 4
    #q_marker_inv[0] = marker.pose.orientation.x
    #q_marker_inv[1] = marker.pose.orientation.y
    #q_marker_inv[2] = marker.pose.orientation.z
    #q_marker_inv[3] = -marker.pose.orientation.w

    # rotation of static marker in map
    #q_trans = [0] * 4
    #q_trans[0] = t_map.transform.rotation.x
    #q_trans[1] = t_map.transform.rotation.y
    #q_trans[2] = t_map.transform.rotation.z
    #q_trans[3] = t_map.transform.rotation.w

    # Calculate resulting rotation between map and odom
    #q_r = quaternion_multiply(q_trans, q_marker_inv)
    #t_quat = Quaternion(q_r[0], q_r[1], q_r[2], q_r[3]) # change format of quaternion to fit message type
    #t_map_odom.transform.rotation = t_quat

    # Save position of detected marker in different format to work with quaternion_multiply
    #v_marker = [0] * 4
    #v_marker[0] = marker.pose.position.x
    #v_marker[1] = marker.pose.position.y
    #v_marker[2] = marker.pose.position.z
    #v_marker[3] = 0 # to match length of quaternion vector

    # Define rotation quaternion to use on detected marker position TO BE CHANGED
    #q_map_marker = [0] * 4
    #q_map_marker[0] = 0#t_map.transform.rotation.x
    #q_map_marker[1] = 0#t_map.transform.rotation.y
    #q_map_marker[2] = 0#t_map.transform.rotation.z
    #q_map_marker[3] = 1#t_map.transform.rotation.w

    # Caculate rotated detected marker position in map(?) frame
    #v_res = qv_mult(q_r, v_marker)

    # Calculate resulting position vector, map->odom
    #t_map_odom.transform.translation.x = v_res[0] - t_map.transform.translation.x
    #t_map_odom.transform.translation.y = v_res[1] - t_map.transform.translation.y
    #t_map_odom.transform.translation.z = v_res[2] - t_map.transform.translation.z

    # Find rotation between map and odom
    # Rotation of the detected marker in odom
    rotation_odom_marker = PyKDL.Rotation.Quaternion(marker.pose.orientation)

    # Rotation of a static marker in the map
    rotation_map_marker = PyKDL.Rotation.Quaternion(t_map.transform.rotation)

    transform = rotation_odom_marker * rotation_map_marker.Inverse()
    (x, y, z, w) = transform.GetQuaternion()  # Get quaternion result
    # t_map_odom.transform.rotation = rotation_odom_marker * rotation_map_marker.Inverse()

    # Calculate resulting position vector, map->odom
    t_map_odom.transform.translation.x = x - t_map.transform.translation.x
    t_map_odom.transform.translation.y = y - t_map.transform.translation.y
    t_map_odom.transform.translation.z = z - t_map.transform.translation.z

    br.sendTransform(t_map_odom)


def qv_mult(q1, v1):
    # v_new = (q1 * v1) * q1_conjugate
    return quaternion_multiply(
        quaternion_multiply(q1, v1),
        quaternion_conjugate(q1)
    )[:3] # remove extra value from vector to match xyz

def main():
    rospy.spin()


if __name__ == '__main__':
    rospy.init_node('broadcast_map_odom')

    tf_buf = tf2_ros.Buffer()
    tf_lstn = tf2_ros.TransformListener(tf_buf)
    br = tf2_ros.TransformBroadcaster()
    sub_marker = rospy.Subscriber('/aruco/markers', MarkerArray, marker_callback)
    tf_timeout = rospy.get_param('~tf_timeout', 0.1)
    frame_id = rospy.get_param('~frame_id', 'cf1/odom')
    main()
