#!/usr/bin/env python

import sys
import math
import json
import numpy as np

import rospy
import tf2_ros 
import tf2_geometry_msgs
from tf.transformations import quaternion_from_euler, quaternion_multiply, quaternion_matrix
from geometry_msgs.msg import TransformStamped, Vector3, PoseStamped, Quaternion
from std_msgs.msg import String
from aruco_msgs.msg import MarkerArray

SIGN_INFO = None

def callback(msg):
    global SIGN_INFO
    SIGN_INFO = parser(msg)

def parser(str):
    data = str.data.split(',')
    label = data[0]
    stamp = rospy.Time(nsecs=int(data[1]))
    frame = data[2]
    position = np.fromstring(data[3])
    orientation = np.fromstring(data[4])

    pose = PoseStamped()
    pose.header.stamp = stamp
    pose.header.frame_id = frame
    pose.pose.position.x = position[0]
    pose.pose.position.y = position[1]
    pose.pose.position.z = position[2]
    # TODO maybe math.radians in quat from euler?
    print("orientation[0]", orientation[0])


    (pose.pose.orientation.x,
     pose.pose.orientation.y,
     pose.pose.orientation.z,
     pose.pose.orientation.w) = quaternion_from_euler((orientation[0]),
                                                        (orientation[1]),
                                                        (orientation[2]))

    # TODO try to apply Ryz
    # print("pose.pose.orientation.x", pose.pose.orientation.x)
    # quat_og = quaternion_matrix([pose.pose.orientation.x, pose.pose.orientation.y,
    #                      pose.pose.orientation.z, pose.pose.orientation.w])
    # R_yz = quaternion_matrix([0.4469983, -0.4469983, 0.7240368, 0.2759632])
    # quat_fixed = quaternion_multiply(quat_og, R_yz)
    # print("quat_fixed[0]", quat_fixed[0][0])
    # pose.pose.orientation.x = quat_fixed[0, 0]
    # pose.pose.orientation.y = quat_fixed[0, 1]
    # pose.pose.orientation.z = quat_fixed[0, 2]
    # pose.pose.orientation.w = quat_fixed[0, 3]
    # pose.pose.orientation = quat_fixed.copy()

    return [pose, label]

def transform_sign(m, label):

    timeout = rospy.Duration(0.5)

    # TODO: change this to a while loop?
    if not tf_buf.can_transform(m.header.frame_id, 'map', m.header.stamp, timeout):
        rospy.logwarn_throttle(5.0, 'No transform from %s to map' % m.header.frame_id)
        return

    m_pose = PoseStamped()
    m_pose.pose = m.pose
    m_pose.header = m.header

    m_trans = tf_buf.transform(m_pose, 'map')

    t = TransformStamped()
    t.header.frame_id = m_trans.header.frame_id
    t.header.stamp = m_trans.header.stamp
    t.child_frame_id = "landmark/detected_" + label
    x = m_trans.pose.position.x
    y = m_trans.pose.position.y
    z = m_trans.pose.position.z
    t.transform.translation = Vector3(*[x, y, z])
    t.transform.rotation = m_trans.pose.orientation
    return t


rospy.init_node('sign_publisher')
sign_sub = rospy.Subscriber('/sign_poses', String, callback)
broadcaster = tf2_ros.TransformBroadcaster()
tf_buf   = tf2_ros.Buffer()
tf_lstn  = tf2_ros.TransformListener(tf_buf)

def main():
    rate = rospy.Rate(10)  # Hz
    while not rospy.is_shutdown():
        if SIGN_INFO:
            sign_transform = transform_sign(SIGN_INFO[0], SIGN_INFO[1])
            # print(sign_transform)
            if sign_transform:
                broadcaster.sendTransform(sign_transform)
                print("detected sign transform sent")
        rate.sleep()

if __name__ == "__main__":
    main()