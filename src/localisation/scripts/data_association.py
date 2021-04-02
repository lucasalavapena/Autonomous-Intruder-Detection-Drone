#!/usr/bin/env python

import rospy
import tf2_ros
import tf2_geometry_msgs
import numpy as np
from tf.transformations import quaternion_multiply, euler_from_quaternion, quaternion_from_euler
from aruco_msgs.msg import MarkerArray
from geometry_msgs.msg import PoseStamped, TransformStamped, Quaternion
from std_msgs.msg import Bool

transforms = None

def marker_callback(msg):
    global transforms
    transforms = []
    is_localized()
    for m in msg.markers:
        temp = broadcast_transform(m)
        transforms.append(temp)


def marker_identification(m):
    # Find transform of pose of static marker in map
    try:
        t_map = tf_buf.lookup_transform('map', "aruco/marker" + str(m.id), m.header.stamp, rospy.Duration(tf_timeout))
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        return

    # Find transform of pose of detected marker in map
    try:
        detected_map = tf_buf.lookup_transform('map', "aruco/detected" + str(m.id), m.header.stamp, rospy.Duration(tf_timeout))
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        return

    # Compare positions and orientations of detected vs map markers

    orientation_error = 30

    map_roll, map_pitch, map_yaw = euler_from_quaternion(t_map.transform.rotation)
    det_roll, det_pitch, det_yaw = euler_from_quaternion(detected_map.transform.rotation)

    diff_roll = map_roll - det_roll
    diff_pitch = map_pitch - det_pitch
    diff_yaw = map_yaw - det_yaw

    # Calculate resulting rotation between map and odom
    q_r = quaternion_multiply(t_map, detected_map)
    roll, pitch, yaw = euler_from_quaternion((q_r[0], q_r[1], q_r[2], q_r[3]))

    if roll <= orientation_error and pitch <= orientation_error and yaw <= orientation_error:
        return t_map
    else:
        return None

def update_time(t):
    t.header.stamp = rospy.Time.now()
    return t


def is_localized():
    pub.publish(Bool(data=True))


def main():
    rate = rospy.Rate(20)  # Hz
    while not rospy.is_shutdown():
        if transforms is not None:
            for t in transforms:
                if t is not None:
                    br.sendTransform(update_time(t))
                    is_localized()
        rate.sleep()


rospy.init_node('data_association')
tf_buf = tf2_ros.Buffer()
tf_lstn = tf2_ros.TransformListener(tf_buf)
#br = tf2_ros.TransformBroadcaster()

sub_marker = rospy.Subscriber('/aruco/markers', MarkerArray, marker_callback)
pub = rospy.Publisher('localisation/is_localised', Bool, queue_size=10)
tf_timeout = rospy.get_param('~tf_timeout', 0.1)
frame_id = rospy.get_param('~frame_id', 'cf1/odom')


if __name__ == '__main__':
    main()
