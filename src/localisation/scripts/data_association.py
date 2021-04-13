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
    for m in msg.markers:
        temp = marker_identification(m)
        pub.publish(temp)


def marker_identification(m):
    best_marker = None
    best_delta = 100

    # Find transform of pose of detected marker in map
    try:
        detected_map = tf_buf.lookup_transform('map', "aruco/detected" + str(m.id), m.header.stamp, rospy.Duration(tf_timeout))
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        print('return')
        return

    more_markers = True
    n = 0
    while more_markers:
        # Find transform of pose of static marker in map
        marker_name = "aruco/marker" + str(m.id) + '_' + str(n)
        try:
            t_map = tf_buf.lookup_transform('map', marker_name, m.header.stamp, rospy.Duration(tf_timeout))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            print('exception')
            break
        print('here')
        print(t_map)
        n += 1
        print(t_map)
        # Compare positions and orientations of detected vs map markers
        orientation_error = 30

        # Calculate resulting rotation between map and odom
        q_r = quaternion_multiply(t_map, detected_map)
        d_roll, d_pitch, d_yaw = euler_from_quaternion((q_r[0], q_r[1], q_r[2], q_r[3]))
        delta = np.sqrt(d_roll**2 + d_pitch**2 + d_yaw**2)
        print(delta)

        if np.abs(d_roll) <= orientation_error \
                and np.abs(d_pitch) <= orientation_error \
                and np.abs(d_yaw) <= orientation_error:
            if best_marker is None:
                best_marker = t_map
                best_delta = delta
            elif delta < best_delta:
                best_marker = t_map
                best_delta = delta
    print(best_marker)
    return best_marker


def update_time(t):
    t.header.stamp = rospy.Time.now()
    return t


def main():
    rospy.spin()


rospy.init_node('data_association')
tf_buf = tf2_ros.Buffer()
tf_lstn = tf2_ros.TransformListener(tf_buf)

sub_marker = rospy.Subscriber('/aruco/markers', MarkerArray, marker_callback)
pub = rospy.Publisher('/DA/best_marker', TransformStamped, queue_size=10)
tf_timeout = rospy.get_param('~tf_timeout', 0.1)


if __name__ == '__main__':
    main()
