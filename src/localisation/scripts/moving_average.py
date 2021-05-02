#!/usr/bin/env python

import time
import math
import rospy
import tf2_ros
import tf2_geometry_msgs
import numpy as np
from geometry_msgs.msg import TransformStamped
from tf.transformations import euler_from_quaternion, quaternion_from_euler


x_vec = []
y_vec = []
yaw_vec = []
x = None
y = None
rot = None


def pose_callback(msg):
    global x_vec, y_vec, yaw_vec, x, y, rot
    p, q = transform_stamped_to_pq(msg)
    roll, pitch, yaw = euler_from_quaternion(q)

    # if x and np.abs(p[0]-x) < 0.2 and np.abs(p[1]-y) < 0.2 and np.abs(yaw-rot) < np.pi/12:
    #     print('here')
    #     x_vec.append(p[0])
    #     y_vec.append(p[1])
    #     yaw_vec.append(yaw)
    # elif not x:
    #     x_vec.append(p[0])
    #     y_vec.append(p[1])
    #     yaw_vec.append(yaw)

    x_sum = 0
    y_sum = 0
    rot_sum = 0

    if len(x_vec) >= 10:
        for n in range(len(x_vec)):
            x_sum = x_sum + x_vec[n]
            y_sum = y_sum + y_vec[n]
            rot_sum = rot_sum + yaw_vec[n]

        x = x_sum / len(x_vec)
        y = y_sum / len(y_vec)
        rot = rot_sum / len(yaw_vec)

        msg.transform.translation.x = x  # x
        msg.transform.translation.y = y  # y

        (msg.transform.rotation.x,
         msg.transform.rotation.y,
         msg.transform.rotation.z,
         msg.transform.rotation.w) = quaternion_from_euler(0, 0, rot)  # yaw

        x_vec.pop(0)
        y_vec.pop(0)
        yaw_vec.pop(0)
        pub.publish(msg)


def transform_to_pq(msg):
    """Convert a C{geometry_msgs/Transform} into position/quaternion np arrays

    @param msg: ROS message to be converted
    @return:
      - p: position as a np.array
      - q: quaternion as a numpy array (order = [x,y,z,w])
    """
    p = np.array([msg.translation.x, msg.translation.y, msg.translation.z])
    q = np.array([msg.rotation.x, msg.rotation.y,
                  msg.rotation.z, msg.rotation.w])
    return p, q


def transform_stamped_to_pq(msg):
    """Convert a C{geometry_msgs/TransformStamped} into position/quaternion np arrays

    @param msg: ROS message to be converted
    @return:
      - p: position as a np.array
      - q: quaternion as a numpy array (order = [x,y,z,w])
    """
    return transform_to_pq(msg.transform)


print('Starting...')
rospy.init_node('kf')
pose_sub = rospy.Subscriber('/localisation/moving_average_input', TransformStamped, pose_callback, queue_size=1,
                            buff_size=2**24)
pub = rospy.Publisher('/localisation/moving_average_output', TransformStamped, queue_size=10)
print('Ready')


def main():
    rospy.spin()


if __name__ == '__main__':
    main()
