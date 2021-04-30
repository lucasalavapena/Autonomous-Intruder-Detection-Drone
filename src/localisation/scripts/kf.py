#!/usr/bin/env python

import time
import math
import rospy
import tf2_ros
import tf2_geometry_msgs
import numpy as np
from geometry_msgs.msg import TransformStamped
from tf.transformations import euler_from_quaternion, quaternion_from_euler


def pose_callback(msg):
    global x, P
    p, q = transform_stamped_to_pq(msg)
    roll, pitch, yaw = euler_from_quaternion(q)
    z = np.diag(np.array([p[0], p[1], yaw]))  # [x, y, yaw]
    x, P = kf(x, P, z, R, Q, F, H)

    msg.transform.translation.x = np.diag(x)[0]  # x
    msg.transform.translation.y = np.diag(x)[1]  # y

    (msg.transform.rotation.x,
     msg.transform.rotation.y,
     msg.transform.rotation.z,
     msg.transform.rotation.w) = quaternion_from_euler(0, 0, np.diag(x)[2])  # yaw

    pub.publish(msg)


def kf(x, P, z, R, Q, F, H):
    """
    :param x: initial state
    :param P: initial uncertainty convariance matrix
    :param z: observed position (same shape as H*x)
    :param R: measurement noise (same shape as H), Constant
    :param Q: process noise (same shape as P), Constant
    :param F: next state function: x_prime = F*x, Constant
    :param H: observation model: position = H*x, Constant
    :return x, P: the updated and predicted new values for (x, P)
    """
    # UPDATE x, P based on measurement m
    # distance between measured and current position-belief
    y = z.T - H * x
    S = H * P * H.T + R  # residual covariance
    K = P * H.T * np.linalg.inv(S)  # Kalman gain
    x = x + K * y
    I = np.eye(F.shape[0])  # identity matrix
    P = (I - K * H) * P

    # PREDICT x, P based on motion
    x = F * x
    P = F * P * F.T + Q
    return x, P


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
pose_sub = rospy.Subscriber('/kf/input', TransformStamped, pose_callback, queue_size=1, buff_size=2**24)
pub = rospy.Publisher('kf/output', TransformStamped, queue_size=10)

#  Initial State Matrix
x = np.array([[0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0]])

# Initial State uncertainty covariance
P = np.array([[1.0, 0.0, 0.0],
              [0.0, 1.0, 0.0],
              [0.0, 0.0, 1.0]])

# Process noise
Q = np.array([[1.0, 0.0, 0.0],
              [0.0, 1.0, 0.0],
              [0.0, 0.0, 1.0]])*10

R = np.array([[1.0, 0.0, 0.0],
              [0.0, 1.0, 0.0],
              [0.0, 0.0, 1.0]])*10

# Next state function F and measurement function H
I = np.eye(3)
Z = np.zeros([3, 3])
#F = np.array(np.hstack((np.vstack((I, Z)), np.vstack((I, I)))))
F = I
#H = np.array(np.hstack((I, Z)))
H = I
print('Ready')


def main():
    rospy.spin()


if __name__ == '__main__':
    main()
