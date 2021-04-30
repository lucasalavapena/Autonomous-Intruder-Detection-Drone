#!/usr/bin/env python

import rospy
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import TransformStamped
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from numpy import dot, sum, tile, linalg, exp, log, pi, diag, eye, zeros, array
from numpy.linalg import inv, det
from numpy.random import randn

first_time = True

A = eye(3)
B = zeros([3, 3])
Q = 1
R = array([1, 1])
H = array([1, 1, 1])
U = zeros([3, 1])
X = None
P = None
Y = None


def pose_callback(msg):
    global X, P, Y, first_time

    p, q = transform_stamped_to_pq(msg)
    roll, pitch, yaw = euler_from_quaternion(q)
    X = array([[p[0]], [p[1]], [yaw]])  # [x, y, yaw]

    if first_time:
        P = eye(3)*0.1
        Y = array([[X[0, 0] + abs(0.1 * randn(1)[0])], [X[1, 0] + abs(0.1 * randn(1)[0])]])

    X, P = kf_predict(X, P, A, Q, B, U)
    print('after predict')
    print(P)
    X, P, K, IM, IS, LH = kf_update(X, P, Y, H, R)
    print('after update')
    print(P)
    Y = array([[X[0, 0] + abs(0.1 * randn(1)[0])], [X[1, 0] + abs(0.1 * randn(1)[0])]])

    msg.transform.translation.x = X[0]  # x
    msg.transform.translation.y = X[1]  # y

    (msg.transform.rotation.x,
     msg.transform.rotation.y,
     msg.transform.rotation.z,
     msg.transform.rotation.w) = quaternion_from_euler(0, 0, X[2])  # yaw

    pub.publish(msg)
    first_time = False


def kf_predict(X, P, A, Q, B, U):
    X = dot(A, X) + dot(B, U)
    P = dot(A, dot(P, A.T)) + Q
    return X, P


def kf_update(X, P, Y, H, R):
    IM = dot(H, X)
    IS = R + dot(H, dot(P, H.T))
    K = dot(P, dot(H.T, 1/IS))
    X = X + dot(K, (Y - IM))
    P = P - dot(K, dot(IS, K.T))
    LH = gauss_pdf(Y, IM, IS)
    return X, P, K, IM, IS, LH


def gauss_pdf(X, M, S):
    if M.shape()[1] == 1:
        DX = X - tile(M, X.shape()[1])
        E = 0.5 * sum(DX * (dot(inv(S), DX)), axis=0)
        E = E + 0.5 * M.shape()[0] * log(2 * pi) + 0.5 * log(det(S))
        P = exp(-E)
    elif X.shape()[1] == 1:
        DX = tile(X, M.shape()[1]) - M
        E = 0.5 * sum(DX * (dot(inv(S), DX)), axis=0)
        E = E + 0.5 * M.shape()[0] * log(2 * pi) + 0.5 * log(det(S))
        P = exp(-E)
    else:
        DX = X - M
        E = 0.5 * dot(DX.T, dot(inv(S), DX))
        E = E + 0.5 * M.shape()[0] * log(2 * pi) + 0.5 * log(det(S))
        P = exp(-E)

    return P[0], E[0]


def transform_to_pq(msg):
    """Convert a C{geometry_msgs/Transform} into position/quaternion np arrays

    @param msg: ROS message to be converted
    @return:
      - p: position as a np.array
      - q: quaternion as a numpy array (order = [x,y,z,w])
    """
    p = array([msg.translation.x, msg.translation.y, msg.translation.z])
    q = array([msg.rotation.x, msg.rotation.y, msg.rotation.z, msg.rotation.w])
    return p, q


def transform_stamped_to_pq(msg):
    """Convert a C{geometry_msgs/TransformStamped} into position/quaternion np arrays

    @param msg: ROS message to be converted
    @return:
      - p: position as a np.array
      - q: quaternion as a numpy array (order = [x,y,z,w])
    """
    return transform_to_pq(msg.transform)


def main():
    rospy.spin()


if __name__ == '__main__':
    print('Starting...')
    rospy.init_node('kf2')
    pose_sub = rospy.Subscriber('/kf2/input', TransformStamped, pose_callback, queue_size=1, buff_size=2 ** 24)
    pub = rospy.Publisher('kf2/output', TransformStamped, queue_size=10)
    print('Ready')
    main()

