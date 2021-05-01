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

# MOTION MODEL
# Coefficients of motion variables
A = eye(3)
# Control from motion model
B = zeros([3, 3])
u = zeros([3, 1])
# Covariance of motion model
R = eye(3)

# MEASUREMENT MODEL
# Coefficients of measurement variables
C = array([1, 1, 1])
# Covariance of measurement model
Q = 1

mu = None
sigma = None
z = None


def pose_callback(msg):
    global mu, sigma, z, first_time

    p, q = transform_stamped_to_pq(msg)
    roll, pitch, yaw = euler_from_quaternion(q)
    # Mean of the state: [x, y, yaw]
    mu = array([[p[0]], [p[1]], [yaw]])  

    if first_time:
        # Covariance of the state
        sigma = eye(3)*0.1 
        # Measurment model
        z = array([[mu[0, 0], [mu[1, 0]], [mu[2, 0]])

    # Estimated belief
    mu, sigma = kf_predict(mu, sigma, A, R, B, u)
    print('after predict')
    print(sigma)

    # Posterior distribution and Kalman Gain
    mu, sigma, K = kf_update(mu, sigma, z, C, Q)
    print('after update')
    print(sigma)

    # Measurment model
    z = array([[p[0]], [p[1]], [yaw]])

    # Update the transform with the updated variables
    msg.transform.translation.x = mu[0, 0]  # x
    msg.transform.translation.y = mu[1, 0]  # y

    (msg.transform.rotation.x,
     msg.transform.rotation.y,
     msg.transform.rotation.z,
     msg.transform.rotation.w) = quaternion_from_euler(0, 0, mu[2, 0])  # yaw

    pub.publish(msg)
    first_time = False


def kf_predict(mu, sigma, A, R, B, u):
    # Estimated mean of the state
    mu = dot(A, mu) + dot(B, u)

    # Estimated covariance of the state
    sigma = dot(A, dot(sigma, A.T)) + R

    return mu, sigma


def kf_update(mu, sigma, z, C, Q):
    # Kalman Gain
    K = dot(sigma, dot(C.T, inv(dot(C, dot(cov, C.T)) + Q)))

    # Innovation
    eta = z - dot(C, mu)

    # Compute posterior
    mu = mu + dot(K, eta) # Mean of the state
    I = eye(2)
    sigma = dot((I - dot(K, C)), sigma) # Covariance of the state
    
    return mu, sigma, K

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
    rospy.init_node('kf3')
    pose_sub = rospy.Subscriber('/kf3/input', TransformStamped, pose_callback, queue_size=1, buff_size=2 ** 24)
    pub = rospy.Publisher('kf3/output', TransformStamped, queue_size=10)
    print('Ready')
    main()

