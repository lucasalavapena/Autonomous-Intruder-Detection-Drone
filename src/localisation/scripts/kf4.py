#!/usr/bin/env python

import rospy
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import TransformStamped
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import numpy as np

dt = 0.05

class kalman_filter:
    def __init__(self):
        # initial state (location and velocity)
        self.x = np.zeros((6, 1))  # [x, y, theta, x', y', theta']

        # initial uncertainty: 0 for positions x and y, 1000 for the two velocities
        self.P = np.zeros((6,6))
        self.P[3, 3] = self.P[4, 4] = self.P[5, 5] = 1000
        # next state function: generalize the 2d version to 4d
        self.F = np.eye(6)
        self.F[0,3] = self.F[1,4] = self.F[2,5] = dt
        # measurement function: reflect the fact that we observe x and y but not the two velocities

        self.H = np.zeros((3, 6))
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = 1
        # measurement uncertainty: use 2x2 matrix with 0.1 as main diagonal
        self.R = np.eye(3) * 100
        self.Q = np.eye(6) * 0.01
        self.I = np.eye(6)

        self.first_time = True

        self.pose_sub = rospy.Subscriber('/kf4/input', TransformStamped, self.pose_callback,
                                         queue_size=1, buff_size=2 ** 24)
        self.pub = rospy.Publisher('/kf4/output', TransformStamped, queue_size=10)

    def pose_callback(self, msg):
        p, q = self.transform_stamped_to_pq(msg)
        roll, pitch, yaw = euler_from_quaternion(q)
        if self.first_time:
            self.x[0] = p[0]
            self.x[1] = p[1]
            self.x[2] = yaw
            self.first_time = False
            
            self.publish(msg, p[0], p[1], yaw)
        else:
            # print("Uncertainty before update:\n{}".format(self.P))
            self.kf(p[0], p[1], yaw, msg)
            # print("Uncertainty after update:\n{}".format(self.P))

    def kf(self, x_measured, y_measured, yaw_measured, msg):
        # measurement
        Z = np.array([[x_measured], [y_measured], [yaw_measured]])
        Z[2] = (Z[2] + np.pi) % (2 * np.pi) - np.pi
        # print("Z:\n{}".format(Z))
        # current sense
        # err between actual observation and expected observation
        y = Z - np.dot(self.H, self.x)
        y[2] = (y[2] + np.pi) % (2 * np.pi) - np.pi

        # print("Norm of innovation: {}".format(np.linalg.norm(y)))

        S = np.dot(np.dot(self.H, self.P), np.transpose(self.H)) + self.R
        K = np.dot(np.dot(self.P, np.transpose(self.H)), np.linalg.inv(S))

        # posterier mu and sigma
        self.x = self.x + np.dot(K, y)
        self.P = np.dot((self.I - np.dot(K, self.H)), self.P)

        self.publish(msg, self.x[0], self.x[1], self.x[2])

        # predict
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), np.transpose(self.F)) + self.Q

    def publish(self, msg, x, y, yaw):
        # Update the transform with the updated variables
        msg.transform.translation.x = x
        msg.transform.translation.y = y
        
        (msg.transform.rotation.x,
         msg.transform.rotation.y,
         msg.transform.rotation.z,
         msg.transform.rotation.w) = quaternion_from_euler(0, 0, -yaw)  # yaw
        self.pub.publish(msg)
        print("Kalman results - x: {X}, y: {Y}, yaw: {YAW}".format(X=x, Y=y, YAW=yaw))

    def transform_to_pq(self, msg):
        """Convert a C{geometry_msgs/Transform} into position/quaternion np arrays

        @param msg: ROS message to be converted
        @return:
          - p: position as a np.array
          - q: quaternion as a numpy array (order = [x,y,z,w])
        """
        p = np.array([msg.translation.x, msg.translation.y, msg.translation.z])
        q = np.array([msg.rotation.x, msg.rotation.y, msg.rotation.z, msg.rotation.w])
        return p, q

    def transform_stamped_to_pq(self, msg):
        """Convert a C{geometry_msgs/TransformStamped} into position/quaternion np arrays

        @param msg: ROS message to be converted
        @return:
          - p: position as a np.array
          - q: quaternion as a numpy array (order = [x,y,z,w])
        """
        return self.transform_to_pq(msg.transform)


if __name__ == '__main__':
    print('Starting...')
    print('Ready')

    rospy.init_node('kf4')
    kf = kalman_filter()
    rospy.spin()
