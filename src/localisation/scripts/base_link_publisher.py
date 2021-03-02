#!/usr/bin/env python

import rospy
import tf2_ros
import tf2_geometry_msgs
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import PoseStamped, TransformStamped


def pose_callback(msg):
    broadcast_pose(msg)


def broadcast_pose(m):
    # Create new message with time stamps and frames
    t = TransformStamped()
    t.header = m.header
    t.header.frame_id = 'cf1/base_stabilized'
    t.child_frame_id = frame_id
    roll, pitch, yaw = euler_from_quaternion((m.pose.orientation.x,
                                              m.pose.orientation.y,
                                              m.pose.orientation.z,
                                              m.pose.orientation.w))
    (t.transform.rotation.x,
     t.transform.rotation.y,
     t.transform.rotation.z,
     t.transform.rotation.w) = quaternion_from_euler(roll, pitch, 0)

    br.sendTransform(t)


def main():
    rospy.spin()


if __name__ == '__main__':
    rospy.init_node('base_link_publisher')

    tf_buf = tf2_ros.Buffer()
    tf_lstn = tf2_ros.TransformListener(tf_buf)
    br = tf2_ros.TransformBroadcaster()
    sub = rospy.Subscriber('/cf1/pose', PoseStamped, pose_callback)
    tf_timeout = rospy.get_param('~tf_timeout', 0.1)
    frame_id = rospy.get_param('~frame_id', 'cf1/base_link')
    main()
