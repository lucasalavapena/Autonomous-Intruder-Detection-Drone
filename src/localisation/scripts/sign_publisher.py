#!/usr/bin/env python

import numpy as np

import rospy
import tf2_ros 
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import String
from tf.transformations import quaternion_from_euler,      \
                                translation_matrix,         \
                                quaternion_matrix,          \
                                quaternion_from_matrix,     \
                                euler_from_quaternion

pi_half = np.pi/2

def callback(msg):
    parse(msg)

def parse(str):
    data = str.data.split(';;;;')
    label = data[0]

    secs = int(data[1])
    nsecs = int(data[2])
    stamp = rospy.Time(secs=secs, nsecs=nsecs)
    frame = data[3]
    position = np.fromstring(data[4])
    orientation = np.fromstring(data[5])
    orientation[0] += -pi_half
    orientation[2] += -pi_half
    # print("SIGN: roll: {}, pitch: {}, yaw: {}".format(orientation[0], orientation[1], orientation[2]))
    orientation = quaternion_from_euler(*orientation.tolist())
    

    transform_sign_and_publish(label, stamp, frame, position, orientation)

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

def transform_sign_and_publish(label, stamp, frame, translation, rotation):

    # timeout = rospy.Duration(0.1)

    # # Lookput transform from camera link to canonical sign
    # try:
    #     t_canon_to_cam = tf_buf.lookup_transform(frame, 'landmark/' + label, rospy.Time.now(), timeout)
    # except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
    #     rospy.logwarn_throttle(5.0, 'sign_publisher.broadcast_transform(marker lookup): {}'.format(e))
    #     return
    # canon_trans, canon_rot = transform_to_pq(t_canon_to_cam.transform)

    # detected_mat = np.dot(translation_matrix(translation), quaternion_matrix(rotation))
    # canon_mat = np.dot(translation_matrix(canon_trans), quaternion_matrix(canon_rot))

    # result_mat = np.dot(canon_mat, detected_mat)
    # rot_result = quaternion_from_matrix(result_mat)

    t = TransformStamped()
    t.header.frame_id = frame
    t.header.stamp = stamp
    t.child_frame_id = "landmark/detected_" + label
    (t.transform.translation.x,
    t.transform.translation.y,
    t.transform.translation.z) = translation.tolist()
    (t.transform.rotation.x,
    t.transform.rotation.y,
    t.transform.rotation.z,
    t.transform.rotation.w) = rotation.tolist()
    broadcaster.sendTransform(t)
    sign_pub.publish(t)


rospy.init_node('sign_publisher')
sign_sub = rospy.Subscriber('/sign_poses', String, callback, queue_size=1, buff_size=2**24)
sign_pub = rospy.Publisher('/sign/detected', TransformStamped, queue_size=1)
broadcaster = tf2_ros.TransformBroadcaster()
tf_buf = tf2_ros.Buffer()
tf_lstn = tf2_ros.TransformListener(tf_buf)


def main():
    rospy.spin()


if __name__ == "__main__":
    main()
