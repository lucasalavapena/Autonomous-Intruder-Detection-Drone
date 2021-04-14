#!/usr/bin/env python

import rospy
import tf2_ros
import tf2_geometry_msgs
import numpy as np
from tf.transformations import quaternion_multiply, euler_from_quaternion, quaternion_from_euler
from aruco_msgs.msg import MarkerArray
from geometry_msgs.msg import PoseStamped, TransformStamped, Quaternion
from std_msgs.msg import Bool, Int16

transforms = []
#unique_id = None


def marker_callback(msg):
    global transforms
    #transforms = []
    is_localized()
    for m in msg.markers:
        #print(unique_id)
        if m.id == unique_id:
            marker_name = "aruco/marker" + str(m.id)
            transforms.append(broadcast_transform(m, marker_name))
        else:
            transforms.append(data_association(m))
        if len(transforms) > 2:
            transforms.pop(0)
    #print(len(transforms))


def unique_callback(msg):
    global unique_id
    unique_id = msg.data


def broadcast_transform(m, marker_name):
    # Find transform of pose of detected marker in odom
    if not tf_buf.can_transform(frame_id, m.header.frame_id, m.header.stamp, rospy.Duration(tf_timeout)):
        rospy.logwarn_throttle(5.0, '%s: No transform from %s to %s', rospy.get_name(), m.header.frame_id, frame_id)
        return
    marker = tf_buf.transform(PoseStamped(header=m.header, pose=m.pose.pose), 'cf1/odom')

    # Find transform of pose of static marker in map
    try:
        t_map = tf_buf.lookup_transform('map', marker_name, m.header.stamp, rospy.Duration(tf_timeout))
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        return

    # Create new message with time stamps and frames
    t = TransformStamped()
    t.header = marker.header
    t.header.frame_id = 'map'
    t.child_frame_id = frame_id

    # Calculate resulting rotation between map and odom
    q_marker = separate_quaternions_marker(marker)
    q_marker[3] = -q_marker[3]
    q_t = separate_quaternions_tf(t_map)
    q_r = quaternion_multiply(q_t, q_marker)
    roll, pitch, yaw = euler_from_quaternion((q_r[0], q_r[1], q_r[2], q_r[3]))
    (t.transform.rotation.x,
     t.transform.rotation.y,
     t.transform.rotation.z,
     t.transform.rotation.w) = quaternion_from_euler(0, 0, yaw)

    #  Calculate the translation vector
    t1 = t_map.transform.translation.x - np.cos(yaw)*marker.pose.position.x + np.sin(yaw)*marker.pose.position.y
    t2 = t_map.transform.translation.y - np.sin(yaw) * marker.pose.position.x - np.cos(yaw) * marker.pose.position.y

    #  Add values to transform
    t.transform.translation.x = t1
    t.transform.translation.y = t2
    t.transform.translation.z = 0.0
    return t


def data_association(m):
    best_marker = None
    best_delta = 100

    # Find transform of pose of detected marker in map
    try:
        detected_map = tf_buf.lookup_transform('map', "aruco/detected" + str(m.id), m.header.stamp, rospy.Duration(tf_timeout))
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
        print(e)
        return

    more_markers = True
    n = 0
    while more_markers:
        # Find transform of pose of static marker in map
        try:
            t_map = tf_buf.lookup_transform('map', "aruco/marker" + str(m.id) + '_' + str(n), m.header.stamp, rospy.Duration(tf_timeout))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            print(e)
            break

        # Compare positions and orientations of detected vs map markers
        orientation_error = 30

        q_map = separate_quaternions_tf(t_map)
        q_det = separate_quaternions_tf(detected_map)
        q_det[3] = -q_det[3]
        # Calculate resulting rotation between map and odom
        q_r = quaternion_multiply(q_map, q_det)
        d_roll, d_pitch, d_yaw = euler_from_quaternion((q_r[0], q_r[1], q_r[2], q_r[3]))
        delta = np.sqrt(d_roll**2 + d_pitch**2 + d_yaw**2)
        #print(delta)

        if np.abs(d_roll) <= orientation_error \
                and np.abs(d_pitch) <= orientation_error \
                and np.abs(d_yaw) <= orientation_error:
            if best_marker is None:
                best_marker = t_map
                best_delta = delta
                marker_name = "aruco/marker" + str(m.id) + '_' + str(n)
            elif delta < best_delta:
                best_marker = t_map
                best_delta = delta
                marker_name = "aruco/marker" + str(m.id) + '_' + str(n)
        n += 1
    print(best_marker)
    return broadcast_transform(m, marker_name)


def update_time(t):
    t.header.stamp = rospy.Time.now()
    return t


def is_localized():
    pub.publish(Bool(data=True))


def separate_quaternions_tf(t):
    q_t = [0] * 4
    q_t[0] = t.transform.rotation.x
    q_t[1] = t.transform.rotation.y
    q_t[2] = t.transform.rotation.z
    q_t[3] = t.transform.rotation.w
    return q_t


def separate_quaternions_marker(m):
    q = [0] * 4
    q[0] = m.pose.orientation.x
    q[1] = m.pose.orientation.y
    q[2] = m.pose.orientation.z
    q[3] = m.pose.orientation.w
    return q


def main():
    rate = rospy.Rate(40)  # Hz
    while not rospy.is_shutdown():
        if transforms is not None:
            for t in transforms:
                if t is not None:
                    br.sendTransform(update_time(t))
        rate.sleep()


rospy.init_node('odom_publisher')
tf_buf = tf2_ros.Buffer()
tf_lstn = tf2_ros.TransformListener(tf_buf)
br = tf2_ros.TransformBroadcaster()
sub_marker = rospy.Subscriber('/aruco/markers', MarkerArray, marker_callback)
sub_unique = rospy.Subscriber('/marker/unique', Int16, unique_callback)
pub = rospy.Publisher('localisation/is_localised', Bool, queue_size=10)
tf_timeout = rospy.get_param('~tf_timeout', 0.5)
frame_id = rospy.get_param('~frame_id', 'cf1/odom')


if __name__ == '__main__':
    main()
