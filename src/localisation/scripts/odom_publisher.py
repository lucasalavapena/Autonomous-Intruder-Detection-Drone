#!/usr/bin/env python

import rospy
import tf2_ros
import tf2_geometry_msgs
import numpy as np
from tf.transformations import quaternion_multiply, euler_from_quaternion, quaternion_from_euler, translation_matrix, quaternion_matrix, translation_from_matrix, quaternion_from_matrix
from aruco_msgs.msg import MarkerArray
from geometry_msgs.msg import PoseStamped, TransformStamped, Quaternion
from std_msgs.msg import Bool, Int16


transforms = []
unique_id = None


def marker_callback(msg):
    global transforms
    #is_localized()
    for m in msg.markers:
        if m.id == unique_id:
            marker_name = "aruco/marker" + str(m.id)
            transforms.append(broadcast_transform(m, marker_name))
        else:
            transforms.append(data_association(m))
        if len(transforms) > 2:
            transforms.pop(0)


def unique_callback(msg):
    global unique_id
    unique_id = msg.data


def broadcast_transform(m, marker_name):
    # Find transform of pose of detected marker in odom
    if not tf_buf.can_transform(m.header.frame_id, frame_id, m.header.stamp, rospy.Duration(tf_timeout)):
        rospy.logwarn_throttle(5.0, '%s: No transform from %s to %s', rospy.get_name(), m.header.frame_id, frame_id)
        return
    detected = tf_buf.transform(PoseStamped(header=m.header, pose=m.pose.pose), 'cf1/odom')
    trans_detected, rot_detected = pose_stamped_to_pq(detected)

    # Find transform of pose of static marker in map
    try:
        t_map = tf_buf.lookup_transform(marker_name, 'map', m.header.stamp, rospy.Duration(tf_timeout))
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        return
    trans_map, rot_map = transform_stamped_to_pq(t_map)

    # Calculate resulting rotation between map and odom
    # Change the detected marker in to 4x4 matrices and combine them
    trans_detected_mat = translation_matrix(trans_detected)
    rot_detected_mat = quaternion_matrix(rot_detected)
    detected_mat = np.dot(trans_detected_mat, rot_detected_mat)

    # Change the map marker in to 4x4 matrices and combine them
    trans_map_mat = translation_matrix(trans_map)
    rot_map_mat = quaternion_matrix(rot_map)
    map_mat = np.dot(trans_map_mat, rot_map_mat)

    # Calculate resulting 4x4 matrix, separate in to 2 matrices
    result_mat = np.dot(detected_mat, map_mat)
    trans_result = translation_from_matrix(result_mat)
    rot_result = quaternion_from_matrix(result_mat)


    # Create new message with time stamps and frames
    t = TransformStamped()
    t.header = m.header
    t.header.frame_id = 'map'
    t.child_frame_id = frame_id

    (t.transform.translation.x,
     t.transform.translation.y,
     t.transform.translation.z) = trans_result
    t.transform.translation.z = 0.0

    roll, pitch, yaw = euler_from_quaternion(rot_result)
    rot_result = quaternion_from_euler(0, 0, yaw)
    (t.transform.rotation.x,
     t.transform.rotation.y,
     t.transform.rotation.z,
     t.transform.rotation.w) = rot_result

    return t


def make_transformstamped(pose, child_frame_id):
    t = TransformStamped()
    t.header = pose.header
    t.child_frame_id = child_frame_id
    t.transform.translation = pose.pose.position
    t.transform.rotation = pose.pose.orientation
    return t.transform.translation, t.transform.rotation


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

        if np.abs(d_roll) <= orientation_error and np.abs(d_pitch) <= orientation_error and np.abs(d_yaw) <= orientation_error:
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


def pose_to_pq(msg):
    """Convert a C{geometry_msgs/Pose} into position/quaternion np arrays

    @param msg: ROS message to be converted
    @return:
      - p: position as a np.array
      - q: quaternion as a numpy array (order = [x,y,z,w])
    """
    p = np.array([msg.position.x, msg.position.y, msg.position.z])
    q = np.array([msg.orientation.x, msg.orientation.y,
                  msg.orientation.z, msg.orientation.w])
    return p, q


def pose_stamped_to_pq(msg):
    """Convert a C{geometry_msgs/PoseStamped} into position/quaternion np arrays

    @param msg: ROS message to be converted
    @return:
      - p: position as a np.array
      - q: quaternion as a numpy array (order = [x,y,z,w])
    """
    return pose_to_pq(msg.pose)


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


def update_time(t):
    t.header.stamp = rospy.Time.now()
    return t


def is_localized():
    pub.publish(Bool(data=True))


def main():
    rate = rospy.Rate(40)  # Hz
    while not rospy.is_shutdown():
        if transforms:
            if transforms[-1] is not None:
                br.sendTransform(update_time(transforms[-1]))
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
