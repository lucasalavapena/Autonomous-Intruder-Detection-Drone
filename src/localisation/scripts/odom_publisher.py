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
            marker_name_extension = str(m.id)
            transforms.append(broadcast_transform(m, marker_name_extension))
        else:
            transforms.append(data_association(m))
        if len(transforms) > 2:
            transforms.pop(0)


def unique_callback(msg):
    global unique_id
    unique_id = msg.data


def broadcast_transform(m, marker_name_extension):
    # Find transform of pose of detected marker in odom
    try:
        detected = tf_buf.lookup_transform('aruco/detected' + marker_name_extension, 'cf1/odom', m.header.stamp, rospy.Duration(tf_timeout))
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
        print('odom_publisher.broadcast_transform(detected lookup): ', e)
        return
    trans_detected, rot_detected = transform_stamped_to_pq(detected)

    # Find transform of pose of static marker in map
    try:
        t_map = tf_buf.lookup_transform('map', 'aruco/marker' + marker_name_extension, m.header.stamp, rospy.Duration(tf_timeout))
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
        print('odom_publisher.broadcast_transform(marker lookup): ', e)
        return
    trans_map, rot_map = transform_stamped_to_pq(t_map)

    # Calculate resulting rotation between map and odom
    # Change the detected marker in to 4x4 matrices and combine them
    detected_mat = np.dot(translation_matrix(trans_detected), quaternion_matrix(rot_detected))

    # Change the map marker in to 4x4 matrices and combine them
    map_mat = np.dot(translation_matrix(trans_map), quaternion_matrix(rot_map))

    # Calculate resulting 4x4 matrix, separate in to 2 matrices
    result_mat = np.dot(map_mat, detected_mat)
    trans_result = translation_from_matrix(result_mat)
    rot_result = quaternion_from_matrix(result_mat)

    # Create new message with time stamps and frames
    t = TransformStamped()
    t.header = m.header
    t.header.frame_id = 'map'
    t.child_frame_id = 'cf1/odom'

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


def data_association(m):
    best_marker = None
    best_delta = 100

    # Find transform of pose of detected marker in map
    try:
        detected = tf_buf.lookup_transform('map', "aruco/detected" + str(m.id), m.header.stamp, rospy.Duration(tf_timeout))
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
        print('odom_publisher.data_association: ', e)
        return
    trans_detected, rot_detected = transform_stamped_to_pq(detected)
    detected_mat = np.dot(translation_matrix(trans_detected), quaternion_matrix(rot_detected))

    more_markers = True
    n = 0
    while more_markers:
        # Find transform of pose of static marker in map
        try:
            t_map = tf_buf.lookup_transform('map', "aruco/marker" + str(m.id) + '_' + str(n), m.header.stamp, rospy.Duration(tf_timeout))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            print(e)
            break
        trans_map, rot_map = transform_stamped_to_pq(t_map)
        map_mat = np.dot(translation_matrix(trans_map), quaternion_matrix(rot_map))

        # Compare positions and orientations of detected vs map markers
        result_mat = np.dot(map_mat, detected_mat)
        trans_result = translation_from_matrix(result_mat)
        rot_result = quaternion_from_matrix(result_mat)

        d_roll, d_pitch, d_yaw = euler_from_quaternion(rot_result)
        orientation_error = 30
        if np.abs(d_yaw) <= orientation_error:
            delta = np.linalg.norm(trans_result)
            if best_marker is None:
                best_marker = t_map
                best_delta = delta
                marker_name_extension = str(m.id) + '_' + str(n)
            elif delta < best_delta:
                best_marker = t_map
                best_delta = delta
                marker_name_extension = str(m.id) + '_' + str(n)
        n += 1
    print(best_marker)
    return broadcast_transform(m, marker_name_extension)


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


print('Starting...')
rospy.init_node('odom_publisher')
tf_buf = tf2_ros.Buffer()
tf_lstn = tf2_ros.TransformListener(tf_buf)
br = tf2_ros.TransformBroadcaster()
sub_marker = rospy.Subscriber('/aruco/markers', MarkerArray, marker_callback)
sub_unique = rospy.Subscriber('/marker/unique', Int16, unique_callback)
pub = rospy.Publisher('localisation/is_localised', Bool, queue_size=10)
tf_timeout = rospy.get_param('~tf_timeout', 0.1)
print('Ready')


if __name__ == '__main__':
    main()
