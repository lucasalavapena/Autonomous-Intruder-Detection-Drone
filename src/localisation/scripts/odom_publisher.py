#!/usr/bin/env python
import time
import math
import rospy
import tf2_ros
import tf2_geometry_msgs
import numpy as np
from aruco_msgs.msg import MarkerArray
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Bool, Int16, String
from tf.transformations import euler_from_quaternion,       \
                                quaternion_from_euler,      \
                                translation_matrix,         \
                                quaternion_matrix,          \
                                translation_from_matrix,    \
                                quaternion_from_matrix


transforms = []
unique_id = None


def marker_callback(msg):
    """
    msg contains list of detected aruco markers and for each marker, determine if it's the unique marker or a
    non-unique marker. If unique, send msg to broadcast_transform(). Otherwise send msg to data_association().
    """
    global transforms
    is_localized()
    for m in msg.markers:
        if m.id == unique_id:
            marker_name_extension = str(m.id)
            transforms.append(broadcast_transform(m, marker_name_extension))
        else:
            pass
            result = data_association(m)
            if result:
                transforms.append(result)
        if len(transforms) > 2:
            transforms.pop(0)


def unique_callback(msg):
    """
    Set global unique_id to the unique id from topic /marker/unique
    """
    global unique_id
    unique_id = msg.data


def broadcast_transform(m, marker_name_extension):
    """
    By assuming that the pose of detected marker = pose of static marker in map, get transforms from map->static marker
    and detected marker->odom (which is the inverse) use matrix dot multiplication to find relative difference between
    the transforms. This difference is then the resulting translation and rotation vectors describing map->cf1/odom.
    Because the problem is simplified with frames cf1/base_link, cf1/base_stabilized and cf1/base_footprint, z, roll
    and pitch are set to 0.

    :param m: message containing marker information
    :param marker_name_extension: if non-unique marker ID, describes the unique name extension
    :broadcast t: broadcast calculated map->cf1/odom transfer with z, roll, pitch = 0
    """
    # Find transform of pose of odom in detected marker frame
    try:
        detected = tf_buf.lookup_transform('aruco/detected' + str(m.id), 'cf1/odom', m.header.stamp, rospy.Duration(tf_timeout))
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
        print('odom_publisher.broadcast_transform(detected lookup): ', e)
        return
    trans_detected, rot_detected = transform_stamped_to_pq(detected)

    # Find transform of pose of static marker in map frame
    try:
        t_map = tf_buf.lookup_transform('map', 'aruco/marker' + marker_name_extension, m.header.stamp)
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
    """
    Using transform detected marker->odom, calculate relative difference in yaw rotation using map->static markers of
    [id] and send the best match to broadcast_transform() (if match).
    :param m:
    :return t: Transform of map->odom using the most correct marker found.
    """
    start = time.time()
    best_marker = None
    marker_name_extension = 'None found'
    best_delta = 100
    best_yaw = 100
    str1 = str2 = str3 = ""
    # Find transform of pose of map in detected marker frame
    try:
        detected = tf_buf.lookup_transform("aruco/detected" + str(m.id), 'map',  m.header.stamp, rospy.Duration(tf_timeout))
        # detected = tf_buf.lookup_transform("map",  'cf1/base_link',  m.header.stamp, rospy.Duration(tf_timeout))

    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
        # print('odom_publisher.data_association: ', e)
        return
    trans_detected, rot_detected = transform_stamped_to_pq(detected)
    detected_mat = np.dot(translation_matrix(trans_detected), quaternion_matrix(rot_detected))

    # Find transform of the position of the drone in map
    try:
        drone_loc = tf_buf.lookup_transform("map",  'cf1/base_link',  m.header.stamp, rospy.Duration(tf_timeout))
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
        # print('odom_publisher.data_association: ', e)
        return
    trans_loc, rot_loc = transform_stamped_to_pq(drone_loc)
    loc_mat = np.dot(translation_matrix(trans_loc), quaternion_matrix(rot_loc))

    # rospy.loginfo_throttle(2, "trans_loc \n{}".format(trans_loc))
    n = 0
    while True:
        # Find transform of pose of static marker in map
        try:
            t_map = tf_buf.lookup_transform('map', "aruco/marker" + str(m.id) + '_' + str(n), m.header.stamp)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            # print(e)
            rospy.loginfo_throttle(1, "break bc n = %d", n)
            break  # if n exceeds number for last non-unique static marker in map, end the loop.
        trans_map, rot_map = transform_stamped_to_pq(t_map)
        map_mat = np.dot(translation_matrix(trans_map), quaternion_matrix(rot_map))

        # Compare positions and orientations of detected vs map markers
        result_mat = np.dot(map_mat, detected_mat)
        trans_result = translation_from_matrix(result_mat)
        rot_result = quaternion_from_matrix(result_mat)
        # if n == 0:
        #     rospy.loginfo_throttle(5, "n=0  result_mat \n{}".format(result_mat))
        # elif n == 1:
        #     rospy.loginfo_throttle(5, "n=1  result_mat \n{}".format(result_mat))
        # elif n == 2:
        #     rospy.loginfo_throttle(5, "n=2  result_mat \n{}".format(result_mat))

        d_roll, d_pitch, d_yaw = euler_from_quaternion(rot_result)
        orientation_error = 1 * math.pi/6

        distance_from_sign = trans_loc - trans_map


        if n == 0:
            str0 = "n=0  distance_from_sign \n{} \ntrans_loc: \n{}\n".format(distance_from_sign, trans_loc)
            rospy.loginfo_throttle(1, "n=0  distance_from_sign \n{} \ntrans_loc: \n{}\n".format(distance_from_sign,
                                                                                                            trans_loc))

        elif n == 1:
            rospy.loginfo_throttle(1, "n=1  distance_from_sign \n{} \ntrans_locc: \n{}\n".format(distance_from_sign,
                                                                                                           trans_loc))
            str1 = "n=1  distance_from_sign \n{} \ntrans_loc: \n{}\n".format(distance_from_sign, trans_loc)

        elif n == 2:
            str2 = "n=2  distance_from_sign \n{} \ntrans_loc: \n{}\n".format(distance_from_sign, trans_loc)
            rospy.loginfo_throttle(1, "n=2  distance_from_sign \n{} \ntrans_loc: \n{}\n".format(distance_from_sign,
                                                                                                           trans_loc))
        result_str = str1 + str1 + str2
        debug_pub.publish(result_str)
        # rospy.loginfo_throttle(2, "trans_loc \n{}".format(trans_loc))
        # d_yaw = 89999

        if np.abs(d_yaw) <= orientation_error:
            delta = np.linalg.norm(distance_from_sign)

            if n == 0:
                rospy.loginfo_throttle(0.6, "trans_result {}".format(trans_loc))
                rospy.loginfo_throttle(0.6, "delta_norm for {} is {};\n yaw info: best {} curr {}".format(str(m.id) + '_' + str(n), delta, best_yaw, d_yaw))
            elif n == 1:
                rospy.loginfo_throttle(0.6, "trans_result {}".format(trans_loc))
                rospy.loginfo_throttle(0.6, "delta_norm for {} is {};\n yaw info: best {} curr {}".format(str(m.id) + '_' + str(n), delta, best_yaw, d_yaw))
            elif n == 2:
                rospy.loginfo_throttle(0.6, "trans_result {}".format(trans_loc))
                rospy.loginfo_throttle(0.6, "delta_norm for {} is {};\n yaw info: best {} curr {}".format(str(m.id) + '_' + str(n), delta, best_yaw, d_yaw))
            #

            if (best_marker is None or d_yaw <= best_yaw) and delta < best_delta:
                best_marker = t_map
                best_delta = delta
                best_yaw = d_yaw
                marker_name_extension = str(m.id) + '_' + str(n)
                rospy.loginfo_throttle(0.6, "delta_norm for {} is {};\n yaw info: best {} curr {}".format(str(m.id) + '_' + str(n), delta, best_yaw, d_yaw))

                # print(marker_name_extension)
        n += 1
    # print('best: ' + marker_name_extension)
    end = time.time()
    if best_delta == 100 or best_yaw == 100:  # If no marker found was good enough, return nothing
        return None
    rospy.loginfo_throttle(1, "Sending transform...n = %d", n)
    return broadcast_transform(m, marker_name_extension)


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
    """
    Take an existing transform and update the time in the header stamp

    :param t: TransformStamped with time = now
    :return:
    """
    t.header.stamp = rospy.Time.now()
    return t


def is_localized():
    """
    Publish a boolean True to 'localisation/is_localised' topic
    """
    pub.publish(Bool(data=True))


def main():
    rate = rospy.Rate(20)  # Hz
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
debug_pub = rospy.Publisher('lucas/debug', String, queue_size=2**10)

tf_timeout = rospy.get_param('~tf_timeout', 0.1)
print('Ready')


if __name__ == '__main__':
    main()
