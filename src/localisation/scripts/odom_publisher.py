#!/usr/bin/env python
import time
import math
import rospy
import tf2_ros
import tf2_geometry_msgs
import numpy as np
from aruco_msgs.msg import MarkerArray
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Bool, Int16MultiArray, String
from tf.transformations import euler_from_quaternion,       \
                                quaternion_from_euler,      \
                                translation_matrix,         \
                                quaternion_matrix,          \
                                translation_from_matrix,    \
                                quaternion_from_matrix


unique_id = None
non_unique_id = None
filt_tresh = np.pi/12
THROTTLE_PERIOD = 1
VERBOSE = True

def marker_callback(msg):
    """
    msg contains list of detected aruco markers and for each marker, determine if it's the unique marker or a
    non-unique marker. If unique, send msg to broadcast_transform(). Otherwise send msg to data_association().
    """
    global filt_tresh
    is_localized()
    for m in msg.markers:
        p, q = msg_to_pq(m)
        delta = np.linalg.norm(p)
        #print('delta ' + str(delta))
        roll, pitch, yaw = euler_from_quaternion(q)
        # print("ARUCO: roll: {}, pitch: {}, yaw: {}".format(roll, pitch, yaw))
        #print('roll: ' + str(abs(abs(roll)-np.pi/2)) + ' pitch: ' + str(pitch) + ' yaw: ' + str(abs(abs(yaw)-np.pi/2)))
        #print('treshold: ' + str(filt_tresh))
        if delta < 1.5 and abs(pitch) < filt_tresh and abs(abs(roll)-np.pi/2) < filt_tresh and abs(abs(yaw)-np.pi/2) < filt_tresh:
            if m.id == unique_id:
                marker_name_extension = str(m.id)
                broadcast_transform(m, marker_name_extension)
            else:
                data_association(m)
        else:
            pass
            # print('filtered')


def unique_callback(msg):
    """
    Set global unique_id to the unique id from topic /marker/unique
    """
    global unique_id
    global non_unique_id
    unique_id = msg.data[0]
    non_unique_id = msg.data[1]


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

    pub_odom.publish(t)


def data_association(m):
    """
      Using transform detected marker->odom, calculate relative difference in yaw rotation using map->static markers of
      [id] and send the best match to broadcast_transform() (if match).
      :param m:
      :return t: Transform of map->odom using the most correct marker found.
      """
    best_marker = None
    marker_name_extension = 'None found'
    best_delta = 100
    best_yaw = 100

    frame_detected = 'aruco/detected' + str(m.id)

    #m.header.stamp = rospy.Time(0)

    n = 0
    while True:
        frame_map = "aruco/marker" + str(non_unique_id) + '_' + str(n)
        # Find transform of pose of static marker in map
        try:
            #print(frame_map)
            t_map = tf_buf.lookup_transform(frame_detected, frame_map, m.header.stamp, rospy.Duration(tf_timeout))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            print(e)
            break  # if n exceeds number for last non-unique static marker in map, end the loop.
        trans_map, rot_map = transform_stamped_to_pq(t_map)

        delta = np.linalg.norm(trans_map)
        roll, pitch, yaw = euler_from_quaternion(rot_map)
        yaw = roll # because of aruco marker orientation
        # rospy.loginfo('\n\nn: %s,\n frame_detected: %s,\n frame_map: %s,\n rot_map: %s,\n roll: %s,\n pitch: %s,\n yaw: %s\n\n',
        #               n, frame_detected, frame_map, rot_map, roll, pitch, yaw)

        orientation_error = math.pi / 6
        if np.abs(yaw) <= orientation_error:
            #print("delta_norm for {} is {};\n yaw info: best {} curr {}".format(str(non_unique_id) + '_' + str(n), delta, best_yaw, yaw))
            # if VERBOSE:
            # rospy.loginfo('\n\nn: %s,\n yaw: %s,\n best_yaw: %s,\n delta: %s,\n best_delta: %s\n\n',
            #                            n, yaw, best_yaw, delta, best_delta)

            if (best_marker is None or np.abs(yaw) <= best_yaw) and delta < best_delta:
                best_marker = t_map
                best_delta = delta
                best_yaw = np.abs(yaw)
                marker_name_extension = str(non_unique_id) + '_' + str(n)
        n += 1
    # print('best: ' + marker_name_extension)
    if best_delta == 100 or best_yaw == 100:  # If no marker found was good enough, return nothing
        return None
    # rospy.loginfo('\n\nbroadcasting %s\n\n',
    #               marker_name_extension)
    broadcast_transform(m, marker_name_extension)


def msg_to_pq(msg):
    """Convert a C{geometry_msgs/Transform} into position/quaternion np arrays

    @param msg: ROS message to be converted
    @return:
      - p: position as a np.array
      - q: quaternion as a numpy array (order = [x,y,z,w])
    """
    p = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
    q = np.array([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                  msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
    return p, q


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


def is_localized():
    """
    Publish a boolean True to 'localisation/is_localised' topic
    """
    pub.publish(Bool(data=True))


def main():
    rospy.spin()


print('Starting...')
rospy.init_node('odom_publisher')
tf_buf = tf2_ros.Buffer()
tf_lstn = tf2_ros.TransformListener(tf_buf)
br = tf2_ros.TransformBroadcaster()
sub_marker = rospy.Subscriber('/aruco/markers', MarkerArray, marker_callback, queue_size=1, buff_size=2**24)
sub_unique = rospy.Subscriber('/marker/unique', Int16MultiArray, unique_callback, queue_size=1, buff_size=2**24)
pub = rospy.Publisher('localisation/is_localised', Bool, queue_size=10)
pub_odom = rospy.Publisher('/kf4/input', TransformStamped, queue_size=10)
#pub_odom = rospy.Publisher('/localisation/moving_average_input', TransformStamped, queue_size=10)

tf_timeout = rospy.get_param('~tf_timeout', 0.1)
print('Ready')


if __name__ == '__main__':
    main()
