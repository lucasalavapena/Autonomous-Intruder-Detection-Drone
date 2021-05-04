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


filt_tresh = np.pi/12

def sign_callback(transform):
    """
    msg contains list of detected aruco markers and for each marker, determine if it's the unique marker or a
    non-unique marker. If unique, send msg to broadcast_transform(). Otherwise send msg to data_association().
    """
    global filt_tresh
    # p, q = transform_stamped_to_pq(transform)
    # delta = np.linalg.norm(p)
    #print('delta ' + str(delta))
    # roll, pitch, yaw = euler_from_quaternion(q)
    #print('roll: ' + str(abs(abs(roll)-np.pi/2)) + ' pitch: ' + str(pitch) + ' yaw: ' + str(abs(abs(yaw)-np.pi/2)))
    #print('treshold: ' + str(filt_tresh))
    # if delta < 1.5 and abs(pitch) < filt_tresh and abs(abs(roll)-np.pi/2) < filt_tresh and abs(abs(yaw)-np.pi/2) < filt_tresh:
    sign_filtering(transform)
    # print("USED TRANSFORM HELL YEAH BOOOOJ")


def broadcast_sign_transform(s):
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
        detected = tf_buf.lookup_transform(s.child_frame_id, 'cf1/odom', s.header.stamp, rospy.Duration(tf_timeout))
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
        print('odom_publisher.broadcast_transform(detected lookup): ', e)
        return
    trans_detected, rot_detected = transform_stamped_to_pq(detected)

    # Find transform of pose of static marker in map frame
    try:
        t_map = tf_buf.lookup_transform('map', 'landmark/' + s.child_frame_id[18:], s.header.stamp)
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
    t.header = s.header
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

def sign_filtering(s):
    """
      Using transform detected marker->odom, calculate relative difference in yaw rotation using map->static markers of
      [id] and send the best match to broadcast_transform() (if match).
      :param m:
      :return t: Transform of map->odom using the most correct marker found.
      """

    #m.header.stamp = rospy.Time(0)

    frame_map =  'landmark/' + s.child_frame_id[18:]
    # Find transform of pose of static marker in map
    try:
        #print(frame_map)
        t_map = tf_buf.lookup_transform(s.child_frame_id, frame_map, s.header.stamp, rospy.Duration(tf_timeout))
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
        print(e)
        return  # if n exceeds number for last non-unique static marker in map, end the loop.
    trans_map, rot_map = transform_stamped_to_pq(t_map)

    delta = np.linalg.norm(trans_map)
    roll, pitch, yaw = euler_from_quaternion(rot_map)
    yaw = roll # because of aruco marker orientation


    orientation_error = math.pi / 30
    if np.abs(yaw) <= orientation_error and delta <= 0.25:
        broadcast_sign_transform(s)
        print("\n########## USED DETECTED SIGN TRANSFORM ##########\nyaw: {YAW}, delta: {DELTA}\n".format(YAW=abs(yaw), DELTA=delta))
    else:
        print("sign filtered | yaw: {YAW}, delta: {DELTA}".format(YAW=abs(yaw), DELTA=delta))



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


def main():
    rospy.spin()


#print('Starting...')
rospy.init_node('sign_odom_publisher')
tf_buf = tf2_ros.Buffer()
tf_lstn = tf2_ros.TransformListener(tf_buf)
br = tf2_ros.TransformBroadcaster()
sign_sub = rospy.Subscriber('/sign/detected', TransformStamped, sign_callback, queue_size=1, buff_size=2**24)
pub_odom = rospy.Publisher('/kf4/output', TransformStamped, queue_size=10)
#pub_odom = rospy.Publisher('/localisation/moving_average_input', TransformStamped, queue_size=10)

tf_timeout = rospy.get_param('~tf_timeout', 0.1)
print('Ready')


if __name__ == '__main__':
    main()
