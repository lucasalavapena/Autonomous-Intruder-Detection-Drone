#!/usr/bin/env python

import sys
import math
import json

import rospy
import tf2_ros 
from tf.transformations import quaternion_from_euler
from geometry_msgs.msg import TransformStamped, Vector3
from std_msgs.msg import Int16MultiArray

"""
Read in the world.json file describing the chosen world using an argument, and for each aruco marker create a transform
map->aruco/marker[id], and if necessary add a unique identifier to markers with non-unique id's. Assumes only 2 
different ID's in map. One ID is unique, and one ID is not unique. Publishes both ID's as a list.
"""


def transform_from_marker(m, n, unique):
    """
    Create a transformation from map to the point of the aruco marker.
    :param m:
    :return t: TransposeStamped message with transformation map->aruco/marker[id]_uniqueID
    """
    t = TransformStamped()
    t.header.frame_id = 'map'

    if m['id'] == unique:
        t.child_frame_id = 'aruco/marker' + str(m['id'])
    else:
        t.child_frame_id = 'aruco/marker' + str(m['id']) + '_' + str(n)  # ex: aruco/marker0_0

    t.transform.translation = Vector3(*m['pose']['position'])
    roll, pitch, yaw = m['pose']['orientation']
    (t.transform.rotation.x,
     t.transform.rotation.y,
     t.transform.rotation.z,
     t.transform.rotation.w) = quaternion_from_euler(math.radians(roll),
                                                     math.radians(pitch),
                                                     math.radians(yaw))
    return t


def main(argv=sys.argv):

    # Initialize node, publisher and broadcaster
    rospy.init_node('display_markers_map')
    pub = rospy.Publisher('/marker/unique', Int16MultiArray, queue_size=10)
    broadcaster = tf2_ros.StaticTransformBroadcaster()

    # Let ROS filter through the arguments
    args = rospy.myargv(argv=argv)

    # Load world JSON
    with open(args[1], 'rb') as f:
        world = json.load(f)

    # Create a transform for each marker
    ids = [marker['id'] for marker in world['markers']]  # Get all ID's
    unique_ind = [ids.index(i) for i in set(ids) if ids.count(i) == 1]  # Find index of unique ID
    unique = ids[unique_ind[0]]  # Get unique ID

    id_list = [unique]
    for x in set(ids):
        if not x == unique:
            id_list.append(x)

    transforms = []
    n = 0
    for m in world['markers']:
        if m['id'] == unique:
            t = transform_from_marker(m, n, unique)
            transforms.append(t)
        else:
            t = transform_from_marker(m, n, unique)
            transforms.append(t)
            n += 1  # Only increase in number for the non-unique ID

    # Publish these transforms statically forever
    broadcaster.sendTransform(transforms)
    while not rospy.is_shutdown():  # Publish the set of ID's
        array = Int16MultiArray(data=id_list)
        pub.publish(array)
        rospy.sleep(0.1)


if __name__ == "__main__":
    main()
