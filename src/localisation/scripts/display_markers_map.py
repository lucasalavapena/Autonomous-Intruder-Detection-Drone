#!/usr/bin/env python

import sys
import math
import json

import rospy
import tf2_ros 
from tf.transformations import quaternion_from_euler
from geometry_msgs.msg import TransformStamped, Vector3
from std_msgs.msg import Int16


def transform_from_marker(m, n, unique):
    t = TransformStamped()
    t.header.frame_id = 'map'

    if m['id'] == unique:
        t.child_frame_id = 'aruco/marker' + str(m['id'])
    else:
        t.child_frame_id = 'aruco/marker' + str(m['id']) + '_' + str(n)

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

    rospy.init_node('display_markers_map')
    pub = rospy.Publisher('/marker/unique', Int16, queue_size=10)
    broadcaster = tf2_ros.StaticTransformBroadcaster()

    # Let ROS filter through the arguments
    args = rospy.myargv(argv=argv)

    # Load world JSON
    with open(args[1], 'rb') as f:
        world = json.load(f)

    #f = open('/home/joakim/dd2419_project/src/course_packages/dd2419_resources/worlds_json/DA_test.world.json', 'rb')
    #world = json.load(f)

    # Create a transform for each marker
    ids = [marker['id'] for marker in world['markers']]
    unique_ind = [ids.index(i) for i in set(ids) if ids.count(i) == 1]
    unique = ids[unique_ind[0]]

    transforms = []
    n = 0
    for m in world['markers']:
        if m['id'] == unique:
            t = transform_from_marker(m, n, unique)
            transforms.append(t)
        else:
            t = transform_from_marker(m, n, unique)
            transforms.append(t)
            n += 1

    # Publish these transforms statically forever
    broadcaster.sendTransform(transforms)
    while not rospy.is_shutdown():
        pub.publish(unique)


if __name__ == "__main__":
    main()
