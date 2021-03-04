#!/usr/bin/env python

import rospy
import tf2_ros
import tf2_geometry_msgs
from std_msgs.msg import Bool


def is_localised():
    msg = Bool()
    time = rospy.Time.now()
    rospy.sleep(1)
    try:
        if tf_buf.can_transform(frame_id, child_frame_id, time, rospy.Duration(tf_timeout)):
            msg.data = True
        else:
            msg.data = False
            rospy.logwarn_throttle(5.0, '%s: No transform from %s to %s', rospy.get_name(), child_frame_id, frame_id)
    except rospy.exceptions.ROSTimeMovedBackwardsException:
        pass
    return msg


def main():
    pub = rospy.Publisher('localisation/is_localised', Bool, queue_size=10)
    rate = rospy.Rate(10)  # 10Hz
    while not rospy.is_shutdown():
        pub.publish(is_localised())
        try:
            rate.sleep()
        except rospy.exceptions.ROSTimeMovedBackwardsException:
            continue


if __name__ == "__main__":
    rospy.init_node('is_localized_publisher')

    tf_buf = tf2_ros.Buffer()
    tf_lstn = tf2_ros.TransformListener(tf_buf)
    tf_timeout = rospy.get_param('~tf_timeout', 0.1)
    frame_id = rospy.get_param('~frame_id', 'map')
    child_frame_id = rospy.get_param('~child_frame_id', 'cf1/odom')

    main()
