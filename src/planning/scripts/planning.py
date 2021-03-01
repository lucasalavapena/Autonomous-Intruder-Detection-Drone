#!/usr/bin/env python
import math
import rospy
import tf2_ros
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import PoseStamped
from crazyflie_driver.msg import Position

from planning_utils import Map, RRT


ERROR_TOLERANCE = 0.1

current_info = None

def goal_callback(msg):
    global current_info

    rospy.loginfo_throttle(5, 'New position read:\n%s', msg)
    current_info = msg

def goal_is_met(goal, current_info):
    if (goal.pose.position.x + ERROR_TOLERANCE > current_info.pose.position.x > goal.pose.position.x - ERROR_TOLERANCE and
            goal.pose.position.y + ERROR_TOLERANCE > current_info.pose.position.y > goal.pose.position.y - ERROR_TOLERANCE and
            goal.pose.position.z + ERROR_TOLERANCE > current_info.pose.position.z > goal.pose.position.z - ERROR_TOLERANCE):
        return True
    else:
        return False
    return False


def create_msg(x, y, z):
    msg = PoseStamped()
    msg.header.frame_id = 'map'
    msg.pose.position.x = x
    msg.pose.position.y = y
    msg.pose.position.z = z

    return msg


def publish_cmd(goal):

    timeout = rospy.Duration(0.1)
    while not tf_buf.can_transform(goal.header.frame_id, 'cf1/odom', goal.header.stamp, timeout):
        rospy.logwarn_throttle(5.0, 'No transform from %s to cf1/odom' % goal.header.frame_id)
        return
    goal_odom = tf_buf.transform(goal, 'cf1/odom')

    cmd = Position()

    cmd.header = goal_odom.header

    cmd.x = goal_odom.pose.position.x
    cmd.y = goal_odom.pose.position.y
    cmd.z = goal_odom.pose.position.z

    roll, pitch, yaw = euler_from_quaternion((goal_odom.pose.orientation.x,
                                              goal_odom.pose.orientation.y,
                                              goal_odom.pose.orientation.z,
                                              goal_odom.pose.orientation.w))

    cmd.yaw = math.degrees(yaw)
    pub_cmd.publish(cmd)


rospy.init_node('planning')
sub_goal = rospy.Subscriber('/cf1/pose', PoseStamped, goal_callback)
pub_cmd = rospy.Publisher('/cf1/cmd_position', Position, queue_size=2)
tf_buf = tf2_ros.Buffer()
tf_lstn = tf2_ros.TransformListener(tf_buf)


def main():
    world_map = Map("course_packages/dd2419_resources/worlds_json/planning_test_map.json")
    path = RRT(0, 0, 1, 1.9, world_map)

    goals = [create_msg(x, y, 0.5) for (x, y) in path]
    rate = rospy.Rate(10)  # Hz
    i = 0
    while not rospy.is_shutdown():
        goal = goals[i]
        publish_cmd(goal)
        if current_info is not None:
            if goal_is_met(goal, current_info) and goal != goals[-1]:
                i += 1

if __name__ == '__main__':
    main()