#!/usr/bin/env python
from __future__ import print_function

import math
import rospy
import tf2_ros
from tf.transformations import euler_from_quaternion
from nav_msgs.msg import OccupancyGrid
import tf2_geometry_msgs
from geometry_msgs.msg import PoseStamped
from crazyflie_driver.msg import Position
import os.path
from planning_utils import Map, RRT
from exploration_utils import DoraTheExplorer

class PathPlanner:

    def __init__(self, Explorer):
        self.sub_goal = rospy.Subscriber('/cf1/pose', PoseStamped, self.goal_callback)
        self.pub_cmd = rospy.Publisher('/cf1/cmd_position', Position, queue_size=2)
        self.tf_buf = tf2_ros.Buffer()
        self.tf_lstn = tf2_ros.TransformListener(self.tf_buf)


        self.current_goal_odom = None
        self.current_info = None
        self.pose_map = None
        self.ERROR_TOLERANCE = 0.06

        self.explorer = Explorer
        self.occ_grid_pub = rospy.Publisher('/explorer_occ_map/', OccupancyGrid, queue_size=1)
        self.occ_grid = OccupancyGrid()

    def goal_callback(self, msg):

        # rospy.loginfo_throttle(5, 'New position read:\n%s', msg)
        self.current_info = msg
        # print(self.current_info)
        # self.current_info.header.stamp = rospy.Time.now()
        self.convert_pose_to_map()

    def goal_is_met(self, goal, current_info):
        # rospy.loginfo_throttle(5, 'current_info:\n%s', current_info)
        # rospy.loginfo_throttle(5, 'goal:\n%s', goal)
        self.occ_grid_pub =
        if (goal.x + self.ERROR_TOLERANCE > current_info.pose.position.x > goal.x - self.ERROR_TOLERANCE and
                goal.y + self.ERROR_TOLERANCE > current_info.pose.position.y > goal.y - self.ERROR_TOLERANCE and
                goal.z + self.ERROR_TOLERANCE > current_info.pose.position.z > goal.z - self.ERROR_TOLERANCE):
            return True
        else:
            return False

    def create_msg(self, x, y, z):
        msg = PoseStamped()
        msg.header.frame_id = 'map'
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z
        msg.pose.orientation.y = 0
        msg.pose.orientation.x = 0
        msg.pose.orientation.z = 0
        msg.pose.orientation.w = 1
        return msg

    def convert_pose_to_map(self):
        timeout = rospy.Duration(0.5)
        while not self.tf_buf.can_transform(self.current_info.header.frame_id, 'map', self.current_info.header.stamp, timeout):
            rospy.logwarn_throttle(5.0, 'hello there..No transform from %s to map' % self.current_info.header.frame_id)
            return
        self.pose_map = self.tf_buf.transform(self.current_info, 'map')

    def rotation_is_met(self, target_yaw, error_d_tol=7.5):
        """

        :param target_yaw:
        :param error_d_tol: error tolerance for yaw in degrees
        :return:
        """
        _, _, actual_yaw = euler_from_quaternion((self.current_info.pose.orientation.x,
                                           self.current_info.pose.orientation.y,
                                            self.current_info.pose.orientation.z,
                                           self.current_info.pose.orientation.w))

        actual_yaw_d = math.degrees(actual_yaw) % 360
        rospy.loginfo_throttle(5, "actual angles: %f target angle: %f", actual_yaw_d, target_yaw)

        if ( actual_yaw_d + error_d_tol > target_yaw > actual_yaw_d - error_d_tol):
            return True
        else:
            return False

    def d360_yaw(self):
        cmd = Position()

        cmd.x = self.current_info.pose.position.x
        cmd.y = self.current_info.pose.position.y
        cmd.z = self.current_info.pose.position.z

        _, _, initial_yaw = euler_from_quaternion((self.current_info.pose.orientation.x,
                                           self.current_info.pose.orientation.y,
                                            self.current_info.pose.orientation.z,
                                           self.current_info.pose.orientation.w))

        initial_yaw = math.degrees(initial_yaw)


        delta_yaw = [45 * i for i in range(1, 8)]

        for d_yaw in delta_yaw:
            cmd.yaw = (initial_yaw + d_yaw) % 360
            while not self.rotation_is_met(cmd.yaw):
                self.pub_cmd.publish(cmd)


    def publish_cmd(self, goal):
        goal.header.stamp = rospy.Time.now()
        timeout = rospy.Duration(0.5)

        while not self.tf_buf.can_transform(goal.header.frame_id, 'cf1/odom', goal.header.stamp, timeout):
            rospy.logwarn_throttle(5.0, 'No transform from %s to cf1/odom' % goal.header.frame_id)
            return
        goal_odom = self.tf_buf.transform(goal, 'cf1/odom')

        # rospy.loginfo_throttle(5, 'goal in odom:\n%s', goal_odom)

        cmd = Position()

        cmd.header.stamp = rospy.Time.now()
        cmd.header.frame_id = goal_odom.header.frame_id

        cmd.x = goal_odom.pose.position.x
        cmd.y = goal_odom.pose.position.y
        cmd.z = goal_odom.pose.position.z

        roll, pitch, yaw = euler_from_quaternion((goal_odom.pose.orientation.x,
                                                goal_odom.pose.orientation.y,
                                                goal_odom.pose.orientation.z,
                                                goal_odom.pose.orientation.w))

        cmd.yaw = math.degrees(yaw)
        self.pub_cmd.publish(cmd)
        self.current_goal_odom = cmd


def exploration(method="next_best_view"):
    # TODO this needs to use RTT
    rospy.init_node('planning')
    my_path = os.path.abspath(os.path.dirname(__file__))
    file = "lucas_room_screen.json"
    map_path = os.path.join(my_path, "../..", "course_packages/dd2419_resources/worlds_json", file)
    # print(map_path)
    planner = PathPlanner()
    rospy.sleep(2)
    Dora = DoraTheExplorer(map_path)
    path = Dora.generate_next_best_view()
    print("path", path)
    goals = [planner.create_msg(x, y, 0.5) for (x, y) in path]
    rate = rospy.Rate(10)  # Hz
    i = 0
    while not rospy.is_shutdown():
        goal = goals[i]
        planner.publish_cmd(goal)
        if planner.current_info is not None:
            print("planner goal", planner.current_goal_odom, "current pose in odom", planner.current_info)
            if planner.goal_is_met(planner.current_goal_odom, planner.current_info) and goal != goals[-1]:
                # TODO do 360 degree rotation slowly
                planner.d360_yaw()
                print("Goal met")
                # print(goal)
                i += 1




def main(file="planning_test_map.json"):
    rospy.init_node('planning')
    my_path = os.path.abspath(os.path.dirname(__file__))
    file = "lucas_room_screen.json"
    map_path = os.path.join(my_path, "../..", "course_packages/dd2419_resources/worlds_json", file)
    # print(map_path)
    planner = PathPlanner()
    rospy.sleep(2)
    world_map = Map(map_path)
    path = RRT(0, 0, 1.4, 0.8, world_map)
    print("path", path)
    goals = [planner.create_msg(x, y, 0.5) for (x, y) in path]
    rate = rospy.Rate(10)  # Hz
    i = 0
    while not rospy.is_shutdown():
        goal = goals[i]
        planner.publish_cmd(goal)
        if planner.current_info is not None:
            print("planner goal", planner.current_goal_odom, "current pose in odom",planner.current_info)
            if planner.goal_is_met(planner.current_goal_odom, planner.current_info) and goal != goals[-1]:
                print("Goal met")
                # print(goal)
                i += 1

if __name__ == '__main__':
    exploration()