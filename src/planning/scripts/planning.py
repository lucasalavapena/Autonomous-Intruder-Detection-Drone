#!/usr/bin/env python
from __future__ import print_function

import math
import rospy
import tf2_ros
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from nav_msgs.msg import OccupancyGrid
import tf2_geometry_msgs
from geometry_msgs.msg import PoseStamped
from crazyflie_driver.msg import Position
import os.path
from planning_utils import Map, RRT
from exploration_utils import DoraTheExplorer
import numpy as np

class PathPlanner:
    def __init__(self, Explorer, error_tol=0.03):
        self.sub_goal = rospy.Subscriber('/cf1/pose', PoseStamped, self.pose_callback)
        self.pub_cmd = rospy.Publisher('/cf1/cmd_position', Position, queue_size=2)
        self.tf_buf = tf2_ros.Buffer()
        self.tf_lstn = tf2_ros.TransformListener(self.tf_buf)


        self.current_goal_odom = None
        self.current_info = None
        self.pose_map = None
        self.ERROR_TOLERANCE = error_tol

        self.explorer = Explorer
        self.occ_grid_pub = rospy.Publisher('/explorer_occ_map', OccupancyGrid, queue_size=2)
        self.occ_grid_pub.publish(self.explorer.occ_grid)

    def pose_callback(self, msg):
        """
        callback for pose subsciber
        :param msg:msg from subscriber
        :return:
        """
        self.current_info = msg
        self.convert_pose_to_map()

    def goal_is_met(self, goal):
        """
        checks if the current goal is meet within a certain tolerance
        :param goal: goal position
        :return: true if the goal is met else false
        """
        # if self.current_info:
        #     rospy.loginfo_throttle(1, 'current_info:\n%s', self.current_info)
        #     rospy.loginfo_throttle(1, 'goal:\n%s', goal)
        if (goal.x + self.ERROR_TOLERANCE > self.current_info.pose.position.x > goal.x - self.ERROR_TOLERANCE and
                goal.y + self.ERROR_TOLERANCE > self.current_info.pose.position.y > goal.y - self.ERROR_TOLERANCE and
                goal.z + self.ERROR_TOLERANCE > self.current_info.pose.position.z > goal.z - self.ERROR_TOLERANCE):
            # print("goal met")
            # print("--------"*5)
            # print("current", current_info.pose.position)
            # print("goal", goal)
            # print("--------"*5)
            return True
        else:
            return False

    def create_msg(self, x, y, z, yaw_angle=None):
        """
        creates a posestamped message to be used for setting set points
        :param x:
        :param y:
        :param z:
        :return:
        """
        msg = PoseStamped()
        msg.header.frame_id = 'map'
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z
        if yaw_angle is None:
            msg.pose.orientation.y = 0
            msg.pose.orientation.x = 0
            msg.pose.orientation.z = 1
            msg.pose.orientation.w = 6.123234e-17
        else:
            msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = quaternion_from_euler(0, 0, yaw_angle)
        return msg

    def convert_pose_to_map(self):
        """
        converts the odom pose to a map pose (mainly used for planning purposes)
        :return:
        """
        timeout = rospy.Duration(0.25)
        while not self.tf_buf.can_transform(self.current_info.header.frame_id, 'map', self.current_info.header.stamp, timeout):
            rospy.logwarn_throttle(5.0, 'hello there..No transform from %s to map' % self.current_info.header.frame_id)
            return
        self.pose_map = self.tf_buf.transform(self.current_info, 'map')

    def rotation_is_met(self, target_yaw, error_d_tol=7.5):
        """
        checks if a target rotation is met
        :param target_yaw:
        :param error_d_tol: error tolerance for yaw in degrees
        :return:
        """
        _, _, actual_yaw = euler_from_quaternion((self.current_info.pose.orientation.x,
                                           self.current_info.pose.orientation.y,
                                            self.current_info.pose.orientation.z,
                                           self.current_info.pose.orientation.w))

        actual_yaw_d = math.degrees(actual_yaw) % 360
        # rospy.loginfo_throttle(5, "actual angles: %f target angle: %f", actual_yaw_d, target_yaw)

        if ( actual_yaw_d + error_d_tol > target_yaw > actual_yaw_d - error_d_tol):
            return True
        else:
            return False

    def publish_occ(self):
        """
        publish the occupancy grid and checks
        :return:
        """
        self.occ_grid_pub.publish(self.explorer.occ_grid)
        # Check if we need to reset our explorer here
        if np.sum(self.explorer.occ_grid.data != 0) / float(self.explorer.occ_grid.data.size) > 0.95:
            self.explorer.generate_map_occupancy()

    def d360_yaw(self):
        """
        360 degree yaw ie a whole rotation to see the entire surroundings of the map
        :return:
        """
        cmd = Position()

        cmd.x = self.current_info.pose.position.x
        cmd.y = self.current_info.pose.position.y
        cmd.z = self.current_info.pose.position.z

        _, _, initial_yaw = euler_from_quaternion((self.current_info.pose.orientation.x,
                                           self.current_info.pose.orientation.y,
                                            self.current_info.pose.orientation.z,
                                           self.current_info.pose.orientation.w))

        initial_yaw = math.degrees(initial_yaw)


        delta_yaw = [30 * i for i in range(1, 12)] # every 30 degrees so that we are careful, our fov is actully 140

        for i, d_yaw in enumerate(delta_yaw):
            cmd.yaw = (initial_yaw + d_yaw) % 360
            start = rospy.get_rostime()
            until = (start + rospy.Duration(0.1))
            if i % 3 == 0:
                until = (start + rospy.Duration(1))
            while not(self.rotation_is_met(cmd.yaw) and rospy.get_rostime() > until):
                # rospy.loginfo_throttle(0.2, "curr: %d until: %d", rospy.get_rostime().secs, until.secs)
                self.pub_cmd.publish(cmd)




    def publish_cmd(self, goal):
        """
        publishes goal position command
        :param goal: goal (pose) in map to be transfer to odom
        :return:
        """
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

def dora_test():
    rospy.init_node('planning')

    # world_map = Map("course_packages/dd2419_resources/worlds_json/planning_test_map.json")
    my_path = os.path.abspath(os.path.dirname(__file__))
    file = "dora_adventure_map.world.json" #"lucas_room_screen.world.json"
    map_path = os.path.join(my_path, "../..", "course_packages/dd2419_resources/worlds_json", file)
    Dora = DoraTheExplorer(map_path)
    print(Dora.generate_best_path((0.5, 0.3), False))

def transform_test():
    rospy.init_node('planning')
    # world_map = Map("course_packages/dd2419_resources/worlds_json/planning_test_map.json")
    my_path = os.path.abspath(os.path.dirname(__file__))
    file = "dora_adventure_map.world.json" #"lucas_room_screen.world.json"
    map_path = os.path.join(my_path, "../..", "course_packages/dd2419_resources/worlds_json", file)
    Dora = DoraTheExplorer(map_path)
    planner = PathPlanner(Dora)
    rate = rospy.Rate(20)  # Hz

    while not rospy.is_shutdown():
        rate.sleep()
        if planner.pose_map is not None:
            x = planner.pose_map.pose.position.x
            y = planner.pose_map.pose.position.y
            # print("pose", x, y)

def test_occ_map():
    rospy.init_node('planning')
    my_path = os.path.abspath(os.path.dirname(__file__))
    file = "DA_test.world.json"
    map_path = os.path.join(my_path, "../..", "course_packages/dd2419_resources/worlds_json", file)
    # print(map_path)
    Dora = DoraTheExplorer(map_path)
    planner = PathPlanner(Dora)
    print("test")
    rospy.sleep(2)
    world_map = Map(map_path)
    rate = rospy.Rate(10)  # Hz
    while not rospy.is_shutdown():
        rate.sleep()
        if planner.pose_map is not None:
            x = planner.pose_map.pose.position.x
            y = planner.pose_map.pose.position.y

            next_best_point, _ = planner.explorer.generate_next_best_view((x, y))

            path = RRT(x, y, next_best_point[0], next_best_point[1], world_map)
            rospy.loginfo_throttle(5, 'Path:\n%s', path)
            planner.goal_is_met(None, None)
            rospy.sleep(3)



def main(file="planning_test_map.json"):
    rospy.init_node('planning')
    my_path = os.path.abspath(os.path.dirname(__file__))
    file = "lucas_room_screen.json"
    map_path = os.path.join(my_path, "../..", "course_packages/dd2419_resources/worlds_json", file)
    # print(map_path)

    planner = PathPlanner(Dora)
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
    test_occ_map()