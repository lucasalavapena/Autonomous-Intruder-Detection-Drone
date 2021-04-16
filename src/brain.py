#!/usr/bin/env python

import rospy
import os
import tf2_ros
import tf2_geometry_msgs
from planning.scripts import planning, planning_utils, exploration_utils

from std_msgs.msg import Bool

is_localised = None


def is_localised_callback(msg):
    global is_localised
    # print(msg.data)
    is_localised = msg.data
    # print('Drone is localised. Safe to fly.')

def main():
    print("RUNNING...")
    rate = rospy.Rate(20)  # Hz
    my_path = os.path.abspath(os.path.dirname(__file__))
    # map_path = os.path.join(my_path, "course_packages/dd2419_resources/worlds_json", "lucas_room_screen.world.json")
    map_path = os.path.join(my_path, "course_packages/dd2419_resources/worlds_json", "DA_test.world.json")
    Dora = exploration_utils.DoraTheExplorer(map_path)
    planner = planning.PathPlanner(Dora)
    world_map = planning_utils.Map(map_path)


    while not rospy.is_shutdown():
        rate.sleep()
        if is_localised:
            if planner.pose_map is not None:
                
                print("RRT start")
                x = planner.pose_map.pose.position.x
                y = planner.pose_map.pose.position.y

                next_best_point, _ = planner.explorer.generate_next_best_view((x, y))

                path = planning_utils.RRT(x, y, next_best_point[0], next_best_point[1], world_map)
                rospy.loginfo_throttle(5, 'Path:\n%s', path)

                path_msg = [planner.create_msg(a, b, 0.3) for (a, b) in path]
                print("are you ready to rumble on your marks get set go")
                for pnt in path_msg:
                    print("lets publish")
                    planner.publish_cmd(pnt)
                    rospy.loginfo_throttle(5, 'map loc:\n%s %s', planner.pose_map.pose.position.x, planner.pose_map.pose.position.y)

                    while not planner.goal_is_met(planner.current_goal_odom, planner.current_info):
                        planner.publish_cmd(pnt)
                        rate.sleep()
                    planner.d360_yaw()

def test_occ_map():
    my_path = os.path.abspath(os.path.dirname(__file__))
    file = "DA_test.world.json"
    map_path = os.path.join(my_path, "../..", "course_packages/dd2419_resources/worlds_json", file)
    # print(map_path)
    Dora = DoraTheExplorer(map_path)
    planner = PathPlanner(Dora)
    rospy.sleep(2)
    world_map = Map(map_path)
    path = RRT(0, 0, 1.4, 0.8, world_map)
    print("path", path)
    rate = rospy.Rate(10)  # Hz
    while not rospy.is_shutdown():
        rate.sleep()
        if is_localised:
            if planner.pose_map is not None:
                x = planner.pose_map.pose.position.x
                y = planner.pose_map.pose.position.y

                next_best_point, _ = planner.explorer.generate_next_best_view((x, y))

                path = RRT(x, y, next_best_point[0], next_best_point[1], world_map)
                rospy.loginfo_throttle(5, 'Path:\n%s', path)
                planner.goal_is_met(None, None)
                rospy.sleep(3)

rospy.init_node('brain')
sub = rospy.Subscriber('localisation/is_localised', Bool, is_localised_callback)

if __name__ == "__main__":
    test_occ_map()
