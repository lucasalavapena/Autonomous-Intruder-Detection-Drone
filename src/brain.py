#!/usr/bin/env python

import rospy
import os
import tf2_ros
import tf2_geometry_msgs
from planning.scripts import planning, planning_utils
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
    planner = planning.PathPlanner()
    my_path = os.path.abspath(os.path.dirname(__file__))
    map_path = os.path.join(my_path, "course_packages/dd2419_resources/worlds_json", "lucas_room_screen.world.json")
    world_map = planning_utils.Map(map_path)
    while not rospy.is_shutdown():
        # print(is_localised)
        rate.sleep()
        # print("is localized: ",is_localised)
        if is_localised:
            print("localised")
            setpoints = [[1.6, 1], [0.4, 0.4]]#, [], [], []]#[[0.5, 0.5], [2.0, 0.3], [2.0, 1.1], [0.3, 1.1], [0.4, 0.4]]
            ind = 0
            # print("planner.current_info: ", planner.current_info)
            if planner.current_info is not None:
                print("RRT stat")
                x = planner.current_info.pose.position.x
                y = planner.current_info.pose.position.y
                path = planning_utils.RRT(x, y, setpoints[ind][0], setpoints[ind][1], world_map)
                print(path)
                rospy.loginfo_throttle(5, 'Path:\n%s', path)
                #
                # print(setpoints[ind])
                # print("PATH: ", path)
                path_msg = [planner.create_msg(x, y, 0.5) for (x, y) in path]
                for pnt in path_msg:
                    planner.publish_cmd(pnt)
                    while not planner.goal_is_met(planner.current_goal_odom, planner.current_info):
                        # print("Current Position: {},{}".format(planner.current_info.pose.position.x, planner.current_info.pose.position.y))
                        # print("Goal Position: {},{}".format(pnt.pose.position.x, pnt.pose.position.y))
                        planner.publish_cmd(pnt)
                        rate.sleep()

                    print("GOT TO CHECKPOINT")
                ind += 1
                if ind == 4:
                    ind = 0
                #print(setpoints[ind])


rospy.init_node('brain')
sub = rospy.Subscriber('localisation/is_localised', Bool, is_localised_callback)

if __name__ == "__main__":
    main()
