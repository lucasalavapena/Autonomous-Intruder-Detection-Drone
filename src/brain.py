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
    Dora = exploration_utils.DoraTheExplorer(map_path, (start_x, start_y))
    planner = planning.PathPlanner(Dora)
    world_map = planning_utils.Map(map_path)

    ind = 0

    path_compute = False

    while not rospy.is_shutdown():
        rate.sleep()
        if is_localised:
            # print(planner.pose_map )
            if planner.pose_map is not None:
                
                print("RRT start")
                x = planner.pose_map.pose.position.x
                y = planner.pose_map.pose.position.y
                
                if not path_compute:
                    planner.explorer.update_current_position((x, y))
                    overall_path = planner.explorer.generate_next_best_view()
                    path_compute = True



                path = planning_utils.RRT(x, y, overall_path[ind][0], overall_path[ind][1], world_map)
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
                    print("GOT TO CHECKPOINT", " goal: {} {} current:{} {}".format(planner.current_goal_odom.x,
                                                                             planner.current_goal_odom.y,
                                                                             planner.current_info.pose.position.x,
                                                                            planner.current_info.pose.position.y) )
                print("Finished point {}".format(ind))
                ind += 1
                print("ind {}".format(ind) )
                if ind == len(overall_path):
                    ind = 0
                #print(setpoints[ind])


rospy.init_node('brain')
sub = rospy.Subscriber('localisation/is_localised', Bool, is_localised_callback)

if __name__ == "__main__":
    main()
