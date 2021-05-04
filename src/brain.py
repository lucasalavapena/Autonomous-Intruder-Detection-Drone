#!/usr/bin/env python

import rospy
import os
import tf2_ros
import tf2_geometry_msgs
import numpy as np
import argparse
from planning.scripts import planning, planning_utils, exploration_utils
from tf.transformations import euler_from_quaternion

from std_msgs.msg import Bool, Float32

is_localised = None

emergency_landing = False

def is_localised_callback(msg):
    global is_localised
    # print(msg.data)
    is_localised = msg.data
    # print('Drone is localised. Safe to fly.')


def battery_callback(data):
    global emergency_landing
    battery = (data.data - 3.0) / (4.23 - 3.0) * 100

    if battery < 10:
        emergency_landing = True

def main(args):
    print("RUNNING...")
    rate = rospy.Rate(20)  # Hz
    my_path = os.path.abspath(os.path.dirname(__file__))
    map_path = os.path.join(my_path, "course_packages/dd2419_resources/worlds_json",
                            "{}.world.json".format(args.map))
    Dora = exploration_utils.DoraTheExplorer(map_path, discretization=args.discretization,
                                             CrazyFlie_Render=args.render_distance)
    planner = planning.PathPlanner(Dora)
    world_map = planning_utils.Map(map_path, expansion_factor=args.expansion_factor)
    rospy.sleep(5) # to one time to record the bag and prepare
    has_taken_off = False

    while not rospy.is_shutdown():
        rate.sleep()
        if is_localised:
            if planner.pose_map is not None:

                # emergency landing
                # if emergency_landing:
                #     print("emergency landing")
                #     break

                print("RRT start")
                #
                # if not has_taken_off:
                #     x = planner.pose_map.pose.position.x
                #     y = planner.pose_map.pose.position.y
                #     has_taken_off = True
                # else:
                #     x = next_best_point[0]
                #     y = next_best_point[1]


                x = planner.pose_map.pose.position.x
                y = planner.pose_map.pose.position.y
                # has_taken_off = True


                quat = np.array([planner.pose_map.pose.orientation.x, planner.pose_map.pose.orientation.y,
                                 planner.pose_map.pose.orientation.z, planner.pose_map.pose.orientation.w])
                _, _, curr_yaw = euler_from_quaternion(quat)

                print("current yaw is", curr_yaw)
                next_best_point, score = planner.explorer.generate_next_best_view((x, y))
                planner.publish_occ() # publishes occupancy grid

                if next_best_point is None and score is None:
                    print("results are None")
                    continue


                path = planning_utils.RRT(x, y, next_best_point[0], next_best_point[1], curr_yaw, world_map)
                rospy.loginfo_throttle(1, 'Path:\n%s', path)

                # If path is empty we are outside of the map
                # try to go into the map
                # if len(path) == 0:
                #     path = planning_utils.escape_boundary(x, y, next_best_point[0], next_best_point[1], world_map)

                path_msg = [planner.create_msg(x_loc, y_loc, 0.3, yaw_angle=theta) for (x_loc, y_loc, theta) in path]



                # First it should always go straight up to make it easy for it
                if path:
                    if not has_taken_off:
                        path_msg.insert(0, planner.create_msg(x, y, 0.3))
                        has_taken_off = True
                        # planner.publish_occ()
                else:
                    print('No Path')
                # rospy.sleep(5)

                for pnt in path_msg:
                    planner.publish_cmd(pnt)

                    while not planner.goal_is_met(planner.current_goal_odom):
                        planner.publish_cmd(pnt)
                        rate.sleep()

                    if not has_taken_off:
                        has_taken_off = True
                        planner.d360_yaw()

                if has_taken_off:
                    planner.d360_yaw()
                    # planner.explorer.update_occ_grid((next_best_point[0], next_best_point[1]))
                    planner.publish_occ()

                # planner.publish_occ()

                print("Completed best view point")

# arguments
parser = argparse.ArgumentParser(
    description="BRAINSSSSS"
)
parser.add_argument('map', type=str, help='filename of world.json map')
parser.add_argument('--discretization', type=float, default=0.05, help='discretization of map for explorer in m/cell')
parser.add_argument('--expansion-factor', type=float, default=0.1, help='expansion_factor of map')
parser.add_argument('--render-distance', type=float, default=0.85, help='CrazyFlie Render Distance')


args = parser.parse_args()

rospy.init_node('brain')
sub = rospy.Subscriber('localisation/is_localised', Bool, is_localised_callback)
rospy.Subscriber("/cf1/battery", Float32, battery_callback)
if __name__ == "__main__":
    if args.map:
        main(args)
    else:
        print("You need to give the file name [without the .world.json] of the map in the world_json dir")