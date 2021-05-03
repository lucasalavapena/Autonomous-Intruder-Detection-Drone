from __future__ import division
from planning_utils import Map
import math
from itertools import product
import numpy as np
import os.path
import sys
import rospy
from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import Pose


import matplotlib.pyplot as plt
import time
import random
np.set_printoptions(threshold=sys.maxsize)

CrazyFlie_FOV = 140
# CrazyFlie_Render = 0.85 # 0.8

class CrazyflieCamera:
    """
    Crazyflie Camera class - kinda useless tbh
    """
    def __init__(self, FOV, render_distance):
        self.FOV = FOV #degrees
        self.render_distance = render_distance# m - distance we can see in front of us



class DoraTheExplorer:
    def __init__(self, map_path, discretization=0.05, CrazyFlie_Render=0.1, expansion_factor=0.1):
        """
        DoraTheExplorer constructor
        :param map_path: path to the world.json file
        :param discretization: discretization in m/cell
        """
        self.Map = Map(map_path, expansion_factor=expansion_factor)
        self.camera = CrazyflieCamera(CrazyFlie_FOV - 20, CrazyFlie_Render)
        self.mesh_grid = None
        self.visited_grid = None

        self.occ_grid = OccupancyGrid()
        self.occ_grid.header.stamp = rospy.Time.now()
        # Hard-coded

        self.points_set = None
        self.discretization = discretization
        self.generate_map_occupancy()
        # self.current_position = current_position
        self.path = []

    # def update_current_position(self, position):
    #     self.current_position = position

    def generate_map_occupancy(self):
        """
        generates key parameters based on the map information
        :return:
        """
        no_x = int(math.ceil((self.Map.airspace[3] - self.Map.airspace[0]) / self.discretization)) + 1
        no_y = int(math.ceil((self.Map.airspace[4] - self.Map.airspace[1]) / self.discretization)) + 1


        x_values = list(range(no_x))
        y_values = list(range(no_y))
        x_values = [i * self.discretization for i in x_values]
        y_values = [i * self.discretization for i in y_values]

        # self.free_space_grid = np.zeros(no_x, no_y)
        self.mesh_grid = [list(product([x_value], y_values)) for x_value in x_values]
        self.points_set = set(list(product(x_values, y_values)))
        self.visited_grid = np.zeros((no_x, no_y))
        # self.occ_grid.info = MapMetaData(width=no_y, height=no_x, resolution=self.discretization,
        #                                  map_load_time=rospy.Time.now())
        self.occ_grid.info = MapMetaData(width=no_x, height=no_y, resolution=self.discretization,
                                         map_load_time=rospy.Time.now())
        self.occ_grid.data = np.zeros((no_x, no_y)).ravel()
        # Hard coded
        self.occ_grid.info.origin = Pose()
        self.occ_grid.info.origin.position.x = 0.0
        self.occ_grid.info.origin.position.y = 0.0
        self.occ_grid.info.origin.position.z = 0.0
        self.occ_grid.info.origin.orientation.x = 0.0
        self.occ_grid.info.origin.orientation.y = 0.0
        self.occ_grid.info.origin.orientation.z = 0.0
        self.occ_grid.info.origin.orientation.w = 1.0


    def viewable_points(self, point, mode="Test"):
        """
        checks how many points are viewable from a specific point
        :param point: point as a tuple
        :param mode: mode to see if it should update parameters or simply check the viewability
        :return: visited grid of sorts (either a temp one or the actual one)
        """
        def is_neighbourhood_visited(visited_grid, number_of_idx, x_idx, y_idx):
            """
            checks if the neighbourhood of a point have already have been visited
            :param visited_grid:
            :param number_of_idx:
            :param x_idx:
            :param y_idx:
            :return: boolean
            """
            for i in range(-number_of_idx, number_of_idx + 1):
                for j in range(-number_of_idx, number_of_idx + 1):
                    if i ** 2 + j ** 2 <= number_of_idx and 0 <= x_idx + i < visited_grid.shape[0] and 0 <= y_idx + j < visited_grid.shape[1]:
                        if visited_grid[x_idx + i, y_idx + j] == 0:
                            return False
            return True

        x_idx = None
        y_idx = None
        # get the index of the point wrt discretised map
        for i in range(len(self.mesh_grid)):
            if self.mesh_grid[i][0][0] - self.discretization/2 <= point[0] <= self.mesh_grid[i][0][0] + self.discretization/2:
                x_idx = i

        for j in range(self.visited_grid.shape[1]):
            if self.mesh_grid[0][j][1] - self.discretization/2 <= point[1] <= self.mesh_grid[0][j][1] + self.discretization/2:
                y_idx = j

        if x_idx is None or y_idx is None:
            return None

        # to cover the the 0.5 m
        number_of_idx = int(self.camera.render_distance / self.discretization)

        if mode == "Test":
            visited_temp = np.zeros(self.visited_grid.shape)
            for i in range(-number_of_idx, number_of_idx + 1):
                for j in range(-number_of_idx, number_of_idx + 1):
                    if i**2 + j**2 <= number_of_idx**2 and 0 <= x_idx + i < visited_temp.shape[0] and 0 <= y_idx + j < visited_temp.shape[1]:
                        visited_temp[x_idx + i, y_idx + j] = 1
            return visited_temp

        elif mode == "Current":
            for i in range(-number_of_idx, number_of_idx + 1):
                for j in range(-number_of_idx, number_of_idx + 1):
                    if i**2 + j**2 <= number_of_idx**2 and 0 <= x_idx + i < self.visited_grid.shape[0] and 0 <= y_idx + j < self.visited_grid.shape[1]:
                        if self.visited_grid[x_idx + i, y_idx + j] == 0:
                            self.visited_grid[x_idx + i, y_idx + j] = 1
                            # only remove points if all its neighbourhood range are already removed.
                            if is_neighbourhood_visited(self.visited_grid, number_of_idx, x_idx, y_idx):
                                self.points_set.remove(self.mesh_grid[x_idx + i][y_idx + j])
            return self.visited_grid


        else:
            print("error")


    def update_occ_grid(self, curr_position=None):
        self.visited_grid = self.viewable_points(curr_position, "Current")
        self.occ_grid.header.stamp = rospy.Time.now()
        self.occ_grid.info.map_load_time = self.occ_grid.header.stamp
        self.occ_grid.data = 100 * np.transpose(self.visited_grid).astype("int8").ravel()


    def generate_next_best_view(self, curr_position=None):
        """
        generates the next best point to visit (given your previous visit history) for a single point
        :param curr_position: current location as a tuple
        :return: next best location as a tuple, usefuleness score of the point
        """

        # 1. decide how much we can view
        self.update_occ_grid(curr_position)

        # print(self.occ_grid.data, self.occ_grid.data.shape)
        # 2. generate random points or all of them and check new things they can view
        best_result = [None, 0]

        for point in self.points_set:
            if self.Map.is_passable(*point):
                visited = self.viewable_points(point, "Test")
                if visited is None:
                    return None, None
                usefulness = np.sum(np.logical_or(self.visited_grid, visited))
                if usefulness > best_result[1]:
                    best_result = [point, usefulness]

        return best_result[0], best_result[1]

    def generate_best_path(self, curr_position, display_flag=False):
        """

        :param curr_position: current location as a tuple
        :param display: a flag to display an image of the visited array
        :return: path to visited the best
        """
        prev_best = -1
        # best_result[1] != prev_best
        while (len(self.points_set) != 0):

            next_point, next_point_score = self.generate_next_best_view(curr_position)

            if display_flag:
                plt.matshow(self.visited_grid)
                plt.show()
            # 3.
            if next_point_score == prev_best:
                break
            if next_point is not None:
                self.path.append(next_point)
                curr_position = next_point

            prev_best = next_point_score

        print("path computed")
        return self.path
def test():
    """
    test Dora as a node (due to the rospy usage with the OccupancyGrid)
    :return:
    """
    # world_map = Map("course_packages/dd2419_resources/worlds_json/planning_test_map.json")
    rospy.init_node('Dora')

    my_path = os.path.abspath(os.path.dirname(__file__))
    file = "joakimV2.world.json" #"lucas_room_screen.world.json"
    map_path = os.path.join(my_path, "../..", "course_packages/dd2419_resources/worlds_json", file)
    Dora = DoraTheExplorer(map_path, expansion_factor=0.22)
    print(Dora.generate_best_path((0.5, 0.5), True))
if __name__ == "__main__":
    test()
