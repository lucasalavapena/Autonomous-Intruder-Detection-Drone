from __future__ import division
from planning_utils import Map
import math
from itertools import product
import numpy as np
import os.path

class CrazyfliCamera:
    def __init__(self, FOV, render_distance):
        self.FOV = FOV #degrees
        self.render_distance = render_distance# m - distance we can see in front of us



class DoraTheExplorer:
    def __init__(self, map_path):
        self.Map = Map(map_path)
        self.camera = CrazyfliCamera(140, 0.5)
        self.mesh_grid = None
        self.visited_grid = None
        self.points_set = None
        self.discretization = 0.05
        self.generate_map_occupancy()
        self.current_position = (0.8, 0.8)
        self.path = []

    def generate_map_occupancy(self):
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

        # for i, row in enumerate(self.mesh_grid):
        #     for j, coords in enumerate(row):
        #         if not self.Map.is_passable(coords[0], coords[1]):
        #             self.visited_grid[i, j] = 1


    def viewable_points(self, point, mode="Test"):

        for i in range(len(self.mesh_grid)):
            if self.mesh_grid[i][0][0] - self.discretization/2 <= point[0] <= self.mesh_grid[i][0][0] + self.discretization/2:
                x_idx = i

        for j in range(self.visited_grid.shape[1]):
            if self.mesh_grid[0][j][1] - self.discretization/2 <= point[1] <= self.mesh_grid[0][j][1] + self.discretization/2:
                y_idx = j

        # to cover the the 0.5 m
        number_of_idx = int(self.camera.render_distance / self.discretization)

        if mode == "Test":
            visited_temp = np.zeros(self.visited_grid.shape)
            for i in range(-number_of_idx, number_of_idx + 1):
                for j in range(-number_of_idx, number_of_idx + 1):
                    if i**2 + j**2 <= number_of_idx:
                        visited_temp[x_idx + i, y_idx + j] = 1
            return visited_temp

        elif mode == "Current":
            for i in range(-number_of_idx, number_of_idx + 1):
                for j in range(-number_of_idx, number_of_idx + 1):
                    if i**2 + j**2 <= number_of_idx:
                        self.visited_grid[x_idx + i, y_idx + j] = 1
                        self.points_set.remove(self.mesh_grid[x_idx + i][y_idx + j])
            return self.visited_grid

        else:
            print("error")


    def generate_next_best_view(self):



        # 1. decide how much we can view
        self.visited_grid = self.viewable_points(self.current_position, "Current")

        # 2. generate random points or all of them and check new things they can view


        # 3.





def test():
    # world_map = Map("course_packages/dd2419_resources/worlds_json/planning_test_map.json")
    my_path = os.path.abspath(os.path.dirname(__file__))
    file = "lucas_room_screen.world.json"
    map_path = os.path.join(my_path, "../..", "course_packages/dd2419_resources/worlds_json", file)
    Dora = DoraTheExplorer(map_path)
    Dora.generate_next_best_view()
if __name__ == "__main__":
    test()

