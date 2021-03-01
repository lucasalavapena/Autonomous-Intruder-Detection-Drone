#!/usr/bin/env python

import json
import random
import math

def RRT(curr_x, curr_y, goal_x, goal_y, Map):
    start_node = Node(curr_x, curr_y)
    goal_node = Node(goal_x, goal_y)
    Tree = [start_node]
    path_found = False

    while not path_found:
        rand_node = random_node(Map.airspace, goal_node)
        # closest node in tree
        nearest_node = find_nearest_node(Tree, rand_node)
        Tree, flag = grow(rand_node, nearest_node)


def distance(from_node, to_node):
    return math.hypot(from_node.x - to_node.x, from_node.y - to_node.y)

def find_nearest_node(tree, new_node):
    nearest = tree[0]
    dist = float('inf')

    for node in tree:
        new_dist = distance(node, new_node)
        if new_dist < dist:
            nearest = node
            dist = new_dist

    return nearest


def random_node(airspace, goal, prob_goal=0.2):
    x_start, y_start, _, x_end, y_end, _ = airspace
    if random.uniform(0, 1) < prob_goal:
        return goal.copy()
    else:
        return Node(random.uniform(x_start, x_end), random.uniform(y_start, y_end))


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.path_x = []
        self.path_y = []
        self.parent = None



# The class representing the room and its limits
class Map:
    def __init__(self, map_path):

        self.map = self.generate_map(map_path)
        self.obstacles = []
        self.airspace = None

    def build_map(self, map_path):
        """
        builds the map, by
        :param map_path:
        :return:
        """
        with open(map_path) as json_file:
            map_data = json.load(json_file)

        x_start, y_start, z_start = map_data["airspace"]["min"]
        x_end, y_end, z_end = map_data["airspace"]["max"]
        self.airspace = ((x_start, y_start, z_start, x_end, y_end, z_end))

        for obs in map_data["walls"]:
            x_start, y_start, z_start = obs["plane"]["start"]
            x_end, y_end, z_end = obs["plane"]["stop"]
            self.obstacles.append((x_start, y_start, z_start, x_end, y_end, z_end))


    def is_passable(self, x, y):
        """
        checks if a
        """
        # check if it is within the airspace
        if x > self.airspace[3] or x < self.airspace[0]  or y > self.airspace[4] or y < self.airspace[1]:
            return False
        # check against obstacles
        for obs in self.obstacles:
            x_start, y_start, _, x_end, y_end, _ = obs
            if max(x_start, x_end) >=  x >= min(x_start, x_end) and max(y_start, y_end) >=  y >= min(y_start, y_end):
                return False
        return True



if __name__ == "__main__":
    pass
