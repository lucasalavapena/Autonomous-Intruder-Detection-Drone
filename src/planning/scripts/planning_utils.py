#!/usr/bin/env python

import json
import random
import math

def RRT(curr_x, curr_y, goal_x, goal_y, Map):
    start_node = Node(curr_x, curr_y)
    goal_node = Node(goal_x, goal_y)
    Tree = [start_node]

    while True:
        rand_node = random_node(Map.airspace, goal_node)

        # closest node in tree
        nearest_node = find_nearest_node(Tree, rand_node)
        new_node = grow(rand_node, nearest_node, Map)

        if new_node != -1:
            Tree.append(new_node)

        if distance(Tree[-1], goal_node) == 0:
            break
    controls = generate_controls(Tree[-1])

    return controls

def generate_controls(node):
    controls = [(node.x, node.y)]
    tmp = node
    parent = tmp.parent

    while parent is not None:
        controls.append((parent.x, parent.y))
        tmp = parent
        parent = tmp.parent

    return controls

def grow(rand_node, nearest_node, Map, steps = 100):

    delta_x = (rand_node.x - nearest_node.x) / steps
    delta_y = (rand_node.y - nearest_node.y) / steps

    for i in range(1, steps):
        x = nearest_node.x + i * delta_x
        y = nearest_node.y + i * delta_y
        if not Map.is_passable(x, y):
            return -1
    return rand_node

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
        return goal
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
        self.obstacles = []
        self.airspace = None
        self.map = self.build_map(map_path)


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

        # check if obstacles exist
        if "walls" in map_data.keys():
            for obs in map_data["walls"]:
                x_start, y_start, z_start = obs["plane"]["start"]
                x_end, y_end, z_end = obs["plane"]["stop"]
                self.obstacles.append((x_start, y_start, z_start, x_end, y_end, z_end))
        else:
            print("No obstacles")



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


def test():
    # world_map = Map("src/course_packages/dd2419_resources/worlds_json/planing-test.json")
    world_map = Map("src/course_packages/dd2419_resources/worlds_json/tutorial_1.world.json")
    result = RRT(0, 0, 1, 1.9, world_map)
    print(result)
if __name__ == "__main__":
    test()
