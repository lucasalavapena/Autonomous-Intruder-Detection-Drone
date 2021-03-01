#!/usr/bin/env python

import json
import random
import math

def RRT(curr_x, curr_y, goal_x, goal_y, Map):
    start_node = Node(curr_x, curr_y)
    goal_node = Node(goal_x, goal_y)
    Tree = [start_node]

    while not path_found:
        rand_node = random_node(Map.airspace, goal_node)
        # closest node in tree
        nearest_node = find_nearest_node(Tree, rand_node)
        Tree = grow(rand_node, nearest_node, Map)

        if distance(Tree[-1], goal_node) == 0:
            break

    if limit:
        times = np.arange(0, 20, 0.01)
        controls = np.zeros(len(times) - 1)
    else:
        controls = generate_controls(Tree[-1])

    return controls

def generate_controls(node):
    controls = []
    parent = node.parent
    tmp = node

    while parent is not None:
        phis = tmp.path_phi
        controls = phis + controls
        tmp = tmp.parent
        parent = tmp.parent

    return controls

def grow(rand_node, nearest_node, Map):
    coords = get_path(rand_node, nearest_node)

    for (x,y) in coords:
        Map.is_passable(x, y)


def grow(from_node, to_node, car, obs, x_lims, y_lims):
    steps = 100
    tmp = from_node
    phi = calc_phi(from_node, to_node)

    x_path = [tmp.x]
    y_path = [tmp.y]
    phis = [phi]

    for i in range(steps):
        xn, yn, thetan = step(car, tmp.x, tmp.y, tmp.theta, phi, dt=0.01)
        tmp = Node(xn, yn, thetan)

        phi = calc_phi(tmp, to_node)

        if distance(tmp, to_node) < 1.0:  # close enough to stop
            break

        x_path.append(xn)
        y_path.append(yn)
        phis.append(phi)

    tmp.path_x = x_path[:-1]
    tmp.path_y = y_path[:-1]
    tmp.path_phi = phis[:-1]
    tmp.parent = from_node

    for x, y in zip(tmp.path_x, tmp.path_y):
        if not valid(Node(x, y, 0), obs, x_lims, y_lims):
            return -1
    return tmp

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
