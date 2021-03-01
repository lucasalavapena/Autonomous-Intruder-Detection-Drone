#!/usr/bin/env python


import json


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
