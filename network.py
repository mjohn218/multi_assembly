import os
import pickle as pk
from bind import bind

class network:
    def __init__(self, poses:dict, config:dict):
        self.poses = poses
        self.config = config
        self.network = {} # start_unit: (score, stop_unit)
        for unit in config.keys():
            self.network[config] = (None, None)

    def build_network(self):
        unused = list(self.network.keys())
        while len(unused) != 0:
            unit = unused.pop()
            for other in unused:
                entry = bind((unit, self.poses[unit]), (other, self.poses[other]))
                self.network.update(entry[0])
                self.poses.update(entry[1])
            unused = list(self.network.keys())
