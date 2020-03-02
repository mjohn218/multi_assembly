from pyrosetta.toolbox import cleanATOM
from pyrosetta import *
from pyrosetta.rosetta.core.scoring import *
import os
import pickle as pk

class network:
    def __init__(self, poses:dict, config:dict):
        self.poses = poses
        self.config = config
        self.network = {} # start_unit: (score, stop_unit)
        for unit in config.keys():
            self.network[config] = (None, None)

    def build_network(self):
        unused = network.keys()
        while len(unused) != 0:
            unit = unused.pop()
            

