import os
import pickle as pk
import heapq
from bind import bind


class Network:
    def __init__(self, poses:dict, config:dict):
        self.poses = poses
        self.config = config
        self.network = {} # start_unit: (score, stop_unit)
        for unit in config.keys():
            self.network[unit] = [(None, None, None)]

    def allowed_neighbor(self, unit_built_of, neighbor) -> bool:
        is_valid = False
        neighbor_built_of = neighbor.split('_')
        neighbor_built_of.remove('')
        for i in neighbor_built_of:
            if i in unit_built_of:
                return False
        for i in unit_built_of:
            if i in neighbor_built_of:
                return False
            i = i+'_'
            for j in neighbor_built_of:
                j = j+'_'
                if j in self.config[i]:
                    is_valid = True
        target = unit_built_of.copy()
        target.extend(neighbor_built_of)
        target.sort()
        target_name = '_'.join(target)+'_'
        if neighbor in self.network:
            for entry in self.network[neighbor]:
                if entry[1] == target_name:
                    return False
        return is_valid


    def build_network(self):
        """
        Construct the network of all possible states.
        To be stored in the network instance variable.
        Format {start_node: (score, target)}
        :return: The network dictionary
        """
        unused = list(self.network.keys())
        is_change_made = True
        while is_change_made:
            is_change_made = False
            while len(unused) != 0:
                unit = unused.pop()
                if len(unit) == 4:
                    print('here')
                built_of = unit.split('_')
                built_of.remove('')
                for other in unused:
                    if len(other) == 4 and len(unit) == 4:
                        print('yayaya')
                    if self.allowed_neighbor(built_of, other):
                        is_change_made = True
                        entry = bind((unit, self.poses[unit]), (other, self.poses[other]))
                        for key in entry[0].keys():
                            if key in self.network:
                                if self.network[key][0][0] is None:
                                    self.network[key] = [entry[0][key]] # remove None placeholder
                                else:
                                    self.network[key].append(entry[0][key])
                            if list(entry[1].keys())[0] not in self.network:
                                self.network[list(entry[1].keys())[0]] = [(None, None, None)]
                            self.poses.update(entry[1])
            unused = list(self.network.keys())
        return self.network

