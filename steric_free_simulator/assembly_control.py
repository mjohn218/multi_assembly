from typing import Tuple, Dict, List

import numpy as np
from networkit import *

class meta_assembly:
    def __init__(self, bngl_path:str, concs:list, on_rates:list, off_rates:list, nfsim_dir:str):
        self.bngl = open(bngl_path, 'r')
        self.init = concs
        # {this, [pop, [neighbor, on_rate_const, off_rate_const]]
        self.units: Dict[frozenset, List[int, List[frozenset, float, float]]]
        self.total_rules = 13
        self.dt = .1

    def match_maker(self, orig, next) -> List[int, List[frozenset, float, float]]:
        partners = []
        for n in self.units[orig].extend(self.units[next][1]):
            if orig - n[0] == 0 and next - n[0] == 0:
                # would not be sterically hindered addition
                partners.append(n)
        return partners
            # TODO: modify rates for trimerization

    def _reaction_prob(self, reactants: Tuple, k: float) -> float:
        rate = np.prod(reactants)*k
        prob = 1 - np.exp(-1*rate * self.dt)
        return prob

    # search tree and write rules.
    def _resolve_tree(self, time_arr: list):
        for t in time_arr:
            keys = list(self.units.keys())
            dist = self.compute_dist()
            for this in keys:
                res = self.units[this][1]
                for next in res:
                    next = next[0]
                    # compute rate and reaction probability
                    p = self._reaction_prob((self.units[this][1], self.units[next][0]), self.units[this][1][1])
                    r = np.random.rand(1)
                    if r < p:
                        new_complex = frozenset(this.union(next))
                        if new_complex not in self.units:
                            partners = self.match_maker(this, next)
                            self.units[new_complex] = [0, partners]
                            self.total_rules += len(partners)
                        else:
                            self.units[new_complex][0] += 1

