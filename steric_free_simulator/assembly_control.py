from typing import Tuple, Dict, List, Union
import numpy as np
import re
import sys
from itertools import chain, combinations


def powerset(inset):
    return chain.from_iterable(combinations(inset, r) for r in range(len(inset)+1))

class MetaAssembly:
    def __init__(self, bngl_path: str, dt: float, steps: int):
        self.bngl = open(bngl_path, 'r')
        self.units: Dict[frozenset, List[int, List[Union[frozenset, float, float]]]] = {}
        self.observable: Dict[frozenset, List[str, np.ndarray]] = {}
        self.steps = steps
        self.dt = dt
        self.parse_bngl(open(bngl_path, 'r'))

    def match_maker(self, orig, next) -> List[Union[int, List[Union[frozenset, float, float]]]]:
        partners = []
        for n in self.units[orig][1] + self.units[next][1]:
            if len(orig - n[0]) == len(orig) and len(next - n[0]) == len(next):
                # would not be sterically hindered addition
                if n not in partners:
                    partners.append(n)
        all_part = powerset(partners)
        partners = []
        for p in all_part:
            if len(p) != 0:
                partners.append(p[0])
        return partners
            # TODO: modify rates for trimerization

    def _reaction_prob(self, reactants: Tuple, k: float) -> float:
        rate = np.prod(reactants)*k
        prob = 1 - np.exp(-1*rate * self.dt)
        return prob

    def _update_observables(self, time: int):
        for mol in list(self.observable.keys()):
            if mol in self.units:
                self.observable[mol][1][time] = self.units[mol][0]
            else:
                self.observable[mol][1][time] = 0

    # search tree and write rules.
    def _resolve_tree(self):
        # TODO: Split in to resolve tree and simulate methods
        for t in range(self.steps):
            keys = list(self.units.keys())
            for this in keys:
                res = self.units[this][1]
                for neighbor_info in res:
                    next = neighbor_info[0]
                    # compute rate and reaction probability
                    p = self._reaction_prob((self.units[this][0], self.units[next][0]), neighbor_info[1])
                    r = np.random.rand(1)
                    new_complex = frozenset(this.union(next))
                    if r < p:
                        if new_complex not in self.units:
                            partners = self.match_maker(this, next)
                            self.units[new_complex] = [0, partners]
                        else:
                            self.units[new_complex][0] += 1
                        self.units[this][0] -= 1
                        self.units[next][0] -= 1
                    if new_complex in self.units:
                        p_rev = self._reaction_prob((self.units[new_complex][0]), neighbor_info[2])
                        r = np.random.rand(1)
                        if r < p_rev:
                            self.units[new_complex][0] -= 1
                            self.units[this][0] += 1
                            self.units[next][0] += 1
            self._update_observables(t)
        return self.units

    def _place_into(self, unit: frozenset, n_info: List[Union[frozenset, float, float]]):
        # place neighbor into memory
        # currently only one node for each available interface!
        for i in range(len(self.units[unit][1])):
            if self.units[unit][1][i][0] is None:
                self.units[unit][1][i] = n_info
                return
        raise (IndexError, "species does not have proper interface. ")

    def parse_param(self, line):
        items = line.split()
        return items

    def parse_species(self, line, params):
        items = line.split()
        sp_info = re.split('\\)|,|\\(', items[0])
        try:
            init_pop = int(items[1])
        except ValueError:
            init_pop = int(params[items[1]])
        num_interfaces = len(sp_info[1:]) - 1
        self.units[frozenset(sp_info[0])] = [init_pop, [[None, None, None]]*num_interfaces]

    def parse_rule(self, line, params):
        items = re.split(r' |, ', line)
        r_info = re.split('\\(.\\)+.|\\(.\\)<->', items[0])
        try:
            k_on = int(items[1])
        except ValueError:
            k_on = float(params[items[1]])
        if len(items) > 2:
            try:
                k_off = int(items[2])
            except ValueError:
                k_off = float(params[items[2]])
        else:
            k_off = 0
        self._place_into(frozenset([r_info[0]]), [frozenset(r_info[1]), k_on, k_off])
        self._place_into(frozenset([r_info[1]]), [frozenset(r_info[0]), k_on, k_off])

    def parse_observables(self, line):
        items = line.split()
        descriptor = items[0]
        obs_info = re.split('\\)|,|\\(|\\.', items[1])
        mol = set()
        for tkn in obs_info:
            if frozenset(tkn) in self.units:
                mol.add(tkn)
        self.observable[frozenset(mol)] = [descriptor, np.empty(self.steps, dtype=np.int)]

    def parse_bngl(self, f):
        parameters = dict()
        cur_block = ''
        for line in f:
            line = line.strip()
            if len(line) > 0 and line[0] != '#':
                if "begin parameters" in line:
                    cur_block = 'param'
                elif "begin species" in line:
                    cur_block = 'species'
                elif "begin rules" in line:
                    cur_block = 'rules'
                elif "begin observables" in line:
                    cur_block = 'observables'
                elif "end" in line:
                    cur_block = ' '
                else:
                    if cur_block == 'param':
                        items = self.parse_param(line)
                        parameters[items[0]] = items[1]
                    elif cur_block == 'species':
                        self.parse_species(line, parameters)
                    elif cur_block == 'rules':
                        self.parse_rule(line, parameters)
                    elif cur_block == 'observables':
                        self.parse_observables(line)

    def plot_observables(self):
        from matplotlib import pyplot as plt
        for mol in list(self.observable.keys()):
            plt.scatter(np.arange(self.steps)*self.dt, self.observable[mol][1])
        plt.show()


if __name__ == '__main__':
    bngls_path = sys.argv[1]  # path to bngl
    dt = float(sys.argv[2])  # time step in seconds
    iter = int(sys.argv[3])  # number of time steps to simulate
    m = MetaAssembly(sys.argv[1], dt, iter)
    m._resolve_tree()
    m.plot_observables()



