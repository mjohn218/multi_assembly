from reaction_network import ReactionNetwork
import numpy as np
from typing import Tuple


class Simulator:

    def __init__(self, net: ReactionNetwork, steps: int, dt: float, obs=None):
        self.steps = steps
        self.dt = dt
        self.rn = net

    def _possible_reactions(self, new_nodes: set, node_set: set):
        new_reactions = set()
        for node in new_nodes:
            out_edges = self.rn.network.out_edges(node, data=True)
            for edge in out_edges:
                euid = edge[2]['uid']
                successor = edge[1]
                coreactant = None
                # find matching predecessor if node_exists
                for in_edge in self.rn.network.in_edges(successor, data=True):
                    if in_edge[:2] != edge[:2] and in_edge[2]['uid'] == euid:
                        coreactant = in_edge[0]
                        break
                if coreactant is None:
                    # first order
                    new_reactions.add(frozenset([edge[:2]]))
                else:
                    # determine if coreactant is nonzero
                    if coreactant in node_set:
                        new_reactions.add(frozenset([(coreactant, successor), tuple(edge[:2])]))
        return new_reactions

    def _reaction_prob(self, reactants: set) -> Tuple[float, ...]:
        prob = [0., 0.]
        data = self.rn.network.edges[min(reactants)]
        kon = data['k_on']
        if len(reactants) == 1:
            edge = list(reactants)[0]
            copies = self.rn.network.nodes[edge[0]]['copies']
            # 1M = 1 mol / L = 6.02e23 copies / L
            k_close = kon * data['lcf'] * 100
            # rxn is intra molecular i.e. loop closure
            rate = k_close * copies
        else:
            edge = None
            rate = kon
            for edge in reactants:
                rate *= self.rn.network.nodes[edge[0]]['copies']
        prob[0] = float(1 - np.exp(-1 * rate * self.dt))
        offrate = data['k_off'] * self.rn.network.nodes[edge[1]]['copies']
        prob[1] = float(1 - np.exp(-1 * offrate * self.dt))
        return tuple(prob)

    def _forward(self, reaction: set, prob: float) -> Tuple[int, set]:
        r = np.random.rand(1)
        newly_nonzero = None
        newly_zero = set()
        if r < prob:
            e = None  # suppress warning
            # decrease reactant concentrations
            for e in reaction:
                self.rn.network.nodes[e[0]]['copies'] -= 1
                if self.rn.network.nodes[e[0]]['copies'] == 0:
                    newly_zero.add(e[0])
            # increase product concentration
            if self.rn.network.nodes[e[1]]['copies'] == 0:
                newly_nonzero = e[1]
            self.rn.network.nodes[e[1]]['copies'] += 1
        return newly_nonzero, newly_zero

    def _reverse(self, reaction: set, prob: float) -> Tuple[set, int]:
        r = np.random.rand(1)
        newly_nonzero = set()
        newly_zero = None
        if r < prob:
            e = None  # suppress warning
            for e in reaction:
                # increase product concentrations
                if self.rn.network.nodes[e[0]]['copies'] == 0:
                    newly_nonzero.add(e[0])
                self.rn.network.nodes[e[0]]['copies'] += 1
            # decrease reactant concentration
            self.rn.network.nodes[e[1]]['copies'] -= 1
            if self.rn.network.nodes[e[1]]['copies'] == 0:
                newly_zero = e[1]
        return newly_nonzero, newly_zero

    def simulate(self):
        """
        modifies reaction network
        :return:
        """
        newly_nonzero = set(range(self.rn.num_monomers))
        nonzero = newly_nonzero.copy()
        reactions = set()  # set of sets of reaction edges
        # get list of edges id pairs present in non-zero population.
        for step in range(self.steps):
            newly_zero = set()
            reactions = reactions.union(self._possible_reactions(newly_nonzero, nonzero))
            newly_nonzero = set()
            for reaction in reactions:
                prob = self._reaction_prob(reaction)
                lnz, lz = self._forward(reaction, prob[0])
                if lnz not in nonzero: newly_nonzero.add(lnz)
                for n in lz:
                    if n in nonzero: newly_zero.add(n)
                lnz, lz = self._reverse(reaction, prob[1])
                for n in lnz:
                    if n not in nonzero: newly_nonzero.add(n)
                if lz in nonzero: newly_zero.add(lz)
            nonzero = nonzero.union(newly_nonzero)
            # below is necessary to prune impossible reactions, overall more efficient this way
            if len(newly_zero) > 0:
                for n in newly_zero: nonzero.remove(n)
                for rxn in list(reactions):
                    for e in list(rxn):
                        if True in [n in newly_zero for n in e] and rxn in reactions:
                            reactions.remove(rxn)
            # update observables
            for obs in self.rn.observables.keys():
                self.rn.observables[obs][1].append(self.rn.network.nodes[obs]['copies'])
