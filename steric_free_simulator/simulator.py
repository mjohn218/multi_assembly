import math

from reaction_network import ReactionNetwork
import numpy as np
from typing import Tuple, Union

from torch import Tensor
from torch import nn
import torch


class Simulator:

    def __init__(self, net: ReactionNetwork, runtime: float, dt: float = 1, obs=None, optimize_dt=True):
        self.dt = dt
        self.rn = net
        if optimize_dt:
            self.optimize_step()
        self.steps = int(runtime / self.dt)
        self._R = 8.3145
        self._T = 273

    def optimize_step(self) -> float:
        """
        find the dt that is accurate while also efficient and return it
        """
        reactions = self._possible_reactions(
            set(range(self.rn.num_monomers)), set(range(self.rn.num_monomers)))
        for r in reactions:
            prob = self._reaction_prob(set(r))
            if len(r) > 1 and (prob[0] >= .99 or prob[1] >= .99):
                self.dt /= 2
                self.optimize_step()
        return self.dt

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

    def _compute_rate_constants(self, activation_energy: Tensor, dGrxn):
        kon = self.rn.A * torch.exp(Tensor(-1)*activation_energy / (self._R * self._T))
        koff = self.rn.A * torch.exp((Tensor(-1)*activation_energy + dGrxn) / (self._R * self._T))
        return kon, koff

    def _reaction_prob(self, reactants: set) -> Tuple[Tensor, ...]:
        prob = [Tensor(0.), Tensor(0.)]
        data = self.rn.network.edges[min(reactants)]
        if self.use_energies:
            kon, koff = self._compute_rate_constants(data['activation_energy'], data['rxn_score'])
            self.rn.network.edges[min(reactants)]['k_on'] = kon
            self.rn.network.edges[min(reactants)]['k_off'] = koff
        else:
            kon = Tensor(data['k_on'])
            koff = Tensor(data['k_off'])
        if len(reactants) == 1:
            edge = list(reactants)[0]
            copies = self.rn.network.nodes[edge[0]]['copies']
            # 1M = 1 mol / L = 6.02e23 copies / L
            k_close = kon * data['lcf'] * 100000
            # rxn is intra molecular i.e. loop closure
            rate = k_close * copies
        else:
            edge = None
            rate = kon
            for edge in reactants:
                rate = rate * self.rn.network.nodes[edge[0]]['copies']
        prob[0] = (1 - torch.exp(-1 * rate * self.dt))
        offrate = koff * self.rn.network.nodes[edge[1]]['copies']
        prob[1] = (1 - torch.exp(-1 * offrate * self.dt))
        if len(reactants) != 1 and (math.isclose(1, prob[0].item(), abs_tol=.01) or
                                    math.isclose(1, prob[1].item(), abs_tol=.01)):
            print("WARNING: Reaction probability seems to be saturated, "
                  "consider reducing time step size.")
        return tuple(prob)

    def _forward(self, reaction: set, prob: Tensor) -> Tuple[int, set]:
        r = np.random.rand(1)
        newly_nonzero = None
        newly_zero = set()
        if r < prob.item():
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

    def _reverse(self, reaction: set, prob: Tensor) -> Tuple[set, int]:
        r = np.random.rand(1)
        newly_nonzero = set()
        newly_zero = None
        if r < prob.item():
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

    def _compute_yield(self):
        nodes = self.rn.network.nodes(data=True)
        max_poss = min([node[1]['copies'] for node in nodes[:self.rn.num_monomers]])
        total_complete = nodes[-1]['copies']
        return total_complete / max_poss

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

