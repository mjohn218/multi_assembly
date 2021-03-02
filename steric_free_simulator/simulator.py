import math

from reaction_network import ReactionNetwork
import numpy as np
from typing import Tuple, Union

import matplotlib
from matplotlib import pyplot as plt

from torch import DoubleTensor as Tensor
from torch import nn
import torch

import pandas as pd

plt.ion()


class Simulator:

    def __init__(self, net: ReactionNetwork, runtime: float, dt: float = 1, lin_factor: float = 5., optimize_dt=True):
        self.dt = dt
        self.rn = net
        self.score_proportionality = .1
        self._linearity_factor = Tensor([lin_factor]) # higher produces *more* discrete reaction updates, but reduces optimization effectivity.
        self.use_energies = self.rn.is_energy_set
        if optimize_dt:
            self.optimize_step()
        self.steps = int(runtime / self.dt)
        self.runtime = runtime

    def optimize_step(self, depth=0) -> float:
        """
        find the dt that is accurate while also efficient and return it
        """
        reactions = self._possible_reactions(
            set(range(self.rn.num_monomers)), set(range(self.rn.num_monomers)))
        depth += 1
        if depth == 50:
            return self.dt
        for r in reactions:
            with torch.no_grad():
                prob = self._reaction_prob(set(r))
            if len(r) > 1 and (prob[0] >= .9 or prob[1] >= .9):
                self.dt /= 2
                self.optimize_step(depth)
                return self.dt
            elif len(r) > 1 and (prob[0] <= .1 or prob[1] <= .1):
                self.dt *= 2
                self.optimize_step(depth)
                return self.dt
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

    def _compute_disassociation(self, kon: Tensor, dGrxn):
        koff = kon / (torch.exp(-Tensor([self.score_proportionality]) * Tensor([dGrxn])))
        return koff

    def _reaction_prob(self, reactants: set) -> Tuple[Tensor, ...]:
        prob = [Tensor([0.]), Tensor([0.])]
        data = self.rn.network.edges[min(reactants)]
        kon = torch.abs(Tensor(data['k_on']))
        if self.use_energies:
            koff = self._compute_disassociation(kon, data['rxn_score'])
            for reactant in reactants:
                self.rn.network.edges[reactant]['k_off'] = koff
        else:
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
        # if len(reactants) != 1 and (math.isclose(1, prob[0].item(), abs_tol=.01) or
        #                             math.isclose(1, prob[1].item(), abs_tol=.01)):
            # print("WARNING: Reaction probability seems to be saturated, "
            #       "consider reducing time step size.")
        return tuple(prob)

    def _compute_copy_change(self, p: Tensor, method='sigmoid'):
        if method == 'sigmoid':
            r = torch.rand(1, dtype=torch.double)
            x = self._linearity_factor * (p - r)
            result = torch.sigmoid(x)
        elif method == 'linear':
            result = p
        else:
            raise ValueError('Invalid copy update mode')
        return result

    def _forward(self, reaction: set, prob: Tensor) -> Tuple[int, set]:
        newly_nonzero = None
        newly_zero = set()
        e = None  # suppress warning
        # decrease reactant concentrations
        delta_copies = self._compute_copy_change(prob)
        for e in reaction:
            self.rn.network.nodes[e[0]]['copies'] = self.rn.network.nodes[e[0]]['copies'] - delta_copies
            if self.rn.network.nodes[e[0]]['copies'] < 1:
                newly_zero.add(e[0])
        # increase product concentration
        if self.rn.network.nodes[e[1]]['copies'] < 1:
            newly_nonzero = e[1]
        self.rn.network.nodes[e[1]]['copies'] = self.rn.network.nodes[e[1]]['copies'] + delta_copies
        return newly_nonzero, newly_zero

    def _reverse(self, reaction: set, prob: Tensor) -> Tuple[set, int]:
        newly_nonzero = set()
        newly_zero = None
        delta_copies = self._compute_copy_change(prob)
        e = None  # suppress warning
        for e in reaction:
            # increase product concentrations
            if self.rn.network.nodes[e[0]]['copies'] < 1:
                newly_nonzero.add(e[0])
            self.rn.network.nodes[e[0]]['copies'] = self.rn.network.nodes[e[0]]['copies'] + delta_copies
        # decrease reactant concentration
        self.rn.network.nodes[e[1]]['copies'] = self.rn.network.nodes[e[1]]['copies'] - delta_copies
        if self.rn.network.nodes[e[1]]['copies'] < 1:
            newly_zero = e[1]
        return newly_nonzero, newly_zero

    def simulate(self, verbose=False):
        """
        modifies reaction network
        :return:
        """
        newly_nonzero = set(range(self.rn.num_monomers))
        nonzero = newly_nonzero.copy()
        reactions = set()  # set of sets of reaction edges
        # get list of edges id pairs present in non-zero population.
        nodes = dict(self.rn.network.nodes(data=True))
        node_list = [nodes[key] for key in sorted(nodes.keys())]
        max_poss_yield = torch.min(Tensor([node['copies'] for node in node_list[:self.rn.num_monomers]]), dim=0)[0]
        for step in range(self.steps):
            newly_zero = set()
            reactions = reactions.union(self._possible_reactions(newly_nonzero, nonzero))
            if verbose and (step % 500) == 0:
                print("Processing " + str(len(reactions)) + " reactions for step " + str(step))
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
                self.rn.observables[obs][1].append(self.rn.network.nodes[obs]['copies'].item())
        total_complete = node_list[-1]['copies']
        final_yield = total_complete / max_poss_yield
        return final_yield

    def observables_to_csv(self, out_path):
        data = {}
        for key in self.rn.observables:
            entry = self.rn.observables[key]
            data[entry[0]] = entry[1]
        df = pd.DataFrame(data)
        df.to_csv(out_path)

    def plot_observables(self, iter_num=None):
        t = np.arange(self.steps) * self.dt
        for key in self.rn.observables.keys():
            data = np.array(self.rn.observables[key][1])
            plt.scatter(t, data,
                        cmap='plasma',
                        s=.1,
                        label=self.rn.observables[key][0])
        plt.legend(loc='best')
        plt.show(block=False)




