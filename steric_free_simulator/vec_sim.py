import math

from vectorized_rxn_net import VectorizedRxnNet
from reaction_network import ReactionNetwork
import numpy as np
from typing import Tuple, Union

import matplotlib
from matplotlib import pyplot as plt

from torch import DoubleTensor as Tensor
from torch import nn
import torch

import pandas as pd


class VecSim:

    def __init__(self, net: Union[ReactionNetwork, VectorizedRxnNet], runtime: float):
        if type(net) is ReactionNetwork:
            self.rn = VectorizedRxnNet(net)
        else:
            self.rn = net
        self.score_proportionality = Tensor([.1])
        self.use_energies = self.rn.is_energy_set
        self.runtime = runtime
        self.observables = self.rn.observables
        self.A = Tensor([1.])
        self._R = Tensor([10.])
        self._T = Tensor([10.])
        self.steps = []

    def _compute_constants(self, EA: Tensor, dGrxn: Tensor) -> Tensor:
        kon = self.A * torch.exp(-EA / (self._R * self._T))
        koff = self.A * torch.exp(-(EA - dGrxn) / (self._R * self._T))
        k = torch.cat([kon, koff], dim=0)
        return k.clone()

    def simulate(self, verbose=False):
        """
        modifies reaction network
        :return:
        """
        cur_time = 0
        # update observables
        max_poss_yield = torch.min(self.rn.copies_vec[:self.rn.num_monomers].clone())
        while cur_time < self.runtime:
            for obs in self.rn.observables.keys():
                self.rn.observables[obs][1].append(self.rn.copies_vec[int(obs)].item())
            k = self._compute_constants(self.rn.EA, self.rn.rxn_score_vec)
            copy_prod_vec = self.rn.get_copy_prod_vector()
            rxn_rates = copy_prod_vec * k
            total_rate = torch.sum(rxn_rates)
            step = 1 / total_rate
            rate_step = rxn_rates * step
            delta_copies = torch.matmul(self.rn.M, rate_step)
            self.rn.copies_vec = self.rn.copies_vec + delta_copies
            cur_time = cur_time + step
            self.steps.append(cur_time.clone())

        total_complete = self.rn.copies_vec[-1]
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




