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
import psutil
import pandas as pd

import sys


class VecSim:
    """
    Run a vectorized gillespie simulation
    """

    def __init__(self, net: Union[ReactionNetwork, VectorizedRxnNet],
                 runtime: float,
                 score_constant: float = 1.,
                 freq_fact: float = 10.,
                 volume=1e-5,
                 device='cuda:0'):
        """

        Args:
            net: The reaction network to run the simulation on.
            runtime: Length (in seconds) of the simulation.
            score_constant: User defined parameter, equals Joules / Rosetta Score Unit.
            freq_fact: User defined parameter, collision frequency.
            volume: Volume of simulation in Micro Liters.
        """
        if torch.cuda.is_available() and "cpu" not in device:
            self.dev = torch.device(device)
            print("Using " + device)
        else:
            self.dev = torch.device("cpu")
            print("Using CPU")

        if type(net) is ReactionNetwork:
            self.rn = VectorizedRxnNet(net, dev=self.dev)
        else:
            self.rn = net
        self.score_proportionality = Tensor([.1]).to(self.dev)
        self.use_energies = self.rn.is_energy_set
        self.runtime = runtime
        self.observables = self.rn.observables
        self._constant = Tensor([score_constant]).to(self.dev)
        self.A = Tensor([freq_fact]).to(self.dev)
        self._R = Tensor([8.314]).to(self.dev)
        self._T = Tensor([273.15]).to(self.dev)
        self.volume = Tensor([volume * 1e-6]).to(self.dev)  # convert microliters to liters
        self.steps = []

    def _compute_constants(self, EA: Tensor, dGrxn: Tensor) -> Tensor:
        kon = self.A * torch.exp(-EA / (self._R * self._T))
        koff = self.A * torch.exp(-(EA - self._constant * dGrxn) / (self._R * self._T))
        k = torch.cat([kon, koff], dim=0)
        return k.clone().to(self.dev)

    def simulate(self, verbose=False):
        """
        modifies reaction network
        :return:
        """
        cur_time = 0
        cutoff = 1000000
        # update observables
        max_poss_yield = torch.min(self.rn.copies_vec[:self.rn.num_monomers].clone()).to(self.dev)
        while cur_time < self.runtime:
            for obs in self.rn.observables.keys():
                try:
                    self.rn.observables[obs][1].append(self.rn.copies_vec[int(obs)].item())
                except IndexError:
                    print('bkpt')
            k = self._compute_constants(self.rn.EA, self.rn.rxn_score_vec)
            concentration_prod_vec = self.rn.get_copy_prod_vector(volume=self.volume)
            rxn_rates = concentration_prod_vec * k
            total_rate = torch.sum(rxn_rates)
            step = 1 / total_rate
            rate_step = rxn_rates * step
            delta_copies = torch.matmul(self.rn.M, rate_step)
            self.rn.copies_vec = torch.max(self.rn.copies_vec + delta_copies, torch.zeros(self.rn.copies_vec.shape, device=self.dev))
            cur_time = cur_time + step
            self.steps.append(cur_time.item())

            if len(self.steps) > cutoff:
                print("WARNING: sim was stopped early due to exceeding set max steps", sys.stderr)
                break
        total_complete = self.rn.copies_vec[-1]
        final_yield = total_complete / max_poss_yield
        return final_yield.to(self.dev)

    def observables_to_csv(self, out_path):
        data = {}
        for key in self.rn.observables:
            entry = self.rn.observables[key]
            data[entry[0]] = entry[1]
        df = pd.DataFrame(data)
        df.to_csv(out_path)
