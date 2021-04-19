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


def _make_finite(t):
    temp = t.clone()
    temp[t == -np.inf] = -2. ** 32.
    temp[t == np.inf] = 2. ** 32.
    return temp


class VecSim:
    """
    Run a vectorized gillespie simulation
    """

    def __init__(self, net: VectorizedRxnNet,
                 runtime: float,
                 score_constant: float = 1.,
                 volume=1,
                 device='cuda:0'):
        """

        Args:
            net: The reaction network to run the simulation on.
            runtime: Length (in seconds) of the simulation.
            score_constant: User defined parameter, equals Joules / Rosetta Score Unit.
            volume: Volume of simulation in Micro Meters. Default is .001 uL = 1 um^3
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
        self.use_energies = self.rn.is_energy_set
        self.runtime = runtime
        self.observables = self.rn.observables
        self._constant = Tensor([score_constant]).to(self.dev)
        self.avo = Tensor([6.022e23])
        self.volume = Tensor([volume * 1e-15]).to(self.dev)  # convert cubic micro-micrometers to liters
        self.steps = []

    def simulate(self, verbose=False):
        """
        modifies reaction network
        :return:
        """
        cur_time = 0
        cutoff = 50000
        # update observables
        max_poss_yield = torch.min(self.rn.copies_vec[:self.rn.num_monomers].clone()).to(self.dev)
        l_k = self.rn.compute_log_constants(self.rn.kon, self.rn.rxn_score_vec, self._constant)
        while cur_time < self.runtime:
            for obs in self.rn.observables.keys():
                try:
                    self.rn.observables[obs][1].append(self.rn.copies_vec[int(obs)].item())
                except IndexError:
                    print('bkpt')
            l_conc_prod_vec = self.rn.get_log_copy_prod_vector(volume=self.volume)
            # copies_prod = copies_prod + 2e-32
            # l_conc_prod_vec = torch.log(copies_prod)
            l_rxn_rates = l_conc_prod_vec + l_k
            l_total_rate = torch.logsumexp(l_rxn_rates, dim=0)
            l_step = 0 - l_total_rate
            rate_step = torch.exp(l_rxn_rates + l_step)
            delta_copies = torch.matmul(self.rn.M, rate_step)
            initial_monomers = self.rn.initial_copies
            min_copies = torch.ones(self.rn.copies_vec.shape, device=self.dev) * np.inf
            min_copies[0:initial_monomers.shape[0]] = initial_monomers
            self.rn.copies_vec = torch.max(self.rn.copies_vec + delta_copies, torch.zeros(self.rn.copies_vec.shape, device=self.dev))
            #self.rn.copies_vec = torch.min(self.rn.copies_vec, min_copies)
            initial_copies = self.rn.initial_copies
            step = torch.exp(l_step)
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
