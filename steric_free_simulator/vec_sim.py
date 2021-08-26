from steric_free_simulator.vectorized_rxn_net import VectorizedRxnNet
from steric_free_simulator import ReactionNetwork
import numpy as np

from torch import DoubleTensor as Tensor
import torch
import pandas as pd
from matplotlib import pyplot as plt

import sys


def _make_finite(t):
    temp = t.clone()
    temp[t == -np.inf] = -2. ** 32.
    temp[t == np.inf] = 2. ** 32.
    return temp


class VecSim:
    """
    Run a vectorized deterministic simulation. All data and parameters are represented as
    Torch Tensors, allowing for gradients to be tracked. This simulator was designed to
    fill three primary requirements.
        - The simulation must be fully differentiable.
    """

    def __init__(self, net: VectorizedRxnNet,
                 runtime: float,
                 device='cuda:0'):
        """

        Args:
            net: The reaction network to run the simulation on.
            runtime: Length (in seconds) of the simulation.

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
        self._constant = 1.
        self.avo = Tensor([6.022e23])
        self.steps = []

    def simulate(self, verbose=False):
        """
        modifies reaction network
        :return:
        """
        cur_time = 0
        cutoff = 10000000
        # update observables
        max_poss_yield = torch.min(self.rn.copies_vec[:self.rn.num_monomers].clone()).to(self.dev)
        l_k = self.rn.compute_log_constants(self.rn.kon, self.rn.rxn_score_vec, self._constant)
        while cur_time < self.runtime:
            for obs in self.rn.observables.keys():
                try:
                    self.rn.observables[obs][1].append(self.rn.copies_vec[int(obs)].item())
                except IndexError:
                    print('bkpt')
            l_conc_prod_vec = self.rn.get_log_copy_prod_vector()
            l_rxn_rates = l_conc_prod_vec + l_k
            l_total_rate = torch.logsumexp(l_rxn_rates, dim=0)
            #l_total_rate = l_total_rate + torch.log(torch.min(self.rn.copies_vec))
            l_step = 0 - l_total_rate
            rate_step = torch.exp(l_rxn_rates + l_step)
            delta_copies = torch.matmul(self.rn.M, rate_step)

            if (torch.min(self.rn.copies_vec + delta_copies) < 1e-8):
                #print("Taking smaller timestep")
                #Take a smaller time step
                l_total_rate = l_total_rate - torch.log(torch.min(self.rn.copies_vec[torch.nonzero(self.rn.copies_vec)]))
                l_step = 0 - l_total_rate
                rate_step = torch.exp(l_rxn_rates + l_step)
                delta_copies = torch.matmul(self.rn.M, rate_step)


            # print("-----------------------------")
            # print("Total number of A: ", self.rn.copies_vec[0]+self.rn.copies_vec[4]+self.rn.copies_vec[5]+self.rn.copies_vec[6])
            # print("Rxn rates: ", l_rxn_rates)
            # print("Total rxn rate: ",l_total_rate)
            # print("Copies: ",self.rn.copies_vec)
            # print("Next step size: ",l_step)
            # print("Rate step: ", rate_step)
            # print("Matrix: ",self.rn.M)
            # print("delta_copies: ", delta_copies)

            #print("Current time: ",cur_time)
            # Prevent negative copy cumbers explicitly (possible due to local linear approximation)
            initial_monomers = self.rn.initial_copies
            min_copies = torch.ones(self.rn.copies_vec.shape, device=self.dev) * np.inf
            min_copies[0:initial_monomers.shape[0]] = initial_monomers
            self.rn.copies_vec = torch.max(self.rn.copies_vec + delta_copies, torch.zeros(self.rn.copies_vec.shape,
                                                                                          dtype=torch.double,
                                                                                          device=self.dev))
            #print("Final copies: ", self.rn.copies_vec[-1])
            step = torch.exp(l_step)
            if cur_time + step > self.runtime:
                print("Current time: ",cur_time)
                print("Next time: ",cur_time + step)
                print("Next time larger than simulation runtime. Ending simulation.")
                for obs in self.rn.observables.keys():
                    try:
                        self.rn.observables[obs][1].pop()
                    except IndexError:
                        print('bkpt')
                break
            cur_time = cur_time + step
            self.steps.append(cur_time.item())

            if len(self.steps) > cutoff:
                print("WARNING: sim was stopped early due to exceeding set max steps", sys.stderr)
                break
        total_complete = self.rn.copies_vec[-1]
        final_yield = total_complete / max_poss_yield
        return final_yield.to(self.dev)

    def plot_observable(self,nodes_list, ax=None):
        t = np.array(self.steps)
        for key in self.observables.keys():

            if self.observables[key][0] in nodes_list:
                data = np.array(self.observables[key][1])
                if not ax:
                    plt.plot(t, data, label=self.observables[key][0])
                else:
                    ax.plot(t, data, label=self.observables[key][0])
        lgnd = plt.legend(loc='best')
        plt.ticklabel_format(style='sci',scilimits=(-3,3))
        plt.ylabel(r'Conc in M')
        plt.xlabel('Time (s)')
        for i in range(len(lgnd.legendHandles)):
            lgnd.legendHandles[i]._sizes = [30]

    def observables_to_csv(self, out_path):
        data = {}
        for key in self.rn.observables:
            entry = self.rn.observables[key]
            data[entry[0]] = entry[1]
        df = pd.DataFrame(data)
        df.to_csv(out_path)
