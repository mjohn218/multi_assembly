import torch
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import psutil 
from steric_free_simulator import Simulator
from steric_free_simulator import VecSim
from steric_free_simulator import VectorizedRxnNet


class Optimizer:

    def __init__(self, reaction_network,
                 sim_runtime: float,
                 optim_iterations: int,
                 learning_rate: float,
                 sim_mode="vectorized",
                 resample_time_step=False,
                 score_constant: float = 1.,
                 freq_fact: float = 10.,
                 volume=1e-5):
        if sim_mode == 'infinite':
            self.sim_class = Simulator
            self.rn = reaction_network
        elif sim_mode == 'vectorized':
            self.sim_class = VecSim
            self.rn = VectorizedRxnNet(reaction_network)
            resample_time_step = False
        else:
            raise TypeError("sim mode not available")
        self.sim_runtime = sim_runtime
        param_itr = self.rn.get_params()
        self.optimizer = torch.optim.Adam(param_itr, learning_rate)
        self.resample_time_step = resample_time_step
        self.optim_iterations = optim_iterations
        self.sim_observables = []
        self.parameter_history = []
        self.yield_per_iter = []
        self.is_optimized = False
        self.dt = None
        self._sim_score_constant = score_constant
        self._sim_freq_factor = freq_fact
        self._sim_volume = volume

    def plot_observable(self, iteration):
        t = self.sim_observables[iteration]['steps']
        for key in self.sim_observables[iteration].keys():
            if key == 'steps':
                continue
            data = np.array(self.sim_observables[iteration][key][1])
            plt.scatter(t, data,
                        cmap='plasma',
                        s=.1,
                        label=self.sim_observables[iteration][key][0])
        plt.legend(loc='best')
        plt.title = 'Sim iteration ' + str(iteration)
        plt.show()

    def plot_yield(self):
        steps = np.arange(self.optim_iterations)
        data = np.array(self.yield_per_iter, dtype=np.float)
        data[data < .1] = np.mean(data[data > .1])
        plt.plot(steps, data)
        plt.ylim((0, 1))
        plt.title = 'Yield at each iteration'
        plt.show()

    def optimize(self):
        for i in range(self.optim_iterations):
            # reset for new simulator
            self.rn.reset()
            if self.resample_time_step or i == 0:
                sim = self.sim_class(self.rn,
                                     self.sim_runtime,
                                     score_constant=self._sim_score_constant,
                                     freq_fact=self._sim_freq_factor,
                                     volume=self._sim_volume)
                if type(sim) is Simulator:
                    sim.optimize_step()
            else:
                sim = self.sim_class(self.rn, self.sim_runtime,
                                     score_constant=self._sim_score_constant,
                                     freq_fact=self._sim_freq_factor,
                                     volume=self._sim_volume)

            # preform simulation
            self.optimizer.zero_grad()
            total_yield = sim.simulate()
            self.yield_per_iter.append(total_yield.item())
            print('yield on sim iteration ' + str(i) + ' was ' + str(total_yield.item() * 100)[:4] + '%')

            # update tracked data
            self.sim_observables.append(self.rn.observables.copy())
            self.sim_observables[-1]['steps'] = np.array(sim.steps)
            if type(sim) is Simulator:
                self.parameter_history.append(nx.get_edge_attributes(self.rn.network, 'k_on').copy())
            elif type(sim) is VecSim:
                self.parameter_history.append(self.rn.EA.clone().detach().numpy())

            # preform gradient step
            if i != self.optim_iterations - 1:
                physics_penalty = torch.sum(10 * F.relu(-1*(self.rn.EA - .1)))  # stops zeroing or negating params
                cost = -total_yield + physics_penalty
                if type(sim) is Simulator:
                    og_params = np.array([p.item() for p in self.rn.get_params()])
                elif type(sim) is VecSim:
                    og_params = self.rn.EA.clone().detach()

                cost.backward()
                self.optimizer.step()

                if type(sim) is Simulator:
                    new_params = np.array([p.item() for p in self.rn.get_params()])
                elif type(sim) is VecSim:
                    new_params = self.rn.EA.clone().detach()

                print('param update: ' + str(new_params - og_params))

            values = psutil.virtual_memory()
            mem = values.available / (1024.0 ** 3)
            if mem < 8:
                # kill program if it uses to much ram
                return self.rn
            if i == self.optim_iterations - 1:
                print("optimization complete")
                return self.rn
                # print('avg parameter update: ' + str(np.mean(og_params - new_params)))
                # print('max parameter update: ' + str(np.max(og_params - new_params)))
                # print('min parameter update: ' + str(np.min(og_params - new_params)))
            del sim
