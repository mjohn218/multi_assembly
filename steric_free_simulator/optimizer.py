import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from steric_free_simulator import Simulator


class Optimizer:

    def __init__(self, reaction_network, sim_runtime, optim_iterations, learning_rate, resample_time_step=False):
        self.rn = reaction_network
        self.sim_runtime = sim_runtime
        param_itr = self.rn.get_params()
        self.optimizer = torch.optim.SGD(param_itr, learning_rate)
        self.resample_time_step = resample_time_step
        self.optim_iterations = optim_iterations
        self.sim_observables = []
        self.parameter_history = []
        self.yield_per_iter = []
        self.is_optimized = False
        self.dt = None

    def plot_observable(self, iteration):
        num_steps = int(self.sim_runtime / self.dt)
        t = np.arange(num_steps) * self.dt
        for key in self.sim_observables[iteration].keys():
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
        plt.title = 'Yield at each iteration'
        plt.show()

    def optimize(self):
        for i in range(self.optim_iterations):
            # reset for new simulator
            self.rn.reset()
            if self.resample_time_step or i == 0:
                sim = Simulator(self.rn, self.sim_runtime)
                sim.optimize_step()
                print("time step: " + str(sim.dt))
                sim.steps = int(self.sim_runtime / sim.dt)
                self.dt = sim.dt
            else:
                sim = Simulator(self.rn, self.sim_runtime, dt=self.dt, optimize_dt=False)

            # preform simulation
            self.optimizer.zero_grad()
            total_yield = sim.simulate()
            self.yield_per_iter.append(total_yield.item())
            print('yield on sim iteration ' + str(i) + ' was ' + str(total_yield.item() * 100)[:4] + '%')

            # update tracked data
            self.sim_observables.append(self.rn.observables.copy())
            self.parameter_history.append(nx.get_edge_attributes(self.rn.network, 'k_on').copy())

            # preform gradient step
            if i != self.optim_iterations - 1:
                cost = -total_yield
                og_params = np.array([p.item() for p in self.rn.get_params()])
                cost.backward()
                self.optimizer.step()

                new_params = np.array([p.item() for p in self.rn.get_params()])
                print('param update: ' + str(og_params - new_params))
                # print('avg parameter update: ' + str(np.mean(og_params - new_params)))
                # print('max parameter update: ' + str(np.max(og_params - new_params)))
                # print('min parameter update: ' + str(np.min(og_params - new_params)))
            del sim
