import torch
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
import psutil
from steric_free_simulator import VecSim
from steric_free_simulator import VectorizedRxnNet


class Optimizer:

    def __init__(self, reaction_network,
                 sim_runtime: float,
                 optim_iterations: int,
                 learning_rate: float,
                 device='cpu'):
        if torch.cuda.is_available() and "cpu" not in device:
            self.dev = torch.device(device)
            print("Using " + device)
        else:
            self.dev = torch.device("cpu")
            device = 'cpu'
            print("Using CPU")
        self._dev_name = device
        self.sim_class = VecSim
        if type(reaction_network) is not VectorizedRxnNet:
            try:
                self.rn = VectorizedRxnNet(reaction_network, dev=self.dev)
            except Exception:
                raise TypeError(" Must be type ReactionNetwork or VectorizedRxnNetwork.")
        else:
            self.rn = reaction_network
        self.sim_runtime = sim_runtime
        param_itr = self.rn.get_params()
        self.optimizer = torch.optim.Adam(param_itr, learning_rate)
        self.lr = learning_rate
        self.optim_iterations = optim_iterations
        self.sim_observables = []
        self.parameter_history = []
        self.yield_per_iter = []
        self.is_optimized = False
        self.dt = None

    def plot_observable(self, iteration, ax=None):
        t = self.sim_observables[iteration]['steps']
        for key in self.sim_observables[iteration].keys():
            if key == 'steps':
                continue
            data = np.array(self.sim_observables[iteration][key][1])
            if not ax:
                plt.plot(t, data, label=self.sim_observables[iteration][key][0])
            else:
                ax.plot(t, data, label=self.sim_observables[iteration][key][0])
        lgnd = plt.legend(loc='best')
        for i in range(len(lgnd.legendHandles)):
            lgnd.legendHandles[i]._sizes = [30]
        plt.title = 'Sim iteration ' + str(iteration)
        plt.show()

    def plot_yield(self):
        steps = np.arange(len(self.yield_per_iter))
        data = np.array(self.yield_per_iter, dtype=np.float)
        plt.plot(steps, data)
        plt.ylim((0, 1))
        plt.title = 'Yield at each iteration'
        plt.show()

    def optimize(self):
        for i in range(self.optim_iterations):
            # reset for new simulator
            self.rn.reset()
            sim = self.sim_class(self.rn,
                                 self.sim_runtime,
                                 device=self._dev_name)

            # preform simulation
            self.optimizer.zero_grad()
            total_yield = sim.simulate()
            self.yield_per_iter.append(total_yield.item())
            print('yield on sim iteration ' + str(i) + ' was ' + str(total_yield.item() * 100)[:4] + '%')

            # update tracked data
            self.sim_observables.append(self.rn.observables.copy())
            self.sim_observables[-1]['steps'] = np.array(sim.steps)
            self.parameter_history.append(self.rn.kon.clone().detach().to(torch.device('cpu')).numpy())

            # preform gradient step
            if i != self.optim_iterations - 1:
                k = torch.exp(self.rn.compute_log_constants(self.rn.kon, self.rn.rxn_score_vec,
                                                            scalar_modifier=1.))
                physics_penalty = torch.sum(10 * F.relu(-1 * (k - self.lr * 10))).to(self.dev)  # stops zeroing or negating params
                cost = -total_yield + physics_penalty
                cost.backward()
                self.optimizer.step()
                new_params = self.rn.kon.clone().detach()
                print('current params: ' + str(new_params))

            values = psutil.virtual_memory()
            mem = values.available / (1024.0 ** 3)
            if mem < .5:
                # kill program if it uses to much ram
                return self.rn
            if i == self.optim_iterations - 1:
                print("optimization complete")
                return self.rn

            del sim


if __name__ == '__main__':
    from steric_free_simulator import ReactionNetwork
    base_input = './input_files/dimer.bngl'
    rn = ReactionNetwork(base_input, one_step=True)
    rn.reset()
    rn.intialize_activations()
    optim = Optimizer(reaction_network=rn,
                      sim_runtime=.001,
                      optim_iterations=10,
                      learning_rate=10,)
    vec_rn = optim.optimize()
