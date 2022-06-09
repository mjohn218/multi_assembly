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
                 device='cpu',method='Adam'):
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
        if method =='Adam':
            self.optimizer = torch.optim.Adam(param_itr, learning_rate)
        elif method =='RMSprop':
            self.optimizer = torch.optim.RMSprop(param_itr, learning_rate)
        self.lr = learning_rate
        self.optim_iterations = optim_iterations
        self.sim_observables = []
        self.parameter_history = []
        self.yield_per_iter = []
        self.flux_per_iter = []
        self.is_optimized = False
        self.dt = None
        self.final_solns = []
        self.final_yields = []

    def plot_observable(self, iteration, nodes_list,ax=None):
        t = self.sim_observables[iteration]['steps']

        for key in self.sim_observables[iteration].keys():
            if key == 'steps':
                continue

            elif self.sim_observables[iteration][key][0] in nodes_list:
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

    def plot_yield(self,flux_bool=False):
        steps = np.arange(len(self.yield_per_iter))
        data = np.array(self.yield_per_iter, dtype=np.float)
        flux = np.array(self.flux_per_iter,dtype=np.float)
        plt.plot(steps, data,label='Yield')
        if flux_bool:
            plt.plot(steps,flux,label='Flux')
        #plt.ylim((0, 1))
        plt.title = 'Yield at each iteration'
        plt.show()

    def optimize(self,optim='yield',node_str=None,max_yield=0.5):
        print("Reaction Parameters before optimization: ")
        print(self.rn.get_params())
        counter = 0
        for i in range(self.optim_iterations):
            # reset for new simulator
            self.rn.reset()
            sim = self.sim_class(self.rn,
                                 self.sim_runtime,
                                 device=self._dev_name)

            # preform simulation
            self.optimizer.zero_grad()
            total_yield,total_flux = sim.simulate(optim,node_str)
            #print("Type/class of yield: ", type(total_yield))

            #Check change in yield from last gradient step. Break if less than a tolerance
            # if i > 1 and (total_yield - self.yield_per_iter[-1] >0 and total_yield - self.yield_per_iter[-1] < 1e-8):
            #     counter+=1
            #     print(total_yield,self.yield_per_iter[-1])
            #     if counter >10 :
            #         print("Max tolerance reached. Stopping optimization")
            #         print('yield on sim iteration ' + str(i) + ' was ' + str(total_yield.item() * 100)[:4] + '%')
            #         return self.rn
            # else:
            #     counter = 0
            self.yield_per_iter.append(total_yield.item())
            # self.flux_per_iter.append(total_flux.item())
            # update tracked data
            self.sim_observables.append(self.rn.observables.copy())
            self.sim_observables[-1]['steps'] = np.array(sim.steps)
            self.parameter_history.append(self.rn.kon.clone().detach().to(torch.device('cpu')).numpy())

            if optim =='yield':
                print('yield on sim iteration ' + str(i) + ' was ' + str(total_yield.item() * 100)[:4] + '%')
                # preform gradient step
                if i != self.optim_iterations - 1:
                    # if self.rn.coupling:
                    #     k = torch.exp(self.rn.compute_log_constants(self.rn.params_kon, self.rn.params_rxn_score_vec,
                    #                                             scalar_modifier=1.))
                    #     physics_penalty = torch.sum(10 * F.relu(-1 * (k - self.lr * 10))).to(self.dev)  # stops zeroing or negating params
                    #     cost = -total_yield + physics_penalty
                    #
                    #     cost.backward(retain_graph=True)
                    # elif self.rn.partial_opt:
                    #     k = torch.exp(self.rn.compute_log_constants(self.rn.params_kon, self.rn.params_rxn_score_vec,
                    #                                             scalar_modifier=1.))
                    #     physics_penalty = torch.sum(10 * F.relu(-1 * (k - self.lr * 10))).to(self.dev)  # stops zeroing or negating params
                    #     cost = -total_yield + physics_penalty
                    #
                    #     cost.backward(retain_graph=True)
                    # else:
                    #     k = torch.exp(self.rn.compute_log_constants(self.rn.kon, self.rn.rxn_score_vec,
                    #                                             scalar_modifier=1.))
                    #     physics_penalty = torch.sum(10 * F.relu(-1 * (k - self.lr * 10))).to(self.dev)  # stops zeroing or negating params
                    #     cost = -total_yield + physics_penalty
                    #
                    #     cost.backward()
                    if self.rn.assoc_is_param:
                        k = torch.exp(self.rn.compute_log_constants(self.rn.kon, self.rn.rxn_score_vec,
                                                            scalar_modifier=1.))
                        physics_penalty = torch.sum(10 * F.relu(-1 * (k - self.lr * 10))).to(self.dev)
                    elif self.rn.copies_is_param:
                        c = self.rn.c_params.clone().detach()
                        physics_penalty = torch.sum(10 * F.relu(-1 * (c))).to(self.dev)
                              # stops zeroing or negating params
                    cost = -total_yield + physics_penalty

                    cost.backward()

                    self.optimizer.step()
                    #print("Previous reaction rates: ",str(self.rn.kon.clone().detach()))
                    if self.rn.coupling:
                        new_params = self.rn.params_kon.clone().detach()
                        for rc in range(len(self.rn.kon)):
                            if rc in self.rn.cid.keys():
                                self.rn.kon[rc] = self.rn.params_kon[self.rn.coup_map[self.rn.cid[rc]]]
                            else:
                                self.rn.kon[rc] = self.rn.params_kon[self.rn.coup_map[rc]]
                    elif self.rn.partial_opt:
                        new_params = self.rn.params_kon.clone().detach()
                        for r in range(len(new_params)):
                            self.rn.kon[self.rn.optim_rates[r]] = self.rn.params_kon[r]
                    elif self.rn.copies_is_param:
                        new_params = self.rn.c_params.clone().detach()
                    else:
                        new_params = self.rn.kon.clone().detach()
                    #print('New reaction rates: ' + str(self.rn.kon.clone().detach()))
                    # new_params = self.rn.kon.clone().detach()
                    print('current params: ' + str(new_params))

                    if total_yield-max_yield > 0:
                        self.final_yields.append(total_yield)
                        self.final_solns.append(new_params)

            # elif optim =='flux':
            #     print('Flux on sim iteration ' + str(i) + ' was ' + str(total_flux.item()))
            #     # preform gradient step
            #     if i != self.optim_iterations - 1:
            #         k = torch.exp(self.rn.compute_log_constants(self.rn.kon, self.rn.rxn_score_vec,
            #                                                     scalar_modifier=1.))
            #         physics_penalty = torch.sum(10 * F.relu(-1 * (k - self.lr * 10))).to(self.dev)  # stops zeroing or negating params
            #         cost = -total_flux + physics_penalty
            #         cost.backward()
            #         self.optimizer.step()
            #         new_params = self.rn.kon.clone().detach()
            #         print('current params: ' + str(new_params))

            values = psutil.virtual_memory()
            mem = values.available / (1024.0 ** 3)
            if mem < .5:
                # kill program if it uses to much ram
                print("Killing optimization because too much RAM being used.")
                print(values.available,mem)
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
