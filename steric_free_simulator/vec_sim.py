from steric_free_simulator.vectorized_rxn_net import VectorizedRxnNet
from steric_free_simulator import ReactionNetwork
import numpy as np

from torch import DoubleTensor as Tensor
import torch
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import random
from scipy import signal
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
                 device='cuda:0',calc_flux=False):
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
        self.flux_vs_time = self.rn.flux_vs_time
        self.net_flux = dict()
        self.switch=False
        self.uid_flux = torch.zeros(1,2*self.rn.reaction_network._rxn_count)
        self.calc_flux=calc_flux

        if self.rn.rxn_coupling:
            self.coupled_kon = torch.zeros(len(self.rn.kon), requires_grad=True).double()


    def simulate(self, optim='yield',node_str=None,verbose=False,switch=False,switch_time=0,switch_rates=None,corr_rxns=[[0],[1]],conc_scale=1.0):
        """
        modifies reaction network
        :return:
        """
        cur_time = 0
        cutoff = 10000000
        # update observables
        max_poss_yield = torch.min(self.rn.copies_vec[:self.rn.num_monomers].clone()).to(self.dev)

        if self.rn.max_subunits !=-1:
            max_poss_yield = max_poss_yield/self.rn.max_subunits
            print("Max Poss Yield: ",max_poss_yield)
        t95_flag=True
        t85_flag=True
        t50_flag=True
        t99_flag=True
        t85=-1
        t95=-1
        t50=-1
        t99=-1

        if self.rn.chap_is_param:
            mask = torch.ones([len(self.rn.copies_vec[:self.rn.num_monomers])],dtype=bool)
            for species,uids in self.rn.chap_uid_map.items():
                mask[species]=False
            max_poss_yield = torch.min(self.rn.copies_vec[:self.rn.num_monomers][mask].clone()).to(self.dev)

        if self.rn.rxn_coupling:
            #new_kon = torch.zeros(len(self.rn.kon), requires_grad=True).double()
            # print("Coupling")
            for i in range(len(self.rn.kon)):
                # print(i)
                if i in self.rn.rx_cid.keys():
                    #new_kon[i] = 1.0
                    self.coupled_kon[i] = max(self.rn.kon[rate] for rate in self.rn.rx_cid[i])
                    # print("Max rate for reaction %s chosen as %.3f" %(i,self.coupled_kon[i]))
                else:
                    self.coupled_kon[i] = self.rn.kon[i]
            l_k = self.rn.compute_log_constants(self.coupled_kon,self.rn.rxn_score_vec, self._constant)
        else:
            l_k = self.rn.compute_log_constants(self.rn.kon, self.rn.rxn_score_vec, self._constant)
            print("Simulation rates: ",torch.exp(l_k))

        while cur_time < self.runtime:

            for obs in self.rn.observables.keys():
                try:
                    self.rn.observables[obs][1].append(self.rn.copies_vec[int(obs)].item())
                    #self.flux_vs_time[obs][1].append(self.net_flux[self.flux_vs_time[obs][0]])
                except IndexError:
                    print('bkpt')
            l_conc_prod_vec = self.rn.get_log_copy_prod_vector()
            # if self.rn.boolCreation_rxn:
                # l_conc_prod_vec[-1]=torch.log(torch.pow(Tensor([0]),Tensor([1])))
            l_rxn_rates = l_conc_prod_vec + l_k
            l_total_rate = torch.logsumexp(l_rxn_rates, dim=0)
            #l_total_rate = l_total_rate + torch.log(torch.min(self.rn.copies_vec))
            l_step = 0 - l_total_rate
            rate_step = torch.exp(l_rxn_rates + l_step)
            # conc_scale = 1  #Units uM
            # if torch.min(self.rn.copies_vec[torch.nonzero(self.rn.copies_vec)]) < conc_scale:
            #     conc_scale = torch.min(self.rn.copies_vec[torch.nonzero(self.rn.copies_vec)]).item()

            delta_copies = torch.matmul(self.rn.M, rate_step)*conc_scale
            #Calculate rxn_flux
            if self.calc_flux:
                rxn_flux = self.rn.get_reaction_flux()





            if (torch.min(self.rn.copies_vec + delta_copies) < 0):
                temp_copies = self.rn.copies_vec + delta_copies
                min_idx = torch.argmin(temp_copies)
                min_value = self.rn.copies_vec[min_idx]

                delta_copy = torch.matmul(self.rn.M[min_idx,:],rate_step)
                modulator = min_value/abs(delta_copy)

                # print("Taking smaller timestep")
                # print(self.rn.copies_vec + delta_copies)
                # print("Previous rate step: ",rate_step)

                #Take a smaller time step
                # l_total_rate = l_total_rate - torch.log(torch.min(self.rn.copies_vec[torch.nonzero(self.rn.copies_vec)]))
                l_total_rate = l_total_rate - torch.log(modulator)
                l_step = 0 - l_total_rate
                rate_step = torch.exp(l_rxn_rates + l_step)
                delta_copies = torch.matmul(self.rn.M, rate_step)

                # print("New rate step: ",rate_step)


            # print("-----------------------------")
            # print("Total number of A: ", self.rn.copies_vec[0]+self.rn.copies_vec[4]+self.rn.copies_vec[5]+self.rn.copies_vec[6]) #For trimer model
            # print("Total number of A: ", self.rn.copies_vec[0]+self.rn.copies_vec[2]+2*self.rn.copies_vec[3]+2*self.rn.copies_vec[4]) #For repeat model
            # print("Total number of A: ", self.rn.copies_vec[0]+self.rn.copies_vec[1]*2+3*self.rn.copies_vec[2])   #For homotrimer model
            # print("Total COpies of A: ", self.rn.copies_vec[0]+self.rn.copies_vec[3]+self.rn.copies_vec[4]+self.rn.copies_vec[7]+2*(self.rn.copies_vec[5]+self.rn.copies_vec[8]+self.rn.copies_vec[9]+self.rn.copies_vec[10]))
            # print("Prod of conc.: ", torch.exp(l_conc_prod_vec))
            # print(l_k)
            # print("Rxn rates: ", torch.exp(l_rxn_rates))
            # print("Total rxn rate: ",l_total_rate)
            # print("Rate step: ",rate_step)
            # print("Copies: ",self.rn.copies_vec)
            #
            # print("Next step size: ",l_step)
            # print("Sum of steps: ", torch.sum(rate_step))
            # print("Matrix: ",self.rn.M)
            # print("delta_copies: ", delta_copies)
            # print("SUM: ",torch.sum(delta_copies))
            #
            # print("Current time: ",cur_time)
            # Prevent negative copy cumbers explicitly (possible due to local linear approximation)
            initial_monomers = self.rn.initial_copies
            min_copies = torch.ones(self.rn.copies_vec.shape, device=self.dev) * np.inf
            min_copies[0:initial_monomers.shape[0]] = initial_monomers
            self.rn.copies_vec = torch.max(self.rn.copies_vec + delta_copies, torch.zeros(self.rn.copies_vec.shape,
                                                                                          dtype=torch.double,
                                                                                          device=self.dev))
            # print("Final copies: ", self.rn.copies_vec)


            step = torch.exp(l_step)
            # print("Full step: ",step)
            if cur_time + step > self.runtime:
                # print("Current time: ",cur_time)
                print("Next time: ",cur_time + step)
                print("Number of steps: ", len(self.steps))
                print("Next time larger than simulation runtime. Ending simulation.")
                for obs in self.rn.observables.keys():
                    try:
                        self.rn.observables[obs][1].pop()
                    except IndexError:
                        print('bkpt')
                break

            #Add a switching criteria. Jump rates to optimized value
            # if switch and (cur_time + step > switch_time):
            #     print("Rates switched")
            #     self.switch = True
            #     # print("New rates: ",self.rn.kon)
            #     l_k = self.rn.compute_log_constants(switch_rates, self.rn.rxn_score_vec, self._constant)
            #     print("Time: ", cur_time+step)
            #     print(l_k)
            #     switch=False

            cur_time = cur_time + step*conc_scale

            if self.rn.copies_vec[-1]/max_poss_yield > 0.5 and t50_flag:
                t50=cur_time
                t50_flag=False
            if self.rn.copies_vec[-1]/max_poss_yield > 0.85 and t85_flag:
                t85=cur_time
                t85_flag=False
            if self.rn.copies_vec[-1]/max_poss_yield > 0.95 and t95_flag:
                t95=cur_time
                t95_flag=False
            if self.rn.copies_vec[-1]/max_poss_yield > 0.99 and t99_flag:
                t99=cur_time
                t99_flag=False
            self.steps.append(cur_time.item())
            if self.calc_flux:
                self.uid_flux = torch.cat((self.uid_flux,rxn_flux),0)


            # print("Current time: ",cur_time)
            #Calculate the flux
            # self.net_flux = self.rn.calculate_total_flux()
            # for obs in self.rn.flux_vs_time.keys():
            #     try:
            #         self.flux_vs_time[obs][1].append(self.net_flux[self.flux_vs_time[obs][0]])
            #     except IndexError:
            #         print('bkpt')

            if len(self.steps) > cutoff:
                print("WARNING: sim was stopped early due to exceeding set max steps", sys.stderr)
                break
            # if len(self.steps)%10==0:
            #     break
            #     print("Current Time: ",cur_time)
        if self.rn.chaperone:
            total_complete = self.rn.copies_vec[-2]/max_poss_yield
        else:
            total_complete = self.rn.copies_vec[-1]/max_poss_yield

        # total_complete = torch.max(torch.DoubleTensor([self.rn.copies_vec[3],self.rn.copies_vec[4],self.rn.copies_vec[5]]))
        # final_yield = torch.abs(0.66932 - (total_complete / max_poss_yield))
        # final_yield = total_complete/max_poss_yield
        final_yield = total_complete
        print("Final Yield: ", final_yield)
        if optim=='flux_coeff':
            final_yield = self.calc_corr_coeff(corr_rxns)
            # print(final_yield)
            return(Tensor([final_yield]).to(self.dev),None)


        if optim == 'flux':
            if node_str != None:
                return(final_yield.to(self.dev),self.net_flux[node_str].to(self.dev))
            else:
                return(final_yield.to(self.dev),self.net_flux[list(self.net_flux.keys())[-1]].to(self.dev))
        else:
            # return (final_yield.to(self.dev),self.net_flux[list(self.net_flux.keys())[-1]].to(self.dev))
            return(final_yield.to(self.dev),(t50,t85,t95,t99))

    def plot_observable(self,nodes_list, ax=None,flux=False,legend=True,seed=None,color_input=None):
        t = np.array(self.steps)
        colors_list = list(mcolors.CSS4_COLORS.keys())
        random.seed(a=seed)
        if not flux:
            counter=0
            for key in self.observables.keys():

                if self.observables[key][0] in nodes_list:
                    data = np.array(self.observables[key][1])
                    if color_input is not None:
                        clr=color_input[counter]
                    else:
                        clr=random.choice(colors_list)
                    if not ax:
                        plt.plot(t, data, label=self.observables[key][0],color=clr)
                    else:
                        ax.plot(t, data, label=self.observables[key][0],color=clr)
                counter+=1
        else:
            for key in self.flux_vs_time.keys():
                if self.flux_vs_time[key][0] in nodes_list:
                    data2 = np.array(self.flux_vs_time[key][1])
                    #print(data2)
                    if not ax:
                        plt.plot(t, data2, label=self.flux_vs_time[key][0],color=random.choice(colors_list))
                    else:
                        ax.plot(t, data2, label=self.flux_vs_time[key][0],color=random.choice(colors_list))
        if legend:
            lgnd = plt.legend(loc='best')
            for i in range(len(lgnd.legendHandles)):
                lgnd.legendHandles[i]._sizes = [30]

        plt.ticklabel_format(style='sci',scilimits=(-3,3))
        plt.tick_params(axis='both',labelsize=14.0)
        f_dict = {'fontsize':14}
        plt.ylabel(r'Conc in $\mu M$',fontdict=f_dict)
        plt.xlabel('Time (s)',fontdict=f_dict)

    def observables_to_csv(self, out_path):
        data = {}
        for key in self.rn.observables:
            entry = self.rn.observables[key]
            data[entry[0]] = entry[1]
        df = pd.DataFrame(data)
        df.to_csv(out_path)

    def calc_corr_coeff(self,rid):
        flux_data=self.uid_flux.detach().numpy()[:-1,:]
        total_coeff = 0
        total_lag = 0
        print(rid)
        for i in range(len(rid[0])):
            x=flux_data[:,rid[0][i]]
            y=flux_data[:,rid[1][i]]
            # print(x)
            corr_array = signal.correlate(x,y,mode='full')
            lag1 = np.argmax(corr_array)-np.floor(corr_array.shape[0]/2)
            coeff = np.corrcoef(x,y,rowvar=False)

            # print(coeff)
            total_coeff += coeff[0,1]
            total_lag+=lag1

        return(total_coeff/len(rid[0]))
        # return(total_lag/len(rid[0]))
