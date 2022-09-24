import torch
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
import psutil
from steric_free_simulator import VecSim
from steric_free_simulator import VectorizedRxnNet
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import MultiplicativeLR
import random



class Optimizer:

    def __init__(self, reaction_network,
                 sim_runtime: float,
                 optim_iterations: int,
                 learning_rate: float,
                 device='cpu',method='Adam',lr_change_step=None,gamma=None):
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
            if self.rn.dissoc_is_param:
                param_list = []
                print(param_itr)
                for i in range(len(param_itr)):
                    # print("#####")
                    # print(param_itr[i])
                    param_list.append({'params':param_itr[i],'lr':torch.mean(param_itr[i]).item()*learning_rate})
                # param_list = [{'params':param_itr[0],'lr':learning_rate[0]},{'params':param_itr[1],'lr':learning_rate[1]}]
                # print("Params List: ")
                self.optimizer = torch.optim.RMSprop(param_list)
            elif self.rn.dG_is_param:
                # torch.autograd.set_detect_anomaly(True)
                self.lr_group=[]
                if self.rn.dG_mode==1:
                    param_list=[]
                    for i in range(len(param_itr)):
                        param_list.append({'params':param_itr[i],'lr':torch.mean(param_itr[i]).item()*learning_rate})
                    self.optimizer = torch.optim.RMSprop(param_list)
                else:
                    param_list=[]

                    for i in range(len(param_itr)):
                        learn_rate = random.uniform(learning_rate,1e-1)
                        param_list.append({'params':param_itr[i],'lr':torch.mean(param_itr[i]).item()*learn_rate})
                        self.lr_group.append(learn_rate)
                    self.optimizer = torch.optim.RMSprop(param_list)
                    # print("Params: ",param_itr)
                    # self.optimizer = torch.optim.RMSprop(param_itr,torch.mean(param_itr[0]).item()*learning_rate)
            elif self.rn.chap_is_param:
                param_list = []
                for i in range(len(param_itr)):
                    param_list.append({'params':param_itr[i], 'lr':torch.mean(param_itr[i]).item()*learning_rate})
                self.optimizer = torch.optim.RMSprop(param_list)
            else:
                self.optimizer = torch.optim.RMSprop(param_itr, learning_rate)
        if self.rn.dissoc_is_param:
            self.lr = learning_rate[1]
        else:
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
        if lr_change_step is not None:
            if gamma == None:
                gamma = 0.5
            # self.scheduler = StepLR(self.optimizer,step_size=lr_change_step,gamma=gamma)
            # self.scheduler = ReduceLROnPlateau(self.optimizer,'max',patience=30)
            if self.rn.chap_is_param:
                self.scheduler = MultiplicativeLR(self.optimizer,lr_lambda=[self.lambda_c,self.lambda_k])
            elif self.rn.dG_is_param:
                if self.rn.dG_mode==1:
                    self.scheduler = MultiplicativeLR(self.optimizer,lr_lambda=[self.lambda1,self.lambda2])
                elif self.rn.dG_mode==2:
                    self.scheduler = MultiplicativeLR(self.optimizer,lr_lambda=self.lambda1)
                elif self.rn.dG_mode ==3:
                    # self.scheduler = MultiplicativeLR(self.optimizer,lr_lambda=self.lambda2)
                    self.lambda_ct = -1
                    lambda_list = []
                    print("*******Using lambda_master for LR Scheduling*****")
                    # for i in range(len(self.rn.params_k)):
                    #     lambda_list.append(self.lambda_master)

                    # self.scheduler = MultiplicativeLR(self.optimizer,lr_lambda=[self.lambda3,self.lambda4,self.lambda5])

                    self.scheduler = MultiplicativeLR(self.optimizer,lr_lambda=[self.lambda_master for i in range(len(self.rn.params_k))])
            self.lr_change_step = lr_change_step
        else:
            self.lr_change_step = None

    def lambda1(self,opt_itr):
        new_lr = torch.min(self.rn.params_k[0]).item()*self.lr
        curr_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        return(new_lr/curr_lr)
    def lambda2(self,opt_itr):
        if self.rn.dG_mode==1:
            new_lr = torch.min(self.rn.params_k[1]).item()*self.lr
            curr_lr = self.optimizer.state_dict()['param_groups'][1]['lr']
        else:
            new_lr = torch.min(self.rn.params_k).item()*self.lr
            curr_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        return(new_lr/curr_lr)

    def lambda_c(self,opt_itr):
        # alpha = []
        # for i in range(len(self.rn.params_k)):
        #     new_lr = self.rn.params_k[i].item()*self.lr_group[i]
        #     curr_lr = self.optimizer.state_dict()['param_groups'][i]['lr']
        #     alpha.append(new_lr/curr_lr)
        new_lr = torch.min(self.rn.chap_params[0]).item()*10*self.lr
        curr_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        return(new_lr/curr_lr)

    def lambda_k(self,opt_itr):
        new_lr = torch.min(self.rn.chap_params[1]).item()*self.lr
        curr_lr = self.optimizer.state_dict()['param_groups'][1]['lr']
        return(new_lr/curr_lr)
    def lambda5(self,opt_itr):
        new_lr = torch.min(self.rn.params_k[2]).item()*self.lr_group[2]
        curr_lr = self.optimizer.state_dict()['param_groups'][2]['lr']
        return(new_lr/curr_lr)

    def update_counter(self):
        lr_ct =1
        # if self.counter == len(self.rn.params_k):
        #     self.counter=0
    def lambda_master(self,opt_itr):
        # update_counter()
        # print("***** LAMBDA MASTER:  {:d}*****".format(self.lambda_counter))
        # self.counter+=
        self.lambda_ct+=1
        return(torch.min(self.rn.params_k[self.lambda_ct%len(self.rn.params_k)]).item()*self.lr_group[self.lambda_ct%len(self.rn.params_k)]/self.optimizer.state_dict()['param_groups'][self.lambda_ct%len(self.rn.params_k)]['lr'])



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

    def optimize(self,optim='yield',node_str=None,max_yield=0.5,corr_rxns=[[1],[5]]):
        print("Reaction Parameters before optimization: ")
        print(self.rn.get_params())

        print("Optimizer State:",self.optimizer.state_dict)
        counter = 0
        calc_flux_optim=False
        if optim=='flux_coeff':
            calc_flux_optim=True
        for i in range(self.optim_iterations):
            # reset for new simulator
            self.rn.reset()
            sim = self.sim_class(self.rn,
                                 self.sim_runtime,
                                 device=self._dev_name,calc_flux=calc_flux_optim)
            # print(sim.calc_flux)

            # preform simulation
            self.optimizer.zero_grad()
            total_yield,total_flux = sim.simulate(optim,node_str,corr_rxns=corr_rxns)
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
                # print(self.rn.copies_vec)
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

                        # cost.backward()
                    if self.rn.assoc_is_param:
                        if self.rn.partial_opt:
                            k = torch.exp(self.rn.compute_log_constants(self.rn.params_kon, self.rn.params_rxn_score_vec,scalar_modifier=1.))
                            physics_penalty = torch.sum(10 * F.relu(-1 * (k - self.lr * 10))).to(self.dev)  # stops zeroing or negating params
                            cost = -total_yield + physics_penalty
                            cost.backward(retain_graph=True)
                        else:
                            k = torch.exp(self.rn.compute_log_constants(self.rn.kon, self.rn.rxn_score_vec,
                                                            scalar_modifier=1.))
                            physics_penalty = torch.sum(10 * F.relu(-1 * (k - self.lr * 10))).to(self.dev)
                            cost = -total_yield + physics_penalty
                            cost.backward()
                    elif self.rn.copies_is_param:
                        c = self.rn.c_params.clone().detach()
                        physics_penalty = torch.sum(10 * F.relu(-1 * (c))).to(self.dev)# stops zeroing or negating params
                        cost = -total_yield + physics_penalty
                        cost.backward()
                    elif self.rn.chap_is_param:
                        c = self.rn.chap_params[0].clone().detach()
                        k = self.rn.chap_params[1].clone().detach()
                        physics_penalty = torch.sum(10 * F.relu(-1 * (c))).to(self.dev) + torch.sum(10 * F.relu(-1 * (k - self.lr))).to(self.dev) #+ torch.sum(00 * F.relu(c-1e2)).to(self.dev)
                        print("Penalty: ",physics_penalty)
                        cost = -total_yield + physics_penalty
                        cost.backward(retain_graph=True)
                    elif self.rn.dissoc_is_param:
                        if self.rn.partial_opt:
                            k = torch.exp(self.rn.compute_log_constants(self.rn.kon, self.rn.rxn_score_vec,scalar_modifier=1.))
                            new_l_k = torch.cat([k,torch.log(self.rn.params_koff)],dim=0)
                            physics_penalty = torch.sum(10 * F.relu(-1 * (new_l_k))).to(self.dev)  # stops zeroing or negating params
                            cost = -total_yield + physics_penalty
                            cost.backward(retain_graph=True)
                        else:

                            k = torch.exp(self.rn.compute_log_constants(self.rn.kon, self.rn.rxn_score_vec,
                                                                scalar_modifier=1.))
                            physics_penalty = torch.sum(10 * F.relu(-1 * (k - self.lr * 10))).to(self.dev)
                            cost = -total_yield + physics_penalty
                            # print(self.optimizer.state_dict)
                            cost.backward()
                            metric = torch.mean(self.rn.params_koff[0].clone().detach()).item()
                    elif self.rn.dG_is_param:
                        k = torch.exp(self.rn.compute_log_constants(self.rn.kon, self.rn.rxn_score_vec,
                                                            scalar_modifier=1.))
                        g = self.rn.compute_total_dG(k)
                        print("Total Complex dG = ",g)

                        dG_penalty = F.relu((g-(self.rn.complx_dG+2))) + F.relu(-1*(g-(self.rn.complx_dG-2)))
                        print("Current On rates: ", k[:len(self.rn.kon)])
                        physics_penalty = torch.sum(1 * F.relu(-1 * (k - self.lr * 10))).to(self.dev) + torch.sum(100 * F.relu((k - 1e2))).to(self.dev)
                        cost = -total_yield + physics_penalty + 10*dG_penalty
                        # print(self.optimizer.state_dict)
                        cost.backward(retain_graph=True)
                        metric = torch.mean(self.rn.params_k[1].clone().detach()).item()

                    self.optimizer.step()
                    # self.scheduler.step(metric)
                    if self.lr_change_step is not None:
                        self.scheduler.step()
                    #Changing learning rate
                    if (self.lr_change_step is not None) and (i%100 ==0) and (i>0):
                        print("New learning rate : ")
                        for param_groups in self.optimizer.param_groups:
                            print(param_groups['lr'])

                    #print("Previous reaction rates: ",str(self.rn.kon.clone().detach()))
                    if self.rn.coupling:
                        new_params = self.rn.params_kon.clone().detach()
                        for rc in range(len(self.rn.kon)):
                            if rc in self.rn.cid.keys():
                                self.rn.kon[rc] = self.rn.params_kon[self.rn.coup_map[self.rn.cid[rc]]]
                            else:
                                self.rn.kon[rc] = self.rn.params_kon[self.rn.coup_map[rc]]
                    elif self.rn.partial_opt and self.rn.assoc_is_param:
                        new_params = self.rn.params_kon.clone().detach()
                        for r in range(len(new_params)):
                            self.rn.kon[self.rn.optim_rates[r]] = self.rn.params_kon[r]
                    elif self.rn.copies_is_param:
                        new_params = self.rn.c_params.clone().detach()
                    elif self.rn.chap_is_param:
                        new_params = [l.clone().detach() for l in self.rn.chap_params]
                    elif self.rn.dissoc_is_param:
                        if self.rn.partial_opt:
                            new_params = self.rn.params_koff.clone().detach()
                            self.rn.params_kon = self.rn.params_koff/(self.rn._C0*torch.exp(self.rn.params_rxn_score_vec))
                            for r in range(len(new_params)):
                                self.rn.kon[self.rn.optim_rates[r]] = self.rn.params_kon[r]
                            print("Current On rates: ", self.rn.kon)
                        else:
                            print("Current On rates: ", torch.exp(k)[:len(self.rn.kon)])
                            new_params = [l.clone().detach() for l in self.rn.params_koff]
                    elif self.rn.dG_is_param:
                        # print("Current On rates: ", torch.exp(k)[:len(self.rn.kon)])

                        if self.rn.dG_mode==1:
                            new_params = [l.clone().detach() for l in self.rn.params_k]
                        else:
                            new_params = [l.clone().detach() for l in self.rn.params_k]
                            # new_params = self.rn.params_k.clone().detach()

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

            elif optim=='flux_coeff':
                print("Optimizing Flux Correlations")
                print('yield on sim iteration ' + str(i) + ' was ' + str(total_yield.item()))
                if i != self.optim_iterations - 1:
                        k = torch.exp(self.rn.compute_log_constants(self.rn.kon, self.rn.rxn_score_vec,
                                                                    scalar_modifier=1.))
                        physics_penalty = torch.sum(10 * F.relu(-1 * (k - self.lr * 10))).to(self.dev)  # stops zeroing or negating params
                        cost = -total_yield + physics_penalty
                        cost.backward()
                        self.optimizer.step()
                        new_params = self.rn.kon.clone().detach()
                        print('current params: ' + str(new_params))

                        # if total_yield-max_yield > 0:
                        #     self.final_yields.append(total_yield)
                        #     self.final_solns.append(new_params)




            values = psutil.virtual_memory()
            mem = values.available / (1024.0 ** 3)
            if mem < .5:
                # kill program if it uses to much ram
                print("Killing optimization because too much RAM being used.")
                print(values.available,mem)
                return self.rn
            if i == self.optim_iterations - 1:
                print("optimization complete")
                print("Final params: " + str(new_params))
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
