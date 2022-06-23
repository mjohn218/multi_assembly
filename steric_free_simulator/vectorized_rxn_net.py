from typing import Tuple

from steric_free_simulator import ReactionNetwork
from steric_free_simulator import reaction_network as RN

import networkx as nx

import torch
from torch import DoubleTensor as Tensor
from torch import rand
from torch import nn



class VectorizedRxnNet:
    """
    Provides a lightweight class that represents the core information needed for
    simulation as torch tensors. Acts as a base object for optimization simulations.
    Data structure is performance optimized, not easily readable / accessible.

    Units:
    units of Kon assumed to be [copies]-1 S-1, units of Koff S-1
    units of reaction scores are treated as J * c / mol where c is a user defined scalar
    """

    def __init__(self, rn: ReactionNetwork, assoc_is_param=True, copies_is_param=False, dissoc_is_param=False, dG_is_param=False,dev='cpu',coupling=False,cid={-1:-1}, rxn_coupling=False, rx_cid={-1:-1},optim_rates=None):
        """

        :param rn: The reaction network template
        :param assoc_is_param: whether the association constants should be treated as parameters for optimization
        :param copies_is_param: whether the initial copy numbers should be treated as parameters for optimization
        :param dev: the device to use for torch tensors
        :param coupling : If two reactions have same kon. i.e. Addition of new subunit is same as previous subunit
        :param cid : Reaction ids in a dictionary format. {child_reaction:parent_reaction}. Set the rate of child_reaction to parent_reaction
        """
        #rn.reset()
        self.dev = torch.device(dev)
        self._avo = Tensor([6.02214e23])  # copies / mol
        self._R = Tensor([8.314])  # J / mol * K
        self._T = Tensor([273.15])  # K
        self._C0 = Tensor([1e6])    #Std. Conc in uM

        #Variables for zeroth order reactions
        self.boolCreation_rxn = rn.boolCreation_rxn
        self.creation_nodes = rn.creation_nodes
        self.creation_rxn_data = rn.creation_rxn_data

        #Variables for Destruction order reactions
        self.boolDestruction_rxn = rn.boolDestruction_rxn
        self.destruction_nodes = rn.destruction_nodes
        self.destruction_rxn_data = rn.destruction_rxn_data

        self.M, self.kon, self.rxn_score_vec, self.copies_vec = self.generate_vectorized_representation(rn)
        self.coupling = coupling
        self.rxn_coupling = rn.rxn_coupling
        self.num_monomers = rn.num_monomers
        if optim_rates is not None:
            self.partial_opt = True
            self.optim_rates = optim_rates
        else:
            self.partial_opt = False
        self.cid = cid
        self.rx_cid = rn.rxn_cid
        self.coup_map = {}

        #Make new param Tensor (that will be optimized) if coupling is True
        if self.coupling == True:
            c_rxn_count = len(cid.values())
            ind_rxn_count = rn._rxn_count - c_rxn_count
            self.params_kon = torch.zeros([ind_rxn_count], requires_grad=True).double()   #Create param Tensor for only the independant reactions
            self.params_rxn_score_vec = torch.zeros([ind_rxn_count]).double()
            #self.kon.requires_grad_(False)
            rid=0
            for i in range(rn._rxn_count):
                if i not in cid.keys():
                    ##Independent reactions
                    self.params_kon[rid] = self.kon.clone().detach()[i]
                    self.params_rxn_score_vec[rid] = self.rxn_score_vec[i]
                    self.coup_map[i]=rid           #Map reaction index for independent reactions in self.kon to self.params_kon. Used to set the self.kon from self.params_kon
                    rid+=1
            self.params_kon.requires_grad_(True)

            self.initial_params = Tensor(self.params_kon).clone().detach()
        elif self.partial_opt == True:
            c_rxn_count = len(self.optim_rates)
            self.params_kon = torch.zeros([c_rxn_count], requires_grad=True).double()   #Create param Tensor for only the independant reactions
            self.params_rxn_score_vec = torch.zeros([c_rxn_count]).double()
            for i in range(c_rxn_count):
                self.params_kon[i] = self.kon.clone().detach()[self.optim_rates[i]]
                self.params_rxn_score_vec[i] = self.rxn_score_vec[i]
            self.params_kon.requires_grad_(True)
            self.initial_params = Tensor(self.params_kon).clone().detach()
        elif dissoc_is_param:
            # self.params_koff = torch.zeros([rn._rxn_count],requires_grad=True).double()
            # self.params_koff = torch.exp(self.rxn_score_vec)*self.kon* self._C0
            self.params_koff = self.kon*1e-2
            self.kon = self.params_koff/(self._C0*torch.exp(self.rxn_score_vec))
            self.params_koff = nn.Parameter(self.params_koff, requires_grad=True)
            self.initial_params = Tensor(self.params_koff).clone().detach()
        elif dG_is_param:
            self.initial_dG = self.rxn_score_vec.clones().detach()
            k_off = self.kon*self._C0*torch.exp(self.rxn_score_vec)
            self.params_k = torch.cat([self.kon,k_off],dim=0)
            self.params_k = nn.Parameter(self.params_k, requires_grad=True)
            self.initial_params = Tensor(self.params_k).clone().detach()

        else:
            self.initial_params = Tensor(self.kon).clone().detach()
        self.initial_copies = self.copies_vec.clone().detach()
        self.assoc_is_param = assoc_is_param
        self.copies_is_param = copies_is_param
        self.dissoc_is_param = dissoc_is_param
        self.dG_is_param = dG_is_param
        if assoc_is_param:
            if self.coupling:
                self.params_kon = nn.Parameter(self.params_kon, requires_grad=True)
            elif self.partial_opt:
                self.params_kon = nn.Parameter(self.params_kon, requires_grad=True)
            else:
                self.kon = nn.Parameter(self.kon, requires_grad=True)
        if copies_is_param:
            print("COPIES ARE PARAMS:::")
            self.c_params = nn.Parameter(self.initial_copies[:rn.num_monomers], requires_grad=True)
        self.observables = rn.observables
        self.flux_vs_time = rn.flux_vs_time
        self.is_energy_set = rn.is_energy_set
        self.num_monomers = rn.num_monomers
        self.reaction_ids = []
        self.reaction_network = rn
        self.to(dev)

    def reset(self, reset_params=False):
        self.copies_vec = self.initial_copies.clone()
        if self.copies_is_param:
            self.copies_vec[:self.num_monomers] = self.c_params.clone()
        # print("Initial copies: ", self.initial_copies.clone())
        if reset_params:
            if self.coupling:
                self.params_kon = nn.Parameter(self.initial_params.clone(), requires_grad=True)
            elif self.partial_opt:
                self.params_kon = nn.Parameter(self.initial_params.clone(), requires_grad=True)
            elif self.dissoc_is_param:
                self.params_koff = nn.Parameter(self.initial_params.clone(), requires_grad=True)
            elif self.dG_is_param:
                self.params_k = nn.Parameter(self.initial_params.clone(), requires_grad=True)
            else:
                self.kon = nn.Parameter(self.initial_params.clone(), requires_grad=True)
        for key in self.observables:
            self.observables[key] = (self.observables[key][0], [])

    def get_params(self):
        if self.assoc_is_param and self.copies_is_param:
            if self.coupling:
                return [self.params_kon,self.c_params]
            elif self.partial_opt:
                return [self.params_kon,self.c_params]
            else:
                return [self.kon, self.c_params]
        elif self.copies_is_param:
            return [self.c_params]
        elif self.assoc_is_param:
            if self.coupling:
                return [self.params_kon]
            elif self.partial_opt:
                return [self.params_kon]
            else:
                return [self.kon]
        elif self.dissoc_is_param:
            return [self.params_koff]
        elif self.dG_is_param:
            return [self.params_k]

    def to(self, dev):
        self.M = self.M.to(dev)
        if self.coupling:
            self.params_kon = nn.Parameter(self.params_kon.data.clone().detach().to(dev), requires_grad=True)
        elif self.partial_opt:
            self.params_kon = nn.Parameter(self.params_kon.data.clone().detach().to(dev), requires_grad=True)
        elif self.dissoc_is_param:
            self.params_koff = nn.Parameter(self.params_koff.data.clone().detach().to(dev), requires_grad=True)
        elif self.dG_is_param:
            self_params_k = nn.Parameter(self.params_k.data.clone().detach().to(dev), requires_grad=True)
        else:
            self.kon = nn.Parameter(self.kon.data.clone().detach().to(dev), requires_grad=True)
        self.copies_vec = self.copies_vec.to(dev)
        self.initial_copies = self.initial_copies.to(dev)
        self.rxn_score_vec = self.rxn_score_vec.to(dev)
        self.dev = dev
        return self

    def generate_vectorized_representation(self, rn: ReactionNetwork) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Get a matrix mapping reactions to state updates. Since every reaction has a forward
        and reverse, dimensions of map matrix M are (rxn_count*2 x num_states). The forward
        reactions are placed in the first half along the reaction axis, and the reverse
        reactions in the second half. Note that the reverse map is simply -1 * the forward map.

        Returns: M, k_on_vec, rxn_score_vec, copies_vec
            M: Tensor, A matrix that maps a vector in reaction space to a vector in state space
                shape (num_states, rxn_count * 2).
            k_vec: Tensor, A vector of rate constants in reaction space. shape (rxn_count).
            rxn_score_vec: Tensor, A vector of the rosetta resolved reaction scores in reaction space.
                shape (rxn_count), though note both halves are symmetric.
            copies_vec: Tensor, A vector of the state copy numbers in state space. shape (num_states).
        """
        num_states = len(rn.network.nodes)
        # initialize tensor representation dimensions
        M = torch.zeros((num_states, rn._rxn_count * 2)).double()
        kon = torch.zeros([rn._rxn_count], requires_grad=True).double()
        rxn_score_vec = torch.zeros([rn._rxn_count]).double()
        copies_vec = torch.zeros([num_states]).double()

        for n in rn.network.nodes():
            print(RN.gtostr(rn.network.nodes[n]['struct']))
            copies_vec[n] = rn.network.nodes[n]['copies']
            print("Reactant Sets:")

            #First check if there are any zeroth order reactions
            if self.boolCreation_rxn or self.boolDestruction_rxn:
                if n in self.creation_nodes:
                    reaction_id = self.creation_rxn_data[n]['uid']
                    kon[reaction_id]=self.creation_rxn_data[n]['k_on']
                    M[n,reaction_id]=1.
                if n in self.destruction_nodes:
                    reaction_id = self.destruction_rxn_data[n]['uid']
                    kon[reaction_id]=self.destruction_rxn_data[n]['k_on']
                    M[n,reaction_id]=-1.
            for r_set in rn.get_reactant_sets(n):
                r_tup = tuple(r_set)
                print(r_tup)
                data = rn.network.get_edge_data(r_tup[0], n)
                reaction_id = data['uid']
                try:
                    kon[reaction_id] = data['k_on']
                except Exception:
                    kon[reaction_id] = 1.
                rxn_score_vec[reaction_id] = data['rxn_score']
                # forward
                if len(r_tup) == 2:   #Bimolecular reaction; Two reactants
                    M[n, reaction_id] = 1.
                    for r in r_tup:
                        M[r, reaction_id] = -1.
                elif len(r_tup) == 1:  #Only one reactant; Have to check if its a Bimolecular
                    if rn.network.nodes[n]['struct'].number_of_edges()>0:
                        #This means there is a bond formation. Therefore it has to be a Bimolecular
                        #But it has same reactant. Reaction stoich = 2
                        M[n,reaction_id] = 1.
                        for r in r_tup:
                            M[r,reaction_id] = -2.
        # generate the reverse map explicitly
        # M[0,11]=0
        M[:, rn._rxn_count:] = -1 * M[:, :rn._rxn_count]

        #To adjust for creation reactions. No reversible destruction
        if self.boolCreation_rxn or self.boolDestruction_rxn:
            num_creat_dest_rxn = len(self.creation_rxn_data) + len(self.destruction_rxn_data)

            new_M = M[:,:-num_creat_dest_rxn:]
            return new_M,kon,rxn_score_vec, copies_vec
        print(M)
        print(kon)

        # print(copies_vec)

        return M, kon, rxn_score_vec, copies_vec

    def compute_log_constants(self, kon: Tensor, dGrxn: Tensor, scalar_modifier) -> Tensor:
        """
        Returns log(k) for each reaction concatenated with log(koff) for each reaction
        """
        # above conversions cancel
        # std_c = Tensor([1e6])  # units umols / L
        l_kon = torch.log(kon)  # umol-1 s-1
        # l_koff = (dGrxn * scalar_modifier / (self._R * self._T)) + l_kon + torch.log(std_c)       #Units of dG in J/mol
        l_koff = (dGrxn * scalar_modifier) + l_kon + torch.log(self._C0)       #Units of dG in J/mol
        l_k = torch.cat([l_kon, l_koff], dim=0)
        if self.boolCreation_rxn or self.boolDestruction_rxn:
            num_creat_dest_rxn = len(self.creation_rxn_data) + len(self.destruction_rxn_data)
            new_l_k = l_k[:-num_creat_dest_rxn]
            return new_l_k.clone().to(self.dev)
        elif self.dissoc_is_param:
            new_l_koff = torch.log(self.params_koff)
            # print("Simulation offrates: ",torch.exp(new_l_koff))
            new_l_kon = new_l_koff - (dGrxn * scalar_modifier) - torch.log(self._C0)
            new_l_k = torch.cat([new_l_kon,new_l_koff],dim=0)
            return(new_l_k.clone().to(self.dev))
        elif self.dG_is_param:
            return(torch.log(self.params_k))
        else:
            return l_k.clone().to(self.dev)

    def get_log_copy_prod_vector(self):
        """
          get the vector storing product of copies for each reactant in each reaction.
        Returns: Tensor
            A tensor with shape (rxn_count * 2)
        """
        r_filter = -1 * self.M.T.clone()        #Invert signs of reactants amd products.
        r_filter[r_filter == 0] = -1            #Also changing molecules not involved in reactions to -1. After this, only reactants in each rxn are positive.

        #Old code
        # c_temp_mat = torch.mul(r_filter, self.copies_vec)
        # l_c_temp_mat = torch.log(c_temp_mat)
        # l_c_temp_mat[c_temp_mat < 0] = 0      #Make zero for non-reactants with non-zero copy number -> Flag 1
        # c_mask = r_filter + self.copies_vec
        # l_c_temp_mat[c_mask == -1] = 0  # 0 = log(1)  #Make zero for non-reactants with zero copy number -> Flag 2
        # l_c_prod_vec = torch.sum(l_c_temp_mat, dim=1)  # compute log products

        #New code
        #Use a non_reactant mask
        nonreactant_mask = r_filter<0       #Combines condition of Flag1 and Flag2. Basically just selecting all non_reactants w.r.t to each reaction
        c_temp_mat = torch.pow(self.copies_vec,r_filter)        #Different from previous where torch.mul was used. The previous only works for stoich=1, since X^1=X*1. But in mass action kinetics, conc. is raised to the power
        l_c_temp_mat = torch.log(c_temp_mat)                #Same as above
        l_c_temp_mat[nonreactant_mask]=0                    #Setting all conc. values of non-reactants to zero before taking the sum. Matrix dim - No. of rxn x No. of species
        l_c_prod_vec = torch.sum(l_c_temp_mat, dim=1)       #Summing for each row to get prod of conc. of reactants for each reaction

        return l_c_prod_vec

    def update_reaction_net(self, rn, scalar_modifier: int = 1):
        for n in rn.network.nodes:
            rn.network.nodes[n]['copies'] = self.copies_vec[n].item()
            for r_set in rn.get_reactant_sets(n):
                r_tup = tuple(r_set)
                reaction_id = rn.network.get_edge_data(r_tup[0], n)['uid']
                for r in r_tup:
                    k = self.compute_log_constants(self.kon, self.rxn_score_vec, scalar_modifier)
                    k = torch.exp(k)
                    rn.network.edges[(r, n)]['k_on'] = k[reaction_id].item()
                    rn.network.edges[(r, n)]['k_off'] = k[reaction_id + int(k.shape[0] / 2)].item()
        return rn

    def get_max_edge(self,n):
        """
        Calculates the max rate (k_on) for a given node
        To find out the maximum flow path to the final complex starting from the current node.

        Can also calculate the total rate of consumption of a node by summing up all rates.
        Can tell which component is used quickly.
        """
        try:
            edges = self.reaction_network.network.out_edges(n)
            #Loop over all edges
            #Get attributes
            kon_max = -1
            next_node = -1

            kon_sum = 0
            total_flux_outedges = 0
            total_flux_inedges = 0
            if len(edges)==0:
                return(False)

            for edge in edges:
                data = self.reaction_network.network.get_edge_data(edge[0],edge[1])
                #print(data)
                #Get uid
                uid = data['uid']

                #Get updated kon
                temp_kon = self.kon[uid]
                kon_sum+=temp_kon

                if temp_kon > kon_max:
                    kon_max = temp_kon
                    next_node=edge[1]

            return(kon_max,next_node,kon_sum)
        except Exception as err:
            raise(err)

    def get_node_flux(self,n):
        node_map = {}
        for node in self.reaction_network.network.nodes():
            node_map[RN.gtostr(self.reaction_network.network.nodes[node]['struct'])] = node
        total_flux_outedges = 0
        total_flux_inedges = 0
        #Go over all the out edges
        edges_out = self.reaction_network.network.out_edges(n)
        if len(edges_out)>0:

            for edge in edges_out:
                data = self.reaction_network.network.get_edge_data(edge[0],edge[1])
                #print(data)
                #Get uid
                uid = data['uid']

                #Get updated kon
                temp_kon = self.kon[uid]

                #Calculate k_off also
                std_c = Tensor([1e6])
                l_kon = torch.log(temp_kon)
                l_koff = (self.rxn_score_vec[uid]) + l_kon + torch.log(std_c)
                koff = torch.exp(l_koff)

                #Getting conc. of reactants and products
                #Get product
                prod = RN.gtostr(self.reaction_network.network.nodes[edge[1]]['struct'])
                #Get other reactant
                react = "".join(sorted(list(set(prod) - set(RN.gtostr(self.reaction_network.network.nodes[edge[0]]['struct']) ))))

                #Net flux from this edge = Generation - consumption
                edge_flux = koff*self.copies_vec[edge[1]] - temp_kon*(self.copies_vec[edge[0]])*(self.copies_vec[node_map[react]])
                #edge_flux = koff*vec_rn.copies_vec[edge[1]]

                # print("Reaction: ", RN.gtostr(rn.network.nodes[edge[0]]['struct']), "+",react," -> ",prod)
                # print("Net flux: ",edge_flux)
                # print("kon : ",temp_kon)
                # print("koff: ",koff)
                # print("Reaction data OUTWARD: ")
                # print(data)

                total_flux_outedges+=edge_flux

        #Now go over all the in edges
        edges_in = self.reaction_network.network.in_edges(n)
        react_list = []
        if len(edges_in) > 0:
            for edge in edges_in:
                if edge[0] in react_list:
                    continue
                data = self.reaction_network.network.get_edge_data(edge[0],edge[1])
                uid = data['uid']


                #Get generation rates; which would be kon
                temp_kon = self.kon[uid]

                #Get consumption rates; which is k_off
                std_c = Tensor([1e6])
                l_kon = torch.log(temp_kon)
                l_koff = (self.rxn_score_vec[uid]) + l_kon + torch.log(std_c)
                koff = torch.exp(l_koff)

                #Get conc. of reactants and products
                prod = RN.gtostr(self.reaction_network.network.nodes[edge[1]]['struct'])
                #Get other reactant
                react = "".join(sorted(list(set(prod) - set(RN.gtostr(self.reaction_network.network.nodes[edge[0]]['struct']) ))))
                react_list.append(node_map[react])
                #Net flux from this edge = Generation - consumption
                edge_flux_in = temp_kon*(self.copies_vec[edge[0]])*(self.copies_vec[node_map[react]])- koff*self.copies_vec[edge[1]]
                #edge_flux_in = koff*vec_rn.copies_vec[edge[1]]



                # print("Reaction: ", prod ," -> ",RN.gtostr(self.reaction_network.network.nodes[edge[0]]['struct']), "+",react)
                # print("Net flux: ",edge_flux_in)
                # print("kon : ",temp_kon)
                # print("koff: ",koff)
                # print("Raction data INWARD: ")
                # print(data)

                total_flux_inedges+=edge_flux_in
        net_node_flux = total_flux_outedges + total_flux_inedges

        return(net_node_flux)

    def get_reaction_flux(self):
        react_flux = torch.zeros(1,2*self.reaction_network._rxn_count)
        uid_dict = {}
        react_dict = {}
        for n in self.reaction_network.network.nodes():
            #print(n)
            #print(rn.network.nodes()[n])
            for r_set in self.reaction_network.get_reactant_sets(n):
                r_tup = tuple(list(r_set)+[n])
                data = self.reaction_network.network.get_edge_data(r_tup[0], n)
                reaction_id = data['uid']
                react_dict[r_tup]=reaction_id
            for k,v in self.reaction_network.network[n].items():
                uid = v['uid']
                r1 = set(RN.gtostr(self.reaction_network.network.nodes[n]['struct']))
                p = set(RN.gtostr(self.reaction_network.network.nodes[k]['struct']))
                r2 = p-r1
                reactants = ("".join(r1),"".join(r2))
                uid_dict[(n,k)] = uid


        #Get flux in each reaction
        #The react_dict is of the form {(reactant,reactant,product):uid}
        #Easy access to species concentrations from uid to calc flux
        for species,uid in react_dict.items():
            #Get generation rates; which would be kon
            temp_kon = self.kon[uid]
            #Get consumption rates; which is k_off
            std_c = Tensor([1e6])
            l_kon = torch.log(temp_kon)
            l_koff = (self.rxn_score_vec[uid]) + l_kon + torch.log(std_c)
            koff = torch.exp(l_koff)

            #Net flux from this edge = Generation - consumption
            assoc_flux = temp_kon*(self.copies_vec[species[0]])*(self.copies_vec[species[1]])
            reverse_flux = koff*self.copies_vec[species[2]]
            react_flux[0,uid]=assoc_flux
            react_flux[0,self.reaction_network._rxn_count + uid] = reverse_flux

        return(react_flux)


    def calculate_total_flux(self):
        net_flux = {}
        for n in self.reaction_network.network.nodes():
            n_str = RN.gtostr(self.reaction_network.network.nodes[n]['struct'])
            node_flux = self.get_node_flux(n)
            net_flux[RN.gtostr(self.reaction_network.network.nodes[n]['struct'])] = node_flux

        return(net_flux)
