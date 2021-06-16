from typing import Tuple

from steric_free_simulator import ReactionNetwork

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

    def __init__(self, rn: ReactionNetwork, assoc_is_param=True, copies_is_param=False, dev='cpu'):
        """

        :param rn: The reaction network template
        :param assoc_is_param: whether the association constants should be treated as parameters for optimization
        :param copies_is_param: whether the initial copy numbers should be treated as parameters for optimization
        :param dev: the device to use for torch tensors
        """
        rn.reset()
        self.dev = torch.device(dev)
        self._avo = Tensor([6.02214e23])  # copies / mol
        self._R = Tensor([8.314])  # J / mol * K
        self._T = Tensor([273.15])  # K
        self.M, self.kon, self.rxn_score_vec, self.copies_vec = self.generate_vectorized_representation(rn)
        self.initial_params = Tensor(self.kon).clone().detach()
        self.initial_copies = self.copies_vec.clone().detach()
        self.assoc_is_param = assoc_is_param
        self.copies_is_param = copies_is_param
        if assoc_is_param:
            self.kon = nn.Parameter(self.kon, requires_grad=True)
        if copies_is_param:
            self.c_params = nn.Parameter(self.initial_copies[:rn.num_monomers], requires_grad=True)
        self.observables = rn.observables
        self.is_energy_set = rn.is_energy_set
        self.num_monomers = rn.num_monomers
        self.reaction_ids = []
        self.to(dev)

    def reset(self, reset_params=False):
        self.copies_vec = self.initial_copies.clone()
        if reset_params:
            self.kon = nn.Parameter(self.initial_params.clone(), requires_grad=True)
        for key in self.observables:
            self.observables[key] = (self.observables[key][0], [])

    def get_params(self):
        if self.assoc_is_param and self.copies_is_param:
            return [self.kon, self.c_params]
        elif self.copies_is_param:
            return [self.c_params]
        elif self.assoc_is_param:
            return [self.kon]

    def to(self, dev):
        self.M = self.M.to(dev)
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
            copies_vec[n] = rn.network.nodes[n]['copies']
            for r_set in rn.get_reactant_sets(n):
                r_tup = tuple(r_set)
                data = rn.network.get_edge_data(r_tup[0], n)
                reaction_id = data['uid']
                try:
                    kon[reaction_id] = data['k_on']
                except Exception:
                    kon[reaction_id] = 1.
                rxn_score_vec[reaction_id] = data['rxn_score']
                # forward
                M[n, reaction_id] = 1.
                for r in r_tup:
                    M[r, reaction_id] = -1.
        # generate the reverse map explicitly
        M[:, rn._rxn_count:] = -1 * M[:, :rn._rxn_count]
        return M, kon, rxn_score_vec, copies_vec

    def compute_log_constants(self, kon: Tensor, dGrxn: Tensor, scalar_modifier) -> Tensor:
        """
        Returns log(k) for each reaction concatenated with log(koff) for each reaction
        """
        # above conversions cancel
        std_c = Tensor([1.])  # units mols / L
        l_kon = torch.log(kon)  # mol-1 s-1
        l_koff = (dGrxn * scalar_modifier / (self._R * self._T)) + l_kon + torch.log(std_c)
        l_k = torch.cat([l_kon, l_koff], dim=0)
        return l_k.clone().to(self.dev)

    def get_log_copy_prod_vector(self):
        """
          get the vector storing product of copies for each reactant in each reaction.
        Returns: Tensor
            A tensor with shape (rxn_count * 2)
        """
        r_filter = -1 * self.M.T.clone()
        r_filter[r_filter == 0] = -1
        c_temp_mat = torch.mul(r_filter, self.copies_vec)
        l_c_temp_mat = torch.log(c_temp_mat)
        l_c_temp_mat[c_temp_mat < 0] = 0
        c_mask = r_filter + self.copies_vec
        l_c_temp_mat[c_mask == -1] = 0  # 0 = log(1)
        l_c_prod_vec = torch.sum(l_c_temp_mat, dim=1)  # compute log products
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
