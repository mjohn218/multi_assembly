from typing import Tuple

from steric_free_simulator import ReactionNetwork

import networkx as nx

import torch
from torch import DoubleTensor as Tensor
from torch import rand
from torch import nn


class VectorizedRxnNet:
    """
    Provides an wrapper for a rxn network that represents the core information needed for
    simulation as torch tensors. Acts as a base object for optimization simulations.
    """

    def __init__(self, rn: ReactionNetwork, dev):
        rn.reset()
        rn.intialize_activations()
        self.dev = torch.device('cpu')
        self.M, EA, self.rxn_score_vec, self.copies_vec = self.generate_vectorized_representation(rn)
        self.initial_EA = Tensor(EA).clone().detach()
        self.initial_copies = self.copies_vec.clone().detach()
        self.EA = nn.Parameter(EA, requires_grad=True)
        self.observables = rn.observables
        self.is_energy_set = rn.is_energy_set
        self.num_monomers = rn.num_monomers
        self.to(dev)

    def reset(self):
        self.copies_vec = self.initial_copies.clone()
        for key in self.observables:
            self.observables[key] = (self.observables[key][0], [])


    def get_params(self):
        yield self.EA

    def to(self, dev):
        self.M = self.M.to(dev)
        self.EA = nn.Parameter(self.EA.data.clone().detach().to(dev), requires_grad=True)
        self.copies_vec = self.copies_vec.to(dev)
        self.initial_copies = self.initial_copies.to(dev)
        self.rxn_score_vec = self.rxn_score_vec.to(dev)
        self.dev = dev
        return self

    @staticmethod
    def generate_vectorized_representation(rn: ReactionNetwork) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
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
        EA = torch.zeros([rn._rxn_count], requires_grad=True).double()
        rxn_score_vec = torch.zeros([rn._rxn_count]).double()
        copies_vec = torch.zeros([num_states]).double()

        for n in rn.network.nodes():
            copies_vec[n] = rn.network.nodes[n]['copies']
            for r_set in rn.get_reactant_sets(n):
                r_tup = tuple(r_set)
                data = rn.network.get_edge_data(r_tup[0], n)
                reaction_id = data['uid']
                try:
                    EA[reaction_id] = data['activation_energy']
                except Exception:
                    EA[reaction_id] = 5.
                rxn_score_vec[reaction_id] = data['rxn_score']
                # forward
                M[n, reaction_id] = 1.
                for r in r_tup:
                    M[r, reaction_id] = -1.
        # generate the reverse map explicitly
        M[:, rn._rxn_count:] = -1 * M[:, :rn._rxn_count]
        return M, EA, rxn_score_vec, copies_vec

    def get_copy_prod_vector(self, volume: float):
        """
          get the vector storing product of copies for each reactant in each reaction.
        Returns: Tensor
            A tensor with shape (rxn_count * 2)
        """
        r_filter = -1 * self.M.T.clone()
        r_filter[r_filter == 0] = -1

        c_temp_mat = torch.mul(r_filter, self.copies_vec)

        c_temp_mat = c_temp_mat / volume  # get copies per liter

        c_temp_mat[c_temp_mat < 0] = 1  # don't want to zero reactions that don't use all species! This doesn't matter for grad since we don't care about the comp graph branch anyway.

        c_mask = r_filter + self.copies_vec

        c_temp_mat[c_mask == -1] = 1

        c_prod_vec = torch.prod(c_temp_mat, dim=1)  # compute products
        return c_prod_vec

    def update_reaction_net(self, rn, k =  None):
        for n in rn.network.nodes:
            rn.network.nodes[n]['copies'] = self.copies_vec[n].item()
            for r_set in rn.get_reactant_sets(n):
                r_tup = tuple(r_set)
                reaction_id = rn.network.get_edge_data(r_tup[0], n)['uid']
                for r in r_tup:
                    if k is not None:
                        rn.network.edges[(r, n)]['k_on'] = k[reaction_id].item()
                        rn.network.edges[(r, n)]['k_off'] = k[reaction_id + int(k.shape[0] / 2)].item()
                    rn.network.edges[(r, n)]['activation_energy'] = self.EA[reaction_id].item()
        return rn