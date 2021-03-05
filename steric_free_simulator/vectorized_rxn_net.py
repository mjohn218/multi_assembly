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

    def __init__(self, rn: ReactionNetwork):
        self.M, k_vec, self.rxn_score_vec, self.copies_vec = self.generate_vectorized_representation(rn)
        self.k_vec = nn.Parameter(k_vec, requires_grad=True)

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
        M = torch.zeros((num_states, rn._rxn_count * 2), dtype=torch.double)
        k_vec = torch.zeros([rn._rxn_count], dtype=torch.double)
        rxn_score_vec = torch.zeros([rn._rxn_count], dtype=torch.double)
        copies_vec = torch.zeros([num_states], dtype=torch.double)

        for n in rn.network.nodes():
            copies_vec[n] = rn.network.nodes[n]['copies']
            for r_set in rn.get_reactant_sets(n):
                r_tup = tuple(r_set)
                data = rn.network.get_edge_data(r_tup[0], n)
                reaction_id = data['uid']
                k_vec[reaction_id] = data['k_on']
                k_vec[reaction_id + rn._rxn_count] = data['k_off']
                rxn_score_vec[reaction_id] = data['rxn_score']
                # forward
                M[n, reaction_id] = 1.
                for r in r_tup:
                    M[r, reaction_id] = -1.
        # generate the reverse map explicitly
        M[:, rn._rxn_count:] = -1 * M[:, :rn._rxn_count]
        return Tensor(M), Tensor(k_vec), Tensor(rxn_score_vec), Tensor(copies_vec)

    def update_reaction_net(self, rn):
        raise NotImplementedError
