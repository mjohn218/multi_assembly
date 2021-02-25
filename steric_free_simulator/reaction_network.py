import re
import sys
import networkx as nx
import random

import torch
from torch import DoubleTensor as Tensor
from torch import rand
from torch import nn

LOOP_COOP_DEFAULT = 1


def _equal(n1, n2):
    nm = nx.algorithms.isomorphism.categorical_node_match("label", None)
    int_n1 = nx.convert_node_labels_to_integers(n1, label_attribute="label")
    int_n2 = nx.convert_node_labels_to_integers(n2, label_attribute="label")
    return nx.is_isomorphic(int_n1, int_n2, node_match=nm)


def gtostr(g: nx.DiGraph) -> str:
    stout = ""
    for n in g.nodes():
        stout += str(n)
    # make invarient
    stout = ''.join(sorted(stout))
    return stout


class ReactionNetwork:
    def __init__(self, bngl_path: str, one_step: bool):
        self.network: nx.DiGraph() = nx.DiGraph()
        self.allowed_edges = {}
        self._node_count = 0
        self._rxn_count = 0
        self.num_monomers = 0
        self.is_one_step = one_step
        # default observables are monomers and final complex
        self.observables = dict()
        # resolve graph
        self._initial_copies = {}
        self.parse_bngl(open(bngl_path, 'r'))
        self.resolve_tree(one_step)
        self.parameters = {}  # gradient params
        self.is_energy_set = False

        # pre-exponential factor hyperparameter
        self.A = None

    def get_params(self):
        """
        returns an iterator over optimization parameters
        :return:
        """
        for key in self.parameters:
            yield self.parameters[key]

    def get_reactant_sets(self, node_id: int):
        """
        Returns a iterator over coreactants for a given node (i.e. product)
        :param node_id: the node to know reactant sets for
        :return:
        """
        all_predecessors = set(self.network.in_edges(node_id))
        while len(all_predecessors) > 0:
            found = False
            reactant = all_predecessors.pop()
            predecessors = {reactant[0]}
            reactant_data = self.network[reactant[0]][reactant[1]]
            poss_coreactant = None
            # find complete reactant sets
            for poss_coreactant in all_predecessors:
                poss_coreactant_data = self.network[poss_coreactant[0]][poss_coreactant[1]]
                if reactant_data['uid'] == poss_coreactant_data['uid']:
                    found = True
                    break
            if found:
                all_predecessors.remove(poss_coreactant)
                predecessors.add(poss_coreactant[0])
            yield predecessors


    def parse_param(self, line):
        # Reserved Params
        # loop_coop: the loop cooperativity factor (f) = exp(-dG_coop / kb*T)
        #           range of 0 to 1, with binding being strongly forward at 1
        items = line.split(None, 1)
        items[1] = eval(items[1])
        return items

    def parse_species(self, line, params):
        items = line.split()
        sp_info = re.split('\\)|,|\\(', items[0])
        try:
            init_pop = int(items[1])
        except ValueError:
            init_pop = int(params[items[1]])
        state_net = nx.Graph()
        state_net.add_node(sp_info[0])
        self.network.add_node(self._node_count, struct=state_net, copies=Tensor([float(init_pop)]))
        self._initial_copies[self._node_count] = Tensor([float(init_pop)])
        self._node_count += 1

    def parse_rule(self, line, params):
        items = re.split(r' |, ', line)
        r_info = re.split('\\(.\\)+.|\\(.\\)<->', items[0])
        try:
            k_on = None # float(items[1])
        except ValueError:
            k_on = None # float(params[items[1]])
        if len(items) > 2:
            try:
                k_off = None # float(items[2])
            except ValueError:
                k_off = None # float(params[items[2]])
        else:
            k_off = None # 0
        self.allowed_edges[tuple(sorted([r_info[0], r_info[1]]))] = [k_on, k_off, LOOP_COOP_DEFAULT]

    def parse_bngl(self, f):
        parameters = dict()
        cur_block = ''
        for line in f:
            line = line.strip()
            if len(line) > 0 and line[0] != '#':
                if "begin parameters" in line:
                    cur_block = 'param'
                elif "begin species" in line:
                    cur_block = 'species'
                elif "begin rules" in line:
                    cur_block = 'rules'
                elif "begin observables" in line:
                    cur_block = 'observables'
                elif "end" in line:
                    cur_block = ' '
                else:
                    if cur_block == 'param':
                        items = self.parse_param(line)
                        parameters[items[0]] = items[1]
                    elif cur_block == 'species':
                        self.parse_species(line, parameters)
                    elif cur_block == 'rules':
                        self.parse_rule(line, parameters)

        # attach loop cooperativity param to rules (python 3.6+ only due to dict ordering changes)
        if "loop_coop" in parameters:
            if len(parameters['loop_coop']) != len(self.allowed_edges):
                raise ValueError('num loop_coop must equal to num allowed_edges')
            keys = list(self.allowed_edges.keys())
            for i, lcf in enumerate(parameters['loop_coop']):
                if lcf > 1 or lcf < 0:
                    raise ValueError('loop cooperativity factor must be between 0 and 1')
                self.allowed_edges[keys[i]][2] = lcf
        self.num_monomers = self._node_count

    def reset(self):
        for key in self._initial_copies:
            self.network.nodes[key]['copies'] = self._initial_copies[key]
        self.observables = {}
        # add default observables
        for i in range(self.num_monomers):
            self.observables[i] = (gtostr(self.network.nodes[i]['struct']), [])
        fin_dex = len(self.network.nodes) - 1
        self.observables[fin_dex] = (gtostr(self.network.nodes[fin_dex]['struct']), [])

    def intialize_activations(self, A: float = 1, mode="middle"):
        """
        function to set and initialize activation energy parameters for reaction network.
        :return:
        """
        if not self.is_energy_set:
            raise ValueError("The network free energies must be calculated for activation params to be used")
        self.A = A  # set pre-exponential hyperparam
        # reaction rates may not match activation energies before sim start.
        for node in self.network.nodes:
            for reactant_set in self.get_reactant_sets(node):
                # same Tensor used in all three places (i.e. ptr)
                if mode == 'uniform':
                    activation_energy = nn.Parameter(rand(1, dtype=torch.double) * Tensor([10.]), requires_grad=True)
                elif mode == 'middle':
                    activation_energy = nn.Parameter(Tensor([5.]), requires_grad=True)
                self.parameters[tuple(list(reactant_set) + [node])] = activation_energy
                for source in reactant_set:
                    self.network.edges[(source, node)]['activation_energy'] = activation_energy

    def _add_graph_state(self, connected_item: nx.Graph, source_1: int, source_2: int = None, template_edge_id=None):
        if type(source_1) is not int:
            source_1 = int(source_1[0])
        if source_2 is not None and type(source_2) is not int:
            source_2 = int(source_2[0])
        node_exists = [x for x in self.network.nodes(data=True) if
                       _equal(x[1]['struct'], connected_item)]
        if len(node_exists) == 0:
            self.network.add_node(self._node_count, struct=connected_item, copies=Tensor([0.]))
            self._initial_copies[self._node_count] = Tensor([0.])
            new_node = self._node_count
            self._node_count += 1
        elif len(node_exists) > 1:
            raise Exception("Duplicate nodes in reaction Network")
        else:
            new_node = node_exists[0][0]

        if template_edge_id is not None:
            self.network.add_edge(source_1, new_node,
                                  k_on=self.allowed_edges[template_edge_id][0],
                                  k_off=self.allowed_edges[template_edge_id][1],
                                  lcf=self.allowed_edges[template_edge_id][2],
                                  uid=self._rxn_count)
            if source_2 is not None:
                self.network.add_edge(source_2, new_node,
                                      k_on=self.allowed_edges[template_edge_id][0],
                                      k_off=self.allowed_edges[template_edge_id][1],
                                      lcf=self.allowed_edges[template_edge_id][2],
                                      uid=self._rxn_count)
        else:
            self.network.add_edge(source_1, new_node,
                                  k_on=1,
                                  k_off=.1,
                                  lcf=1,
                                  uid=self._rxn_count)
            if source_2 is not None:
                self.network.add_edge(source_2, new_node,
                                      k_on=1,
                                      k_off=.1,
                                      lcf=1,
                                      uid=self._rxn_count)
        self._rxn_count += 1
        if len(node_exists) == 0:
            return (new_node, self.network.nodes[new_node])
        else:
            return None

    def match_maker(self, n1, n2=None, one_step=False) -> list:
        """
        determines if a valid edge can be added between two network states, and preforms
        addition if possible
        :param one_step: whether to do one step binding
        :param n1: node in network
        :param n2: node in network
        :return:
        """
        nodes_added = []
        orig = n1[1]['struct']
        if n2 is not None:
            nextn = n2[1]['struct']
            item = nx.compose(orig, nextn)
        else:
            item = orig
        connected_item = item.copy()
        for poss_edge in list(self.allowed_edges.keys()):
            if False not in [item.has_node(n) for n in poss_edge] and \
                    (n2 is None or
                     (True in [orig.has_node(n) for n in poss_edge] and
                      True in [nextn.has_node(n) for n in poss_edge]))\
                    and not item.has_edge(poss_edge[0], poss_edge[1]):
                # adds edge to new struct, should happen only once per match step if not one step
                if not one_step:
                    connected_item = item.copy()
                connected_item.add_edge(poss_edge[0], poss_edge[1])
            else:
                continue
            if not one_step:
                new_node = self._add_graph_state(connected_item, n1, source_2=n2, template_edge_id=poss_edge)
                if new_node is not None:
                    nodes_added.append(new_node)
        # resolving cooperative network
        if one_step:
            new_node = self._add_graph_state(connected_item, n1, source_2=n2)
            if new_node is not None:
                nodes_added.append(new_node)

        return nodes_added

    def is_hindered(self, n1, n2):
        node_set1 = set(n1[1]['struct'].nodes())
        node_set2 = set(n2[1]['struct'].nodes())
        return len(node_set1 - node_set2) < len(node_set1)

    def resolve_tree(self, is_one_step):
        new_nodes = list(self.network.nodes(data=True))
        while len(new_nodes) > 0:
            node = new_nodes.pop(0)
            for anode in list(self.network.nodes(data=True)):
                if not self.is_hindered(node, anode):
                    new_nodes += self.match_maker(node, anode, is_one_step)
            # must also try internal bonds
            new_nodes += self.match_maker(node)

        # add default observables
        for i in range(self.num_monomers):
            self.observables[i] = (gtostr(self.network.nodes[i]['struct']), [])
        fin_dex = len(self.network.nodes) - 1
        self.observables[fin_dex] = (gtostr(self.network.nodes[fin_dex]['struct']), [])



if __name__ == '__main__':
    bngls_path = sys.argv[1]  # path to bngl
    dt = float(sys.argv[2])  # time step in seconds
    iter = int(sys.argv[3])  # number of time steps to simulate
    m = ReactionNetwork(sys.argv[1])
    print('done')
