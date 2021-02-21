from typing import Tuple

from pyrosetta import rosetta
from pyrosetta import pose_from_pdb
from pyrosetta import init as rosetta_init
import pyrosetta.rosetta.protocols.relax as relax
from pyrosetta import get_fa_scorefxn
from pyrosetta.toolbox import cleanATOM
from pyrosetta import Pose

from reaction_network import ReactionNetwork
from reaction_network import gtostr

import os
import pickle as pk
import shutil
import networkx


def strip_pdb_ext(pdb_file: str) -> str:
    name = pdb_file[:-4]
    if '.clean' in name:
        name = name[:-6]
    return name


class EnergyExplorer:

    def __init__(self, net: ReactionNetwork, subunit_dir: str):
        rosetta_init()
        self.net = net
        self.monomer_pdb = [os.path.join(subunit_dir, subunit)
                            for subunit in os.listdir(os.path.join(subunit_dir, 'monomers'))]
        self.sub_dir = subunit_dir
        self.written = dict()
        self.scorefxn = get_fa_scorefxn  # note this score function is only guaranteed consistent in this instance!
        self._preprocess()
        for file in os.listdir(subunit_dir):
            if ".clean.pdb" in file:
                m_pose = os.path.join(subunit_dir, file)
                self.written[strip_pdb_ext(file)] = self._pose_from_pdb(m_pose)
        for i in range(self.net.num_monomers):
            self.net.network.nodes[i]['score'] = self.written[gtostr(self.net.network.nodes[i]['struct'])][1]

    def _preprocess(self):
        for mon_file in self.monomer_pdb:
            fname = os.path.split(mon_file)[1]
            if strip_pdb_ext(fname) not in self.written:
                shutil.copyfile(mon_file, os.path.join(self.sub_dir, fname))
                cleanATOM(os.path.join(self.sub_dir, fname))

    def _pose_from_pdb(self, pdb_file: str) -> Tuple[Pose, float]:
        new_pose = pose_from_pdb(pdb_file)
        relax.relax_pose(pose=new_pose, scorefxn=self.scorefxn, tag='')
        score = self.scorefxn(new_pose)
        return new_pose, score

    def score_reaction(self, reactant_ids: list):
        names = [''.join(sorted(gtostr(self.net.network.nodes[rid]['struct']))) for rid in reactant_ids]
        prebound_score = sum([self.written[n][1] for n in names])
        new_pdb_str = ''
        for name in names:
            reactant_file = os.path.join(self.sub_dir, name + '.clean.pdb')
            with open(reactant_file, 'r') as f:
                new_pdb_str += f.read()
        new_pdb_name = ''.join(sorted(names))
        new_pdb_path = os.path.join(self.sub_dir, new_pdb_name + '.clean.pdb')
        with open(new_pdb_path, 'w') as f:
            f.write(new_pdb_str)
        new_pose, bound_score = self._pose_from_pdb(new_pdb_path)
        self.written[new_pdb_name] = (new_pose, bound_score)
        return bound_score, prebound_score

    def explore_network(self):
        for node_id in self.net.network.nodes():
            name = gtostr(self.net.network.nodes[node_id]['struct'])
            name = ''.join(sorted(name))
            predecessors = list(self.net.network.predecessors(node_id))
            if name not in self.written:
                # this pattern has not yet been processed.
                r_score, p_score = self.score_reaction(predecessors)
                self.net.network.nodes[node_id]['first'] = True # attribute to tell whether score is inherited from previous node with this pattern
            else:
                # we will write the score to every node, and tell whether is energetically meaningful with "first" attribute.
                r_score = p_score = self.written[name][1]
                self.net.network.nodes[node_id]['first'] = False
            self.net.network.nodes[node_id]['score'] = p_score  # add score attribute
            for n in predecessors:
                self.net.network.edges[(n, node_id)]['rxn_score'] = p_score - r_score




