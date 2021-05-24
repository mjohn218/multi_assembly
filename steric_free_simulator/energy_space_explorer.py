from typing import Tuple, Union

from pyrosetta import rosetta
from pyrosetta import pose_from_pdb
from pyrosetta import dump_pdb
from pyrosetta import init as rosetta_init
import pyrosetta.rosetta.protocols.relax as relax
from rosetta.protocols.relax import *
from pyrosetta import get_fa_scorefxn
from pyrosetta.toolbox import cleanATOM
from pyrosetta import Pose

from reaction_network import ReactionNetwork
from reaction_network import gtostr

import os
import shutil
import sys
from pathos.multiprocessing import ProcessingPool as Pool


def strip_pdb_ext(pdb_file: str) -> str:
    name = pdb_file.split('.')[0]
    name = ''.join(sorted(name))
    return name


class EnergyExplorer:
    """
    A class to estimate the relative reaction free energies for a set of pairwise reactions.
    Will output the energy estimates to the command line. It is up to the user to add them
    to an input .pwr file if desired. Keep in mind that the scale of the energy score outputs
    is arbitrary, and will likely need to be changed.

    """
    def __init__(self, net: ReactionNetwork, subunit_dir: str):
        rosetta_init()
        self.net = net
        self.monomer_pdb = [os.path.abspath(os.path.join(subunit_dir, 'monomers', subunit))
                            for subunit in os.listdir(os.path.join(subunit_dir, 'monomers'))]
        self.sub_dir = os.path.abspath(subunit_dir)
        self.written = dict()
        self.scorefxn = get_fa_scorefxn()  # note this score function is only guaranteed consistent in this instance!
        self.relaxer = relax.FastRelax()
        self.relaxer.set_scorefxn(self.scorefxn)
        self.relaxer.max_iter(200)
        self._preprocess()
        self._load_existing()
        subunit_files = set(
            [os.path.join(self.sub_dir, file) if '.clean' in file and '.relaxed' not in file else ''
             for file in os.listdir(subunit_dir)])
        subunit_files.remove('')
        subunit_files = list(subunit_files)

        try:
            with Pool(len(subunit_files)) as p:
                results = p.map(self._pose_from_pdb, subunit_files)
        except Exception:
            print("Lacking serializable rosetta build. Parallel processing disabled. "
                  "Recommend compiling from source with --serialization flag \n "
                  "Continuing to process sequentially",
                  sys.stderr)
            results = [self._pose_from_pdb(sunit) for sunit in subunit_files]

        for i, res in enumerate(results):
            self.written[strip_pdb_ext(os.path.split(subunit_files[i])[1])] = res
        for i in range(self.net.num_monomers):
            self.net.network.nodes[i]['score'] = self.written[gtostr(self.net.network.nodes[i]['struct'])][1]

    def _load_existing(self):
        for file in os.listdir(self.sub_dir):
            if 'relaxed' in file:
                pdb_pose = pose_from_pdb(os.path.join(self.sub_dir, file))
                self.written[strip_pdb_ext(os.path.split(file)[1])] = (pdb_pose, self.scorefxn(pdb_pose))

    def _preprocess(self):
        for mon_file in self.monomer_pdb:
            fname = os.path.split(mon_file)[1]
            if strip_pdb_ext(fname) not in self.written:
                shutil.copyfile(mon_file, os.path.join(self.sub_dir, fname))
                cleanATOM(os.path.join(self.sub_dir, fname))

    def _pose_from_pdb(self, pdb_file: str) -> Tuple[Pose, float]:
        name = strip_pdb_ext(os.path.split(pdb_file)[1])
        name = ''.join(sorted(name))
        if name in self.written:
            return self.written[name]
        new_pose = pose_from_pdb(pdb_file)
        self.relaxer.apply(new_pose)
        new_pose.dump_pdb(os.path.join(os.path.split(pdb_file)[0], name + '.relaxed.clean.pdb'))
        score = get_fa_scorefxn()(new_pose)
        return new_pose, score

    def score_reaction(self, reactant_ids: Union[list, set]):
        names = [''.join(sorted(gtostr(self.net.network.nodes[rid]['struct']))) for rid in reactant_ids]
        new_pdb_name = ''.join(sorted(''.join(sorted(names))))
        prebound_score = sum([self.written[n][1] for n in names])
        if new_pdb_name in self.written:
            return self.written[new_pdb_name][1], prebound_score
        new_pdb_str = ''
        for name in names:
            reactant_file = os.path.join(self.sub_dir, name + '.clean.pdb')
            with open(reactant_file, 'r') as f:
                new_pdb_str += f.read()
        new_pdb_path = os.path.join(self.sub_dir, new_pdb_name + '.clean.pdb')
        with open(new_pdb_path, 'w') as f:
            f.write(new_pdb_str)
        new_pose, bound_score = self._pose_from_pdb(new_pdb_path)
        self.written[new_pdb_name] = (new_pose, bound_score)
        return bound_score, prebound_score

    def explore_network(self):
        for node_id in self.net.network.nodes():
            structure = self.net.network.nodes[node_id]['struct']
            if len(structure) == 2:
                # is a pairwise reaction !
                name = gtostr(structure)
                name = ''.join(sorted(name))
                for predecessors in self.net.get_reactant_sets(node_id):
                    r_score, pr_score = self.score_reaction(predecessors)
                    self.net.network.nodes[node_id]['score'] = pr_score  # add score attribute
                    print('for pairwise product ', name, ' score is ', pr_score - r_score)

