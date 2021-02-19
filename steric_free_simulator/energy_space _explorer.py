from pyrosetta import rosetta
from pyrosetta import pose_from_pdb
from pyrosetta import init as rosetta_init
import pyrosetta.rosetta.protocols.relax as relax
from pyrosetta import get_fa_scorefxn
from reaction_network import ReactionNetwork

class EnergyExplorer:

    def __init__(self, net: ReactionNetwork):
        self.pose_net = net
