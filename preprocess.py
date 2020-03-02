from pyrosetta.toolbox import cleanATOM
from pyrosetta import *
from pyrosetta.rosetta.core.scoring import *
import os
import pickle as pk


def preprocess(path: str, config: dict) -> dict:
    poses = {}
    pyrosetta.init()
    objects = os.listdir('./obj/')
    if 'base_poses.pkl' in objects:
        poses = pk.load('./obj/base_poses.pkl')
    else:
        for pdb in config.keys():
            cleanATOM(path + pdb+'.pdb')
            pose = pose_from_pdb(path + pdb + '.clean.pdb')
            pyrosetta.rosetta.protocols.relax.relax_pose(pose=pose, scorefxn=get_fa_scorefxn(), tag='')
            poses[pdb] = pose
    return poses

