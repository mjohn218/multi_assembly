from pyrosetta.toolbox import cleanATOM
from pyrosetta import init as rosetta_init
from pyrosetta import pose_from_pdb
from pyrosetta.rosetta.core.scoring import *
import pyrosetta.rosetta.protocols.relax as relax
import os
import pickle as pk


def preprocess(path: str, config: dict) -> dict:
    poses = {}
    rosetta_init()
    objects = os.listdir('./obj/')
    if 'base_poses.pkl' in objects:
        file = open('./obj/base_poses.pkl', 'rb')
        poses = pk.load(file)
    else:
        file = open('./obj/base_poses.pkl', 'wb')
        for pdb in config.keys():
            cleanATOM(path + pdb+'.pdb')
            pose = pose_from_pdb(path + pdb + '.clean.pdb')
            relax.relax_pose(pose=pose, scorefxn=get_fa_scorefxn(), tag='')
            poses[pdb] = pose
        pk.dump(poses, file)
    return poses

