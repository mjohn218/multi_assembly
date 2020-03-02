from pyrosetta.toolbox import cleanATOM
from pyrosetta import *
from pyrosetta.rosetta.core.scoring import *


def bind(unit1: tuple, unit2: tuple) -> tuple[dict[str: tuple[float, str]], dict]:
    og_score = unit1[1].score() + unit2[1].score()
    fin1 = open("subunits/" + unit1[0] + ".clean.pdb", "r").read()
    fin2 = open("subunits/" + unit2[0] + ".clean.pdb", "r").read()
    newpdb = fin1 + fin2
    new_pose = pose_from_pdb(newpdb)
    pyrosetta.rosetta.protocols.relax.relax_pose(pose=new_pose, scorefxn=get_fa_scorefxn(), tag='')
    new_score = new_pose.score()
    diff = og_score - new_score
    return ({unit1[0]: (diff, unit1[0] + '_' + unit2[0]),
            unit2[0]: (diff, unit1[0] + '_' + unit2[0])},
            {unit1[0] + '_' + unit2[0]: new_pose})




