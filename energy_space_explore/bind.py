from pyrosetta import pose_from_pdb
from pyrosetta import init as rosetta_init
import pyrosetta.rosetta.protocols.relax as relax
from pyrosetta import get_fa_scorefxn


def bind(unit1: tuple, unit2: tuple): # returns tuple[dict[str: tuple[float, str]], dict]:
    scorefxn = get_fa_scorefxn()
    rosetta_init()
    og_score = scorefxn(unit1[1]) + scorefxn(unit2[1])
    fin1 = open("subunits/" + unit1[0] + ".clean.pdb", "r").read()
    fin2 = open("subunits/" + unit2[0] + ".clean.pdb", "r").read()
    new_unit = (unit1[0]+unit2[0]).split('_')
    new_unit.remove('')
    new_unit.sort()
    new_str = '_'.join(new_unit)+'_'
    to_write = open("subunits/" + new_str + ".clean.pdb", "w")
    to_write.write(fin1 + fin2)
    new_pose = pose_from_pdb("subunits/" + new_str + ".clean.pdb")
    #relax.relax_pose(pose=new_pose, scorefxn=get_fa_scorefxn(), tag='')
    new_score = scorefxn(new_pose)
    diff = og_score - new_score
    return ({unit1[0]: (diff, new_str, unit2[0]),
            unit2[0]: (diff, new_str, unit1[0])},
            {new_str: new_pose})




