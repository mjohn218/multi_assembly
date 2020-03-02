import os
import sys
from preprocess import preprocess

subunit_dir = sys.argv[1]
config_dir = sys.argv[2]

subunits = ''
mol_config = ''
mol_struct = {}

try:
    subunits = os.listdir(subunit_dir)
except IOError:
    print("Bad monomer directory", sys.stderr)
    exit()

try:
    mol_config = open(config_dir, 'r')
except IOError:
    print("Config file not found", sys.stderr)
    exit()

for line in mol_config:
    if line[0] != '#':
        # load molecule structure to dict
        try:
            cols = line.split('\t')
            subunit = cols[0].replace(' ', '')
            if subunit+'.pdb' in subunits:
                mol_struct[cols[0]] = cols[1].replace(' ', '').replace('\n', '').split(',')
            else:
                print("Bad config format", sys.stderr)
                exit()
        except IndexError:
            print("Bad config format", sys.stderr)
            exit()

poses = preprocess(subunit_dir, mol_struct)
print(poses)

