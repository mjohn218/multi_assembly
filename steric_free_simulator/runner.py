from reaction_network import ReactionNetwork
from energy_space_explorer import EnergyExplorer
from simulator import Simulator
from vec_sim import VecSim
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

import pickle
import sys


if __name__ == '__main__':
    input_file = sys.argv[1]
    runtime_s = float(sys.argv[2])

    try:
        # if a pickle is provided load it, otherwise run the energy explorer on the network
        if '.pkl' in sys.argv[3] or '.pickle' in sys.argv[3]:
            with open(sys.argv[3], 'rb') as f:
                rn = pickle.load(f)
        else:
            rn = ReactionNetwork(input_file, one_step=True)
            subunit_dir = sys.argv[3]
            En = EnergyExplorer(rn, subunit_dir)
            En.explore_network()
    except IndexError:
        rn = ReactionNetwork(input_file, one_step=True)

    with open('./saved_nets/ap2_one_step_en.pkl', 'wb') as f:
        pickle.dump(rn, f)

    rn.reset()
    rn.intialize_activations()
    sim = VecSim(rn, runtime_s)
    steps = sim.steps
    print("found best dt to be " + str(sim.dt))
    dt = sim.dt
    sim.optimize(10, lr=1)

    with open('./saved_nets/ap2_one_step_en_optim.pkl', 'wb') as f:
        pickle.dump(rn, f)
    #  percent_yield = sim.simulate()

    # print("end yield is " + str(percent_yield.item() * 100)[:4] + "%")
    #nx.draw(sim.rn.network)
    sim.plot_observables()
    print('done')
