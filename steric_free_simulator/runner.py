from reaction_network import ReactionNetwork
from energy_space_explorer import EnergyExplorer
from simulator import Simulator

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

import pickle
import sys


if __name__ == '__main__':
    input_file = sys.argv[1]
    runtime_s = int(sys.argv[2])

    if sys.argv[3] is not None:
        # if a pickle is provided load it, otherwise run the energy explorer on the network
        if '.pkl' in sys.argv[3] or '.pickle' in sys.argv[3]:
            with open(sys.argv[3], 'rb') as f:
                rn = pickle.load(f)
        else:
            rn = ReactionNetwork(input_file)
            subunit_dir = sys.argv[3]
            En = EnergyExplorer(rn, subunit_dir)
            En.explore_network()
            En.intialize_activations()
    else:
        rn = ReactionNetwork(input_file)

    with open('./saved_nets/ap2_en_net.pkl', 'wb') as f:
        pickle.dump(rn, f)

    sim = Simulator(rn, runtime_s)
    steps = sim.steps
    print("found best dt to be " + str(sim.dt))
    dt = sim.dt
    sim.simulate()

    #nx.draw(sim.rn.network)
    t = np.arange(steps)*dt
    df = pd.DataFrame(sim.rn.observables)
    df.to_csv('./4loop.csv')
    for key in sim.rn.observables.keys():
        plt.scatter(t, sim.rn.observables[key][1],
                    cmap='plasma',
                    s=.1,
                    label=sim.rn.observables[key][0])
    plt.legend(loc='best')
    plt.show()
    print('done')
