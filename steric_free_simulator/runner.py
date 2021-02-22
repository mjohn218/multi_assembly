from reaction_network import ReactionNetwork
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
    rn = ReactionNetwork(input_file)

    sim = Simulator(rn, runtime_s)
    print("found best dt to be " + str(sim.dt))
    dt = sim.dt
    sim.simulate()

with open("./saved_nets/ap2_sim_uneven_assoc.pkl", 'wb') as f:
    pickle.dump(sim, f)
#

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
