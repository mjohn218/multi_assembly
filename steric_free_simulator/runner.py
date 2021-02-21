from reaction_network import ReactionNetwork
from simulator import Simulator
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import pickle

# with open("./saved_nets/arp23_net_final.pkl", 'rb') as f:
#     sim = pickle.load(f)

print('begin')

rn = ReactionNetwork('input_files/ap2.bngl')
# with open("./saved_nets/arp23_net.pkl", 'wb') as f:
#     pickle.dump(rn, f)
# with open("./saved_nets/arp23_net.pkl", 'rb') as f:
#     rn = pickle.load(f)
steps = 200000

sim = Simulator(rn, steps)
print("found best dt to be " + str(sim.dt))
dt = sim.dt
# sim = Simulator(rn, 10000, .001, obs=[0, 1, 2, 3, 4, 5, 9])
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
