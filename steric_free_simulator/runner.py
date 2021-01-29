from reaction_network import ReactionNetwork
from simulator import Simulator
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

rn = ReactionNetwork('./base_rules.bngl')
sim = Simulator(rn, 20000, .0001, obs=[0, 1, 2, 3, 16])
sim.simulate()
#nx.draw(sim.rn.network)
t = np.arange(20000)*.001
plt.scatter(t, sim.obs_key[0], c='green', s=.1)
plt.scatter(t, sim.obs_key[1], c='blue', s=.1)
plt.scatter(t, sim.obs_key[2], c='red', s=.1)
plt.scatter(t, sim.obs_key[3], c='purple', s=.1)
plt.scatter(t, sim.obs_key[16], c='black', s=1)
plt.show()
print('done')
