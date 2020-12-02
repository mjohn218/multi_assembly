from reaction_network import ReactionNetwork
from simulator import Simulator
import networkx as nx
from matplotlib import pyplot as plt

rn = ReactionNetwork('./base_rules.bngl')
sim = Simulator(rn, 10000, .00001)
sim.simulate()
nx.draw(sim.rn.network)
plt.show()
print('done')
