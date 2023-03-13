import numpy as np
import time
from neat.nn.feed_forward import FeedForwardNetwork, FFNN
from util import load_checkpoints

checkpoint_path = "/Users/didrik/Documents/Master/Neuroevolution/src/checkpoints/mazerobot-medium/performance-rep_div_cka/000"
pop = load_checkpoints(checkpoint_path)[0]

network = FeedForwardNetwork.create(pop.best_genome, pop.config)
ffnn = FFNN(pop.best_genome, pop.config)

inputs = []
for i in range(100000):
    inputs.append(np.random.rand(10))

start = time.time()
for in_values in inputs:
    old_output, old_act = network.activate(in_values)
end = time.time()

print(f"Runtime (old version): {end - start} s")

start = time.time()
for in_values in inputs:
    new_output, new_act = ffnn.activate(in_values)
end = time.time()
print(f"Runtime (new version): {end - start} s")






"""

from simulation.environments.retina.retina_environment import HardRetinaEnvironment, VisualObject, Side

env = HardRetinaEnvironment()
#print(env)

print(env.visual_objects[14])
print(env.visual_objects[15])
print()

print(env.visual_objects[32])
print(env.visual_objects[33])
print()
"""
