from simulation.environments.tartarus.tartarus_util import generate_configurations
from simulation.environments.tartarus.tartarus_environment import TartarusEnvironment

config, agent = generate_configurations(6,6, sample=10)

for c, a in zip(config, agent):
    print(c)
    env = TartarusEnvironment(c, fixed_agent_pos=a[0], fixed_agent_dir=[1])
    env.reset()
    print(env.encode_tartarus_state())
    print()

"""
import gymnasium as gym
from pprint import pprint

env = gym.make("BipedalWalker-v3")
env.reset()

pprint(tuple(env.hull.position))
pprint(tuple(env.hull.worldCenter))

pprint(vars(env))

a = []
for i in range(200):
    if i % 4 == 0:
        print(i)
        a.append(i)

print(len(a))


print("START")
from util import load_checkpoints
func = load_checkpoints
print("FINISH")

old_min = -1.0
old_max = 1.0

new_min = 0.0
new_max = 1.0

old_value = 0.0

old_range = (old_max - old_min)
new_range = (new_max - new_min)

new_value = (((old_value - old_min) * new_range) / old_range) + new_min
print(old_value)
print(new_value)
new_value = ((old_value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
print(new_value)

low = np.array([-math.pi, -5.0, -5.0, -5.0, -math.pi, -5.0, -math.pi,
                -5.0, -0.0, -math.pi, -5.0, -math.pi, -5.0, -0.0] + [-1.0] * 10).astype(np.float32)
high = np.array([math.pi, 5.0, 5.0, 5.0, math.pi, 5.0, math.pi,
                 5.0, 5.0, math.pi, 5.0, math.pi, 5.0, 5.0] + [1.0] * 10).astype(np.float32)

print(low)
print(high)
print(zip(low, high))
min_max_range = zip(low, high)
for i, (old_min, old_max) in enumerate(min_max_range):
    print(f"{i} = [{old_min} - {old_max}]")



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
