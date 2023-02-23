
import numpy as np

a = np.zeros((256, 10))
b = np.zeros((256, 16))

print(a.shape)
print(b.shape)

idx = np.round(np.linspace(0, a.shape[0]-1, a.shape[0]//4)).astype(int)
print(idx)

a = a[idx, :]
b = b[idx, :]

print(a.shape)
print(b.shape)



"""
from simulation.environments.retina.retina_environment import HardRetinaEnvironment, VisualObject, Side

env = HardRetinaEnvironment()
print(env)

test_patterns = [(VisualObject("o o o\no o o\no o o", side=Side.LEFT, size=3),
                  VisualObject("o o o\no o o\no o o", side=Side.RIGHT, size=3)),
                 (VisualObject(". . .\no o o\no o o", side=Side.LEFT, size=3),
                  VisualObject("o o o\no o o\n. . .", side=Side.RIGHT, size=3)),
                 (VisualObject("o o o\n. . .\n. . .", side=Side.LEFT, size=3),
                  VisualObject(". . .\n. . .\no o o", side=Side.RIGHT, size=3)),
                 (VisualObject("o . .\no . .\no . .", side=Side.LEFT, size=3),
                  VisualObject(". . o\n. . o\n. . o", side=Side.RIGHT, size=3))]

print("TEST PATTERNS:")
for left, right in test_patterns:
    print(left)
    print(right)
    print("")

print(len(env.visual_objects))
print(env.N)

from random import sample


a = {str(i):i for i in range(10)}
print(a)

def _generate_combinations(keys, samples=100):
    generated_combinations = [tuple()] * (len(keys) * samples)
    for i, key in enumerate(keys):
        sampled_keys = sample(keys[:i] + keys[i + 1:], samples)
        for j, s_key in enumerate(sampled_keys):
            generated_combinations[i * samples + j] = (key, s_key)

    return generated_combinations

keys = list(a.keys())
print(_generate_combinations(keys, samples=3))
"""
