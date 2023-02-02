

import numpy as np

a = np.eye(4)
print(a)
b = np.eye(4)
d = a + b
print(d)
c = np.eye(4)
d = d + c
print(d)

e = np.zeros(c.shape)
print(e)

with np.errstate(divide="ignore"):
    print(np.minimum(-np.log(0.0), 5))

"""
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
