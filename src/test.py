from random import sample
from itertools import product

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
