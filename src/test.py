
from multiprocessing import Pool, cpu_count
from multiprocessing.pool import ApplyResult
import time


class A:

    def __init__(self, idx):
        self.idx = idx


"""
def get_idx(idx):

    time.sleep(3)
    return idx

def main():
    a = [A(i) for i in range(10)]

    with Pool(cpu_count() // 4) as pool:
        jobs = []
        for b in a:
            jobs.append(pool.apply_async(get_idx, [b.idx]))

        pool.close()
        map(ApplyResult.wait, jobs)
        outputs = [result.get() for result in jobs]

        print([c.idx for c in a])
        print(outputs)


if __name__ == "__main__":
    main()



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
