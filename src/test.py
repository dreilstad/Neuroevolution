import numpy as np
from simulation.tartarus_simulation import TartarusSimulator
from multiprocessing import Pool

objectives = ["performance"]
nn = [1,2,3]
pool = Pool(2)
jobs = []
sim = TartarusSimulator(objectives)
for i in range(2):
    jobs.append(pool.apply_async(sim.simulate, (nn)))

# assign the fitness back to each genome
for job in jobs:
    job.get(timeout=None)


