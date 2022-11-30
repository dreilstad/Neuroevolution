"""
Runs evaluation functions in parallel subprocesses
in order to evaluate multiple genomes at once.
"""
import neat
import numpy as np
from multiprocessing import Pool


class ParallelEvaluator(object):
    def __init__(self, num_workers, eval_function, timeout=None):
        """
        eval_function should take one argument, a tuple of
        (genome object, config object), and return
        a single float (the genome's fitness).
        """
        self.num_workers = num_workers
        self.eval_function = eval_function
        self.timeout = timeout
        self.pool = Pool(num_workers)

    def __del__(self):
        self.pool.close() # should this be terminate?
        self.pool.join()

    def evaluate(self, genomes, config, generation):
        jobs = []
        for ignored_genome_id, genome in genomes:
            jobs.append(self.pool.apply_async(self.eval_function, (genome, config)))

        # assign the fitness back to each genome
        for job, (ignored_genome_id, genome) in zip(jobs, genomes):
            genome.fitness = job.get(timeout=self.timeout)
            
            
class MultiObjectiveParallelEvaluator(ParallelEvaluator):
    
    def __init__(self, num_workers, simulator, timeout=None):
        super(MultiObjectiveParallelEvaluator, self).__init__(num_workers, simulator.simulate, timeout)
        self.simulator = simulator

    def evaluate(self, genomes, config, generation):
        jobs = []
        for genome_id, genome in genomes:
            neural_network = neat.nn.FeedForwardNetwork.create(genome, config)
            genome.fitness = neat.nsga2.NSGA2Fitness(*[0.0]*self.simulator.num_objectives)
            jobs.append(self.pool.apply_async(self.eval_function, (genome_id,
                                                                   genome,
                                                                   neural_network,
                                                                   generation)))

        for job, (genome_id, genome) in zip(jobs, genomes):
            simulation_output = job.get(timeout=self.timeout)
            self._assign_output(genome_id, simulation_output)

        self.simulator.assign_fitness(genomes)

    def _assign_output(self, genome_id, simulation_output):
        if self.simulator.performance is not None:
            self.simulator.performance[genome_id] = simulation_output[0]

        if self.simulator.hamming is not None:
            self.simulator.hamming.sequences[genome_id] = simulation_output[1]

        if self.simulator.novelty is not None:
            self.simulator.novelty.behaviors[genome_id] = simulation_output[2]

        if self.simulator.CKA is not None:
            self.simulator.CKA.activations[genome_id] = np.array(simulation_output[3])

        if self.simulator.Q is not None:
            pass



