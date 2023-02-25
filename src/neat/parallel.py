"""
Runs evaluation functions in parallel subprocesses
in order to evaluate multiple genomes at once.
"""
import neat
from multiprocessing import Pool
from multiprocessing.pool import ApplyResult


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

    def evaluate_genomes(self, genomes, config, generation):
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

    def evaluate_genomes(self, genomes, config, generation):

        nodes_of_interest = config.genome_config.input_keys
        if not self.simulator.use_input_nodes_in_mod_div:
            nodes_of_interest = config.genome_config.output_keys

        jobs = []
        for genome_id, genome in genomes:
            neural_network = neat.nn.FeedForwardNetwork.create(genome, config)
            genome.fitness = neat.nsga2.NSGA2Fitness(*[0.0]*self.simulator.num_objectives)
            jobs.append(self.pool.apply_async(self.eval_function, [neural_network]))

        pool.close()
        map(ApplyResult.wait, jobs)
        simulation_outputs = [result.get() for result in jobs]

        for simulation_output, (genome_id, genome) in zip(simulation_outputs, genomes):
            simulation_output["nodes"] = config.genome_config.input_keys + list(genome.nodes)
            simulation_output["edges"] = list(genome.connections)
            simulation_output["nodes_of_interest"] = nodes_of_interest
            self.simulator.assign_output(genome_id, simulation_output, generation)

        self.simulator.assign_fitness(genomes)
