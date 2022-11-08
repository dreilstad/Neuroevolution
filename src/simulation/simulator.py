import neat
import time
from threading import Thread
from objectives.hamming import Hamming
from objectives.novelty_v2 import Novelty


class Simulator:

    def __init__(self, objectives):
        self.objectives = objectives
        self.num_objectives = len(self.objectives)

        # performance objective
        self.performance = None

        # structural diversity objective
        self.Q = None # TODO: structural or behavioral?

        # behavioral diversity objective
        self.hamming = None
        self.novelty = None

        if "performance" in self.objectives:
            self.performance = {}

        if "hamming" in self.objectives:
            self.hamming = Hamming()

        if "beh_div" in self.objectives:
            self.novelty_archive = Novelty()
            self.novelty = {}

    def evaluate_genomes(self, genomes, config, generation):
        for genome_id, genome in genomes:
            neural_network = neat.nn.FeedForwardNetwork.create(genome, config)
            if self.num_objectives > 1:
                genome.fitness = neat.nsga2.NSGA2Fitness(*[0.0]*self.num_objectives)

            task_performance = self.simulate(genome_id, genome, neural_network, generation)
            self.performance[genome_id] = task_performance

        self.assign_fitness(genomes, generation)

    def simulate(self, genome_id, genome, neural_network, generation):
        raise NotImplementedError

    def assign_fitness(self, genomes, generation):
        # do calculation of objectives which need genome-to-population comparison
        if self.hamming is not None:
            self.hamming.assign_hamming_distances(genomes)
        if self.novelty is not None:
            _, _, self.novelty = self.novelty_archive.calculate_novelty(genomes, generation)
        if self.Q is not None:
            pass

        # assign fitness values
        for genome_id, genome in genomes:
            fitnesses = [0.0] * self.num_objectives
            for i, objective in enumerate(self.objectives):
                if objective == "performance":
                    fitnesses[i] = self.performance[genome_id]
                elif objective == "hamming":
                    fitnesses[i] = self.hamming[genome_id]
                elif objective == "beh_div":
                    fitnesses[i] = self.novelty[genome_id]
                elif objective == "Q":
                    pass

            genome.fitness.add(*fitnesses)

