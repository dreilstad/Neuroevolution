import neat
import numpy as np

from objectives.hamming import Hamming
from objectives.novelty_v2 import Novelty
from objectives.cka import CKA
from objectives.modularity import Modularity, ModularDiversity


class Simulator:

    def __init__(self, objectives):
        self.objectives = objectives
        self.num_objectives = len(self.objectives)

        # performance objective
        self.performance = {} if "performance" in self.objectives else None

        # structural diversity objective
        self.Q = Modularity() if "modularity" in self.objectives else None
        self.ModDiv = ModularDiversity() if "mod_div" in self.objectives else None

        # behavioral diversity objective
        self.hamming = Hamming() if "hamming" in self.objectives else None
        self.novelty = Novelty() if "beh_div" in self.objectives else None

        # representational diversity objective
        self.CKA = None
        if "linear_cka" in self.objectives:
            self.CKA = CKA(linear_kernel=True)
        elif "rbf_cka" in self.objectives:
            self.CKA = CKA(linear_kernel=False)

    def evaluate_genomes(self, genomes, config):
        for genome_id, genome in genomes:
            neural_network = neat.nn.FeedForwardNetwork.create(genome, config)
            genome.fitness = neat.nsga2.NSGA2Fitness(*[0.0]*self.num_objectives)

            simulation_output = self.simulate(neural_network)
            self._assign_output(genome_id, simulation_output)

        self.assign_fitness(genomes)

    def simulate(self, neural_network):
        raise NotImplementedError

    def assign_fitness(self, genomes):
        # do calculation of objectives which need genome-to-population comparison
        if self.hamming is not None:
            self.hamming.assign_hamming_distances(genomes)
        if self.novelty is not None:
            self.novelty.calculate_novelty(genomes)
        if self.Q is not None:
            pass
        if self.CKA is not None:
            self.CKA.calculate_CKA_similarities_parallel(genomes)

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
                    fitnesses[i] = self.Q[genome_id]
                elif objective == "linear_cka" or objective == "rbf_cka":
                    fitnesses[i] = 1.0 / self.CKA[genome_id]

            genome.fitness.add(*fitnesses)

    def _assign_output(self, genome_id, simulation_output):
        if self.performance is not None:
            self.performance[genome_id] = simulation_output[0]

        if self.hamming is not None:
            self.hamming.sequences[genome_id] = simulation_output[1]

        if self.novelty is not None:
            self.novelty.behaviors[genome_id] = simulation_output[2]

        if self.CKA is not None:
            self.CKA.activations[genome_id] = np.array(simulation_output[3])

        if self.Q is not None:
            pass

    def _get_novelty_characteristic(self, neural_network):
        raise NotImplementedError

    @staticmethod
    def _binarize_sequence(sequence):
        if sequence is None:
            return None

        theta = np.zeros(len(sequence), dtype=int)

        # binarize values and add to theta
        for i, value in enumerate(sequence):
            if value >= 0.5:
                theta[i] = 1

        return theta
