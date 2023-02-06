import neat
import numpy as np

from objectives.hamming import Hamming
from objectives.novelty import Novelty
from objectives.cka import CKA
from objectives.modularity import Modularity, ModularityDiversity
from objectives.cosine_similarity import CosineSimilarity


class Simulator:

    def __init__(self, objectives, domain):
        self.objectives = objectives
        self.num_objectives = len(self.objectives)
        self.domain = domain
        self.use_input_nodes_in_mod_div = True

        # performance objective
        self.performance = {} if "performance" in self.objectives else None

        # structural diversity objective
        self.Q = {} if "modularity" in self.objectives else None
        self.mod_div = ModularityDiversity() if "mod_div" in self.objectives else None

        # behavioral diversity objective
        self.hamming = Hamming() if "hamming" in self.objectives else None
        self.novelty = Novelty(self.domain) if "beh_div" in self.objectives else None

        # representational diversity objective
        self.cos_sim = CosineSimilarity() if "cos_sim" in self.objectives else None

        self.CKA = None
        if "linear_cka" in self.objectives:
            self.CKA = CKA(linear_kernel=True)
        elif "rbf_cka" in self.objectives:
            self.CKA = CKA(linear_kernel=False)

        # initialized when simulating Mazerobot
        self.history = None

    def evaluate_genomes(self, genomes, config, generation):

        nodes_of_interest = config.genome_config.input_keys
        if not self.use_input_nodes_in_mod_div:
            nodes_of_interest = config.genome_config.output_keys

        for genome_id, genome in genomes:
            neural_network = neat.nn.FeedForwardNetwork.create(genome, config)
            genome.fitness = neat.nsga2.NSGA2Fitness(*[0.0]*self.num_objectives)

            simulation_output = self.simulate(neural_network)
            simulation_output["nodes"] = config.genome_config.input_keys + list(genome.nodes)
            simulation_output["edges"] = list(genome.connections)
            simulation_output["nodes_of_interest"] = nodes_of_interest
            self.assign_output(genome_id, simulation_output, generation)

        self.assign_fitness(genomes)

    def simulate(self, neural_network):
        raise NotImplementedError

    def assign_fitness(self, genomes):
        # do calculation of objectives which need genome-to-population comparison
        if self.hamming is not None:
            self.hamming.calculate_hamming_distances(genomes)
        if self.novelty is not None:
            self.novelty.calculate_novelty()
        if self.mod_div is not None:
            self.mod_div.calculate_modular_diversity(genomes)
        if self.cos_sim is not None:
            self.cos_sim.calculate_cosine_similarities(genomes)
        if self.CKA is not None:
            self.CKA.calculate_CKA_similarities_parallel(genomes)
            #self.CKA.calculate_CKA_similarities_opt_parallel(genomes, samples=len(genomes)//10)

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
                elif objective == "modularity":
                    fitnesses[i] = self.Q[genome_id]
                elif objective == "mod_div":
                    fitnesses[i] = self.mod_div[genome_id]
                elif objective == "cos_sim":
                    fitnesses[i] = 1.0 - self.cos_sim[genome_id]
                elif objective == "linear_cka" or objective == "rbf_cka":
                    fitnesses[i] = 1.0 - self.CKA[genome_id]

            genome.fitness.add(*fitnesses)

        self.Q = {}

    def assign_output(self, genome_id, simulation_output, generation):
        if self.performance is not None:
            self.performance[genome_id] = simulation_output["performance"]

        if self.hamming is not None:
            self.hamming.sequences[genome_id] = simulation_output["hamming"]

        if self.novelty is not None:
            self.novelty.add(genome_id, simulation_output["novelty"])

        if self.CKA is not None:
            self.CKA.activations[genome_id] = np.array(simulation_output["activations"])

        if self.cos_sim is not None:
            self.cos_sim.activations[genome_id] = np.array(simulation_output["activations"]).ravel()

        if self.Q is not None:
            self.Q[genome_id] = Modularity.calculate_modularity(simulation_output["nodes"],
                                                                simulation_output["edges"])

        if self.mod_div is not None:
            self.mod_div.add(genome_id, simulation_output["nodes"], simulation_output["edges"],
                             simulation_output["nodes_of_interest"])

        if self.domain == "mazerobot-medium" or self.domain == "mazerobot-hard":
            record = simulation_output["agent_record"]
            record.agent_id = genome_id
            record.generation = generation
            self.history.add_record(record)

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
