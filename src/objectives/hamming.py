import numpy as np
from itertools import combinations


class Hamming:

    def __init__(self):
        self.hamming = {}
        self.sequences = {}

    def __getitem__(self, key):
        return self.hamming[key]

    def assign_hamming_distances(self, genomes):
        distances = {genome_id: 0.0 for genome_id, _ in genomes}
        for (genome_A_id, genome_A), (genome_B_id, genome_B) in combinations(genomes, 2):
            dist = Hamming.hamming_distance(self.sequences[genome_A_id], self.sequences[genome_B_id])
            distances[genome_A_id] += dist
            distances[genome_B_id] += dist

        # divide by pop_size - 1
        num_other_genomes = (len(genomes) - 1)
        self.hamming = {key: value / num_other_genomes for key, value in distances.items()}

    @staticmethod
    def hamming_distance(theta_1, theta_2):
        hamming_distance = np.count_nonzero(theta_1 != theta_2)
        return hamming_distance
