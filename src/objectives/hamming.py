import numpy as np
from itertools import combinations


class Hamming:

    def __init__(self):
        self.hamming = {}
        self.sequences = {}

    def __getitem__(self, key):
        return self.hamming[key]

    def assign_hamming_distances(self, genomes):
        distances = {}
        for genome_A, genome_B in combinations(genomes, 2):
            dist = Hamming.hamming_distance(self.sequences[genome_A[0]], self.sequences[genome_B[0]])
            try:
                distances[genome_A[0]] += dist
            except KeyError:
                distances[genome_A[0]] = dist

            try:
                distances[genome_B[0]] += dist
            except KeyError:
                distances[genome_B[0]] = dist

        # divide by pop_size - 1
        self.hamming = {key: value / (len(genomes) - 1) for key, value in distances.items()}

    @staticmethod
    def hamming_distance(theta_1, theta_2):
        hamming_distance = np.count_nonzero(theta_1 != theta_2)
        return hamming_distance
