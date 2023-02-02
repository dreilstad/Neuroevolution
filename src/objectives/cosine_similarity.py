import numpy as np
from itertools import combinations


class CosineSimilarity:

    def __init__(self):
        self.similarity = {}
        self.activations = {}

    def __getitem__(self, key):
        return self.similarity[key]

    def calculate_cosine_similarities(self, genomes):
        distances = {genome_id: 0.0 for genome_id, _ in genomes}

        for (genome_A_id, _), (genome_B_id, _) in combinations(genomes, 2):
            dist = self.cosine_similarity(self.activations[genome_A_id],
                                          self.activations[genome_B_id])
            distances[genome_A_id] += dist
            distances[genome_B_id] += dist

        num_other_genomes = (len(genomes) - 1)
        self.similarity = {key: value / num_other_genomes for key, value in distances.items()}
        self.activations = {}

    @staticmethod
    def cosine_similarity(a, b):
        max_len = min(a.shape[0], b.shape[0])
        a = a[:max_len]
        b = b[:max_len]
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))