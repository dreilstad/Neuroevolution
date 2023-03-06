import numpy as np
from itertools import combinations

class CCA:

    def __init__(self):
        self.activations = {}
        self.correlation = {}

    def __getitem__(self, key):
        return self.correlation[key]

    def calculate_CCA_correlations(self, genomes):
        correlation = {genome_id: 0.0 for genome_id, _ in genomes}

        for (genome_A_id, genome_A), (genome_B_id, genome_B) in combinations(genomes, 2):
            X = self.activations[genome_A_id]
            Y = self.activations[genome_B_id]

            # find min N samples, limit to match first dim of X and Y
            min_n = min(X.shape[0], Y.shape[0])
            X = X[:min_n, :]
            Y = Y[:min_n, :]

            cca_value = self.cca(X, Y)

            correlation[genome_A_id] += cca_value
            correlation[genome_B_id] += cca_value

        num_other_genomes = float(len(genomes) - 1)
        self.correlation = {key: value / num_other_genomes for key, value in correlation.items()}
        self.activations = {}

    @staticmethod
    def cca(features_x, features_y):
        """Compute the mean squared CCA correlation (R^2_{CCA}).

        Args:
            features_x: a (num_examples x num_features) matrix of features
            features_y: a (num_examples x num_features) matrix of features
        Returns:
            The mean squared CCA correlations between X and Y
        """

        qx, _ = np.linalg.qr(features_x)
        qy, _ = np.linalg.qr(features_y)
        return np.linalg.norm(qx.T.dot(qy)) ** 2 / min(features_x.shape[1], features_y.shape[1])