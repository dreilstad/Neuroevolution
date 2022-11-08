import numpy as np
from scipy.spatial.distance import cdist

class Novelty:

    @staticmethod
    def novelty(distances, neighbours):
        idx = np.argsort(distances)
        mean_k_dist = np.mean(distances[idx[1:neighbours + 1]])
        return mean_k_dist

    @staticmethod
    def evaluate_performances(bd_set, reference_set, distance_metric="euclidean",
                              k_neighbours=15, pool=None):

        distance_matrix = Novelty.calculate_distances(bd_set, reference_set, distance_metric)
        if pool is not None:
            novelties = [pool.apply(Novelty.novelty, args=(distance, k_neighbours,)) for distance in distance_matrix]
        else:
            novelties = [Novelty.novelty(distance, k_neighbours) for distance in distance_matrix]

        return novelties

    @staticmethod
    def calculate_distances(bd_set, reference_set, distance_metric):

        distance_matrix = None
        if distance_metric == "euclidean":
            distance_matrix = cdist(bd_set, reference_set, metric=distance_metric)
        else:
            raise ValueError(f"Specified distance {distance_metric} not available")

        return distance_matrix

