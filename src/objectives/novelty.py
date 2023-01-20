import numpy as np
from simulation.environments.maze.agent import AgentRecordStore, AgentRecord


class Novelty:

    def __init__(self, k=15, initial_threshold=10.0, archive_seed_size=1):
        self.archive = []
        self.behaviors = {}
        self.novelty = {}

        self.k = k
        self.k_archive = 1
        self.threshold = initial_threshold
        self.archive_seed_size = archive_seed_size

        self.num_added_to_archive = 0
        self.evals_since_archive_addition = 0

    def __getitem__(self, key):
        return self.novelty[key]

    def calculate_novelty(self):

        # reset novelty dict
        self.novelty = {}

        # calculate distances against archive behaviors
        for genome_id, behavior in self.behaviors.items():
            distances_archive = self.euclidean_distance_to_archive(behavior)
            distances_pop = self.euclidean_distance_to_current_pop(behavior)
            distances = np.concatenate((distances_archive, distances_pop))

            # sort list of distances to get the k nearest neighbors
            nearest_neighbors = sorted(distances)[:self.k]
            print(nearest_neighbors)

            # calculate novelty and assign score
            novelty = np.mean(nearest_neighbors)
            self.novelty[genome_id] = novelty

        print(f"archive size = {len(self.archive)}")
        print(f"number added to archive = {self.num_added_to_archive}")

        self._update_archive_settings()

        print(f"number of evals since last archive addition = {self.evals_since_archive_addition}")
        print(f"threshold = {self.threshold}")

        # reset behavior dict
        self.behaviors = {}

    def add(self, genome_id, behavior):

        # save behavior
        self.behaviors[genome_id] = behavior

        # get novelty score and its nearest neighbor
        novelty, nn_ind = self.novelty_score(behavior)

        # add to archive if novelty greater than threshold or ,
        # otherwise check if novelty is larger than its nearest neighbor
        if novelty > self.threshold or len(self.archive) < self.archive_seed_size:
            self.archive.append((behavior, novelty))
            self.num_added_to_archive += 1
            self.evals_since_archive_addition = 0
        else:
            self.evals_since_archive_addition += 1
            #if nn_ind is not None and novelty > self.archive[nn_ind][1]:
            #    self.archive[nn_ind] = (behavior, novelty)

    def novelty_score(self, behavior, default_novelty=0.1):
        if len(self.archive) == 0:
            return default_novelty, None

        # get distance to archive behaviors
        distances = self.euclidean_distance_to_archive(behavior)

        # sort list of distances ot get the k nearest neighbors
        k = min(len(self.archive), self.k_archive)
        nearest_neighbors = sorted(zip(distances, list(range(len(self.archive)))))[:k]
        nearest_neighbors_distances, nearest_neighbors_indices = map(list, zip(*nearest_neighbors))

        # calculate novelty
        novelty = np.mean(nearest_neighbors_distances)

        # get index of nearest neighbor
        nearest_neighbor_ind = None
        if len(nearest_neighbors_indices) > 0:
            nearest_neighbor_ind = nearest_neighbors_indices[0]

        return novelty, nearest_neighbor_ind

    def write_archive_to_file(self, filename):

        history = AgentRecordStore()

        for behavior, _ in self.archive:
            record = AgentRecord()
            record.x = behavior[0]
            record.y = behavior[1]
            history.add_record(record)

        history.dump(filename)

    def euclidean_distance_to_archive(self, behavior):
        distances = np.zeros(len(self.archive))
        for i, (other_behavior, _) in enumerate(self.archive):
            distances[i] = self.euclidean_distance(behavior, other_behavior)

        distances = np.power(distances, 0.5)
        return distances

    def euclidean_distance_to_current_pop(self, behavior):
        distances = np.zeros(len(self.behaviors))
        for i, (_, other_behavior) in enumerate(self.behaviors.items()):
            distances[i] = self.euclidean_distance(behavior, other_behavior)

        distances = np.power(distances, 0.5)
        return distances

    @staticmethod
    def euclidean_distance(vec, other):
        distance = 0
        for i in range(len(vec)):
            distance += pow(vec[i] - other[i], 2.0)
        return distance

    def _update_archive_settings(self):
        # threshold is lowered if more than 1000 evaluations occured since last archive addition,
        # limit from "Novelty-based Multiobjectivization" paper
        if self.evals_since_archive_addition >= 1000:
            self.threshold = self.threshold * 0.95

        # threshold is increased if more than 10 behaviors were added to the archive during the generation
        if self.num_added_to_archive > 5:
            self.threshold = self.threshold * 1.05

        # reset counter
        self.num_added_to_archive = 0

