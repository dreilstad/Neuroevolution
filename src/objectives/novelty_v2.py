import numpy as np
from sklearn.neighbors import NearestNeighbors


class Novelty:

    def __init__(self):
        """
        Initialises list of novelty members.
        https://github.com/12mashok/neat-python
        """
        self.novel_members = {}
        self.novelty_scores = {}
        self.behaviors = {}

        self.pop_knn_neighbours = 20
        self.archive_knn_neighbours = 10
        self.threshold = 0.1

    def __getitem__(self, key):
        return self.novelty_scores[key]

    def calculate_novelty(self, genomes, generation):
        """
        Carries out two steps:

            Step 1: Calculate how novel behavior is by euclidean distance/KNN.
            Step 2: If individuals novelty is high, add to novel_members list.

        """

        # Extract behaviors of all genomes in the population
        #behaviors = {}
        #for genome_id, genome in genomes:
        #    behaviors[genome_id] = genome.behavior

        # Get all behavior values in current population and create a KNN model
        behavior_values = np.array(list(self.behaviors.values()))

        # If neighbours value is out of bounds, set it to max possible value
        if int(self.pop_knn_neighbours) > len(behavior_values):
            pop_knn_neighbours = len(behavior_values)
        else:
            pop_knn_neighbours = int(self.pop_knn_neighbours)

        # FIRST KNN MODEL: fit on the behavior values of current population
        # Obtain normalized data
        normalized_behavior_values = self.normalize_data(behavior_values)

        knn_model = NearestNeighbors(n_neighbors=pop_knn_neighbours, algorithm='ball_tree').fit(
            normalized_behavior_values)

        if len(self.novel_members) < 1:
            models = [knn_model]
        else:
            # Get behaviors of novel member archive and create KNN model
            novel_members = list(self.novel_members.values())
            novel_members_behaviors = np.array([self.behaviors[member_key] for member_key in novel_members])
            novel_members_behaviors.reshape(-1, 1)

            # If neighbours value is out of bounds, set it to max possible value
            if int(self.archive_knn_neighbours) > len(novel_members_behaviors):
                archive_knn_neighbours = len(novel_members_behaviors)
            else:
                archive_knn_neighbours = int(self.archive_knn_neighbours)

            # SECOND KNN MODEL:: Build knn model for novel member archive

            # Obtain normalized data
            normalized_novel_members_behaviors = self.normalize_data(novel_members_behaviors)
            novel_members_knn_model = NearestNeighbors(n_neighbors=archive_knn_neighbours, algorithm='ball_tree').fit(
                normalized_novel_members_behaviors)

            # Gather models
            models = [knn_model, novel_members_knn_model]

        # Novelty is assigned as the average distance to the k-nearest neighbors
        # If genome is novel, average distance will be high.
        self.calculate_population_distances(self.behaviors, genomes, models)

        # Extract fitnesses of all genomes in the population after they have been calculated in previous step
        fitnesses = {}
        for genome_id, genome in genomes:
            # store genome id as value for easy access
            fitnesses[self.novelty_scores[genome_id]] = genome_id

        # Get best genome, it's fitness, and behavior value
        best_fitness = max(list(fitnesses.keys()))
        best_fitness_genome_id = fitnesses[best_fitness]
        best_behavior = self.behaviors[best_fitness_genome_id]
        best_genome = [genome for genome in genomes if genome[0] == best_fitness_genome_id]

        # If novel member archive has less than three members add best genomes
        if len(list(self.novel_members.keys())) < 1:
            self.novel_members[generation] = best_genome[0][0]

        # If knn average of best genome is greater than threshold, add to novel member archive
        else:
            # If distance of best genome is greater than threshold, add to novel member archive
            knn_distance = self.KNN_distance(best_behavior, novel_members_knn_model)

            if knn_distance > float(self.threshold):
                self.novel_members[generation] = best_genome[0][0]

        # Return novel member archive and behavior for entire population
        #return self.novel_members, behavior_values, knn_distances


    def calculate_population_distances(self, behaviors, genomes, knn_models):
        """
        Sets genome.fitness to average distance of n nearest neighbours
        """
        # Get normalization factors
        behavior_values = list(behaviors.values())
        behavior_min = np.amin(behavior_values)
        behavior_max = np.amax(behavior_values)

        # For each genome
        for genome_id, genome in genomes:
            fitness = 0

            # for each knn model provided
            for knn_model in knn_models:
                # Get corresponding behavior and normalize prior to checking knn
                behavior = (behaviors[genome_id] - behavior_min) / (behavior_max - behavior_min)

                # Get average knn distance
                average_distance = self.KNN_distance(behavior, knn_model)

                # Add average distance to fitness. The more novel a genome, the higher its average knn distance, and the higher its fitness
                fitness += average_distance

            # Set genome novelty
            self.novelty_scores[genome_id] = fitness


    def KNN_distance(self, behavior, knn_model):
        """
        Returns average distance of a behavior in a given knn model
        """
        behavior = np.array(behavior)
        distances, indices = knn_model.kneighbors([behavior])
        average_distance = sum(distances[0]) / len(distances[0])
        return average_distance

    def normalize_data(self, data):
        """
        Normalizes data according to X_norm = (X - X_min)/(X_max-X_min)
        """

        if data.ndim == 1:
            return data

        # Transpose data
        data = data.T

        # Get shape
        shape = data.shape

        # The number of features is taken as the smaller dimension
        number_of_dimensions = min(shape)
        X = np.split(data, number_of_dimensions)

        # Normalize each feature separately
        for index, feature_data in enumerate(X):
            feature_data_min = np.amin(feature_data)
            feature_data_max = np.amax(feature_data)
            X[index] = (feature_data - feature_data_min) / (feature_data_max - feature_data_min)

        # Concatenate X back together
        X = np.concatenate(X)

        # Transpose data back
        X = X.T
        return X