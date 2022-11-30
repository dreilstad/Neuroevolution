import numpy as np
from sklearn.neighbors import NearestNeighbors


class Novelty:

    def __init__(self):
        """
        Initialises list of novelty members.
        https://github.com/12mashok/neat-python
        """
        self.novel_members = []
        self.novelty_scores = {}
        self.behaviors = {}

        self.pop_knn_neighbours = 15
        self.archive_knn_neighbours = 15
        self.threshold = 0.2

        self.evaluations_since_last_added = 0

    def __getitem__(self, key):
        return self.novelty_scores[key]

    def calculate_novelty(self, genomes):
        """
        Carries out three steps:

            Step 1: Retrieve behaviors of current population, normalize data and create models
            Step 2: Calculate how novel behavior is by euclidean distance/KNN.
            Step 3: If individuals novelty is high, add to novel_members list.

        """

        # reset novelty scores for previous generation
        self.novelty_scores = {}

        # STEP 1: GET BEHAVIORS OF CURRENT POPULATION, NORMALIZE VALUES and CREATE MODELS

        # Get all behavior values in current population and create a KNN model
        behavior_values = np.array(list(self.behaviors.values()))

        # If neighbours value is out of bounds, set it to max possible value
        pop_knn_neighbours = min(len(behavior_values), self.pop_knn_neighbours)

        # Obtain normalized data
        normalized_behavior_values = self.normalize_data(behavior_values)

        # FIRST KNN MODEL: fit on the behavior values of current population
        knn_model = NearestNeighbors(n_neighbors=pop_knn_neighbours,
                                     algorithm='ball_tree').fit(normalized_behavior_values)

        if len(self.novel_members) < 1:
            models = [knn_model]
        else:
            # Get behaviors of novel member archive and create KNN model
            novel_members_behaviors = np.array(self.novel_members)
            novel_members_behaviors.reshape(-1, 1)

            # If neighbours value is out of bounds, set it to max possible value
            archive_knn_neighbours = min(len(novel_members_behaviors), self.archive_knn_neighbours)

            # Obtain normalized data
            normalized_novel_members_behaviors = self.normalize_data(novel_members_behaviors)

            # SECOND KNN MODEL:: Build knn model for novel member archive
            novel_members_knn_model = NearestNeighbors(n_neighbors=archive_knn_neighbours, algorithm='ball_tree').fit(
                normalized_novel_members_behaviors)

            # Gather models
            models = [knn_model, novel_members_knn_model]

        # STEP 2: CALCULATE NOVELTY SCORES
        # Novelty is assigned as the average distance to the k-nearest neighbors
        # If genome is novel, average distance will be high.
        self.calculate_population_distances(self.behaviors, genomes, models)

        # STEP 3: ADD BEHAVIORS OVER THRESHOLD TO NOVELTY ARCHIVE
        # Extract novelty scores in the population after they have been calculated in previous step
        novel_members_added = 0
        if len(self.novel_members) < 1:
            max_novelty_score_id = max(self.novelty_scores, key=self.novelty_scores.get)
            self.novel_members.append(self.behaviors[max_novelty_score_id])
            novel_members_added += 1
        else:
            for genome_id, _ in self.novelty_scores.items():

                behavior = self.behaviors[genome_id]
                knn_distance = self.KNN_distance(behavior, novel_members_knn_model)

                if knn_distance > self.threshold:
                    self.novel_members.append(behavior)
                    self.evaluations_since_last_added = 0
                    novel_members_added += 1
                else:
                    self.evaluations_since_last_added += 1

        if novel_members_added > 4:
            self.threshold = self.threshold * 1.05

        # using 1000 or 1250 evals based on "Novelty-based Multiobjectivization" paper as limit
        if self.evaluations_since_last_added > 1500:
            self.threshold = self.threshold * 0.95

        # reset behaviors dict
        self.behaviors = {}
        #print(f"novel members size: {len(self.novel_members)}")
        #print(f"threshold: {self.threshold}")
        #print(f"evals since last added: {self.evaluations_since_last_added}")


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