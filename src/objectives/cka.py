import numpy as np
from itertools import combinations

class CKA:

    def __init__(self):
        self.activations = {}
        self.similarity = {}

    def __getitem__(self, key):
        return self.similarity[key]

    def calculate_CKA_similarities(self, genomes):
        similarities = {genome_id: 0.0 for genome_id, _ in genomes}

        for (genome_A_id, genome_A), (genome_B_id, genome_B) in combinations(genomes, 2):
            X = self.activations[genome_A_id]
            Y = self.activations[genome_B_id]

            # find min N samples, limit to match first dim of X and Y
            min_n = min(X.shape[0], Y.shape[0])
            X = X[:min_n, :]
            Y = Y[:min_n, :]

            similarity_value = self.feature_space_linear_cka(X, Y)

            similarities[genome_A_id] += similarity_value
            similarities[genome_B_id] += similarity_value

        num_other_genomes = float(len(genomes) - 1)
        self.similarity = {key: value / num_other_genomes for key, value in similarities.items()}
        self.activations = {}

    @staticmethod
    def gram_linear(x):
        """Compute Gram (kernel) matrix for a linear kernel.

        Args:
            x: A num_examples x num_features matrix of features.

        Returns:
            A num_examples x num_examples Gram matrix of examples.
        """

        return x.dot(x.T)

    @staticmethod
    def gram_rbf(x, threshold=1.0):
        """Compute Gram (kernel) matrix for an RBF kernel.

        Args:
            x: A num_examples x num_features matrix of features.
            threshold: Fraction of median Euclidean distance to use as RBF kernel bandwidth.

        Returns:
            A num_examples x num_examples Gram matrix of examples.
        """

        dot_products = x.dot(x.T)
        sq_norms = np.diag(dot_products)
        sq_distances = -2 * dot_products + sq_norms[:, None] + sq_norms[None, :]
        sq_median_distance = np.median(sq_distances)
        return np.exp(-sq_distances / (2 * threshold**2 * sq_median_distance))

    @staticmethod
    def center_gram(gram, unbiased=False):
        """Center a symmetric Gram matrix.

        This is equvialent to centering the (possibly infinite-dimensional) features
        induced by the kernel before computing the Gram matrix.

        Args:
            gram: A num_examples x num_examples symmetric matrix.
            unbiased: Whether to adjust the Gram matrix in order to compute an unbiased estimate of HSIC.
                      (Note that this estimator may be negative)

        Returns:
            A symmetric matrix with centered columns and rows.
        """

        if not np.allclose(gram, gram.T):
            raise ValueError("Input must be a symmetric matrix")

        gram = gram.copy()

        if unbiased:
            n = gram.shape[0]
            np.fill_diagonal(gram, 0)
            means = np.sum(gram, 0, dtype=np.float64) / (n - 2)
            means -= np.sum(means) / (2 * (n - 1))
            gram -= means[:, None]
            gram -= means[None, :]
            np.fill_diagonal(gram, 0)
        else:
            means = np.mean(gram, 0, dtype=np.float64)
            means -= np.mean(means) / 2
            gram -= means[:, None]
            gram -= means[None, :]

        return gram

    def cka(self, gram_x, gram_y, debiased=False):
        """Compute CKA.

        Args:
            gram_x: A num_examples x num_examples Gram matrix.
            gram_y: A num_examples x num_examples Gram matrix.
            debiased: Use unbiased estimator of HSIC. CKA may still be biased.

        Returns:
            The value of CKA between X and Y.
        """

        gram_x = self.center_gram(gram_x, unbiased=debiased)
        gram_y = self.center_gram(gram_y, unbiased=debiased)

        scaled_hsic = gram_x.ravel().dot(gram_y.ravel())

        normalization_x = np.linalg.norm(gram_x)
        normalization_y = np.linalg.norm(gram_y)
        return scaled_hsic / (normalization_x * normalization_y)

    def feature_space_linear_cka(self, features_x, features_y, debiased=False):
        """Compute CKA with a linear kernel, in feature space.

        This is typically faster than computing the Gram matrix when there are fewer
        features than examples.

        Args:
            features_x: A num_examples x num_features matrix of features.
            features_y: A num_examples x num_features matrix of features.
            debiased: Use unbiased estimator of dot product similarity. CKA may still debiased.
                      (Note that this estimator may be negative)

        Returns:
            The value of CKA between X and Y.
        """

        features_x = features_x - np.mean(features_x, 0, keepdims=True)
        features_y = features_y - np.mean(features_y, 0, keepdims=True)

        dot_product_similarity = np.linalg.norm(features_x.T.dot(features_y))**2
        normalization_x = np.linalg.norm(features_x.T.dot(features_x))
        normalization_y = np.linalg.norm(features_y.T.dot(features_y))

        if debiased:
            n = features_x.shape[0]
            sum_squared_rows_x = np.einsum("ij,ij->i", features_x, features_x)
            sum_squared_rows_y = np.einsum("ij,ij->i", features_y, features_y)
            squared_norm_x = np.sum(sum_squared_rows_x)
            squared_norm_y = np.sum(sum_squared_rows_y)

            dot_product_similarity = self._debiased_dot_product_similarity_helper(dot_product_similarity,
                                                                                  sum_squared_rows_x,
                                                                                  sum_squared_rows_y,
                                                                                  squared_norm_x,
                                                                                  squared_norm_y,
                                                                                  n)
            normalization_x = np.sqrt(self._debiased_dot_product_similarity_helper(normalization_x**2,
                                                                                   sum_squared_rows_x,
                                                                                   sum_squared_rows_x,
                                                                                   squared_norm_x,
                                                                                   squared_norm_x,
                                                                                   n))
            normalization_y = np.sqrt(self._debiased_dot_product_similarity_helper(normalization_y**2,
                                                                                   sum_squared_rows_y,
                                                                                   sum_squared_rows_y,
                                                                                   squared_norm_y,
                                                                                   squared_norm_y,
                                                                                   n))

        epsilon = 0.0000001
        return dot_product_similarity / ((normalization_x * normalization_y) + epsilon)

    @staticmethod
    def _debiased_dot_product_similarity_helper(xty, sum_squared_rows_x, sum_squared_rows_y,
                                                squared_norm_x, squared_norm_y, n):
        """
        Helper for computing debiased dot product similarity (i.e. linear HSIC)
        """

        return xty - n / (n - 2.0) * sum_squared_rows_x.dot(sum_squared_rows_y) \
               + squared_norm_x * squared_norm_y / ((n - 1) * (n - 2))
