import math
import numpy as np
from multiprocessing import Pool, cpu_count
from multiprocessing.pool import ApplyResult
from itertools import combinations
from random import sample

class CKA:

    def __init__(self, linear_kernel, evenly_sampled=1):
        self.activations = {}
        self.similarity = {}
        self.evenly_sampled = evenly_sampled

        if linear_kernel:
            self.cka_similarity_func = linear_CKA
        else:
            self.cka_similarity_func = kernel_CKA

    def __getitem__(self, key):
        return self.similarity[key]

    def calculate_CKA_similarities(self, genomes):
        similarities = {genome_id: 0.0 for genome_id, _ in genomes}

        for (genome_A_id, genome_A), (genome_B_id, genome_B) in combinations(genomes, 2):

            X = self.activations[genome_A_id]
            Y = self.activations[genome_B_id]

            if X.shape[1] > Y.shape[1]:
                X, Y = Y, X

            # find min N samples, limit to match dim of X and Y
            min_n = min(X.shape[0], Y.shape[0])
            X = X[:min_n, :]
            Y = Y[:min_n, :]

            # samples evenly from activations, if evenly_sampled is 1 then all activations are used
            if self.evenly_sampled > 1:
                sample_idx = np.round(np.linspace(0, X.shape[0] - 1, X.shape[0] // self.evenly_sampled)).astype(int)
                X = X[sample_idx, :]
                Y = Y[sample_idx, :]

            similarity_value = self.cka_similarity_func(X, Y)

            similarities[genome_A_id] += similarity_value
            similarities[genome_B_id] += similarity_value

        num_other_genomes = len(genomes) - 1
        self.similarity = {key: value / num_other_genomes for key, value in similarities.items()}
        self.activations = {}

    def calculate_CKA_similarities_parallel(self, genomes):
        similarities = {genome_id: 0.0 for genome_id, _ in genomes}

        with Pool(40) as pool:
            jobs = []
            all_combinations = list(combinations(genomes, 2))
            for (genome_A_id, genome_A), (genome_B_id, genome_B) in all_combinations:
                X = self.activations[genome_A_id]
                Y = self.activations[genome_B_id]
                jobs.append(pool.apply_async(_similarity_parallel,
                                             (X, Y, self.cka_similarity_func, self.evenly_sampled)))

            pool.close()
            map(ApplyResult.wait, jobs)
            similarity_values = [result.get() for result in jobs]

            for similarity_value, ((genome_A_id, genome_A), (genome_B_id, genome_B)) in zip(similarity_values,
                                                                                            all_combinations):
                similarities[genome_A_id] += similarity_value
                similarities[genome_B_id] += similarity_value

        num_other_genomes = len(genomes) - 1
        self.similarity = {key: value / num_other_genomes for key, value in similarities.items()}
        self.activations = {}

    @staticmethod
    def _generate_combinations(keys, samples):
        generated_combinations = [tuple()] * (len(keys) * samples)
        for i, key in enumerate(keys):
            sampled_keys = sample(keys[:i] + keys[i + 1:], samples)
            for j, s_key in enumerate(sampled_keys):
                generated_combinations[i * samples + j] = (key, s_key)

        return generated_combinations


def centering(K):
    n = K.shape[0]
    unit = np.ones([n, n])
    I = np.eye(n)
    H = I - unit / n

    # HKH are the same with KH, KH is the first centering, H(KH) do the second time,
    # results are the same with one time centering
    return np.dot(np.dot(H, K), H)


def rbf(X, sigma=None):
    GX = np.dot(X, X.T)
    KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
    if sigma is None:
        mdist = np.median(KX[KX != 0])
        sigma = math.sqrt(mdist)
    KX *= - 0.5 / (sigma * sigma)
    KX = np.exp(KX)
    return KX


def kernel_HSIC(X, Y, sigma):
    return np.sum(centering(rbf(X, sigma)) * centering(rbf(Y, sigma)))


def linear_HSIC(X, Y):
    L_X = np.dot(X, X.T)
    L_Y = np.dot(Y, Y.T)
    return np.sum(centering(L_X) * centering(L_Y))


def linear_CKA(X, Y):
    hsic = linear_HSIC(X, Y)
    var1 = np.sqrt(linear_HSIC(X, X))
    var2 = np.sqrt(linear_HSIC(Y, Y))

    epsilon = 0.000001
    return hsic / ((var1 * var2) + epsilon)


def kernel_CKA(X, Y, sigma=None):
    hsic = kernel_HSIC(X, Y, sigma)
    var1 = np.sqrt(kernel_HSIC(X, X, sigma))
    var2 = np.sqrt(kernel_HSIC(Y, Y, sigma))

    return hsic / (var1 * var2)


def _similarity_parallel(X, Y, similarity_func, evenly_sampled=1):

    if X.shape[1] > Y.shape[1]:
        X, Y = Y, X

    # find min N samples, limit to match dim of X and Y
    min_n = min(X.shape[0], Y.shape[0])
    X = X[:min_n, :]
    Y = Y[:min_n, :]

    # samples evenly from activations, if evenly_sampled si 1 then all activations are used
    if evenly_sampled > 1:
        sample_idx = np.round(np.linspace(0, X.shape[0] - 1, X.shape[0] // evenly_sampled)).astype(int)
        X = X[sample_idx, :]
        Y = Y[sample_idx, :]

    return similarity_func(X, Y)

