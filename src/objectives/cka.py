import math
import numpy as np
import multiprocessing as mp
from itertools import combinations

class CKA:

    def __init__(self, linear_kernel):
        self.activations = {}
        self.similarity = {}

        if linear_kernel:
            self.cka_similarity_func = linear_CKA
        else:
            self.cka_similarity_func = kernel_CKA

        np.seterr('raise')

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

            similarity_value = self.cka_similarity_func(X, Y)

            similarities[genome_A_id] += similarity_value
            similarities[genome_B_id] += similarity_value


        num_other_genomes = (len(genomes) - 1)
        self.similarity = {key: value / num_other_genomes for key, value in similarities.items()}

    def calculate_CKA_similarities_parallel(self, genomes):
        similarities = {genome_id: 0.0 for genome_id, _ in genomes}

        with mp.Pool(32) as pool:
            jobs = []
            all_combinations = list(combinations(genomes, 2))
            for (genome_A_id, genome_A), (genome_B_id, genome_B) in all_combinations:
                X = self.activations[genome_A_id]
                Y = self.activations[genome_B_id]
                jobs.append(pool.apply_async(_similarity_parallel, (X, Y, self.cka_similarity_func)))

            for job, ((genome_A_id, genome_A), (genome_B_id, genome_B)) in zip(jobs, all_combinations):
                similarity_value = job.get(timeout=None)
                similarities[genome_A_id] += similarity_value
                similarities[genome_B_id] += similarity_value

            num_other_genomes = (len(genomes) - 1)
            self.similarity = {key: value / num_other_genomes for key, value in similarities.items()}


def centering(K):
    n = K.shape[0]
    unit = np.ones([n, n])
    I = np.eye(n)
    H = I - unit / n

    # HKH are the same with KH, KH is the first centering, H(KH) do the second time,
    # results are the same with one time centering
    return np.dot(np.dot(H, K), H)
    # return np.dot(H, K)  # KH

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
    #print(f"var1: {var1} \nvar2: {var2} \nvar1*var2: {var1*var2}")
    epsilon = 0.000001

    return hsic / ((var1 * var2) + epsilon)

def kernel_CKA(X, Y, sigma=None):
    hsic = kernel_HSIC(X, Y, sigma)
    var1 = np.sqrt(kernel_HSIC(X, X, sigma))
    var2 = np.sqrt(kernel_HSIC(Y, Y, sigma))

    return hsic / (var1 * var2)

def _similarity_parallel(X, Y, similarity_func):

    if X.shape[1] > Y.shape[1]:
        X, Y = Y, X

    # find min N samples, limit to match dim of X and Y
    min_n = min(X.shape[0], Y.shape[0])
    X = X[:min_n, :]
    Y = Y[:min_n, :]

    return similarity_func(X, Y)

