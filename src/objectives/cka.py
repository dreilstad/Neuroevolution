import math
import numpy as np
from itertools import combinations

class CKA:

    def __init__(self, linear_kernel):
        self.activations = {}
        self.similarity = {}

        if linear_kernel:
            self.cka_similarity_func = self.linear_CKA
        else:
            self.cka_similarity_func = self.kernel_CKA

    def __getitem__(self, key):
        return self.similarity[key]

    @staticmethod
    def centering(K):
        n = K.shape[0]
        unit = np.ones([n, n])
        I = np.eye(n)
        H = I - unit / n

        # HKH are the same with KH, KH is the first centering, H(KH) do the second time,
        # results are the same with one time centering
        return np.dot(np.dot(H, K), H)
        # return np.dot(H, K)  # KH

    @staticmethod
    def rbf(X, sigma=None):
        GX = np.dot(X, X.T)
        KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
        if sigma is None:
            mdist = np.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        KX *= - 0.5 / (sigma * sigma)
        KX = np.exp(KX)
        return KX


    def kernel_HSIC(self, X, Y, sigma):
        return np.sum(self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma)))


    def linear_HSIC(self, X, Y):
        L_X = np.dot(X, X.T)
        L_Y = np.dot(Y, Y.T)
        return np.sum(self.centering(L_X) * self.centering(L_Y))


    def linear_CKA(self, X, Y):
        hsic = self.linear_HSIC(X, Y)
        var1 = np.sqrt(self.linear_HSIC(X, X))
        var2 = np.sqrt(self.linear_HSIC(Y, Y))

        return hsic / (var1 * var2)


    def kernel_CKA(self, X, Y, sigma=None):
        hsic = self.kernel_HSIC(X, Y, sigma)
        var1 = np.sqrt(self.kernel_HSIC(X, X, sigma))
        var2 = np.sqrt(self.kernel_HSIC(Y, Y, sigma))

        return hsic / (var1 * var2)

    def calculate_CKA_similarities(self, genomes):
        similarities = {genome_id: 0.0 for genome_id, _ in genomes}

        i = 0
        for (genome_A_id, genome_A), (genome_B_id, genome_B) in combinations(genomes, 2):
            print(f"comb {i}")
            X = self.activations[genome_A_id]
            Y = self.activations[genome_B_id]

            if X.shape[1] > Y.shape[1]:
                X, Y = Y, X

            # find min N samples, limit to match dim of X and Y
            min_n = min(X.shape[0], Y.shape[0])
            X = X[:min_n, :]
            Y = Y[:min_n, :]

            similarity_value = self.cka_similarity_func(X, Y)
            print(f"similarity: {similarity_value}\n")
            similarities[genome_A_id] += similarity_value
            similarities[genome_B_id] += similarity_value

            i += 1

        num_other_genomes = (len(genomes) - 1)
        self.similarity = {key: value / num_other_genomes for key, value in similarities.items()}

"""
if __name__=='__main__':
    def shuffle_along_axis(a, axis):
        idx = np.random.rand(*a.shape).argsort(axis=axis)
        return np.take_along_axis(a, idx, axis=axis)

    #X = np.random.randn(100, 64)
    #X_shuffled = shuffle_along_axis(X, axis=1)
    #Y = np.random.randn(100, 64)

    X = np.random.randn(6, 6)
    X_shuffled = np.copy(X)
    for i in range(3):
        np.random.shuffle(X_shuffled[i,:3])
        np.random.shuffle(X_shuffled[i,3:])

    #X_shuffled = np.concatenate((X_shuffled_first_half, X_shuffled_second_half), axis=1)
    print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in X]))
    print("")
    print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in X_shuffled]))
    print("")
    Y = np.random.randn(6, 6)
    print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in Y]))
    print("")


    cka = CKA(True)
    print('Linear CKA, between X and Y: {}'.format(cka.linear_CKA(X, Y)))
    print('Linear CKA, between X and X: {}'.format(cka.linear_CKA(X, X)))
    print('Linear CKA, between X and X_shuffled: {}'.format(cka.linear_CKA(X, X_shuffled)))

    print('RBF Kernel CKA, between X and Y: {}'.format(cka.kernel_CKA(X, Y)))
    print('RBF Kernel CKA, between X and X: {}'.format(cka.kernel_CKA(X, X)))
    print('RBF Kernel CKA, between X and X_shuffled: {}'.format(cka.kernel_CKA(X, X_shuffled)))

"""
