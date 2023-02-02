import math
import time
import numpy as np
import multiprocessing as mp

from itertools import combinations
import objectives.cca as cca
from objectives.cosine_similarity import CosineSimilarity

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

    return hsic / (var1 * var2)


def kernel_CKA(X, Y, sigma=None):
    hsic = kernel_HSIC(X, Y, sigma)
    var1 = np.sqrt(kernel_HSIC(X, X, sigma))
    var2 = np.sqrt(kernel_HSIC(Y, Y, sigma))

    return hsic / (var1 * var2)

def sequential(all_X):
    for X, Y in combinations(all_X, 2):
        linear_CKA(X, Y)



def parallel(all_X, pool):
    jobs = []
    for X, Y in combinations(all_X, 2):
        jobs.append(pool.apply_async(linear_CKA, (X, Y)))
        #print('Linear CKA, between X and Y: {}'.format(linear_CKA(X, Y)))

    for job in jobs:
        job.get(timeout=None)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


if __name__=="__main__":

    cos = CosineSimilarity()

    N = 250
    all_X = []
    for i in range(N):
        all_X.append(np.random.randn(1028, 1028))


    all_combinations = []

    for i in range(100):
        X = np.random.choice(np.arange(N))
        Y = np.random.choice(np.arange(N))
        all_combinations.append((all_X[X], all_X[Y]))

    sim = cosine_similarity(all_combinations[0][0].ravel(), all_combinations[0][1].ravel())
    print(f"CosSim = {sim}")
    print(f"1 - CosSim = {1.0 - sim}")
    print(f"1/CosSim = {1.0 / sim}\n")

    sim = cosine_similarity(all_combinations[0][1].ravel(), all_combinations[0][0].ravel())
    print(f"CosSim = {sim}")
    print(f"1 - CosSim = {1.0 - sim}")
    print(f"1/CosSim = {1.0 / sim}\n")
    exit(0)
    avg_cka_time = 0.0
    avg_cos_sim_time = 0.0
    a = []
    for X, Y in all_combinations:
        start = time.time()
        sim = linear_CKA(X, Y)
        avg_cka_time += time.time() - start
        print(f"CKA = {sim}")
        print(f"1 - CKA = {1.0 - sim}")
        print(f"1/CKA = {1.0/sim}\n")

        start = time.time()
        sim = cosine_similarity(X.ravel(), Y.ravel())
        avg_cos_sim_time += time.time() - start
        a.append(1.0 - sim)
        print(f"CosSim = {sim}")
        print(f"1 - CosSim = {1.0 - sim}")
        print(f"1/CosSim = {1.0/sim}\n")

    print(f"Average CKA time per call: {avg_cka_time / 100} s")
    print(f"Average Cosine Similarity time per call: {avg_cos_sim_time / 100} s")
    print(np.max(a))
    print(np.min(a))
    import matplotlib.pyplot as plt

    q25, q75 = np.percentile(a, [25, 75])
    bin_width = 2 * (q75 - q25) * len(a) ** (-1 / 3)
    bins = round((max(a) - min(a)) / bin_width)
    print("Freedman–Diaconis number of bins:", bins)
    plt.hist(a, density=True, bins=bins)
    plt.ylabel('Probability')
    plt.xlabel('Data')
    plt.show()

    sim = cosine_similarity(all_combinations[0][0].ravel(), all_combinations[0][0].ravel())
    print(f"CosSim = {sim}")
    print(f"1 - CosSim = {1.0 - sim}")
    print(f"1/CosSim = {1.0 / sim}\n")

    #pool = mp.Pool(mp.cpu_count())
    #pool.close()
    """
    print(mp.cpu_count())
    start = time.time()
    with mp.Pool(mp.cpu_count()//2) as pool:
        parallel(all_X, pool)
    parallel_time = time.time() - start

    start = time.time()
    sequential(all_X)
    seq_time = time.time() - start

    print(f"Sequential runtime: {seq_time} s")
    print(f"Parallel runtime: {parallel_time} s")
    """









