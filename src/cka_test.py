import math
import time
import numpy as np
import multiprocessing as mp

from itertools import combinations

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
    jobs = []
    for X, Y in combinations(all_X, 2):
        print('Linear CKA, between X and Y: {}'.format(linear_CKA(X, Y)))



def parallel(all_X, pool):
    jobs = []
    for X, Y in combinations(all_X, 2):
        jobs.append(pool.apply_async(linear_HSIC, (X, Y)))
        #print('Linear CKA, between X and Y: {}'.format(linear_CKA(X, Y)))

    for job in jobs:
        print('Linear CKA, between X and Y: {}'.format(job.get(timeout=None)))

if __name__=="__main__":

    N = 150
    all_X = []
    for i in range(N):
        all_X.append(np.random.randn(64, 32))

    #pool = mp.Pool(mp.cpu_count())
    #pool.close()


    start = time.time()
    with mp.Pool(mp.cpu_count()) as pool:
        parallel(all_X, pool)
    parallel_time = time.time() - start

    start = time.time()
    sequential(all_X)
    seq_time = time.time() - start

    print(f"Sequential runtime: {seq_time} s")
    print(f"Parallel runtime: {parallel_time} s")









