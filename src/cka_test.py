import math
import time
import numpy as np
import multiprocessing as mp
from numba.typed import List
from itertools import combinations



def gram_linear(x):
    return x.dot(x.T)


def gram_rbf(x, threshold=1.0):
    dot_products = x.dot(x.T)
    sq_norms = np.diag(dot_products)
    sq_distances = -2 * dot_products + sq_norms[:, None] + sq_norms[None, :]
    sq_median_distance = np.median(sq_distances)
    return np.exp(-sq_distances / (2 * threshold**2 * sq_median_distance))


def center_gram(gram, unbiased=False):
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


def cka(gram_x, gram_y, debiased=False):
    gram_x = center_gram(gram_x, unbiased=debiased)
    gram_y = center_gram(gram_y, unbiased=debiased)

    scaled_hsic = gram_x.ravel().dot(gram_y.ravel())

    normalization_x = np.linalg.norm(gram_x)
    normalization_y = np.linalg.norm(gram_y)
    return scaled_hsic / (normalization_x * normalization_y)


def _debiased_dot_product_similarity_helper(xty, sum_squared_rows_x, sum_squared_rows_y,
                                            squared_norm_x, squared_norm_y, n):
    return xty - n / (n - 2.0) * sum_squared_rows_x.dot(sum_squared_rows_y) \
           + squared_norm_x * squared_norm_y / ((n - 1) * (n - 2))


def feature_space_linear_cka(features_x, features_y, debiased=False):

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

        dot_product_similarity = _debiased_dot_product_similarity_helper(dot_product_similarity,
                                                                         sum_squared_rows_x,
                                                                         sum_squared_rows_y,
                                                                         squared_norm_x,
                                                                         squared_norm_y,
                                                                         n)
        normalization_x = np.sqrt(_debiased_dot_product_similarity_helper(normalization_x**2,
                                                                          sum_squared_rows_x,
                                                                          sum_squared_rows_x,
                                                                          squared_norm_x,
                                                                          squared_norm_x,
                                                                          n))
        normalization_y = np.sqrt(_debiased_dot_product_similarity_helper(normalization_y**2,
                                                                          sum_squared_rows_y,
                                                                          sum_squared_rows_y,
                                                                          squared_norm_y,
                                                                          squared_norm_y,
                                                                          n))

    return dot_product_similarity / (normalization_x * normalization_y)

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

    epsilon = 0.000001
    return hsic / ((var1 * var2) + epsilon)


def kernel_CKA(X, Y, sigma=None):
    hsic = kernel_HSIC(X, Y, sigma)
    var1 = np.sqrt(kernel_HSIC(X, X, sigma))
    var2 = np.sqrt(kernel_HSIC(Y, Y, sigma))

    return hsic / (var1 * var2)


def sequential(combs):
    for X, Y in combs:
        result = linear_CKA(X, Y)


def parallel(all_X, pool):
    jobs = []
    for X, Y in combinations(all_X, 2):
        jobs.append(pool.apply_async(linear_CKA, (X, Y)))
        #print('Linear CKA, between X and Y: {}'.format(linear_CKA(X, Y)))

    for job in jobs:
        job.get(timeout=None)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def cca(features_x, features_y):
    qx, _ = np.linalg.qr(features_x)
    qy, _ = np.linalg.qr(features_y)
    return np.linalg.norm(qx.T.dot(qy))**2 / min(features_x.shape[1], features_y.shape[1])


if __name__=="__main__":

    N = 50
    all_X = []
    for i in range(N):
        all_X.append(np.random.randn(256, 32))

    """
    start = time.time()
    sequential(all_X)
    end = time.time()
    print(f"Sequential runtime: {end - start} s")
    """
    combs = list(combinations(all_X, 2))


    X = np.random.randn(256, 42)
    Y = np.random.randn(256, 63)
    cka_res = cka(gram_linear(X), gram_linear(Y))
    feature_res = feature_space_linear_cka(X, Y)
    old_res = linear_CKA(X, Y)
    cca_res = cca(X, Y)
    print(f"Linear CKA from Examples: {cka_res}")
    print(f"Linear CKA from Features: {feature_res}")
    print(f"Linear CKA (old): {old_res}")
    print(f"CCA: {cca_res}")

    cka_res = cka(gram_linear(Y), gram_linear(X))
    feature_res = feature_space_linear_cka(Y, X)
    old_res = linear_CKA(Y, X)
    cca_res = cca(Y, X)
    print(f"Linear CKA from Examples: {cka_res}")
    print(f"Linear CKA from Features: {feature_res}")
    print(f"Linear CKA (old): {old_res}")
    print(f"CCA: {cca_res}")

    """
    start = time.time()
    for X, Y in combs:
        linear_CKA(X, Y)
    end = time.time()
    print(f"Old CKA: {end - start} s")
    
    start = time.time()
    for X, Y in combs:
        cka(gram_linear(X), gram_linear(Y))
    end = time.time()
    print(f"New CKA: {end - start} s")

    start = time.time()
    for X, Y in combs:
        feature_space_linear_cka(X, Y)
    end = time.time()
    print(f"New CKA (feature space): {end - start} s")

    start = time.time()
    for X, Y in combs:
        cca(X, Y)
    end = time.time()
    print(f"CCA: {end - start} s")
    """



