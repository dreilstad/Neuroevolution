from itertools import combinations
from random import choice, choices
import numpy as np


def generate_configurations(N, K, sample_size=0):
    box_configs = generate_all_possible_configurations(N - 2, K)
    invalid_config_indices = find_invalid_configuration_indices(box_configs)
    valid_configs = remove_invalid_configurations(box_configs, invalid_config_indices)

    complete_configs = generate_complete_configurations(valid_configs, N, K)
    test_config = choice(complete_configs)

    if sample_size:
        complete_configs = choices(complete_configs, k=sample_size)

    return complete_configs, test_config


def generate_complete_configurations(configurations, N, K):
    complete_configurations = [None] * len(configurations)
    for i, config in enumerate(configurations):
        extended_config = np.zeros((N, N), dtype="int8")
        extended_config[1:N - 1, 1:N - 1] = config
        complete_configurations[i] = extended_config

    return complete_configurations


def generate_all_possible_configurations(N, K):
    # all combinations of K boxes in NxN grid
    which = np.array(list(combinations(range(N * N), K)))
    grid = np.zeros((len(which), N * N), dtype="int8")

    # set
    grid[np.arange(len(which))[None].T, which] = 1
    grid = [g.reshape(N, N) for g in grid]
    return grid


def find_invalid_configuration_indices(configurations):
    # find partially solvable with 4x4 cluster of blocks
    pattern_4x4 = np.array([[1, 1],
                            [1, 1]])
    config_4x4_indices = find_configuration_indices_with_pattern(configurations, pattern_4x4)

    pattern_wilson = np.array([[1, 1, 0],
                               [1, 0, 1],
                               [0, 1, 1]])
    pattern_wilson_mirror = np.array([[0, 1, 1],
                                      [1, 0, 1],
                                      [1, 1, 0]])
    config_wilson_indices = find_configuration_indices_with_pattern(configurations, pattern_wilson) + \
                            find_configuration_indices_with_pattern(configurations, pattern_wilson_mirror)

    return config_4x4_indices + config_wilson_indices


def find_configuration_indices_with_pattern(configurations, pattern):
    indices = []
    for i, config in enumerate(configurations):

        col_match = match_pattern(config, pattern.shape) == pattern.ravel()[:, None]
        out_shape = np.asarray(config.shape) - np.asarray(pattern.shape) + 1
        R, C = np.where(col_match.all(0).reshape(out_shape))

        if len(R) != 0 and len(C) != 0:
            indices.append(i)

    return indices


def remove_invalid_configurations(configurations, id_to_del):
    solvable_configurations = [c for i, c in enumerate(configurations) if i not in id_to_del]
    return solvable_configurations


def match_pattern(A, BLKSZ):
    # Parameters
    M, N = A.shape
    col_extent = N - BLKSZ[1] + 1
    row_extent = M - BLKSZ[0] + 1

    # Get Starting block indices
    start_idx = np.arange(BLKSZ[0])[:, None] * N + np.arange(BLKSZ[1])

    # Get offsetted indices across the height and width of input array
    offset_idx = np.arange(row_extent)[:, None] * N + np.arange(col_extent)

    # Get all actual indices & index into input array for final output
    return np.take(A, start_idx.ravel()[:, None] + offset_idx.ravel())