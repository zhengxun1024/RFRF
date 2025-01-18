import numpy as np
from numba import jit


############################################################
############################################################
################ Find best split functions  ################
############################################################
############################################################


@jit(cache=True, nopython=True)
def get_all_entropy(all_memberships, hesitation):
    n_features = len(all_memberships)
    nof_objects = len(all_memberships[0])
    # 计算总的信息熵
    all_entropy = np.zeros((nof_objects, n_features))
    for i in range(n_features):
        f_memberships = all_memberships[i]
        u = f_memberships
        v = (1 - u)
        v = np.where(v != 1, v * hesitation, v)
        d_IFS_M = (np.abs(u - 1) + np.abs(v - 0) + np.abs(1 - u - v - 0)) / 2
        d_IFS_N = (np.abs(u - 0) + np.abs(v - 1) + np.abs(1 - u - v - 0)) / 2
        max_values = np.maximum(d_IFS_M, d_IFS_N)
        min_values = np.minimum(d_IFS_M, d_IFS_N)
        entropy = min_values / max_values
        entropy = entropy.sum(axis=1)
        all_entropy[:, i] = entropy

    return all_entropy


@jit(cache=True, nopython=True)
def get_chosen_entropy(all_entropy, features_chosen_indices):
    n_features = all_entropy.shape[1]
    total_entropy = all_entropy.sum(axis=1).reshape(-1, 1)
    total_entropy = total_entropy / n_features
    features_entropy = all_entropy[:, features_chosen_indices]

    return total_entropy, features_entropy


@jit(cache=True, nopython=True)
def IENT(y, x_entropy, membership_degree):
    label = np.unique(y)
    total_entropy = x_entropy * membership_degree
    E_total = np.sum(total_entropy)
    IEnt = 0
    # 计算每个类别所占用的熵
    for c in label:
        indices = np.where(y == c)[0]
        if E_total == 0:
            P_i = len(indices) / len(y)
        else:
            c_total_entropy = x_entropy[indices, :] * membership_degree[indices, :]
            E_i = np.sum(c_total_entropy)
            P_i = E_i / E_total
        if P_i != 0:
            IEnt += P_i * np.log2(P_i)

    return -IEnt


@jit(cache=True, nopython=True)
def get_best_split(y, hesitation, X_degree, all_memberships, n_centroids, features_chosen_indices):
    best_igain = -100
    best_attribute = 0
    all_entropy = get_all_entropy(all_memberships, hesitation)
    total_entropy, features_entropy = get_chosen_entropy(all_entropy, features_chosen_indices)

    membership_degree = X_degree[:, 0:1]
    INET_total = IENT(y, total_entropy, membership_degree)

    for i in range(len(features_chosen_indices)):
        features = features_chosen_indices[i]
        f_entropy = features_entropy[:, i]
        f_memberships = all_memberships[features]
        IGain = INET_total
        for j in range(n_centroids[features]):
            indices = np.where(f_memberships[:, j] > 0)[0]
            y_data = y[indices]
            c_entropy = f_entropy[indices].reshape(-1, 1)
            c_degree = membership_degree[indices]
            c_IEnt = IENT(y_data, c_entropy, c_degree)
            IGain -= (len(indices) / len(y)) * c_IEnt

        if IGain > best_igain:
            best_attribute = features
            best_igain = IGain
    return best_attribute

