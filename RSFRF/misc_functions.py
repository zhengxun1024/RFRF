import numba
import numpy as np
from numba import jit
from scipy.spatial.distance import cdist

cache = True


############################################################
############################################################

############################ MISC  #########################
############################################################
############################################################


@jit(cache=cache, nopython=True)
def return_class_probas(y, classes, X_degree):
    """
    The leaf probabilities for each class
    """
    nof_classes = len(classes)
    class_probas = np.zeros(nof_classes)
    for i in range(nof_classes):
        u_degree = np.nansum(X_degree[:, 0][y[:, 0] == classes[i]])
        v_degree = np.nansum(X_degree[:, 1][y[:, 0] == classes[i]])
        class_probas[i] = u_degree - v_degree

    return class_probas


# @jit(cache=True, nopython=True)
def get_split_indices_membership(all_memberships, attribute_indices, centroid):
    f_memberships = all_memberships[attribute_indices]
    c_memberships = f_memberships[:, centroid]
    indices = np.where(c_memberships > 0)[0]
    sub_membership = c_memberships[indices]
    return indices, sub_membership


@jit(cache=True, nopython=True)
def get_split_degree(X_degree, indices, sub_membership, hesitation):
    sub_degree = X_degree[indices]
    sub_nonmembership = (1 - sub_membership) * hesitation
    sub_degree[:, 0] = sub_degree[:, 0] * sub_membership
    sub_degree[:, 1] = sub_degree[:, 1] * sub_nonmembership
    return sub_degree


@jit(cache=True, nopython=True)
def get_split_memberships(indices, all_memberships):
    memberships = numba.typed.List()
    for i in range(len(all_memberships)):
        f_memberships = all_memberships[i]
        sub_memberships = f_memberships[indices]
        memberships.append(sub_memberships)
    return memberships


@jit(cache=True, nopython=True)
def get_split_features(features_indices, best_attribute_indices):
    result = features_indices[features_indices != best_attribute_indices]
    return result


@jit(cache=True, nopython=True)
def choose_features(features_indices):
    """
    function randomly selects the features that will be examined for each split
    """
    if len(features_indices) != 1:
        nof_features = round(np.log2(len(features_indices)))
    else:
        nof_features = 1
    features_chosen = np.random.choice(features_indices, size=nof_features, replace=False)

    return features_chosen


@jit(cache=True, nopython=True)
def pull_values(A, indices):
    """
    Splits an array A to two
    according to lists of indicies
    """
    A_sub = A[indices]

    return A_sub


def K_Means(data, n_clusters, max_iter):
    idx = np.random.choice(data.shape[0], size=n_clusters, replace=False)
    centroids = data[idx, :]

    # 开始迭代
    for i in range(max_iter):
        distances = cdist(data, centroids)
        c_index = np.argmin(distances, axis=1)
        for j in range(n_clusters):
            if j in c_index:
                centroids[j] = np.mean(data[c_index == j], axis=0)
    centroids = np.sort(centroids, axis=0)
    return centroids


# 定义一个能根据特征数据聚类，并且计算每个样本隶属度的方法
def memberships(X_data, n_clusters, S):
    f_centroids = np.zeros((n_clusters+1, X_data.shape[1]))    # 存放所有特征的中心点
    n_centroids = np.zeros(X_data.shape[1], int)
    f_memberships = numba.typed.List()  # 存放所有特征下样本与中心点之间的隶属度

    for i in range(X_data.shape[1]):
        f_data = X_data[:, i].reshape(-1, 1)
        if len(np.unique(f_data)) <= n_clusters:   # 离散特征
            centroids = np.unique(f_data, axis=0)
            f_centroids[:centroids.shape[0], i] = centroids[:, 0]
            f_centroids[-1][i] = 0
            n_centroids[i] = centroids.shape[0]
            F_u = lisan_membership(f_data, centroids)
            f_memberships.append(F_u)

        else:   # 连续特征
            centroids = K_Means(f_data, n_clusters, 300)
            centroids = np.unique(centroids, axis=0)
            f_centroids[:centroids.shape[0], i] = centroids[:, 0]
            f_centroids[-1][i] = 1
            n_centroids[i] = centroids.shape[0]

            ei = np.array([((centroids[i+1][0] - centroids[i][0])/S) for i in range(centroids.shape[0]-1)])
            F_u = lianxu_membership(f_data, centroids, ei)
            f_memberships.append(F_u)

    return f_centroids, n_centroids, f_memberships


@jit(cache=True, nopython=True)
def lisan_membership(f_data, centroids):
    F_u = np.zeros((f_data.shape[0], centroids.shape[0]))
    for i in range(centroids.shape[0]):
        F_u[:, np.array([i])] = np.where(f_data == centroids[i][0], 1, 0)
    return F_u


@jit(cache=True, nopython=True)
def lianxu_membership(f_data, centroids, e_i):
    F_u = np.zeros((f_data.shape[0], centroids.shape[0]))

    F_u[:, 0:1] = np.where(f_data >= centroids[1][0] - e_i[0], 0,
                           np.where(f_data <= centroids[0][0] + e_i[0], 1,
                                    (centroids[1][0] - e_i[0] - f_data) / (centroids[1][0] - centroids[0][0] - 2 * e_i[0])))

    for i in range(1, centroids.shape[0]-1):
        fu = F_u[:, i:i+1]
        fu = np.where(f_data <= centroids[i - 1][0] + e_i[i - 1], 0, fu)
        fu = np.where(f_data >= centroids[i + 1][0] - e_i[i], 0, fu)
        fu = np.where((centroids[i][0] - e_i[i - 1] <= f_data) & (f_data <= centroids[i][0] + e_i[i]), 1, fu)
        fu = np.where((centroids[i - 1][0] + e_i[i - 1] <= f_data) & (f_data <= centroids[i][0] - e_i[i - 1]), (f_data - centroids[i - 1][0] - e_i[i - 1]) / (centroids[i][0] - centroids[i - 1][0] - 2 * e_i[i - 1]), fu)
        fu = np.where((centroids[i][0] + e_i[i] <= f_data) & (f_data <= centroids[i + 1][0] - e_i[i]), (centroids[i + 1][0] - e_i[i] - f_data) / (centroids[i + 1][0] - centroids[i][0] - 2 * e_i[i]), fu)
        F_u[:, i] = fu[:, 0]

    F_u[:, -1:] = np.where(f_data <= centroids[-2][0] + e_i[-1], 0,
                             np.where(f_data >= centroids[-1][0] - e_i[-1], 1,
                                      (f_data - centroids[-2][0] - e_i[-1]) / (centroids[-1][0] - centroids[-2][0] - 2 * e_i[-1])))
    return F_u
