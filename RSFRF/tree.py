import numpy as np
from numba import jit
from . import best_split
from . import misc_functions as m
from dataclasses import dataclass, field
from typing import Any, List, Optional

cache = True

############################################################
############################################################
############               TRAIN                ############
############################################################
############################################################

# 定义树的节点结构
@dataclass
class MultiTreeNode:
    feature: Optional[int] = None       # 当前节点用于分裂的特征索引
    centers: List[float] = field(default_factory=list)  # 分裂的阈值
    children: List["MultiTreeNode"] = field(default_factory=list) # 子节点列表，默认为空
    depth: Optional[int] = None       # 当前节点的深度
    value: Any = None  # 叶节点的值


def fit_tree(X, y, hesitation,tree_max_depth, depth, classes, features_indices, name_features, X_degree, all_memberships, n_centroids, all_centroids):
    """
    function grows a recursive disicion tree according to the objects X and their classifications y
    """
    # 有3个递归结束的情况：
    label = np.unique(y)
    if len(label) == 1 or len(features_indices) == 0:
        class_probas = m.return_class_probas(y, classes, X_degree)
        return class_probas

    max_depth = depth + 1
    if tree_max_depth:
        max_depth = tree_max_depth
    if depth >= max_depth:
        class_probas = m.return_class_probas(y, classes, X_degree)
        return class_probas

    # 下面是正式建树的过程
    features_chosen_indices = m.choose_features(features_indices)
    best_attribute_indices = best_split.get_best_split(y, hesitation, X_degree, all_memberships, n_centroids, features_chosen_indices)
    best_attribute_name = name_features[best_attribute_indices]
    # 初始化决策树
    decision_tree = {best_attribute_indices: {}}
    f_centroids = all_centroids[:, best_attribute_indices]
    for i in range(n_centroids[best_attribute_indices]):
        indices, sub_membership = m.get_split_indices_membership(all_memberships, best_attribute_indices, i)
        if len(indices) == 0:
            class_probas = m.return_class_probas(y, classes, X_degree)
            decision_tree[best_attribute_indices][i] = class_probas
        else:
            x_subdata = m.pull_values(X, indices)
            y_subdata = m.pull_values(y, indices)
            sub_degree = m.get_split_degree(X_degree, indices, sub_membership, hesitation)
            sub_all_memberships = m.get_split_memberships(indices, all_memberships)
            features_indices = m.get_split_features(features_indices, best_attribute_indices)
            decision_tree[best_attribute_indices][i] = fit_tree(x_subdata, y_subdata, hesitation, tree_max_depth, depth+1, classes, features_indices, name_features, sub_degree, sub_all_memberships, n_centroids, all_centroids)
    return decision_tree


############################################################
############################################################
############               PREDICT              ############
############################################################
############################################################


# @jit(cache=cache, nopython=True)
def predict_all(tree_, X, classes_, S, n_centroids, all_centroids, return_leafs):

    nof_objects = X.shape[0]
    nof_classes = len(classes_)
    result = np.zeros((nof_objects, nof_classes))
    for i in range(nof_objects):
        j = np.array([i])
        result[i] = predict_single(tree_, X[j], classes_, S, n_centroids, all_centroids, return_leafs)
    return result


# @jit(cache=cache, nopython=True)
def predict_single(node_tree_results, x, classes_, S, n_centroids, all_centroids, return_leafs):
    """
    function classifies a single object according to the trained tree
    """
    nof_classes = len(classes_)
    summed_prediction = np.zeros(nof_classes)
    if isinstance(node_tree_results, dict):
        idx_feature = list(node_tree_results.keys())[0]
        second_dict = node_tree_results[idx_feature]
        keys = get_reach_node(idx_feature, x, n_centroids, all_centroids, S)
        for k in keys:
            next_node = second_dict[k]
            summed_prediction += predict_single(next_node, x, classes_, S, n_centroids, all_centroids, return_leafs)

    else:
        if return_leafs:
            summed_prediction = node_tree_results
        else:
            max_index = np.argmax(node_tree_results)
            summed_prediction[max_index] = node_tree_results[max_index]
    return summed_prediction


@jit(cache=cache, nopython=True)
def get_reach_node(idx_feature, x, n_centroids, all_centroids, S):
    xi = x[:, np.array([idx_feature])]
    n_centroid = n_centroids[idx_feature]
    centroids = all_centroids[:n_centroid, np.array([idx_feature])]
    issan = all_centroids[-1][idx_feature]
    if issan == 0:
        F_u = m.lisan_membership(xi, centroids)
    else:
        ei = np.array([((centroids[i + 1][0] - centroids[i][0]) / S) for i in range(centroids.shape[0] - 1)])
        F_u = m.lianxu_membership(xi, centroids, ei)
    keys = np.where(F_u > 0)[1]
    return keys


def get_tree_weight(tree_list, num):
    trees = []
    weights = []
    for i in range(num):
        trees.append(tree_list[i][0])
        weights.append(tree_list[i][1])
    emax, emin = np.max(weights), np.min(weights)
    marg = (emax - emin) / 4
    for j in range(num):
        if weights[j] <= (emin + marg):
            trees[j].weight_tree = 1
        else:
            trees[j].weight_tree = (emax + marg - weights[j]) / (emax - emin)
    return trees


def get_tree_vote_prediction(prediction):
    prediction_max = np.max(prediction, axis=1)
    for i in range(len(prediction)):
        prediction[i] = np.where(prediction[i] >= prediction_max[i], prediction[i], 0)
    return prediction