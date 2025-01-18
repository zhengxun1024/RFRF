import numpy as np
from joblib import Parallel, delayed
import threading
from sklearn.metrics import accuracy_score

from . import misc_functions as m
from . import tree


############################################################
############################################################
################ DecisionTreeClassifier Class  #############
############################################################
############################################################

class DecisionTreeClassifier:
    """
    This is the decision tree classifier.
    """
    def __init__(self, criterion='gini', max_depth=None, min_samples_split=2):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

        self.tree_ = None
        self.weight_ = 1

    def fit(self, X, y):
        """
        the DecisionTreeClassifier.fit() function with a similar appearance to that of sklearn
        """
        features_indices = np.arange(X.shape[1])
        t_centroids, n_centroids, t_memberships = m.memberships(X, self.n_clusters, self.S)
        X_degree = np.ones((X.shape[0], 2))

        self.classes_ = np.unique(y)
        self.n_centroids = n_centroids
        self.all_centroids = t_centroids

        self.tree_ = tree.fit_tree(X, y, self.max_depth, self.classes_, features_indices, X_degree, t_memberships, n_centroids, t_centroids)

    def predict(self, X, return_leafs=False):
        """
        The DecisionTreeClassifier.predict_proba() function with a similar appearance to the of sklearn
        """
        result = self.predict_proba(X, return_leafs)
        res = np.argmax(result, axis=1)
        y_pred = np.array([self.classes_[i] for i in res])
        return y_pred

    def predict_proba(self, X, return_leafs=False):
        """
        The DecisionTreeClassifier.predict_proba() function with a similar appearance to the of sklearn
        """
        result = tree.predict_all(self.tree_, X, self.classes_, self.S, self.n_centroids, self.all_centroids, return_leafs)

        return result



############################################################
############################################################
################ RandomForestClassifier Class  #############
############################################################
############################################################

class RandomForestClassifier(DecisionTreeClassifier):
    def __init__(self, n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, bootstrap=True, hesitation=1.0, S=5, n_clusters=6):
        super().__init__(criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split)
        self.n_estimators_ = n_estimators
        self.criterion = criterion
        self.estimators_ = []
        self.bootstrap = bootstrap
        self.hesitation = hesitation
        self.S = S
        self.n_clusters = n_clusters

    def check_input(self, X):
        # Check if array is not already np ndarray
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        return X

    def _choose_objects(self, X, y):
        """
        function builds a sample of the same size as the input data, but chooses the objects with replacement
        according to the given probability matrix
        """
        nof_objects = y.shape[0]
        objects_indices = np.arange(nof_objects)
        objects_chosen = np.random.choice(objects_indices, nof_objects, replace=True)
        X_chosen = X[objects_chosen, :]
        y_chosen = y[objects_chosen, :]

        objects_oob = np.delete(objects_indices, objects_chosen, axis=0)
        X_oob = X[objects_oob, :]
        y_oob = y[objects_oob, :]
        return X_chosen, X_oob, y_chosen, y_oob

    def _fit_single_tree(self, X, y):
        if self.bootstrap:
            X_chosen, X_oob, y_chosen, y_oob = self._choose_objects(X, y)
        else:
            X_chosen, X_oob, y_chosen, y_oob = X, X, y, y

        tree = DecisionTreeClassifier(criterion=self.criterion,
                                      name_features=self.name_features,
                                      hesitation=self.hesitation,
                                      max_depth=self.max_depth,
                                      S=self.S,
                                      n_clusters=self.n_clusters)
        tree.fit(X_chosen, y_chosen)
        y_pred = tree.predict(X_oob)
        error_acc = 1 - accuracy_score(y_oob, y_pred)

        return tree, error_acc

    def fit(self, X, y):
        """
        The RandomForestClassifier.fit() function with a similar appearance to that of sklearn
        """
        self.name_features = np.array(X.columns)

        X = self.check_input(X)
        y = self.check_input(y)

        self.classes_ = np.unique(y)

        # tree_list = [self._fit_single_tree(X, y) for i in range(self.n_estimators_)]
        tree_list = Parallel(n_jobs=-1)(delayed(self._fit_single_tree)(X, y)for i in range(self.n_estimators_))

        trees = tree.get_tree_weight(tree_list, self.n_estimators_)

        self.estimators_ = trees

        return self

    def predict_single_tree(self, predict, weight_tree, X, out, lock, vote):
        # let all the trees vote - the function pick_best find the best class from each tree, and gives zero probability to all the others
        prediction = predict(X)

        with lock:
            if vote == 'tree':
                prediction = tree.get_tree_vote_prediction(prediction)
                prediction = prediction * weight_tree
                out += prediction
            elif vote == 'leaf':
                prediction = prediction * weight_tree
                out += prediction

    def predict_proba(self, X, vote='tree'):
        """
        The RandomForestClassifier.predict() function with a similar appearance to that of sklearn
        """
        X = self.check_input(X)
        proba = np.zeros((X.shape[0], len(self.classes_)), dtype=np.float64)
        lock = threading.Lock()
        Parallel(n_jobs=-1, backend="threading")(delayed(self.predict_single_tree)
                              (tree.predict_proba, tree.weight_tree, X, proba, lock, vote) for tree in self.estimators_)

        # for i, tree in enumerate(self.estimators_):
        #     self.predict_single_tree(tree.predict_proba, tree.weight_tree, X, proba, lock, vote)

        return proba

    def predict(self, X, vote='tree'):
        y_pred_inds = np.argmax(self.predict_proba(X, vote), axis=1)
        y_pred = np.array([self.classes_[i] for i in y_pred_inds])
        return y_pred

    def score(self, X, y):
        y_pred = self.predict(X)
        score = (y_pred == (y)).sum()/len(y)
        return score

    def __str__(self):
        sb = []
        do_not_print = ['estimators_', 'n_estimators', 'max_depth', 'hesitation', 'n_clusters']
        for key in self.__dict__:
            if key not in do_not_print:
                sb.append("{key}='{value}'".format(key=key, value=self.__dict__[key]))

        sb = 'IntuitionisticRandomForestClassifier(' + ', '.join(sb) + ')'
        return sb

