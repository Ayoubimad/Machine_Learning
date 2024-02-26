import numpy as np


class DecisionStump:
    """
    Function that models a WeakClassifier
    """

    def __init__(self):
        # initialize a few stuff
        self._dim = None
        self._threshold = None
        self._label_above_split = None

    def fit(self, X: np.ndarray, Y: np.ndarray):
        n, d = X.shape
        possible_labels = np.unique(Y)

        # select random feature (see np.random.choice)
        self._dim = np.random.choice(a=range(0, d))

        # select random split (see np.random.uniform)
        M, m = np.max(X[:, self._dim]), np.min(X[:, self._dim])
        self._threshold = np.random.uniform(low=m, high=M)

        # select random verse (see np.random.choice)
        self._label_above_split = np.random.choice(a=possible_labels)

    def predict(self, X: np.ndarray):
        num_samples = X.shape[0]
        y_pred = np.zeros(shape=num_samples)
        y_pred[X[:, self._dim] >= self._threshold] = self._label_above_split
        y_pred[X[:, self._dim] < self._threshold] = -1 * self._label_above_split

        return y_pred
