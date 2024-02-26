import numpy as np


class Classifier:
    """
    Implementation of Support Vector Machines for binary classification.
    This version does not use the kernel trick!
    """

    def __init__(self):
        """
        Constructor of the class
        Initializes the weight vector to None.
        """
        self._w = None

    def fit(self, X, Y, _lambda=0.5, epochs=50):
        """
        Training procedure based on gradient descent via the Pegasos algorithm.

        :param epochs: maximum iterations for gradient descent
        :param _lambda: parameter used in the Pegasos algorithm
        :param X: data np.array of shape (n_samples, n_features)
        :param Y: labels np.array of shape (n_samples,) {-1, 1}
        """
        n_samples, n_features = X.shape

        # Initialization
        self._w = np.zeros((n_features,))
        t = 0
        for e in range(epochs):
            for j in range(n_samples):
                t = t + 1
                n_t = 1 / (_lambda * t)
                X_j, y_j = X[j, :], Y[j]

                # Compute the dot product of w and X_j
                w_dot_X_j = np.dot(self._w, X_j)

                if (y_j * w_dot_X_j) < 1:
                    # Update weights based on gradient
                    self._w = (1 - n_t) * self._w + n_t * y_j * X_j
                else:
                    # Update weights with L2 regularization
                    self._w = (1.0 - n_t * _lambda) * self._w

    def predict(self, X):
        """
        Make predictions on input data.

        :param X: data to be predicted. np.array of shape (n_test_examples, n_features)

        :return: predictions in {-1, 1}
        """
        predictions = np.dot(X, self._w)

        # Transform predictions into binary labels {-1, 1}
        predictions = (predictions >= 0).astype(int)
        predictions[predictions == 0] = -1
        return predictions
