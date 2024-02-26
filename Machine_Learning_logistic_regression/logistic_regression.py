import numpy as np


def sigmoid(x):
    """
    Element-wise sigmoid function

    Parameters
    ----------
    x: np.array
        a numpy array of any shape

    Returns
    -------
    np.array
        an array having the same shape of x.
    """

    # Sigmoid activation function, used to map real number to (0,1)
    return 1 / (1 + np.exp(-x))


def loss_function(y_true, y_pred):
    """
    The binary crossentropy loss.

    Parameters
    ----------
    y_true: np.array
        real labels in {0, 1}. shape=(n_examples,)
    y_pred: np.array
        predicted labels in [0, 1]. shape=(n_examples,)

    Returns
    -------
    float
        the value of the binary crossentropy.
    """

    # Compute the average binary cross-entropy between
    # y_true (targets) and y_pred (predictions)
    # y_pred = np.clip(y_pred, eps, 1 - eps)
    loss_value = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return np.mean(loss_value)


def dloss_dw(y_true, y_pred, X):
    """
    Derivative of loss function w.r.t. weights.

    Parameters
    ----------
    y_true: np.array
        real labels in {0, 1}. shape=(n_examples,)
    y_pred: np.array
        predicted labels in [0, 1]. shape=(n_examples,)
    X: np.array
        predicted data. shape=(n_examples, n_features)

    Returns
    -------
    np.array
        derivative of the loss function w.r.t weights.
        Has shape=(n_features,)
    """

    N = X.shape[0]

    # Compute the derivative of the loss function w.r.t. weights.
    return np.dot(X.T, y_pred - y_true) / N


class Classifier:
    def __init__(self):
        """
        Constructor method to initialize the classifier.
        """
        self._w = None

    def __armijo_backtracking(self, X, Y, gradient, alpha, beta, sigma):
        predictions = np.dot(X, self._w)
        labels = Y
        loss = loss_function(y_true=labels, y_pred=predictions)
        while loss_function(Y, sigmoid(np.dot(X, self._w - alpha * gradient))) > loss - alpha * sigma * np.dot(gradient,
                                                                                                               gradient):
            alpha *= beta
        return alpha

    def fit(self, X, Y, n_epochs, alpha=1.0, beta=0.5, sigma=0.5):
        """
        Implement the gradient descent training procedure.

        Parameters
        ----------
        X: np.array
            data. shape=(n_examples, n_features)
        Y: np.array
            labels. shape=(n_examples,)
        n_epochs: int
            number of gradient updates.
        alpha: float
                default step size gradient descend
        beta: float (0,1)
                parameter used to update the alpha in each step of "Backtracking line search with Armijo rule"
        sigma: float (0,1)
                parameter used in "Backtracking line search with Armijo rule"
        """
        n_samples, n_features = X.shape

        # Initialize weights with small random values
        self._w = np.random.randn(n_features) * 0.001

        # gradient descent procedure
        for e in range(n_epochs):
            predictions = np.dot(X, self._w)
            predictions = sigmoid(predictions)
            gradient = dloss_dw(Y, predictions, X)
            alpha = self.__armijo_backtracking(X=X, Y=Y, gradient=gradient, alpha=alpha, beta=beta, sigma=sigma)
            self._w = self._w - alpha * gradient

    def predict(self, X):
        """
        Make predictions on input data.

        Parameters
        ----------
        X: np.array
            data to be predicted. shape=(n_test_examples, n_features)

        Returns
        -------
        prediction: np.array
            prediction in {0, 1}.
            Shape is (n_test_examples,)
        """
        predictions = np.dot(X, self._w)
        predictions = sigmoid(predictions)
        predictions = (predictions >= 0.5).astype(int)
        predictions[predictions == 0] = -1
        return predictions
