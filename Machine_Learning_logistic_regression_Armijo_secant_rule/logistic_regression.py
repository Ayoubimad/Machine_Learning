import random
import numpy as np

class LogisticRegression:
    def __init__(self):
        """
        Initializes LogisticRegression object.
        
        Attributes:
            w: Model weights.
            c: Intercept.
        """
        self.w = None
        self.c = None

    def loss(self, x, y, w, c, _lambda):
        """
        Calculates the logistic loss function.

        Args:
            x: Data in np.array format with shape (n_samples, n_features).
            y: Labels in np.array format with shape (n_samples,).
            w: Weights associated with each feature with shape (n_features,).
            c: Real number "intercept".
            _lambda: Hyperparameter for L2 regularization.

        Returns:
            Loss value.
        """
        loss = -y * (np.dot(x, w) + c)
        loss = 1 + np.exp(loss)
        loss = np.log(loss)
        loss = np.mean(loss) + (_lambda / 2) * (np.linalg.norm(np.append(w, c)) ** 2)
        return loss

    def gradient(self, x, y, w, c, _lambda):
        """
        Calculates the gradient of the logistic loss function.

        Args:
            x: Data in np.array format with shape (n_samples, n_features).
            y: Labels in np.array format with shape (n_samples,).
            w: Weights associated with each feature with shape (n_features,).
            c: Real number "intercept".
            _lambda: Hyperparameter for L2 regularization.

        Returns:
            Gradient of the loss function split into two parts: gradient_w and gradient_c.
        """
        n_samples = x.shape[0]
        predictions = np.dot(x, w) + c
        error = predictions - y
        gradient_w = (1 / n_samples) * np.dot(x.T, error) + (2 * _lambda * w)
        gradient_c = (1 / n_samples) * np.sum(error)
        return gradient_w, gradient_c

    def rho_secants_rule(self, prev_w, w, prev_gradient, gradient):
        """
        Calculates the rho parameter based on the secant rule.

        Args:
            prev_w: Previous weights including intercept c.
            w: Weights including intercept c.
            prev_gradient: Previous gradient including intercept c.
            gradient: Gradient including intercept c.

        Returns:
            Rho parameter based on secant rule.
        """
        s_k = w - prev_w
        y_k = gradient - prev_gradient
        return np.dot(s_k, s_k) / np.dot(s_k, y_k)

    def armijo_backtracking(self, x, y, w, c, step_size, sigma, beta, gradient_w, gradient_c, direction_c,
                            direction_w, _lambda):
        """
        Performs Armijo line search for step size determination.

        Args:
            x: Data.
            y: Labels.
            w: Weights.
            c: Intercept.
            step_size: Initial step size.
            sigma: Threshold parameter.
            beta: Step size reduction factor.
            gradient_w: Gradient of weights.
            gradient_c: Gradient of intercept.
            direction_c: Search direction for intercept.
            direction_w: Search direction for weights.
            _lambda: Hyperparameter for L2 regularization.

        Returns:
            Determined step size (alpha).
        """
        alpha = step_size
        while self.loss(x, y, w + alpha * direction_w, c + alpha * direction_c, _lambda) > \
                self.loss(x, y, w, c, _lambda) + sigma * alpha * np.dot(np.append(gradient_w, gradient_c),
                                                                        np.append(direction_w, direction_c)):
            alpha = alpha * beta
        return alpha

    def direction(self, rho, gradient_w, gradient_c):
        """
        Calculates the search direction based on rho.

        Args:
            rho: Rho parameter.
            gradient_w: Gradient of weights.
            gradient_c: Gradient of intercept.

        Returns:
            Search direction for weights and intercept.
        """
        return -rho * gradient_w, -rho * gradient_c

    def fit(self, x_train, y_train, tao=0.01, beta=0.5, sigma=0.1, step_size=0.1, rho_min=0.1, rho_max=0.9,
            epochs=100, _lambda=0.2, verbose=False):
        """
        Training procedure based on gradient descent.

        Args:
            verbose: Print loss.
            x_train: Training data.
            y_train: Training labels.
            tao: Tolerance for stopping criteria.
            beta: Reduction factor for step size.
            sigma: Threshold parameter for Armijo backtracking.
            step_size: Initial step size.
            rho_min: Minimum value for rho.
            rho_max: Maximum value for rho.
            epochs: Maximum number of training iterations.
            _lambda: Hyperparameter for L2 regularization.
        """
        n_samples, n_features = x_train.shape
        self.w = np.random.rand(n_features, ) * 0.001
        self.c = random.randint(0, 10)
        rho = random.uniform(rho_min, rho_max)
        gradient_w, gradient_c = self.gradient(x_train, y_train, self.w, self.c, _lambda)
        direction_w, direction_c = self.direction(rho, gradient_w, gradient_c)

        for i in range(epochs):
            if np.linalg.norm(np.append(gradient_w, gradient_c)) < tao:
                break
            alpha = self.armijo_backtracking(x_train, y_train, self.w, self.c, step_size, sigma,
                                             beta, gradient_w, gradient_c, direction_c, direction_w, _lambda)
            gradient_w_prev = gradient_w
            gradient_c_prev = gradient_c
            w_prev = self.w
            c_prev = self.c
            self.w = self.w + alpha * direction_w
            self.c = self.c + alpha * direction_c
            gradient_w, gradient_c = self.gradient(x_train, y_train, self.w, self.c, _lambda)
            rho = self.rho_secants_rule(np.append(w_prev, c_prev), np.append(self.w, self.c),
                                        np.append(gradient_w_prev, gradient_c_prev), np.append(gradient_w, gradient_c))
            direction_w, direction_c = self.direction(rho, gradient_w, gradient_c)
            if verbose and i % 10 == 0:
                loss = self.loss(x_train, y_train, self.w, self.c, _lambda)
                print("Loss:", loss)

    def predict(self, x_test):
        """
        Makes predictions on input data.

        Args:
            x_test: Data to be predicted, shape=(n_test_examples, n_features).

        Returns:
            Predictions in {-1, 1}.
        """
        predictions = np.dot(x_test, self.w) + self.c
        predictions = 1 / (1 + np.exp(-predictions))  # sigmoid function
        predictions = (predictions >= 0.5).astype(int)
        predictions[predictions == 0] = -1
        return predictions
