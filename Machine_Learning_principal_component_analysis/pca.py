import numpy as np

class PCA:
    """
    Implementation of Principal Component Analysis (PCA) for dimensionality reduction.
    """

    def __init__(self):
        """
        Initializes PCA object.
        """
        self._x_mean = None
        self._eigenvectors = None

    def transform(self, X, new_features):
        """
        Transforms the shape of the input data.

        Args:
            X: Training set as a NumPy array with shape (n_samples, n_features).
            new_features: New dimensionality.

        Returns:
            Transformed data with shape (n_samples, new_features).

        Raises:
            ValueError: If new_features is greater than or equal to n_features.
        """
        n_samples, n_features = X.shape
        if new_features >= n_features:
            raise ValueError("new_features should be lower...")
        # Compute the mean vector and store it for the inference stage
        self._x_mean = np.mean(X, axis=0)
        # Removing the mean
        X_norm = X - self._x_mean
        # Compute the covariance matrix with the eigen-trick
        covariance_matrix = X_norm.T @ X_norm
        # Compute eigenvectors of the covariance matrix
        eigval, eigvec = np.linalg.eig(covariance_matrix)
        # Sort them (decreasing order w.r.t. eigenvalues)
        sorted_indexes = np.argsort(eigval)[::-1]
        sorted_eigval, sorted_eigvec = eigval[sorted_indexes], eigvec[:, sorted_indexes]
        self._eigenvectors = sorted_eigvec
        # Project the data into the selected principal components
        new_X = X_norm @ self._eigenvectors[:, :new_features]
        return new_X
