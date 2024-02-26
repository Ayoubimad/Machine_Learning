# PCA (Principal Component Analysis) Implementation

## Overview
This Python module implements Principal Component Analysis (PCA) for dimensionality reduction. PCA is used to reduce the dimensionality of high-dimensional data while preserving most of the original information. It achieves this by transforming the data into a new coordinate system, where the axes (principal components) are ordered by the amount of variance in the data they explain.

## Usage
To use this PCA implementation, follow these steps:

1. **Instantiate the PCA object**: Create an instance of the PCA class.
    ```python
    pca = PCA()
    ```

2. **Transform the Data**: Use the `transform()` method to perform dimensionality reduction on your dataset.
    ```python
    new_X = pca.transform(X, new_features)
    ```

    - `X`: The training set as a NumPy array with shape `(n_samples, n_features)`.
    - `new_features`: The desired dimensionality of the transformed data.

## Example
```python
import numpy as np
from pca import PCA

# Create sample data
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Instantiate PCA object
pca = PCA()

# Perform dimensionality reduction
new_X = pca.transform(X, new_features=2)

print(new_X)
```

## Dependencies
- NumPy

## Class Methods
### `transform(X, new_features)`
Transform the shape of `X` (n_samples, n_features) into (n_samples, new_features).

- **Parameters**:
  - `X`: Training set as a NumPy array with shape `(n_samples, n_features)`.
  - `new_features`: New dimensionality.

- **Returns**:
  - `new_X`: Transformed data with shape `(n_samples, new_features)`.

## Author
Ayoub Imad 
