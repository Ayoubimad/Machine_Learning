import os
from pathlib import Path

import numpy as np

# Adding dataset file paths to the list
BASE_DIR = Path(__file__).resolve().parent
dataset_w_8_a = os.path.join(BASE_DIR, "dataset_w8a.txt")
dataset_w_5_a = os.path.join(BASE_DIR, "dataset_w5a.txt")
dataset_w_6_a = os.path.join(BASE_DIR, "dataset_w6a.txt")
easy_dataset = os.path.join(BASE_DIR, "easy_dataset.txt")
datasets = {
    'dataset_w8a.txt': dataset_w_8_a,
    'dataset_w5a.txt': dataset_w_5_a,
    'dataset_w6a.txt': dataset_w_6_a,
    'easy_dataset.txt': easy_dataset,
}


def load_dataset(filename, num_features=300):
    """
    Load a dataset from a file and return it as numpy arrays.

    Parameters
    ----------
    filename: str
        Name of the dataset file to load.
    num_features: int, optional
        Number of features per sample.

    Returns
    -------
    y_train: np.array
        Labels with shape (n_samples,).
    x_train: np.array
        Data features with shape (n_samples, num_features).
    """
    labels = []
    data = []

    with open(datasets[filename], 'r') as file:
        for line in file:
            elements = line.strip().split()
            label = int(elements[0])
            labels.append(label)

            if len(elements) > 1:
                features = [0] * num_features
                for element in elements[1:]:
                    feature_index, feature_value = map(int, element.split(':'))
                    features[feature_index - 1] = feature_value
                data.append(features)
            else:
                features = [0] * num_features
                data.append(features)

    y_train = np.array(labels)
    x_train = np.array(data)
    return y_train, x_train
