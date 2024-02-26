import os
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

plt.ion()

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

    if filename == 'easy_dataset.txt':
        return load_easy_datatset()

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


def load_easy_datatset():
    x_train = []
    y_train = []

    with open(easy_dataset, "r") as file:
        for line in file:
            line = line.strip().split()
            y_train.append(int(line[0]))
            x_train.append([float(value) for value in line[1:]])

    # Converte le liste in array numpy
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    return np.array(y_train), np.array(x_train)


def plot_2D_dataset(x_train, y_train):
    positive_points = x_train[y_train == 1]
    negative_points = x_train[y_train == -1]

    plt.figure(figsize=(8, 6))
    plt.scatter(positive_points[:, 0], positive_points[:, 1], c='b', label='Class 1')
    plt.scatter(negative_points[:, 0], negative_points[:, 1], c='r', label='Class -1')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(loc='best')
    plt.title('2D Dataset Visualization')
    plt.show()
