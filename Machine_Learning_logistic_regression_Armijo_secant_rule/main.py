import time

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from datasets.dataset import *
from logistic_regression import LogisticRegression


def test(dataset_name):
    """
    Load a dataset, split it into training and test sets, train a logistic regression classifier,
    and print the accuracy on the test set.

    Parameters
    ----------
    dataset_name: str
        Name of the dataset to load and test.
    """
    y_train, x_train = load_dataset(dataset_name)
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    classifier = LogisticRegression()
    init = time.time()
    classifier.fit(x_train, y_train)
    last = time.time()
    print(str(last - init))
    predictions = classifier.predict(x_test)
    accuracy = np.sum((y_test == predictions)) / x_test.shape[0]
    print(dataset_name + ", accuracy: " + str(accuracy))


def test_sklearn_dataset(n_samples=50000):
    np.random.seed(0)
    x_train, y_train = make_classification(n_samples=n_samples, n_features=300,
                                           n_classes=2, n_clusters_per_class=1,
                                           n_redundant=0)
    y_train[y_train == 0] = -1
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    classifier = LogisticRegression()
    init = time.time()
    classifier.fit(x_train, y_train)
    last = time.time()
    print(str(last - init))
    predictions = classifier.predict(x_test)
    accuracy = np.sum((y_test == predictions)) / x_test.shape[0]
    print("accuracy: " + str(accuracy))


if __name__ == '__main__':
    # test('easy_dataset.txt')
    test_sklearn_dataset()
    # test('dataset_w5a.txt')
    # test('dataset_w6a.txt')
    #test('dataset_w8a.txt')
