from sklearn.model_selection import train_test_split

from datasets.dataset import *
from svm import Classifier


def test(dataset_name):
    """
    Load a dataset, split it into training and test sets, train svm classifier,
    and print the accuracy on the test set.

    Parameters
    ----------
    dataset_name: str
        Name of the dataset to load and test.
    """
    y_train, x_train = load_dataset(dataset_name)
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    classifier = Classifier()
    classifier.fit(x_train, y_train, 50)
    predictions = classifier.predict(x_test)
    accuracy = np.sum((y_test == predictions)) / x_test.shape[0]
    print(dataset_name + ", accuracy: " + str(accuracy))


if __name__ == '__main__':
    test('dataset_w5a.txt')
    test('dataset_w6a.txt')
    test('dataset_w8a.txt')
