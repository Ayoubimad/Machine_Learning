from sklearn.model_selection import train_test_split

from datasets.dataset import *
from pca import PCA


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
    pca = PCA()
    new_x_train = pca.transform(X=x_train, new_features=100)
    print(new_x_train.shape)


if __name__ == '__main__':
    test('easy_dataset.txt')
    test('dataset_w5a.txt')
    test('dataset_w6a.txt')
    test('dataset_w8a.txt')
