import numpy as np
from decision_stump import DecisionStump


class Classifier:
    """
    Function that models a Adaboost classifier
    """

    def __init__(self, n_learners: int, n_max_trials: int = 200):
        """
        Model constructor

        Parameters
        ----------
        n_learners: int
            number of weak classifiers.
        """

        # initialize a few stuff
        self.n_learners = n_learners
        self.learners = []
        self.alphas = np.zeros(shape=n_learners)
        self.n_max_trials = n_max_trials

    def fit(self, X: np.ndarray, Y: np.ndarray, verbose: bool = False):
        """
        Trains the model.

        Parameters
        ----------
        X: ndarray
            features having shape (n_samples, dim).
        Y: ndarray
            class labels having shape (n_samples,).
        verbose: bool
            whether or not to visualize the learning process.
            Default is False
        """

        # some inits
        n, d = X.shape
        if d != 2:
            verbose = False  # only plot learning if 2 dimensional

        possible_labels = np.unique(Y)

        # only binary problems please
        assert possible_labels.size == 2, 'Error: data is not binary'

        # initialize the sample weights as equally probable
        sample_weights = np.ones(shape=n) / n

        # start training
        for l in range(self.n_learners):

            # choose the indexes of 'difficult' samples (np.random.choice)
            cur_idx = np.random.choice(a=range(0, n), size=n, replace=True, p=sample_weights)

            # extract 'difficult' samples
            cur_X = X[cur_idx]
            cur_Y = Y[cur_idx]

            # search for a weak classifier
            error = 1
            n_trials = 0
            cur_wclass = None
            y_pred = None

            while error > 0.5:

                cur_wclass = DecisionStump()
                cur_wclass.fit(cur_X, cur_Y)
                y_pred = cur_wclass.predict(cur_X)

                # compute error
                error = np.sum(sample_weights[cur_idx[cur_Y != y_pred]])

                n_trials += 1
                if n_trials > self.n_max_trials:
                    # initialize the sample weights again
                    sample_weights = np.ones(shape=n) / n

            # save weak learner parameter
            self.alphas[l] = alpha = np.log((1 - error) / error) / 2

            # append the weak classifier to the chain
            self.learners.append(cur_wclass)

            # update sample weights
            sample_weights[cur_idx[cur_Y != y_pred]] *= np.exp(alpha)
            sample_weights[cur_idx[cur_Y == y_pred]] *= np.exp(-alpha)
            sample_weights /= np.sum(sample_weights)

            if verbose:
                self._plot(cur_X, y_pred, sample_weights[cur_idx],
                           self.learners[-1], l)

    def predict(self, X: np.ndarray):
        """
        Function to perform predictions over a set of samples.

        Parameters
        ----------
        X: ndarray
            examples to predict. shape: (n_examples, d).

        Returns
        -------
        ndarray
            labels for each examples. shape: (n_examples,).

        """
        num_samples = X.shape[0]

        pred_all_learners = np.zeros(shape=(num_samples, self.n_learners))

        for l, learner in enumerate(self.learners):
            pred_all_learners[:, l] = learner.predict(X)

        # weight for learners efficiency
        pred_all_learners *= self.alphas

        # compute predictions
        pred = np.sign(np.sum(pred_all_learners, axis=1))

        return pred
