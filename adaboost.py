import numpy as np


class AdaBoost(object):
    """
    AdaBoost, http://en.wikipedia.org/wiki/AdaBoost
    """
    def __init__(self, predictions, labels, beta_fraction=1.e-4,
                 maxiter=1000000):

        # basic checks
        assert labels.shape[0] == predictions.shape[1], \
            '1D label array must be same size as Ncols in predictions'
        assert np.in1d(predictions.ravel(), np.array[-1, 1]), \
            'Predicted class labels must be either -1 or 1'
        assert np.in1d(labels, np.array[-1, 1]), \
            'Predicted class labels must be either -1 or 1'

        # define
        self.labels = labels
        self.alphas = np.zeros(maxiter)
        self.weights = np.ones_like(labels) / predictions.shape[1]
        self.predictions = predictions
        self.error_rates = np.zeros(maxiter)
        self.classifiers = np.zeros(maxiter, dtype=np.int)

        # get indications
        self.indications = self.get_indications()

        # run AdaBoost
        self.run(maxiter, beta_fraction)

    def get_indications(self):
        """
        Get indications where predicted labels are wrong
        """
        indications = np.zeros_like(self.predictions)
        for i in range(self.predictions.shape[0]):
            ind = np.where(self.predictions[i, :]-self.labels == 0.0)[0]
            indications[i, ind] = 1.0

        return indications

    def get_weighted_error_rate(self):
        """
        Return current weighted error rate.
        """
        eps = self.weights[None, :] * indications
        return np.sum(eps, axis=1)

    def get_alpha(self, error_rate, func='default'):
        """
        Return alpha coefficient, given error rate
        """
        return 0.5 * np.log((1. - error_rate) / error_rate)

    def update_weights(self, alpha, ind):
        """
        Update the running weights
        """
        inside = -alpha * self.labels * self.predictions[ind, :]
        new_weights = self.weights * np.exp(inside)
        self.weights = new_weights / np.sum(new_weights)

    def run(self, maxiter, beta_fraction):
        """
        Run AdaBoost, return boosted predictions
        """
        for i in range(maxiter):

            # get best error rate, assess stopping criterion
            errors = self.get_weighted_error_rate()
            abs_diffs = np.abs(0.5 - errors)
            current_best = np.max(abs_diffs)
            ind = np.where(abs_diffs == current_best)[0]
            if i == 0:
                beta = current_best * beta_fraction
            if current_best < beta:
                self.alphas = self.alphas[:i]
                self.error_rates = self.error_rates[:i]
                self.classifiers = self.classifiers[:i]
                break

            # assign
            self.classifiers[i] = ind
            self.error_rates[i] = errors[ind]
            assert errors[ind] < 0.5, \
                'Error rate is greater than 0.5, what gives?'

            # get alpha and update weights
            self.alphas[i] = self.get_alpha(self.error_rates[i])
            self.update_weights(self.alphas[i], self.classifiers[i])

        if (i == maxiter - 1):
            print 'Warning: ran for maxiters'

        new = self.alphas[:, None] * self.predictions[self.classfiers, :]
        return np.sign(np.sum(new), axis=0)
