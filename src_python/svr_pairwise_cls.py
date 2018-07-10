import numpy as np

from sklearn.svm import SVR

class SVRPairwise (SVR):
    def score (self, X, pairs, sample_weight = None, normalize = True):
        """
        :param X: array-like, data description
            feature-vectors: shape = (n, d)
            -- or --
            kernel-matrix: shape = (l, n),

            with n being the number of test molecules.

        :param pairs: List of tuples (i,j) of length p containing the pairwise relations:
            i elutes before j, ... --> [(i,j), (u,v), ...]

        :param sample_weight: parameter is ignored at the moment!

        :param normalize: boolean, should the number of correctly predicted
            examples be divided by the total number of examples, to get the
            accuracy.

        :return: scalar, pairwise prediction accuracy (if normalize = True)
        """
        return self.score_using_prediction (self.predict (X), pairs, normalize)

    def score_using_prediction (self, Y, pairs, normalize = True):
        """
        :param Y: array, predicted retention times

        :param pairs: List of tuples (i,j) of length p containing the pairwise relations:
            i elutes before j, ... --> [(i,j), (u,v), ...]

        :param normalize: boolean, should the number of correctly predicted
            examples be divided by the total number of examples, to get the
            accuracy.

        :return: scalar, pairwise prediction accuracy (if normalize = True)
        """
        if len (pairs) == 0:
            return (0.0)

        tp = 0.0
        for idx, pair in enumerate (pairs):
            if Y[pair[0]] - Y[pair[1]] == 0 and pair[0] != pair[1]:
                tp += 0.5
            else:
                tp += (np.sign (Y[pair[0]] - Y[pair[1]]) == -1) # equivalent: Y[pair[0]] < Y[pair[1]]

        if normalize:
            return tp / len (pairs)
        else:
            return tp

    def map_values (self, X):
        """
        Function needed for compatibility.

        :param X: array-like, data description
            feature-vectors: shape = (n, d)
            -- or --
            kernel-matrix: shape = (l, n),

            with n being the number of test molecules.

        :return: Regression value for each test example
        """

        return self.predict (X)