####
#
# The MIT License (MIT)
#
# Copyright 2018 Eric Bach <eric.bach@aalto.fi>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
####

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