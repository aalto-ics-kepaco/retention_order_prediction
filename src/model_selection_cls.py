####
#
# The MIT License (MIT)
#
# Copyright 2017, 2018 Eric Bach <eric.bach@aalto.fi>
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

'''

Collection of methods that can be used to estimate the model hyper-parameters
of the pairwise Support Vector Regression (SVR) and Ranking Support Vector Machine
(RankSVM).

'''

import numpy as np
# Import some data structures
from collections import OrderedDict

# Import some helper function from the sklearn package
from sklearn.model_selection._search import ParameterGrid
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.base import clone
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel

# Allow the paralellization of the parameter search
from joblib import Parallel, delayed

# Import the retention order class
from retention_cls import retention_cls

# Import stuff from the rank svm
from rank_svm_cls import get_pairs_from_order_graph, minmax_kernel_mat, tanimoto_kernel_mat

from svr_pairwise_cls import SVRPairwise

def find_hparam_regression (estimator, X, y, param_grid, cv, scaler = None, n_jobs = 1):
    """
    Task: find the hyper-parameter from a set of parameters (param_grid),
          that performs best in an cross-validation setting for the given
          estimator.

    :param estimator: Estimator object, e.g. KernelRankSVC

    :param X: dictionary, (mol-id, system)-tuples as keys and molecular
              features as values:

              Example:
                {("M1", "S1"): feat_11, ...}

    :param y: dictionary, (mol-id, system)-tuples as keys and retention
              times as values

              Example:
                {("M1", "S1"): rt_11, ...}

    :param param_grid: dictionary, defining the grid-search space
        "C": Trade-of parameter for the SVM
        "gamma": width of the rbf/gaussian kernel
        ... etc. ...

        Example:
            {"C": [0.1, 1, 10], "gamma": [0.1, 0.25, 0.5, 1]}

    :param cv: cross-validation generator, see sklearn package, must be
               either a GroupKFold or GroupShuffleSplit object.

    :param scaler: scaler object, per feature scaler, e.g. MinMaxScaler

    :param n_jobs: int, number of jobs run in parallel. Parallelization
                   is performed over the cv-folds.

    :return: dictionary, containing combination of best parameters
                Example:
                    {"C": 1, "gamma": 0.25}

             dictionary, all parameter combinations with corresponding scores
                 Example:
                    [{"C": 1, "gamma": 0.25, "score": 0.98},
                     {"C": 1, "gamma": 0.50, "score": 0.94},
                     ...]

             scalar, number of pairs used to train the final model

             estimator object, fitted using the best parameters
    """
    if not len (X) == len (X) or len (X.keys() - y.keys()) or len (y.keys() - X.keys()):
        raise ValueError ("Keys-set for features and retentions times must be equal.")

    if not isinstance (estimator, SVRPairwise):
        raise ValueError ("Currently parameters can only be estimated for the support vector regression "
                          "class 'SVRPairwise'.")

    # Make a list of all combinations of parameters
    l_params = list (ParameterGrid (param_grid))
    param_scores = np.zeros ((len (l_params),))

    # Get all (mol-id, system)-tuples used for the parameter search
    keys = list (X.keys())

    mol_ids = list (zip (*keys))[0]
    cv_splits = cv.split (range (len (keys)), groups = mol_ids)

    # Precompute the training / test targets to save computation time as
    # we do not need to repeat this for several parameter settings.
    # cv_splits = cv.split (range (len (keys)))
    y_train_sets, y_test_sets = [], []
    X_train_sets, X_test_sets = [], []

    print ("Get pairs for hparam estimation: ", end = "", flush = True)
    for k_cv, (train_set, test_set) in enumerate (cv_splits):
        print ("%d " % k_cv, end = "", flush = True)

        # 0) Get keys (mol-id, system)-tuples, corresponding to the training
        #    and test sets.
        keys_train = [keys[idx] for idx in train_set]
        keys_test = [keys[idx] for idx in test_set]

        # Check for overlap of molecular ids, e.g. InChIs. Between training and test
        # molecular ids should not be shared, e.g. if they appear in different systems
        # at the same time.
        mol_ids_train = [mol_ids[idx] for idx in train_set]
        mol_ids_test = [mol_ids[idx] for idx in test_set]

        if set (mol_ids_train) & set (mol_ids_test):
            if isinstance (cv, GroupKFold) or isinstance(cv, GroupShuffleSplit):
                raise RuntimeError ("As grouped cross-validation is used the training "
                                    "and test molecules, i.e. mol_ids, are not allowed "
                                    "to overlap. This can happen if molecular structures "
                                    "are appearing in different systems. During the "
                                    "learning of hyper-parameter the training set should "
                                    "not contain any structure also in the test set.",
                                    set (mol_ids_train) & set (mol_ids_test))
            else:
                print ("Training and test keys overlaps.", set (mol_ids_train) & set (mol_ids_test))

        # 1) Extract the target values from y (train and test) using the keys
        y_train_sets.append (np.array([y[key] for key in keys_train]))
        y_test_sets.append (np.array([y[key] for key in keys_test]))

        # 2) Extract the features from X (train and test) using the keys
        X_train_sets.append (np.array ([X[key] for key in keys_train]))
        X_test_sets.append (np.array ([X[key] for key in keys_test]))

    print ("")

    for k_param, param in enumerate (l_params):
        fold_scores = Parallel (n_jobs = n_jobs, verbose = False)(
            delayed (_fit_and_score_regression)(param, clone (estimator),
                                                X_train_sets[k_cv], X_test_sets[k_cv],
                                                y_train_sets[k_cv], y_test_sets[k_cv],
                                                scaler)
            for k_cv in range (cv.get_n_splits()))

        param_scores[k_param] = np.mean (fold_scores)

    ## Fit model using the best parameters
    # Find the best params
    best_params = l_params[np.argmax (param_scores)].copy()

    # Fit the model using the best parameters
    best_estimator = clone (estimator)
    best_estimator.set_params (**_filter_params (best_params, best_estimator))

    X = np.array ([X[key] for key in keys])
    y = np.array ([y[key] for key in keys])

    if not scaler is None:
        X = scaler.transform (X)

    best_estimator.fit (X, y)

    # Combine the mean fold scores with the list of parameter sets
    for k_param, _ in enumerate (l_params):
        l_params[k_param]["score"] = param_scores[k_param]

    return best_params, l_params, -1, best_estimator

def find_hparan_ranksvm (estimator, X, y, param_grid, cv, pair_params, scaler = None, n_jobs = 1,
                         fold_score_aggregation = "weighted_average", all_pairs_as_test = True):
    """
    Task: find the hyper-parameter from a set of parameters (param_grid),
          that performs best in an cross-validation setting for the given
          estimator.

    :param estimator: Estimator object, e.g. KernelRankSVC

    :param X: dictionary, (mol-id, system)-tuples as keys and molecular
              features as values:

              Example:
                {("M1", "S1"): feat_11, ...}

    :param y: dictionary, (mol-id, system)-tuples as keys and retention
              times as values

              Example:
                {("M1", "S1"): rt_11, ...}

    :param param_grid: dictionary, defining the grid-search space
        "C": Trade-of parameter for the SVM
        "gamma": width of the rbf/gaussian kernel
        ... etc. ...

        Example:
            {"C": [0.1, 1, 10], "gamma": [0.1, 0.25, 0.5, 1]}

    :param cv: cross-validation generator, see sklearn package, must be
               either a GroupKFold or GroupShuffleSplit object.

    :param pair_params: dictionary, specifying parameters for the order graph:
        "ireverse": scalar, Should cross-system elution transitivity be included
            0: no, 1: yes
        "d_lower": scalar, minimum distance of two molecules in the elution order graph
                   to be considered as a pair.
        "d_upper": scalar, maximum distance of two molecules in the elution order graph
                   to be considered as a pair.
        "allow_overlap": scalar, Should overlap between the upper and lower sets
                         be allowed. Those overlaps originate from retention order
                         contradictions between the different systems.

    :param scaler: scaler object, per feature scaler, e.g. MinMaxScaler

    :param n_jobs: integer, number of jobs run in parallel. Parallelization is performed
        over the cv-folds. (default = 1)

    :fold_score_aggregation: string, (default = "weighted_average")

    :all_pairs_as_test: boolean, should all possible pairs (d_lower = 0, d_upper = np.inf)
        be used during the test. If 'False' than corresponding values are taking from the
        'pair_params' dictionary. (default = True)

    :return: dictionary, containing combination of best parameters
                Example:
                    {"C": 1, "gamma": 0.25}

             dictionary, all parameter combinations with corresponding scores
                 Example:
                    [{"C": 1, "gamma": 0.25, "score": 0.98},
                     {"C": 1, "gamma": 0.50, "score": 0.94},
                     ...]

             scalar, number of pairs used to train the final model

             estimator object, fitted using the best parameters
    """
    if not (isinstance (cv, GroupKFold) or isinstance (cv, GroupShuffleSplit)):
        raise ValueError ("Cross-validation generator must be either of "
                          "class 'GroupKFold' or 'GroupShuffleSplit'. "
                          "Provided class is '%s'." % cv.__class__.__name__)

    if len (X) != len (y) or len (X.keys() - y.keys()) or len (y.keys() - X.keys()):
        raise ValueError ("Keys-set for features and retentions times must "
                          "be equal.")

    # Make a list of all combinations of parameters
    l_params = list (ParameterGrid (param_grid))
    param_scores = np.zeros ((len (l_params),))

    # Get all (mol-id, system)-tuples used for the parameter search
    keys = list (X.keys())

    if len (l_params) > 1:
        mol_ids = list (zip (*keys))[0]
        cv_splits = cv.split (range (len (keys)), groups = mol_ids)

        # Precompute the training / test pairs to save computation time as
        # we do not need to repeat this for several parameter settings.
        pairs_train_sets, pairs_test_sets = [], []
        X_train_sets, X_test_sets = [], []
        n_pairs_test_sets = []

        print ("Get pairs for hparam estimation: ", end = "", flush = True)
        for k_cv, (train_set, test_set) in enumerate (cv_splits):
            print ("%d " % k_cv, end = "", flush = True)

            # 0) Get keys (mol-id, system)-tuples, corresponding to the training
            #    and test sets.
            keys_train = [keys[idx] for idx in train_set]
            keys_test = [keys[idx] for idx in test_set]

            # Check for overlap of molecular ids, e.g. InChIs. Between training and test
            # molecular ids should not be shared, e.g. if they appear in different systems
            # at the same time.
            mol_ids_train = [mol_ids[idx] for idx in train_set]
            mol_ids_test = [mol_ids[idx] for idx in test_set]

            if set (mol_ids_train) & set (mol_ids_test):
                if isinstance (cv, GroupKFold) or isinstance(cv, GroupShuffleSplit):
                    raise RuntimeError ("As grouped cross-validation is used the training "
                                        "and test molecules, i.e. mol_ids, are not allowed "
                                        "to overlap. This can happen if molecular structures "
                                        "are appearing in different systems. During the "
                                        "learning of hyper-parameter the training set should "
                                        "not contain any structure also in the test set.",
                                        set (mol_ids_train) & set (mol_ids_test))
                else:
                    print ("Training and test keys overlaps.", set (mol_ids_train) & set (mol_ids_test))

            # 1) Extract the target values from y (train and test) using the keys
            y_train, y_test = OrderedDict(), OrderedDict()
            for key in keys_train:
                y_train[key] = y[key]
            for key in keys_test:
                y_test[key] = y[key]

            # 2) Calculate the pairs (train and test)
            cretention_train, cretention_test = retention_cls(), retention_cls()

            #   a) load 'lrows' in the retention_cls
            cretention_train.load_data_from_target (y_train)
            cretention_test.load_data_from_target (y_test)

            #   b) build the digraph
            cretention_train.make_digraph (ireverse = pair_params["ireverse"])
            cretention_test.make_digraph (ireverse = pair_params["ireverse"])

            #   c) find the upper and lower set
            cretention_train.dmolecules_inv = cretention_train.invert_dictionary (cretention_train.dmolecules)
            cretention_train.dcollections_inv = cretention_train.invert_dictionary (cretention_train.dcollections)
            cretention_test.dmolecules_inv = cretention_test.invert_dictionary (cretention_test.dmolecules)
            cretention_test.dcollections_inv = cretention_test.invert_dictionary (cretention_test.dcollections)

            #   d) get the pairs from the upper and lower sets
            pairs_train = get_pairs_from_order_graph (cretention_train, keys_train,
                                                      allow_overlap = pair_params["allow_overlap"], n_jobs = n_jobs,
                                                      d_lower = pair_params["d_lower"], d_upper = pair_params["d_upper"])
            pairs_train_sets.append (pairs_train)

            if all_pairs_as_test:
                pairs_test = get_pairs_from_order_graph (cretention_test, keys_test,
                                                         allow_overlap = pair_params["allow_overlap"], n_jobs = n_jobs,
                                                         d_lower = 0, d_upper = np.inf)
            else:
                pairs_test = get_pairs_from_order_graph (cretention_test, keys_test,
                                                         allow_overlap = pair_params["allow_overlap"], n_jobs = n_jobs,
                                                         d_lower = pair_params["d_lower"], d_upper = pair_params["d_upper"])

            pairs_test_sets.append (pairs_test)
            n_pairs_test_sets.append (len (pairs_test))

            # 3) Extract the features from X (train and test) using the keys
            X_train_sets.append (np.array ([X[key] for key in keys_train]))
            X_test_sets.append (np.array ([X[key] for key in keys_test]))

        print ("")

        for k_param, param in enumerate (l_params):
            # Calculate the absolute number of correctly classified pairs
            # for each fold.
            fold_scores = Parallel (n_jobs = n_jobs, verbose = False)(
                delayed (_fit_and_score_ranksvm)(param.copy(), clone (estimator),
                                                 X_train_sets[k_cv], X_test_sets[k_cv],
                                                 pairs_train_sets[k_cv], pairs_test_sets[k_cv],
                                                 scaler)
                for k_cv in range (cv.get_n_splits()))

            if fold_score_aggregation == "average":
                param_scores[k_param] = np.mean (fold_scores / np.array (n_pairs_test_sets))
            elif fold_score_aggregation == "weighted_average":
                param_scores[k_param] = np.sum (fold_scores) / np.sum (n_pairs_test_sets)
            else:
                raise ValueError ("Invalid fold-scoring aggregation: %s." % fold_score_aggregation)

    ## Fit model using the best parameters
    # Find the best params
    best_params = l_params[np.argmax (param_scores)].copy()

    # Fit the model using the best parameters
    best_estimator = clone (estimator)
    best_estimator.set_params (**_filter_params (best_params, best_estimator))

    # Build retention order graph
    cretention = retention_cls()
    cretention.load_data_from_target (y)
    cretention.make_digraph (ireverse = pair_params["ireverse"])
    cretention.dmolecules_inv = cretention.invert_dictionary (cretention.dmolecules)
    cretention.dcollections_inv = cretention.invert_dictionary (cretention.dcollections)

    pairs = get_pairs_from_order_graph (cretention, keys, allow_overlap = pair_params["allow_overlap"],
                                        n_jobs = n_jobs, d_lower = pair_params["d_lower"],
                                        d_upper = pair_params["d_upper"])
    n_pairs_train = len (pairs)
    X = np.array ([X[key] for key in keys])

    if scaler is not None:
        X = scaler.transform (X)

    fit_params = {"FX": X, "pairs": pairs}

    best_estimator.fit (None, y = None, fit_params = fit_params)

    # Combine the mean fold scores with the list of parameter sets
    for k_param, _ in enumerate (l_params):
        l_params[k_param]["score"] = param_scores[k_param]

    return best_params, l_params, n_pairs_train, best_estimator, X, None

def _fit_and_score_regression (param, estimator, X_train, X_test, y_train, y_test, scaler):
    if scaler is not None:
        X_train = scaler.transform (X_train)
        X_test = scaler.transform (X_test)

    # Update the estimators parameters to the current ones
    estimator.set_params (**_filter_params (param, estimator))

    estimator.fit (X_train, y_train)

    # We call the scoring function of the KernelRidge class in sklearn.
    # This scoring function is a little bit different, but seems to be
    # reasonable and widely (?) used for hyper-parameter estimation.
    score = super (SVRPairwise, estimator).score (X_test, y_test)

    return score

def _fit_and_score_ranksvm (param, estimator, X_train, X_test, pairs_train, pairs_test, scaler):
    if scaler is not None:
        X_train = scaler.transform (X_train)
        X_test = scaler.transform (X_test)

    fit_params = {"FX": X_train, "pairs": pairs_train}

    # Update the estimators parameters to the current ones
    estimator.set_params (**_filter_params (param, estimator))
    estimator.fit (None, y = None, fit_params = fit_params)

    return estimator.score (X_test, pairs_test, normalize = False)

def _filter_params (param, estimator):
    """
    Given a set of parameters and an estimator: Remove the parameters
    that are not belonging to the estimator.

    :return: dictionary, containing only the supported parameters
    """
    valid_params = estimator.get_params()
    out = dict()
    for key, value in param.items():
        if key in valid_params.keys():
            out[key] = value
    return out