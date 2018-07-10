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

Test function for the RankSVM implementation.

'''


import sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# from evaluation_scenarios_cls import get_pairs
from rank_svm_cls import KernelRankSVC, get_pairs_single_system, load_data, minmax_kernel
from model_selection_cls import find_hparan_ranksvm

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RepeatedKFold, GroupKFold, PredefinedSplit, ShuffleSplit
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, linear_kernel

from collections import OrderedDict

from helper_cls import dict2str

def create_artificial_dataset (type = "linear", n = 150, distribution = "normal", random_state = None):
    """
    Create artificial dataset for the ranking svm.

    :param type: string, naming the desired dataset
        "linear": Simple linear function with some random noise
    :return:
    """

    np.random.seed (random_state)

    if distribution == "uniform":
        X = np.random.uniform (low = -1.0, high = 1.0, size = (n, 2))
    elif distribution == "normal":
        X = np.random.normal (size = (n, 2))

    if type == "linear":
        def f_lin (x, m, b = 0):
            return (x.dot(m) + b)

        m = np.array ([2, -2])

        f_X = [f_lin (x, m) for x in X]

        #
        # for idx in range (len (X)):
        #     X[idx] /= np.linalg.norm (X[idx])

    elif type == "quadratic":
        def f_qua (x, a, b, c = 0):
            return (x.dot(a).dot(x.T) + x.dot(b) + c)

        a = np.eye (2) + 2
        b = np.array ([2, -2])

        f_X = [f_qua (x, a, b) for x in X]

        #
        # for idx in range (len (X)):
        #     X[idx] /= np.linalg.norm (X[idx])

    elif type == "open_circle":
        def f_angle (x, reference):
            return (np.degrees(np.sign(x[1]) * np.arccos (x.dot (reference))))

        for idx in range (len (X)):
            X[idx] /= np.linalg.norm (X[idx])
            
        X = X[X[:,0] > -0.85]

        reference = np.array ([1.0, 0.0])
        reference /= np.linalg.norm (reference)
        f_X = [f_angle (x, reference) for x in X]

        # for idx in range (len (X)):
        #     X[idx] += np.random.normal (X[idx], 0.025, 2
        #
    np.random.seed (None)

    return X, np.array (f_X).reshape (-1, 1)

def create_artificial_dataset2 (type = "linear", n = 150, distribution = "normal"):
    X, target = create_artificial_dataset (type, n, distribution)

    # Make dictionary structure
    d_X, d_target = OrderedDict(), OrderedDict()
    for idx in range (len (X)):
        key = ("M%d" % idx, "A")
        d_X[key] = X[idx]
        d_target[key] = target[idx]

    return X, target, d_X, d_target

def visualize_dataset (raw, features, axes):
    scaler = MinMaxScaler ((1, 100))
    target = scaler.fit_transform (raw["target"][:, 0].reshape(-1, 1))

    axes[0].scatter (raw["X"][:, 0], raw["X"][:, 1],
                     s = target.flatten(), c = target.flatten())
    axes[0].set_title ("Input space")
    axes[0].set_xlabel ("x_1")
    axes[0].set_ylabel ("x_2")

    idx_pos = (features["y_clf"] ==  1).flatten()
    idx_neg = (features["y_clf"] == -1).flatten()

    axes[1].scatter (features["X_diff"][idx_pos, 0], features["X_diff"][idx_pos, 1],
                     marker = '+', c = "blue")
    axes[1].scatter (features["X_diff"][idx_neg, 0], features["X_diff"][idx_neg, 1],
                     marker = '_', c = "red")
    axes[1].set_title ("Features (diff)")
    axes[1].set_xlabel ("x_1 - x'_1")
    axes[1].set_ylabel ("x_2 - x'_2")

def visualize_dataset2 (x, y, z, axis, title = ""):
    z_scld = MinMaxScaler ((1, 100)).fit_transform (z).flatten()

    axis.scatter (x, y, s = z_scld, c = z_scld)
    axis.set_title (title)
    axis.set_xlabel ('x_1')
    axis.set_ylabel ('x_2')

def visualize_ranksvm (target, target_pred, axis, title = ""):
    axis.scatter (target, target_pred)
    axis.set_title (",".join ([title, str (np.round (sp.stats.kendalltau (target, target_pred)[0], 4))]))
    axis.set_xlabel ("Target")
    axis.set_ylabel ("Pred. target")

def compare_datasets (fig_fn = None):
    fig, axes = plt.subplots (3, 4)

    if not fig_fn is None:
        fig.set_size_inches (12, 8)

    for k, type in enumerate (["linear", "quadratic", "open_circle"]):
        print ("Type: %s" % type)

        X, target = create_artificial_dataset (type = type, n = 50, random_state = 1001)

        # print (X[range(10)])
        # pairs = get_pairs ({"X": X, "target": target})
        # X_diff, y_clf = get_pairwise_features2 (X, pairs, balance_classes = True)
        #
        # # Visualize dataset and feature space
        # visualize_dataset ({"X": X, "target": target}, {"X_diff": X_diff, "y_clf": y_clf}, axes[k])

        # Train a linear rankSVM
        # target_pred_linear = np.zeros (len (X))
        # target_pred_kernel = np.zeros (len (X))
        mean_score = 0.0

        # cv = KFold (n_splits = 10, random_state = 646, shuffle = True)
        cv = PredefinedSplit (np.zeros (len (X)))
        # for k_cv, (_, test_set) in enumerate (cv.split (X)):
        for i_rep in range (10):
            # print ("Fold %d / %d" % (k_cv + 1, cv.n_splits))

            # print (train_set[range(10)], test_set[range(10)])
#
            train_set = test_set = range(len (X))

            pairs_train = get_pairs_single_system (target[train_set], d_lower = 0, d_upper = 4)
            pairs_test = get_pairs_single_system (target[test_set], d_lower = 0, d_upper = np.inf)

            # ranksvm_linear = linear_rank_svm (verbose = False)
            # ranksvm_linear.train (X[train_set], pairs_train, n_jobs = 2, C = [1])
            #
            # target_pred_linear[test_set] = ranksvm_linear.map_values (X[test_set])[0]

            ranksvm_kernel = KernelRankSVC (C = 0.1, verbose = False, kernel = "precomputed",
                                            feature_type = "difference")

            if type == "linear":
                KX_train = linear_kernel (X[train_set], X[train_set])
                KX_train_test = linear_kernel (X[train_set], X[test_set])
            elif type == "quadratic":
                KX_train = polynomial_kernel (X[train_set], X[train_set], degree = 2)
                KX_train_test = polynomial_kernel (X[train_set], X[test_set], degree = 2)
            elif type == "open_circle":
                KX_train = rbf_kernel (X[train_set], X[train_set], gamma = 3)
                KX_train_test = rbf_kernel (X[train_set], X[test_set], gamma = 3)
            else:
                raise ValueError ("Invalid test data type: %s" % type)

            ranksvm_kernel.fit (np.arange (KX_train.shape[0]), None, fit_params = {"KX": KX_train, "pairs": pairs_train})
            score = ranksvm_kernel.score (KX_train_test, pairs_test)
            mean_score += score

            print (score)
            # target_pred_kernel[test_set] = ranksvm_kernel.map_values (KX_train_test)

        print (mean_score / 10)
        # visualize_ranksvm (target[:, 0], target_pred_linear, axes[k, 2])
        # visualize_ranksvm (target[:, 0], target_pred_kernel, axes[k, 3])

    if not fig_fn is None:
        plt.tight_layout()
        plt.savefig (fig_fn)
    else:
        plt.show()

def compare_datasets3 (fig_fn = None):
    fig, axes = plt.subplots (3, 2)
    fig_conv, axes_conv = plt.subplots (3, 3)

    if not fig_fn is None:
        fig.set_size_inches (12, 8)
        fig_conv.set_size_inches (12, 8)

    # for k, type in enumerate (["linear", "quadratic", "open_circle"]):
    for k, type in enumerate (["linear"]):
        print ("Type: %s" % type)

        X, target, d_X, d_target = create_artificial_dataset2 (type = type, n = 400)
        keys = list (d_X.keys())

        visualize_dataset2 (X[:, 0], X[:, 1], target, axes[k, 0], title = "Dataset: %s" % type)

        param_grid = {"C": [1]}

        train_set, test_set = list (ShuffleSplit (n_splits = 1, train_size = 0.75, test_size = 0.25).split (keys))[0]
        keys_train = [keys[idx] for idx in train_set]
        keys_test = [keys[idx] for idx in test_set]

        d_X_train, d_target_train = OrderedDict(), OrderedDict()
        for key in keys_train:
            d_X_train[key] = d_X[key]
            d_target_train[key] = d_target[key]


        if type == "linear":
            ranksvm_kernel = KernelRankSVC (verbose = True, debug = 2,
                                            kernel = "linear",
                                            feature_type = "difference", slack_type = "on_pairs",
                                            step_size_algorithm = "diminishing",
                                            convergence_criteria = "gs_change")
        elif type == "quadratic":
            ranksvm_kernel = KernelRankSVC (verbose = True, debug = 2, kernel = "poly",
                                            feature_type = "difference", slack_type = "on_examples")
            param_grid["degree"] = [2]
        elif type == "open_circle":
            ranksvm_kernel = KernelRankSVC (verbose = True, debug = 2, kernel = "rbf",
                                            feature_type = "difference", slack_type = "on_examples")
            param_grid["gamma"] = [3]
        else:
            raise ValueError ("Invalid test data type: %s" % type)

        best_params, param_scores, n_pairs_train, best_estimator, _, _ = find_hparan_ranksvm (
            ranksvm_kernel, d_X_train, d_target_train, cv = None, param_grid = param_grid,
            pair_params = {"allow_overlap": True, "d_upper": 8, "d_lower": 0, "ireverse": True}, n_jobs = 1, scaler = None)
        print ("Best params:", best_params)

        X_test = np.array ([d_X[key] for key in keys_test])
        target_test = np.array ([d_target[key] for key in keys_test])
        pairs_test = get_pairs_single_system (target_test, d_lower = 0, d_upper = np.inf)

        target_pred = best_estimator.map_values (X_test)
        print ("Score: %f" % best_estimator.score (X_test, pairs_test))

        visualize_ranksvm (target_test, target_pred, axes[k, 1])

        inspect_convergence (best_estimator, np.array(list(d_target_train.values())), axes_conv[k, :])

    if not fig_fn is None:
        plt.tight_layout()
        plt.savefig (fig_fn)
    else:
        plt.show(fig)
        plt.show(fig_conv)

def compare_datasets2 (fig_fn = None):
    fig, axes = plt.subplots (3, 2)

    if not fig_fn is None:
        fig.set_size_inches (12, 8)

    # for k, type in enumerate (["linear", "quadratic", "open_circle"]):
    for k, type in enumerate (["quadratic"]):
        print ("Type: %s" % type)

        X, target, d_X, d_target = create_artificial_dataset2 (type = type, n = 150)
        keys = list (d_X.keys())

        visualize_dataset2 (X[:, 0], X[:, 1], target, axes[k, 0], title = "Dataset: %s" % type)

        target_pred = np.zeros (len (X))
        mean_score = 0.0

        param_grid = {"C": [1]}

        cv = RepeatedKFold (n_splits = 10, n_repeats = 1, random_state = 1)
        for k_cv, (train_set, test_set) in enumerate (cv.split (keys)):
            print ("Fold %d / %d" % (k_cv + 1, cv.get_n_splits()))

            keys_train = [keys[idx] for idx in train_set]
            keys_test = [keys[idx] for idx in test_set]

            d_X_train, d_target_train = OrderedDict(), OrderedDict()
            for key in keys_train:
                d_X_train[key] = d_X[key]
                d_target_train[key] = d_target[key]

            # d_X_train = {key: value for key, value in d_X.items() if key in keys_train}
            # d_target_train = {key: value for key, value in d_target.items() if key in keys_train}

            if type == "linear":
                ranksvm_kernel = KernelRankSVC (verbose = False, kernel = "linear",
                                                feature_type = "difference", slack_type = "on_pairs",
                                                step_size_algorithm = "diminishing_2",
                                                convergence_criteria = "alpha_change_norm")
            elif type == "quadratic":
                ranksvm_kernel = KernelRankSVC (verbose = False, kernel = "poly", feature_type = "difference",
                                                slack_type = "on_pairs")
                param_grid["degree"] = [2]
            elif type == "open_circle":
                ranksvm_kernel = KernelRankSVC (verbose = False, kernel = "rbf", feature_type = "difference",
                                                slack_type = "on_pairs")
                param_grid["gamma"] = [3]
            else:
                raise ValueError ("Invalid test data type: %s" % type)

            cv_inner = GroupKFold (n_splits = 3)
            best_params, param_scores, n_pairs_train, best_estimator, _, _ = find_hparan_ranksvm (
                ranksvm_kernel, d_X_train, d_target_train, cv = cv_inner, param_grid = param_grid,
                pair_params = {"allow_overlap": True, "d_upper": 4, "d_lower": 0, "ireverse": True}, n_jobs = 1)
            print (best_params)

            X_test = np.array ([d_X[key] for key in keys_test])
            target_test = np.array ([d_target[key] for key in keys_test])
            pairs_test = get_pairs_single_system (target_test, d_lower = 0, d_upper = np.inf)

            target_pred[test_set] += best_estimator.map_values (X_test)
            score = best_estimator.score (X_test, pairs_test)
            print (score)
            mean_score += score

        target_pred /= cv.get_n_splits()
        mean_score /= cv.get_n_splits()

        print (mean_score)

        visualize_ranksvm ([d_target[key] for key in keys], target_pred, axes[k, 1])

    if not fig_fn is None:
        plt.tight_layout()
        plt.savefig (fig_fn)
    else:
        plt.show()

def single_dataset (X, target, kernel, convergence_criteria = "alpha_change_max", t_0 = 0.5, tol = 0.001,
                    slack_type = "on_pairs", fig_fn = None, C = 1, step_size_algorithm = "diminishing_2"):
    fig, axes = plt.subplots (2, 3)
    fig.suptitle (dict2str ({"convergence_criteria": convergence_criteria,
                             "step_size_algorithm": step_size_algorithm,
                             "slack_type": slack_type}, sep = " ; "))
    if fig_fn is not None:
        fig.set_size_inches (14, 8)

    if X.shape[1] == 2:
        visualize_dataset2 (X[:, 0], X[:, 1], target, axes[0, 0], title = "Dataset: %s" % type)
    else:
        # Do some low dimensional embedding, e.g. given a set of fingerprints.
        pass

    # Get training and test split
    train_set, test_set = list (ShuffleSplit (n_splits = 1, train_size = 0.75, test_size = 0.25,
                                              random_state = 666).split (X))[0]

    pairs_train = get_pairs_single_system (target[train_set], d_lower = 0, d_upper = 16)
    pairs_train_full = get_pairs_single_system (target[train_set], d_lower = 0, d_upper = np.inf)
    pairs_test = get_pairs_single_system (target[test_set], d_lower = 0, d_upper = np.inf)

    alpha = np.zeros ((len (pairs_train), 1))
    t_0_ = t_0
    k = 1
    max_iter = 25
    n_steps = 80
    f_0s = np.array([]) # primal objective values during optimization
    gs = np.array([]) # dual objective values during optimization
    dgs = np.array([]) # duality gap
    rdgs = np.array([])
    score_train = []
    score_test = []
    rank_corr_train = []
    rank_corr_test = []
    x_space = []

    for ii in range (n_steps):
        print ("%d: " % ii, end = "", flush = True)
        ranksvm = KernelRankSVC (C = C, debug = 1, kernel = kernel, slack_type = slack_type, t_0 = t_0_,
                                 convergence_criteria = convergence_criteria, max_iter = k + (max_iter - 1),
                                 step_size_algorithm = step_size_algorithm, random_state = 101, degree = 2,
                                 gamma = 0.5, tol = tol, verbose = True)

        # Set stepsize to latest stepsize
        ranksvm.t_0 = t_0_
        ranksvm.fit (None, None, fit_params = {"FX": X[train_set], "pairs": pairs_train, "alpha_init": alpha,
                                               "k_init": k})
        # Store the alpha as initial value for the next round
        alpha = ranksvm.alpha.reshape ((-1, 1))

        # Get last stepsize
        print ("Stepsize: t_0 = %f, t_conv = %f; Iteration: k = %d"
               % (t_0, ranksvm._t_convergence, ranksvm._k_convergence))

        if step_size_algorithm == "diminishing_2":
            t_0_ = ranksvm._get_step_size_diminishing_2 (ranksvm._t_convergence)
        if ranksvm._obj_has_converged:
            break
        else:
            k = ranksvm._k_convergence + 1

        # Get internal optimization results
        f_0s = np.concatenate ((f_0s, np.array (ranksvm.f_0s).flatten()))
        gs = np.concatenate ((gs, np.array (ranksvm.gs).flatten()))
        dgs = np.concatenate ((dgs, np.array(ranksvm.dgs).flatten()))
        rdgs = np.concatenate ((rdgs, np.array(ranksvm.rdgs).flatten()))

        # Get pairwise score and rank_correlation (train set)
        score_train.append (ranksvm.score (X[train_set], pairs_train_full))
        rank_corr_train.append (sp.stats.kendalltau (ranksvm.map_values (X[train_set]), target[train_set])[0])

        # Get pairwise score and rank_correlation (test set)
        score_test.append (ranksvm.score (X[test_set], pairs_test))
        rank_corr_test.append (sp.stats.kendalltau (ranksvm.map_values (X[test_set]), target[test_set])[0])

        x_space.append (ranksvm._k_convergence)


    print ("Iterations: %d" % ranksvm._k_convergence)

    # Run RankSVM optimization until convergence
    ranksvm = KernelRankSVC (C = C, verbose = True, kernel = kernel, slack_type = slack_type, t_0 = t_0,
                             convergence_criteria = convergence_criteria, step_size_algorithm = step_size_algorithm,
                             degree = 2, gamma = 0.5, tol = tol, max_iter = max_iter * n_steps,
                             random_state = 101)
    ranksvm.fit (None, None, fit_params = {"FX": X[train_set], "pairs": pairs_train})
    visualize_ranksvm (target[test_set], ranksvm.map_values (X[test_set]), axes[0, 1],
                       title = "Iter: %d (max = %d)" % (ranksvm._k_convergence, ranksvm.max_iter))

    rc_train_line = axes[0, 2].plot (x_space, rank_corr_train, color = "blue", linestyle = "-")
    rc_test_line = axes[0, 2].plot (x_space, rank_corr_test, color = "red", linestyle = "-")
    axes[0, 2].set_xlabel ("Iteration")
    axes[0, 2].set_ylabel ("Rank-correlation")
    axes[0, 2].set_title ("C = %.3f" % ranksvm.C)
    axes[0, 2].grid (True)
    axes[0, 2].legend ((rc_train_line[0], rc_test_line[0]), ("Train", "Test"))

    sc_train_line = axes[1, 0].plot (x_space, score_train, color = "blue", linestyle = "-")
    sc_test_line = axes[1, 0].plot (x_space, score_test, color = "red", linestyle = "-")
    axes[1, 0].set_xlabel ("Iteration")
    axes[1, 0].set_ylabel ("Pairwise accuracy")
    axes[1, 0].set_title ("C = %.3f" % ranksvm.C)
    axes[1, 0].grid (True)
    axes[1, 0].legend ((sc_train_line[0], sc_test_line[0]), ("Train", "Test"))

    # Primal and dual objective
    f_0_line = axes[1, 1].semilogy (f_0s, color = "blue", linestyle = "-")
    g_line = axes[1, 1].semilogy (gs, color = "red", linestyle = "-")
    axes[1, 1].grid (True)
    axes[1, 1].set_xlabel ("Iteration")
    axes[1, 1].set_ylabel ("Objective value")
    axes[1, 1].legend ((f_0_line[0], g_line[0]), ("Primal", "Dual",), title = "Objectives")

    # Duality gap
    dgs_line = axes[1, 2].semilogy (dgs, "green", linestyle = "-")
    rdgs_line = axes[1, 2].semilogy (rdgs, "red", linestyle = "-")
    axes[1, 2].set_xlabel ("Iteration")
    axes[1, 2].set_ylabel ("Duality gap")
    axes[1, 2].grid (True)
    axes[1, 2].legend ((dgs_line[0], rdgs_line[0]), ("Absolute", "Relative",), title = "Objectives")

    if fig_fn is not None:
        plt.savefig (fig_fn)
    else:
        plt.show()

def inspect_convergence (rankSVM, target, axes, lsty = "-"):
    # Rank correlation per iteration
    rank_corr = []
    for iter, target_pred in enumerate (rankSVM.mapped_values):
        tmp = sp.stats.kendalltau (target, target_pred)[0]
        if tmp is not np.nan:
            rank_corr.append (tmp)
        else:
            rank_corr.append (0)

    rc_line = axes[0].semilogx (rank_corr, "blue", linestyle = lsty)
    axes[0].set_xlabel ("Iteration")
    axes[0].set_ylabel ("Rank-correlation")
    axes[0].set_title ("C = %d" % rankSVM.C)
    axes[0].grid (True)

    # Primal and dual objective
    f_0_line = axes[1].semilogx (rankSVM.f_0s, "blue", linestyle = lsty)
    g_line   = axes[1].semilogx (rankSVM.gs, "red", linestyle = lsty)
    axes[1].grid (True)
    axes[1].set_xlabel ("Iteration")

    if lsty == "-":
        axes[1].semilogx([0, len (rankSVM.f_0s)], [0, 0], "black")
        axes[1].legend ((f_0_line[0], g_line[0]), ("Primal", "Dual",), title = "Objectives")

    # Duality gap
    axes[2].loglog (rankSVM.dgs, "green", linestyle = lsty)
    axes[2].set_xlabel ("Iteration")
    axes[2].grid (True)

def run_linear_convegence (odir):
    # Linear setting
    X, target = create_artificial_dataset (type = "linear", n = 500, random_state = 1001)

    conv = {"convergence_criteria": "gs_change", "slack_type": "on_examples", "C": 1,
            "step_size_algorithm": "diminishing", "t_0": 0.1}
    single_dataset (X, target, "linear", convergence_criteria = conv["convergence_criteria"],
                    slack_type = conv["slack_type"], C = conv["C"], t_0 = conv["t_0"],
                    step_size_algorithm = conv["step_size_algorithm"],
                    fig_fn = odir + "linear" + dict2str (conv, sep = "_") + ".png")

    conv = {"convergence_criteria": "gs_change", "slack_type": "on_examples", "C": 1,
            "step_size_algorithm": "diminishing_2", "t_0": 0.5}
    single_dataset (X, target, "linear", convergence_criteria = conv["convergence_criteria"],
                    slack_type = conv["slack_type"], C = conv["C"], t_0 = conv["t_0"],
                    step_size_algorithm = conv["step_size_algorithm"],
                    fig_fn = odir + "linear" + dict2str (conv, sep = "_") + ".png")

    conv = {"convergence_criteria": "alpha_change_max", "slack_type": "on_examples", "C": 1,
            "step_size_algorithm": "diminishing", "t_0": 0.1}
    single_dataset (X, target, "linear", convergence_criteria = conv["convergence_criteria"],
                    slack_type = conv["slack_type"], C = conv["C"], t_0 = conv["t_0"],
                    step_size_algorithm = conv["step_size_algorithm"],
                    fig_fn = odir + "linear" + dict2str (conv, sep = "_") + ".png")

    conv = {"convergence_criteria": "alpha_change_max", "slack_type": "on_examples", "C": 1,
            "step_size_algorithm": "diminishing_2", "t_0": 0.5}
    single_dataset (X, target, "linear", convergence_criteria = conv["convergence_criteria"],
                    slack_type = conv["slack_type"], C = conv["C"], t_0 = conv["t_0"],
                    step_size_algorithm = conv["step_size_algorithm"],
                    fig_fn = odir + "linear" + dict2str (conv, sep = "_") + ".png")

if __name__ == '__main__':
    if len (sys.argv) > 1:
        basepath = sys.argv[1]
    else:
        basepath = "INPUT YOUR BASEPATH HERE"

    odir = basepath + "/src/ranksvm/test/"
    idir = basepath + "/data/processed/PredRet/v2/"

    target, X = load_data (idir, system = "FEM_long", predictor = ["fps"], pred_fn = "fps_maccs_count.csv")
    target = target.rt.values.reshape (-1, 1)
    X = X.drop ("inchi", axis = 1).values

    for C in [1, 10]:
        conv = {"convergence_criteria": "alpha_change_max", "slack_type": "on_pairs", "C": C,
                "step_size_algorithm": "diminishing", "t_0": 0.1}
        single_dataset (X, target, minmax_kernel, convergence_criteria = conv["convergence_criteria"],
                        slack_type = conv["slack_type"], C = conv["C"], t_0 = conv["t_0"],
                        step_size_algorithm = conv["step_size_algorithm"], tol = 0.001,
                        fig_fn = odir + "fps_minmax_" + dict2str (conv, sep = "_") + ".png")