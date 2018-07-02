import numpy as np
import scipy as sp
import pandas as pd
import itertools
import time
import csv
import networkx as nx
import os
import re
import copy

## my own classes, e.g. ranksvm, retention graph, etc ...
from helper_cls import Timer, join_dicts, sample_perc_from_list, get_statistic_about_concordant_and_discordant_pairs
from helper_cls import pairwise, is_sorted
from rank_svm_cls import load_data, KernelRankSVC
from svr_pairwise_cls import SVRPairwise
# load my own kernels
from rank_svm_cls import tanimoto_kernel, tanimoto_kernel_mat, minmax_kernel_mat, minmax_kernel
# load functions for the pair generation
from rank_svm_cls import get_pairs_single_system, get_pairs_multiple_systems
# load function for the model selection
from model_selection_cls import find_hparan_ranksvm, find_hparam_regression

## scikit-learn methods
from sklearn.model_selection import ShuffleSplit, KFold, GroupKFold, GroupShuffleSplit, PredefinedSplit
from sklearn.preprocessing import  StandardScaler, MinMaxScaler, Normalizer, PolynomialFeatures
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.pipeline import Pipeline

## Data structures
from pandas import DataFrame
from collections import OrderedDict

## Allow the paralellization of the candidate graph construction
from joblib import Parallel, delayed

def evaluate_on_target_systems (
        target_systems, training_systems, predictor, pair_params, kernel_params, opt_params, input_dir, estimator,
        feature_type, n_jobs = 1, perc_for_training = 100):
    """
    Task: Evaluate rank-correlation, accuracy, etc. by learning a ranking SVM
          using the given set of training systems and prediction on the given
          set of target systems.

          Leave-n-out works now on the molecular structures only, i.e. during
          the learning, the training molecular structures (regardless of the
          system they have been measured with), will not be part of the test
          set.

    :param target_systems:
    :param training_systems:
    :param predictor:
    :param pair_params:
    :param kernel_params:
    :param opt_params:
    :param input_dir:
    :param n_jobs:
    :return:
    """

    # Variables related to the number of random / cv splits, for inner (*_cv)
    # and outer fold (*_ncv).
    n_splits_shuffle = opt_params["n_splits_shuffle"]
    n_splits_nshuffle = opt_params["n_splits_nshuffle"]
    n_splits_cv = opt_params["n_splits_cv"]
    n_splits_ncv = opt_params["n_splits_ncv"]
    n_rep = opt_params["n_rep"]

    # Should molecules be excluded from the training, if their structure appears
    # in the test _even if_ they have been measured with another system than the
    # (current) target system:
    excl_mol_by_struct_only = opt_params["excl_mol_by_struct_only"]

    # Currently only 'slack_type == "on_pairs"' is supported.
    slack_type = opt_params["slack_type"]

    # Should all possible pairs be used for the (inner) test split during the
    # parameter estimation, regardless of what are the settings for 'd_upper'
    # and 'd_lower'?
    all_pairs_for_test = opt_params["all_pairs_for_test"]

    if not estimator in ["ranksvm", "svr"]:
        raise ValueError ("Invalid estimator: %s" % estimator)

    # RankSVM and SVR regularization parameter
    param_grid = {"C": opt_params["C"]}

    if estimator == "svr":
        # error-tube width of the SVR
        param_grid["epsilon"] = opt_params["epsilon"]

    # Molecule kernel
    if kernel_params["kernel"] == "linear":
        kernel = "linear"
    elif kernel_params["kernel"] in ["rbf", "gaussian"]:
        param_grid["gamma"] = kernel_params["gamma"]
        kernel = "rbf"
    elif kernel_params["kernel"] == "tanimoto":
        if estimator in ["ranksvm"]:
            kernel = tanimoto_kernel
        elif estimator in ["svr"]:
            kernel = tanimoto_kernel_mat
    elif kernel_params["kernel"] == "minmax":
        if estimator in ["ranksvm"]:
            kernel = minmax_kernel
        elif estimator in ["svr"]:
            kernel = minmax_kernel_mat
    else:
        raise ValueError ("Invalid kernel: %s." % kernel_params["kernel"])

    if isinstance (target_systems, str):
        target_systems = [target_systems]
    if isinstance (training_systems, str):
        training_systems = [training_systems]
    all_systems = list (set (target_systems).union (training_systems))

    assert isinstance (target_systems, list) and isinstance (training_systems, list)

    n_target_systems = len (target_systems)
    n_training_systems = len (training_systems)

    print ("Target systems (# = %d): %s" % (n_target_systems, ",".join (target_systems)))
    print ("Training systems (# = %d): %s" % (n_training_systems, ",".join (training_systems)))

    ## Load the target and training systems into directories using (molecule, system)-keys
    ## and retention times respectively molecular features as values

    # If we use molecular descriptors, we need to scale the data, e.g. to [0, 1].
    if kernel_params["scaler"] == "noscaling":
        scaler = None
    elif kernel_params["scaler"] == "minmax":
        scaler = MinMaxScaler()
    elif kernel_params["scaler"] == "std":
        scaler = StandardScaler()
    elif kernel_params["scaler"] == "l2norm":
        scaler = Normalizer()
    else:
        raise ValueError ("Invalid scaler for the molecular features: %s"
                          % kernel_params["scaler"])

    # Handle counting MACCS fingerprints
    if predictor[0] == "maccsCount_f2dcf0b3":
        predictor_c = ["maccs"]
        predictor_fn = "fps_maccs_count.csv"
    else:
        predictor_c = predictor
        predictor_fn = None

    d_rts, d_features, d_system_index = OrderedDict(), OrderedDict(), OrderedDict()
    for k_sys, system in enumerate (all_systems):
        rts, data = load_data (input_dir, system = system, predictor = predictor_c, pred_fn = predictor_fn)

        # Use (mol-id, system)-tupel as key
        keys = list (zip (rts.inchi.values, [system] * rts.shape[0]))

        # Values: retention time, features
        rts  = rts.rt.values.reshape (-1, 1)
        data = data.drop ("inchi", axis = 1).values

        if kernel_params["poly_feature_exp"]:
            # If we use binary fingerprints, we can include some
            # interactions, e.g. x_1x_2, ...
            data = PolynomialFeatures (interaction_only = True, include_bias = False).fit_transform (data)

        # Make ordered directories
        d_rts[system], d_features[system] = OrderedDict(), OrderedDict()

        for i, key in enumerate (keys):
            d_rts[system][key] = rts[i, 0]
            d_features[system][key] = data[i, :]

        # Dictionary containing a unique numeric identifier for each system
        d_system_index[system] = k_sys

        if scaler is not None:
            if getattr (scaler, "partial_fit", None) is not None:
                # 'partial_fit' allows us to learn the parameters of the scaler
                # online. (great stuff :))
                scaler.partial_fit (data)
            else:
                # We have scaler at hand, that does not allow online fitting.
                # This probably means, that this is a scaler, that performs
                # the desired scaling for each example independently, e.g.
                # sklearn.preprocessing.Normalizer.
                pass

    for system in target_systems:
        print ("Target set '%s' contains %d examples." % (system, len (d_rts[system])))

    # Collect all the data that is available for training.
    d_rts_training = join_dicts (d_rts, training_systems)
    d_features_training = join_dicts (d_features, training_systems)

    # (mol-id, system)-tuples used in the training set
    l_keys_training = list (d_features_training.keys())

    # Data frames storing the evaluation measures
    mapped_values = {target_system : DataFrame() for target_system in target_systems}
    accuracies, correlations, simple_statistics= DataFrame(), DataFrame(), DataFrame()
    grid_search_results, grid_search_best_params = DataFrame(), DataFrame()

    for idx_system, target_system in enumerate (target_systems):
        print ("Process target system: %s (%d/%d)."
               % (target_system, idx_system + 1, len (target_systems)))

        # (mol-id, system)-tuples in the target set
        l_keys_target = list (d_features[target_system].keys())

        for i_rep in range (n_rep):
            print ("Repetition: %d/%d" % (i_rep + 1, n_rep))

            # Get a random subset of the training data
            l_keys_training_sub = sample_perc_from_list (l_keys_training, tsystem = target_system,
                                                         perc = perc_for_training, random_state = 747 * i_rep)
            print ("Training set contains %d (%f%%) examples."
                   % (len (l_keys_training_sub), 100 * len (l_keys_training_sub) / len (l_keys_training)))
            for training_system in training_systems:
                n_train_sys_sub = sum (np.array (list (zip (*l_keys_training_sub))[1]) == training_system)
                n_train_sys = sum (np.array (list (zip (*l_keys_training))[1]) == training_system)
                print ("\tSystem %s contributes %d (%f%%) examples."
                       % (training_system, n_train_sys_sub, 100 * n_train_sys_sub / n_train_sys))

            # Check whether the target system has any overlap with training system
            print ("Outer validation split strategy: ", end = "", flush = True)

            l_molids_training = list (zip (*l_keys_training_sub))[0]
            l_molids_target = list (zip (*l_keys_target))[0]

            if (excl_mol_by_struct_only and (len (set (l_molids_training) & set (l_molids_target)) == 0)) or \
                (not excl_mol_by_struct_only and (len (set (l_keys_training_sub) & set (l_keys_target)) == 0)):

                print ("Predefined split:\n"
                       "\tTraining and target do not share molecular structures "
                       "(excl_mol_by_struct_only=%d)" % excl_mol_by_struct_only)
                cv_outer = PredefinedSplit (np.zeros (len (l_keys_target)))

            else:
                # Determine strategy for training / test splits
                if len (l_keys_target) < 75:
                    print ("ShuffleSplit")
                    train_size = 0.75
                    cv_outer = ShuffleSplit (n_splits = n_splits_shuffle, train_size = train_size,
                                             test_size = (1 - train_size), random_state = 320 * i_rep)
                else:
                    print ("KFold")
                    cv_outer = KFold (n_splits = n_splits_cv, shuffle = True, random_state = 320 * i_rep)

            # Performance evaluation using cross-validation / random splits
            for i_fold, (_, test_set) in enumerate (cv_outer.split (l_keys_target)):
                print ("Outer fold: %d/%d" % (i_fold + 1, cv_outer.get_n_splits()))

                # (mol-id, system)-tuples in the test subset of the target set
                l_keys_target_test = [l_keys_target[idx] for idx in test_set]

                # Remove test subset of the target set from the training set.
                # NOTE: The training set might contain the whole target set.
                l_molids_target_test = list (zip (*l_keys_target_test))[0]
                if excl_mol_by_struct_only:
                    l_keys_training_train = [key for key in l_keys_training_sub if key[0] not in l_molids_target_test]
                else:
                    l_keys_training_train = [key for key in l_keys_training_sub if key not in l_keys_target_test]

                if isinstance (cv_outer, PredefinedSplit):
                    print ("Shuffle pre-defined split.")

                    rs_old = np.random.get_state()
                    np.random.seed (320 * i_fold)

                    # If we use the pre-defined splits we need to shuffle by our self.
                    # In that way we prevent bias during the h-param estimation.
                    np.random.shuffle (l_keys_training_train) # Shuffle is done inplace

                    np.random.set_state (rs_old)

                l_molids_training_train = list (zip (*l_keys_training_train))[0]

                if excl_mol_by_struct_only:
                    assert (len (set (l_molids_target_test) & set (l_molids_training_train)) == 0)
                else:
                    assert (len (set (l_keys_target_test) & set (l_keys_training_train)) == 0)

                # Determine strategy for training / test splits (inner)
                print ("Inner (h-param) validation split strategy: ", end = "", flush = True)
                if len (l_keys_training_train) < 75:
                    print ("GroupShuffleSplit")
                    train_size = 0.75
                    cv_inner = GroupShuffleSplit (n_splits = n_splits_nshuffle, train_size = train_size,
                                                  test_size = (1 - train_size), random_state = 350 * i_fold * i_rep)
                else:
                    print ("GroupKFold")
                    cv_inner = GroupKFold (n_splits = n_splits_ncv)

                # Train the rankSVM: Find optimal set of hyper-parameters
                od_rts_training_train, od_features_training_train = OrderedDict(), OrderedDict()
                for key in l_keys_training_train:
                    od_rts_training_train[key] = d_rts_training[key]
                    od_features_training_train[key] = d_features_training[key]

                start_time = time.time()

                if estimator == "ranksvm":
                    best_params, cv_results, n_train_pairs, ranking_model, _, _ = find_hparan_ranksvm (
                        estimator = KernelRankSVC (kernel = kernel, slack_type = slack_type,
                                                   random_state = 319 * i_fold * i_rep),
                        fold_score_aggregation = "weighted_average", X = od_features_training_train,
                         y = od_rts_training_train, param_grid = param_grid, cv = cv_inner,
                        pair_params = pair_params, n_jobs = n_jobs, scaler = scaler,
                        all_pairs_as_test = all_pairs_for_test)
                elif estimator == "svr":
                    best_params, cv_results, n_train_pairs, ranking_model = find_hparam_regression (
                        estimator = SVRPairwise (kernel = kernel), X = od_features_training_train,
                        y = od_rts_training_train, param_grid = param_grid,
                        cv = cv_inner, n_jobs = n_jobs, scaler = scaler)
                else:
                    raise ValueError ("Invalid estimator: %s" % estimator)

                rtime_gcv = time.time() - start_time
                print ("[find_hparam_*] %.3fsec" % rtime_gcv)

                # Store the grid-search statistics for further analyses
                grid_search_results_tmp = DataFrame (cv_results)
                grid_search_results_tmp["target_system"] = target_system
                grid_search_results_tmp["training_systems"] = ";".join (training_systems)
                grid_search_results = grid_search_results.append (grid_search_results_tmp)

                grid_search_best_params_tmp = DataFrame ([best_params])
                grid_search_best_params_tmp["target_system"] = target_system
                grid_search_best_params_tmp["training_systems"] = ";".join (training_systems)
                grid_search_best_params = grid_search_best_params.append (grid_search_best_params_tmp)

                print (grid_search_best_params_tmp)

                ## Do prediction for the test set
                # Calculate: w' * x_i, for all molecules i
                X_test, rts_test = [], []

                for key in l_keys_target_test:
                    rts_test.append (d_rts[target_system][key])
                    X_test.append (d_features[target_system][key])

                rts_test = np.array (rts_test).reshape (-1, 1)
                X_test = np.array (X_test)

                if scaler is not None:
                    X_test = scaler.transform (X_test)

                if estimator == "ranksvm":
                    Y_pred_test = ranking_model.predict (X_test, X_test)
                elif estimator == "svr":
                    Y_pred_test = ranking_model.predict (X_test)
                else:
                    raise ValueError ("Invalid estimator: %s" % estimator)

                wTx = ranking_model.map_values (X_test)

                mapped_values[target_system] = pd.concat ([mapped_values[target_system],
                                                           DataFrame ({"mapped_value" : wTx,
                                                                       "true_rt"      : rts_test.flatten(),
                                                                       "inchi"        : l_molids_target_test})],
                                                          ignore_index = True)

                correlations = correlations.append (
                    {"rank_corr"       : sp.stats.kendalltau (wTx, rts_test)[0],
                     "spear_corr"      : sp.stats.spearmanr (wTx, rts_test)[0],
                     "target_system"   : target_system,
                     "training_system" : ";".join (training_systems)},
                    ignore_index = True)

                n_train_mol = len (set (l_molids_training_train))
                n_test_mol = len (set (l_molids_target_test))
                n_shared_mol = len (set (l_molids_target_test) & (set (l_molids_training_train)))
                p_shared_mol = float (n_shared_mol) / n_test_mol

                # Predict: x_i > x_j or x_i < x_j for all molecule pairs (i, j)
                with Timer ("Get prediction score"):
                    for d_lower, d_upper in itertools.product ([0] + list (range (1, 15, 2)),
                                                               2 ** np.array([0, 1, 2, 3, 4, 5, 6, np.inf])):
                        if d_lower > d_upper:
                            continue

                        pairs_test = get_pairs_single_system (rts_test, d_lower = d_lower, d_upper = d_upper)

                        accuracies = accuracies.append (
                            {"score_w"         : ranking_model.score_using_prediction (
                                Y_pred_test, pairs_test, normalize = False),
                             "score"           : ranking_model.score_using_prediction (Y_pred_test, pairs_test),
                             "n_pairs_test"    : len (pairs_test),
                             "target_system"   : target_system,
                             "training_system" : ";".join (training_systems),
                             "d_lower"         : d_lower,
                             "d_upper"         : d_upper,
                             "i_rep"           : i_rep},
                            ignore_index = True)

                        # Write out how many molecular structures are shared between the target and training systems
                        n_test_pairs = len (pairs_test)
                        simple_statistics = simple_statistics.append (
                            {"n_shared_mol"     : n_shared_mol,
                             "p_shared_mol"     : p_shared_mol,
                             "n_train_mol"      : n_train_mol,
                             "n_test_mol"       : n_test_mol,
                             "n_train_pairs"    : n_train_pairs,
                             "n_test_pairs"     : n_test_pairs,
                             "grid_search_time" : rtime_gcv,
                             "target_system"    : target_system,
                             "training_systems" : ";".join (training_systems),
                             "d_lower"          : d_lower,
                             "d_upper"          : d_upper},
                            ignore_index = True)

    # Average the mapped values over the repetitions
    for target_system in target_systems:
        mapped_values[target_system]["mapped_value_std"] = mapped_values[target_system]["mapped_value"]
        mapped_values[target_system] = mapped_values[target_system].groupby (["inchi"], as_index = False).agg (
            {"mapped_value" : np.mean, "mapped_value_std" : np.std, "true_rt": np.unique})

    # Aggregate the rows in 'correlations' to get the mean- and std-values across the folds.
    correlations["rank_corr_std"] = correlations["rank_corr"]
    correlations["spear_corr_std"] = correlations["spear_corr"]
    correlations = correlations.groupby (["target_system", "training_system"], as_index = False).agg (
        {"rank_corr" : np.mean, "rank_corr_std" : np.std, "spear_corr" : np.mean, "spear_corr_std" : np.std})

    # Aggregate the rows in 'accuracies' to get the expected pairwise accuracy
    accuracies = accuracies.groupby (["target_system", "training_system", "d_lower", "d_upper", "i_rep"],
                                     as_index = False).agg (
        {"score_w": np.sum, "n_pairs_test": np.sum, "score": np.mean})
    accuracies["score_w"] = accuracies["score_w"] / accuracies["n_pairs_test"]
    accuracies.drop (["i_rep", "n_pairs_test"], axis = 1, inplace = True)

     # Calculate expected accuracy across the repetitions
    accuracies["score_w_std"] = accuracies["score_w"]
    accuracies["score_std"] = accuracies["score"]
    accuracies = accuracies.groupby (["target_system", "training_system", "d_lower", "d_upper"], as_index = False).agg (
        {"score_w": np.mean, "score_w_std": np.std, "score": np.mean, "score_std": np.std})

    # Aggregate the simple statistics
    simple_statistics["n_shared_mol_std"] = simple_statistics["n_shared_mol"]
    simple_statistics["p_shared_mol_std"] = simple_statistics["p_shared_mol"]
    simple_statistics["n_train_mol_std"] = simple_statistics["n_train_mol"]
    simple_statistics["n_test_mol_std"] = simple_statistics["n_test_mol"]
    simple_statistics["n_train_pairs_std"] = simple_statistics["n_train_pairs"]
    simple_statistics["n_test_pairs_std"] = simple_statistics["n_test_pairs"]
    simple_statistics["grid_search_time_std"] = simple_statistics["n_test_pairs"]

    simple_statistics = simple_statistics.groupby (["target_system", "training_systems", "d_lower", "d_upper"],
                                                   as_index = False).agg (
        {"n_shared_mol": np.mean, "p_shared_mol": np.mean,
         "n_train_mol": np.mean, "n_test_mol": np.mean,
         "n_train_pairs": np.mean, "n_test_pairs": np.mean,
         "grid_search_time": np.mean,
         "n_shared_mol_std": np.std, "p_shared_mol_std": np.std,
         "n_train_mol_std": np.std, "n_test_mol_std": np.std,
         "n_train_pairs_std": np.std, "n_test_pairs_std": np.std,
         "grid_search_time_std": np.std})

    return mapped_values, correlations, accuracies, simple_statistics, grid_search_results, grid_search_best_params

def build_candidate_structure (
        model, input_dir_candidates, n_jobs = 1, verbose = False, debug = False):
    """

    :param model: dictionary

        :ranking_model: for example KerneRankSVM object

        :kernel_params: dictionary, containing the parameter used to calcualte
            the kernel on the fingerprints

        :training_data: dictionary, data used to train the model

            :X: array-like, shape = (n_train, n_features), molecule features
            :X_C: None

    :param input_dir_candidates: string, directory containing the scoring and
        fingerprints of the candidates.

    :param n_jobs: scaler, number of jobs used to build up the graph in parallel

    :param debug: boolean, of True the edge weight do not contain any
        prediction scores of the ranking model. This option can be used for debugging.
        (default = False)

    :return: networkx.DiGraph, directed graph containing the candidate lists
    """
    # Load candidate fingerprints and retention times related to the candidate lists
    msms_rts = DataFrame.from_csv (input_dir_candidates + "/rts_msms.csv", index_col = None).sort_values ("rt")

    # Load the map from inchikey <-> inchi in order to find the correct candidates
    # in the scoring list.
    d_inchikey2inchi = {inchikey: inchi for inchikey, inchi in msms_rts[["inchikey", "inchi"]].values}

    # Add all the nodes to the candidate list graph.
    # The candidate list graph is model as a directed graph.
    l_data = Parallel (n_jobs = n_jobs, verbose = 12)(
        delayed (_candidate_iokrscores_and_rankingscores)(
            spec = spec_id, layer = spec_idx, input_dir_candidates = input_dir_candidates,
            model = model, msms_rts = msms_rts, d_inchikey2inchi = d_inchikey2inchi)
        for spec_idx, spec_id in enumerate (msms_rts.inchikey.values))

    return l_data

def shortest_path (cand_data, weight_fun, cut_off_n_cand = np.inf, check_input = False, **kwds):
    """

    :param cand_data:
    :param weight_fun:
    :param cut_off_n_cand:
    :param check_input:
    :param exclude_blocked_candidates:
    :param kwds:
    :return:
    """
    kwds_c = copy.deepcopy (kwds)

    n_layer = len (cand_data)

    if check_input:
        rt = -np.inf
        for t in range (n_layer):
            assert (is_sorted (cand_data[t]["iokrscores"], ascending = False))
            assert (cand_data[t]["rt_cand_list"] >= rt)
            rt = cand_data[t]["rt_cand_list"]

    # List of lists to store the accumulated lengths of the paths
    n_cand_first = len (cand_data[0]["iokrscores"])

    # We need to take the scores of the first layer into account!
    S_path = [[- score for score in cand_data[0]["iokrscores"]]]
    dW_path = [[0] * n_cand_first]
    parents = [[-1] * n_cand_first]

    # We need to pass the RankSVM score differences to the weighting function
    kwds_c["dW_path"] = dW_path

    # Dynamically update the path scores
    for t in range (n_layer - 1):
        tp1 = t + 1

        n_cand_t = int (np.minimum (len (cand_data[t]["iokrscores"]), cut_off_n_cand))
        n_cand_tp1 = int (np.minimum (len (cand_data[tp1]["iokrscores"]), cut_off_n_cand))

        S_path.append ([np.inf] * n_cand_tp1)
        dW_path.append ([-np.inf] * n_cand_tp1)
        parents.append ([np.nan] * n_cand_tp1)

        for i in range (n_cand_t):
            # Path length in (t,i)
            s_t_i = S_path[t][i]
            # Sum of pairwise predicitons: sum_{(t-1,k)<(t,i)} w^T(\phi_(t,i)-\phi_(t-1,k))
            dW_t_i = dW_path[t][i]

            for j in range (n_cand_tp1):
                s_update, dw_tp1_j = weight_fun (cand_data, (t, i), (tp1, j), **kwds_c)

                # Path length in (t+1,j)
                s_tp1_j = s_t_i + s_update

                if S_path[tp1][j] > s_tp1_j:
                    S_path[tp1][j] = s_tp1_j
                    dW_path[tp1][j] = dW_t_i + dw_tp1_j
                    parents[tp1][j] = i

    # Find shortest path and its length
    # - Starting node (backwards) is the one that corresponds to the
    #   end-point of the shortest path.
    nodes_shortest_path = [np.argmin (S_path[-1])]
    # - The lenght of the shortest path is the score in this node.
    len_shortest_path = S_path[-1][nodes_shortest_path[0]]

    t = n_layer - 1
    while True:
        par = parents[t][nodes_shortest_path[-1]]
        if par == -1:
            break
        nodes_shortest_path.append (par)
        t -= 1

    nodes_shortest_path.reverse()

    return nodes_shortest_path, len_shortest_path

def shortest_path_exclude_candidates (
        cand_data, weight_fun, cut_off_n_cand = np.inf, check_input = False,
        exclude_blocked_candidates = False, **kwds):
    """

    :param cand_data:
    :param weight_fun:
    :param cut_off_n_cand:
    :param check_input:
    :param exclude_blocked_candidates:
    :param kwds:
    :return:
    """
    kwds_c = copy.deepcopy (kwds)

    n_layer = len (cand_data)

    if check_input:
        rt = -np.inf
        for t in range (n_layer):
            assert (is_sorted (cand_data[t]["iokrscores"], ascending = False))
            assert (cand_data[t]["rt_cand_list"] >= rt)
            rt = cand_data[t]["rt_cand_list"]

            if exclude_blocked_candidates:
                assert ("is_blocked" in cand_data[t].keys())
                assert (len (cand_data[t]["is_blocked"]) == len (cand_data[t]["iokrscores"]))

    # List of lists to store the accumulated lengths of the paths
    n_cand_first = len (cand_data[0]["iokrscores"])

    # We need to take the scores of the first layer into account!
    S_path = [[- score for score in cand_data[0]["iokrscores"]]]
    dW_path = [[0] * n_cand_first]
    parents = [[-1] * n_cand_first]

    # We need to pass the RankSVM score differences to the weighting function
    kwds_c["dW_path"] = dW_path

    # Dynamically update the path scores
    for t in range (n_layer - 1):
        tp1 = t + 1

        n_cand_t = int (np.minimum (len (cand_data[t]["iokrscores"]), cut_off_n_cand))
        n_cand_tp1 = int (np.minimum (len (cand_data[tp1]["iokrscores"]), cut_off_n_cand))

        S_path.append ([np.inf] * n_cand_tp1)
        dW_path.append ([-np.inf] * n_cand_tp1)
        parents.append ([np.nan] * n_cand_tp1)

        for i in range (n_cand_t):
            if exclude_blocked_candidates and cand_data[t]["is_blocked"][i]:
                continue

            # Path length in (t,i)
            s_t_i = S_path[t][i]
            # Sum of pairwise predicitons: sum_{(t-1,k)<(t,i)} w^T(\phi_(t,i)-\phi_(t-1,k))
            dW_t_i = dW_path[t][i]

            for j in range (n_cand_tp1):
                if exclude_blocked_candidates and cand_data[tp1]["is_blocked"][j]:
                    continue

                s_update, dw_tp1_j = weight_fun (cand_data, (t, i), (tp1, j), **kwds_c)

                # Path length in (t+1,j)
                s_tp1_j = s_t_i + s_update

                if S_path[tp1][j] > s_tp1_j:
                    S_path[tp1][j] = s_tp1_j
                    dW_path[tp1][j] = dW_t_i + dw_tp1_j
                    parents[tp1][j] = i

    # Find shortest path and its length
    # - Starting node (backwards) is the one that corresponds to the
    #   end-point of the shortest path.
    nodes_shortest_path = [np.argmin (S_path[-1])]
    # - The lenght of the shortest path is the score in this node.
    len_shortest_path = S_path[-1][nodes_shortest_path[0]]

    t = n_layer - 1
    while True:
        par = parents[t][nodes_shortest_path[-1]]
        if par == -1:
            break
        nodes_shortest_path.append (par)
        t -= 1

    nodes_shortest_path.reverse()

    return nodes_shortest_path, len_shortest_path

def _weight_func_max (cand_data, u, v, **kwargs):
    """
    Task: Calculate the edge-weight for a graph during the shortest path
          algorithm. The weight is calculated on the fly depending on the
          regularisation parameter D. To change D, we need to redefine the
          function when we iterate over the grid.

    Notation: C_k, C_k+1 are the candidates _sets_ of the spectra S_k and S_k+1

    :param u: tuple, (layer, candidate), head-node, (i,j) in the paper

    :param v: tuple, (layer, candidate), tail-node, (r,s) in the paper

    :param **kwds : optional keyword parameters

    :return: tuple, (edge-weight, difference between order scores), edge-weight
        is used to determine the shortest path between first and last layer.
    """
    assert ("D" in kwargs.keys())
    assert ("use_sign" in kwargs.keys())
    assert ("epsilon_rt") in kwargs.keys()
    assert ("use_log") in kwargs.keys()

    # Additional keyword parameters
    D = kwargs["D"]
    # scalar, regularisation parameter controlling the influence of
    #     the weight of the edge (u,v), if D = 0, then only the score of the
    #     candidate are used, and no re-ranking is done.

    use_sign = kwargs["use_sign"]
    # boolean, should only the sign of the penalty be taken to calculate the
    #     shortest path?

    epsilon_rt = kwargs["epsilon_rt"]
    # scalar, order information of two consecutive layers is only taken into
    #     account of their retention time difference is larger than 'epsilon_rt'.

    use_log = kwargs["use_log"]
    # boolean, should the logarithm of the penalty be used instead?

    t,i = u # head-node: Notation in paper (i,j), Section 2.3.2
    tp1,j = v # tail-node: Notation in paper (r,s)

    # IOKR score of the tail-node
    s_tp1_j = cand_data[tp1]["iokrscores"][j]

    # When two spectra to have the same retention-time (but for examples different
    # m/z), than there is no order-information to exploit. We in that case give
    # the penalty zero, so that the best connection is the one connects the
    # highest IOKR scores.
    if np.abs (cand_data[t]["rt_cand_list"] - cand_data[tp1]["rt_cand_list"]) <= epsilon_rt:
        dW_tp1_j = 0
    else:
        dW_tp1_j = cand_data[t]["wtx"][i] - cand_data[tp1]["wtx"][j] # < 0 is good!

    if use_sign:
        dW_tp1_j = np.sign (dW_tp1_j)

    # Calculate penalty term as described in section 2.3.2
    penalty = max (0, dW_tp1_j)

    if use_log:
        # Log transform the penalty to avoid the outliers.
        penalty = np.log (penalty + 1)

    return (- s_tp1_j + D * penalty, dW_tp1_j)

def _load_candidate_fingerprints (spec, input_dir_candidates, predictor):
    """

    :param spec: string, identifier of the spectra and candidate list. Currently
        we use the inchikey of the structure represented by the spectra. The idea
        behind this is, that we can have the same spectra representing different
        compounds, e.g. if we are looking at different stereo-isomers.

    :param l_fps_files:

    :param input_dir_candidates:

    :return: (list, np.array)-tuple:
        :1st: list of string, shape = (n_cand,), InChIs of all candidates
        :2nd: array-like, shape = (n_cand, n_feature), fingerprint matrix of the canidates
    """
    if predictor[0] == "maccs":
        fps_fn = "maccs_binary"
    elif predictor[0] == "maccsCount_f2dcf0b3":
        fps_fn = "maccs_count"
    else:
        raise ValueError ("Unsupported predictor for candidates: %s" % predictor[0])

    # TODO: Make 'scaler' applicable here.

    l_fps_files = os.listdir (input_dir_candidates + "/fingerprints/" + fps_fn + "/")
    cand_fps_fn = list (filter (re.compile ("fps_" + fps_fn + "_list.*=%s.csv" % spec).match, l_fps_files))
    assert (len (cand_fps_fn) == 1)
    cand_fps_fn = input_dir_candidates + "/fingerprints/" + fps_fn + "/" + cand_fps_fn[0]
    cand_fps = DataFrame.from_csv (cand_fps_fn, index_col = None)

    return cand_fps

def _load_candidate_scorings (spec, input_dir_candidates, predictor):
    """

    :param spec: string, identifier of the spectra and candidate list. Currently
        we use the inchikey of the structure represented by the spectra. The idea
        behind this is, that we can have the same spectra representing different
        compounds, e.g. if we are looking at different stereo-isomers.

    :param input_dir_candidates:

    :return: pandas.DataFrame
    """
    if predictor[0] == "maccs":
        fps_fn = "maccs_binary"
    elif predictor[0] == "maccsCount_f2dcf0b3":
        fps_fn = "maccs_count"
    else:
        raise ValueError ("Unsupported predictor for candidates: %s" % predictor[0])

    l_scoring_files = os.listdir (input_dir_candidates + "/scorings/" + fps_fn + "/")
    scores_fn = list (filter (re.compile ("scoring_list.*=%s.csv" % spec).match, l_scoring_files))
    assert (len (scores_fn) == 1)
    scores_fn = input_dir_candidates + "/scorings/" + fps_fn + "/" + scores_fn[0]

    return DataFrame.from_csv (scores_fn, index_col = None).sort_values ("score", ascending = False)


def perform_reranking_of_candidates (cand_data, topk = 25, cut_off_n_cand = np.inf,
                                     weight_function = None, **kwargs):
    """


    :param cand_data:
    :param cut_off_n_cand:
    :param topk:
    :param kwargs:
    :return:
    """
    def _t_map (t, all_cand_blocked):
        return np.arange(0, len (all_cand_blocked))[~all_cand_blocked][t]

    cand_data_c = copy.deepcopy (cand_data)

    if weight_function is None:
        raise ValueError ("A weight function must be provided.")

    for t, _ in enumerate (cand_data_c):
        n_cand_t = len (cand_data_c[t]["iokrscores"])
        cand_data_c[t]["is_blocked"] = np.zeros (n_cand_t, dtype = "bool")

    num_correct = []
    paths = []
    lengths = []
    all_cand_blocked = np.zeros (len (cand_data_c), dtype = "bool")
    for k in range (topk):
        if all (all_cand_blocked):
            break

        # with Timer("Get shortest path for k = " + str (k + 1)):
        path, length = shortest_path_exclude_candidates (
            [cand_data_c[t] for t, _ in enumerate (cand_data_c) if not all_cand_blocked[t]],
            weight_function, exclude_blocked_candidates = True, cut_off_n_cand = cut_off_n_cand,
            check_input = True, **kwargs)

        assert (len (path) == np.sum (~all_cand_blocked))

        num_correct.append (0)
        paths.append (path)
        lengths.append (length)

        for t, cand in enumerate (path):
            num_correct[k] += cand_data_c[_t_map (t, all_cand_blocked)]["is_true_identification"][cand]

            # Block candidates along the current shortest path
            cand_data_c[_t_map (t, all_cand_blocked)]["is_blocked"][cand] = True

        for t, _ in enumerate (cand_data_c):
            all_cand_blocked[t] = all (cand_data_c[t]["is_blocked"])

    # Get the top-k accuracies for different k
    topk_accs = np.cumsum (num_correct) / len (cand_data_c) * 100 # k \in [1,topk]

    return topk_accs, paths, lengths


def _candidate_iokrscores_and_rankingscores (
        spec, input_dir_candidates, model, msms_rts, d_inchikey2inchi, layer):
    """

    :param spec:
    :param l_fps_files:
    :param l_scoring_files:
    :param input_dir_candidates:
    :param model:
    :param msms_rts:
    :param d_inchikey2inchi:
    :parm add_source_and_target_node: boolean,
    :return:
    """
    def _process_single_candidate_list (
            spec, scores, cand_fps, msms_rts, d_inchikey2inchi, layer, model):
        """

        :param spec: string, identifier of the spectra and candidate list. Currently
            we use the inchikey of the structure represented by the spectra. The idea
            behind this is, that we can have the same spectra representing different
            compounds, e.g. if we are looking at different stereo-isomers.

        :param scores: pandas.DataFrame, shape = (n_cand, 2), two-column data-frame:

            "id1",   "score"
             INCHI-1, score_1
             INCHI-2, score_2
             ...

             The table must be sorted according to the score of the candidates descending.

        :param msms_rts:

        :param d_inchikey2inchi: dictionary, mapping from the inchikey (spectra and
            candidate list identifier) to the inchi of the correct / true candidate.

            :key: string, inchikey
            :value: string, inchi

        :param layer:

        :param add_ranking_score: boolean, should the ranking / preference score be
            added to each node, e.g. RankSVM: w^T\phi_i

        :return: scalar, rank of the correct / true candidates (remember we are using
            dense ranks here.)

        :TODO: We could also predict for each candidate a retention time here!
        """
        #
        rt_cand_list = msms_rts[msms_rts.inchikey == spec].rt.values
        assert (len (rt_cand_list) == 1)
        rt_cand_list = rt_cand_list[0]
        # Determine the index, i.e. row in the scores-DataFrame, if the correct / true
        # candidate.
        inchi_correct_cand = d_inchikey2inchi[spec]

        # FIXME: 'id1' corresponds the InChIs of the candidates. We should fix the scoring files.
        assert (sum (scores.id1 == inchi_correct_cand) == 1)

        fps_correct_cand = cand_fps[cand_fps.inchi == inchi_correct_cand].drop ("inchi", axis = 1).values
        K_candcand = pairwise_kernels (
            fps_correct_cand, fps_correct_cand,
            metric = model["ranking_model"].kernel, filter_params = True)

        # Output lists
        # Note: The candidate are (if) sorted according to their IOKR score
        #       see '_load_candidate_scorings'.
        iokrscores = []
        ranks = []
        is_true_identification = []
        d_squared = []
        wtx = []
        id1s = []

        rank = 0 # smallest rank is one which is assigned to the largest score.
        rank_correct_cand = 0
        last_score = np.inf
        for id1, score in scores[["id1", "score"]].values:
            if last_score > score:
                last_score = score
                rank += 1

            # Calculat the similarity (in terms of the used fingerprints for the RankSVM)
            # between the candidate and the correct / true identification.
            fps_cand_id1 = cand_fps[cand_fps.inchi == id1].drop ("inchi", axis = 1).values
            K_ii = pairwise_kernels (
                fps_cand_id1, fps_cand_id1,
                metric = model["ranking_model"].kernel, filter_params = True)
            K_icand = pairwise_kernels (
                fps_cand_id1, fps_correct_cand,
                metric = model["ranking_model"].kernel, filter_params = True)

            iokrscores.append (score)
            ranks.append (rank)
            is_true_identification.append (inchi_correct_cand == id1)
            d_squared.append ((K_ii + K_candcand - 2 * K_icand)[0])
            wtx.append (model["ranking_model"].map_values (fps_cand_id1)[0])
            id1s.append (id1)

            if inchi_correct_cand == id1:
                rank_correct_cand = rank

        assert (rank_correct_cand > 0)

        return {"iokrscores": iokrscores, "ranks": ranks, "is_true_identification": is_true_identification,
                "d_squared": d_squared, "wtx": wtx, "inchis": id1s, "rt_cand_list": rt_cand_list,
                "rank_correct_cand": rank_correct_cand, "inchi_correct_cand": inchi_correct_cand,
                "spec_id": spec}

    # Load candidate scores and fps
    scores = _load_candidate_scorings (spec, input_dir_candidates, model["predictor"])
    cand_fps = _load_candidate_fingerprints (spec, input_dir_candidates, model["predictor"])

    # Add the nodes for all candidates
    return _process_single_candidate_list (
        spec, scores, cand_fps, msms_rts, d_inchikey2inchi, layer, model)


def train_model_using_all_data (
        training_systems, predictor, pair_params, kernel_params, opt_params, input_dir, estimator,
        feature_type, n_jobs = 1):
    """
    Taks: Train a RankSVM model that uses all the data for training. The hyper-parameter
          are chosen using cross-validation.

    :param training_systems:
    :param predictor:
    :param pair_params:
    :param kernel_params:
    :param opt_params:
    :param input_dir:
    :param estimator:
    :param feature_type:
    :param n_jobs:
    :return:
    """

    # Number of cv-splits used for the parameters estimated
    n_splits_cv = opt_params["n_splits_ncv"]

    # Slack type for RankSVM
    slack_type = opt_params["slack_type"]

    # See 'evaluate_on_target_systems' for description:
    all_pairs_for_test = opt_params["all_pairs_for_test"]

    if estimator != "ranksvm":
        raise ValueError ("Invalid estimator: %s" % estimator)

    # RankSVM regularization parameter
    param_grid = {"C": opt_params["C"]}

    if kernel_params["kernel"] == "linear":
        kernel = "linear"
    elif kernel_params["kernel"] in ["rbf", "gaussian"]:
        param_grid["gamma"] = kernel_params["gamma"]
        kernel = "rbf"
    elif kernel_params["kernel"] == "tanimoto":
        if estimator in ["ranksvm", "kernelridge"]:
            kernel = tanimoto_kernel
        elif estimator in ["svr"]:
            kernel = tanimoto_kernel_mat
    elif kernel_params["kernel"] == "minmax":
        if estimator in ["ranksvm", "kernelridge"]:
            kernel = minmax_kernel
        elif estimator in ["svr"]:
            kernel = minmax_kernel_mat
    else:
        raise ValueError ("Invalid kernel: %s." % kernel_params["kernel"])

    if isinstance (training_systems, str):
        training_systems = [training_systems]

    n_training_systems = len (training_systems)
    print ("Training systems (# = %d): %s" % (n_training_systems, ",".join (training_systems)))

    ## Load the target and training systems into directories using (molecule, system)-keys
    ## and retention times respectively molecular features as values

    # If we use molecular descriptors, we need to scale the data, e.g. to [0, 1].
    if kernel_params["scaler"] == "noscaling":
        scaler = None
    elif kernel_params["scaler"] == "minmax":
        scaler = MinMaxScaler()
    elif kernel_params["scaler"] == "std":
        scaler = StandardScaler()
    elif kernel_params["scaler"] == "l2norm":
        scaler = Normalizer()
    else:
        raise ValueError ("Invalid scaler for the molecular features: %s"
                          % kernel_params["scaler"])

    # Handle MACCS counting fingerprints
    if predictor[0] == "maccsCount_f2dcf0b3":
        predictor_c = ["maccs"]
        predictor_fn = "fps_maccs_count.csv"
    else:
        predictor_c = predictor
        predictor_fn = None

    d_rts, d_features, d_system_index = OrderedDict(), OrderedDict(), OrderedDict()
    for k_sys, system in enumerate (training_systems):
        rts, data = load_data (input_dir, system = system, predictor = predictor_c, pred_fn = predictor_fn)

        # Use (mol-id, system)-tupel as key
        keys = list (zip (rts.inchi.values, [system] * rts.shape[0]))

        # Values: retention time, features
        rts  = rts.rt.values.reshape (-1, 1)
        data = data.drop ("inchi", axis = 1).values

        if kernel_params["poly_feature_exp"]:
            # If we use binary fingerprints, we can include some
            # interactions, e.g. x_1x_2, ...
            data = PolynomialFeatures (interaction_only = True, include_bias = False).fit_transform (data)

        # Make ordered directories
        d_rts[system], d_features[system] = OrderedDict(), OrderedDict()

        for i, key in enumerate (keys):
            d_rts[system][key] = rts[i, 0]
            d_features[system][key] = data[i, :]

        # Dictionary containing a unique numeric identifier for each system
        d_system_index[system] = k_sys

        if scaler is not None:
            if getattr (scaler, "partial_fit", None) is not None:
                # 'partial_fit' allows us to learn the parameters of the scaler
                # online. (great stuff :))
                scaler.partial_fit (data)
            else:
                # We have scaler at hand, that does not allow online fitting.
                # This probably means, that this is a scaler, that performs
                # the desired scaling for each example independently, e.g.
                # sklearn.preprocessing.Normalizer.
                pass

    for system in training_systems:
        print ("Training set '%s' contains %d examples." % (system, len (d_rts[system])))

    # Collect all the data that is available for training.
    d_rts_training = join_dicts (d_rts)
    d_features_training = join_dicts (d_features)

    # Train the model
    start_time = time.time()

    best_params, cv_results, n_train_pairs, ranking_model, X_train, _ = find_hparan_ranksvm (
        estimator = KernelRankSVC (kernel = kernel, slack_type = slack_type, random_state = 319),
        fold_score_aggregation = "weighted_average", X = d_features_training, y = d_rts_training,
        param_grid = param_grid, cv = GroupKFold (n_splits = n_splits_cv), pair_params = pair_params,
        n_jobs = n_jobs, scaler = scaler, all_pairs_as_test = all_pairs_for_test)

    # NOTE: X_train and X_C_train (if not None) are already transformed using the specified scaler.

    rtime_gcv = time.time() - start_time
    print ("[find_hparam_*] %.3fsec" % rtime_gcv)
    print (DataFrame (cv_results))

    # Create dictionaries used by the graph building interface
    training_data = {"X_train": X_train, "X_C_train": None}

    return ranking_model, best_params, training_data, kernel_params