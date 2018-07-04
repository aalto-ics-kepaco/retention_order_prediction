'''
Main script to run the evaluations presented in the publication:

    Liquid-Chromatography Retention Order Prediction for Metabolite Identification,
    Eric Bach, Sandor Szedmak, Celine Brouard, Sebastian Böcker and Juho Rousu,
    ...
'''

import sys
import os
import json
import numpy as np
import copy
import scipy as sp

from joblib import Parallel, delayed
from pandas import DataFrame, concat

## Load evaluation scenarios
from evaluation_scenarios_cls import evaluate_on_target_systems

## Load function related to the metabolite identification
from metabolite_identification_cls import perform_reranking_of_candidates, _weight_func_max, \
    build_candidate_structure, train_model_using_all_data

from rank_svm_cls import get_pairs_single_system

## Load some scikit learn helper functions
from sklearn.externals import joblib
from sklearn.model_selection import ShuffleSplit

## Load some helper functions
from helper_cls import dict2str

def parse_sysargs (arguments):
    """
    Task: Parser for the system arguments passed during the script-call, e.g.:
        - Which experiment (here called scenario) should be ran?
        - Which estimator should be used?
        - Which set of systems (contained in the dataset) should be used?
        - Which concrete configuration of the estimator, features, etc. should be used?

    :return: parsed parameters
    """
    # Which estimator should be used? Currently available: "ranksvm" and "svr".
    estimator = arguments[1]

    # Which experiment / scenario should be ran?
    scenario = arguments[2]          # {baseline, ...}

    # Which set of target systems should be considered?
    # For the experiments in:
    # - Section 3.1 in the paper, we use sysset = 10
    # - Section 3.2 in the paper, we use sysset = 10, 10_imp and imp
    sysset = arguments[3]            # {set=all, set=1, ...}

    # Index of the target system for which the evaluation should be performed.
    # This parameter is ment to be used for parallelization. That means, one
    # can run a job for each target system in the system set separetly instead
    # running all within the same job (if tsysidx == -1).
    tsysidx = eval (arguments[4])    # {-1,0,...,|sysset|-1},

    # Read the config file that defines the 'data', 'model', and paramters for the
    # metaboloite identification 'application'.
    with open (arguments[5]) as config_file:
        d_config = json.load (config_file)

        ## Which _data_ should be used?
        base_dir = d_config["data"]["base_dir"]
        dataset = d_config["data"]["dataset"] # e.g. PredRet/v2
        systems = d_config["data"]["systems"][sysset]
        excl_mol_by_struct_only = d_config["data"]["excl_mol_by_struct_only"]

        ## Which _model_ should be used?
        # RankSVM
        if estimator == "ranksvm":
            pair_params = d_config["model"]["ranksvm"]["pair_params"]
            feature_type = "difference"
            slack_type = "on_pairs"
        else:
            pair_params = None
            feature_type = None
            slack_type = None

        # Molecule representation
        kernel = d_config["model"]["molecule_representation"]["kernel"]
        predictor = d_config["model"]["molecule_representation"]["predictor"]
        feature_scaler = d_config["model"]["molecule_representation"]["feature_scaler"]
        poly_feature_exp = d_config["model"]["molecule_representation"]["poly_feature_exp"]

        # Parameters for the model selection
        all_pairs_for_test = d_config["model"]["modelselection"]["all_pairs_for_test"]

        # Parameters for the candidate re-ranking
        dp_weight_function = d_config["application"]["candidate_reranking"]["dp_weight_function"]
        use_log_reranking = d_config["application"]["candidate_reranking"]["use_log"]

    # How many jobs can we run in parallel? The number of jobs is only used for
    # the hyper-parameter estimation.
    n_jobs = eval (arguments[6])

    # Do we run in debug-mode?
    # The debug-mode can be used to test, whether all parts of the evaluation:
    #
    #   1) Loading the data
    #   2) Model estimation
    #   3) Model evaluation
    #   4) Saving of the results
    #
    # is working properly. The main difference between debug- and "normal"-mode is,
    # that in the debug mode the number of cross-validation splits, the number of
    # repitions (of the cross-validation) and the grid of hyper-paramters is reduced.
    # Furthermore, the results are stored in a folder ".../debug/..." rather than
    # ".../final/...".
    debug = eval (arguments[7])

    return scenario, sysset, tsysidx, base_dir, dataset, systems, pair_params, feature_type, predictor,\
           feature_scaler, kernel, n_jobs, debug, estimator, excl_mol_by_struct_only, poly_feature_exp,\
           slack_type, all_pairs_for_test, dp_weight_function, use_log_reranking


def evaluate_system_pairs (
        training_systems, target_systems, input_dir, predictor, n_jobs, feature_type, pair_params,
        kernel_params, opt_params, estimator):
    """
    Given a set of training systems S1 and target systems S2, the evaluation is run on
    all pairs in (training, target) \in S1 x S2. If S1 = S2, we can evaluate how good
    each systems can predict each others system.

    :param training_systems: list of strings, system-ids used for training:
        E.g.: ["FEM_long", "RIKEN"]

    :param target_systems: list of strings, systems-ids used as target systems, i.e.,
        in which the performance is evaluted.

    :param input_dir: string, directory containing the pre-processed dataset under
        investigation.
        E.g.: 'base_dir + "/data/processed/" + dataset + "/"', with dataset = "PredRet/v2"

    :param predictor: list of string, predictors used for the model training.
        E.g.: ["maccs", "circular"] --> both MACCS and circular fingerprints are used
              ["maccs"]             --> only MACCS fingerprints are used

    :param n_jobs: integer, number of jobs used for the parallelization. Several jobs are
        primarily used for hyper-parameter estimation of the order predictor. See
        'model_selection_cls.py' for details.

    :param feature_type: string, feature type that is used for the RankSVM. Currently
        only 'difference' features are supported, i.e., \phi_j - \phi_i is used for
        the decision. If the estimator is not RankSVM, but e.g. Support Vector Regression,
        than tis parameter can be set to None and is ignored.

    :param pair_params: dictionary, containing the paramters used for the creation of
        the RankSVM learning pairs, e.g. minimum and maximum oder distance.

    :param kernel_params: dictionary, containing the parameters for the kernels and
        generally for handling the input features / predictors. See definition of the
        dictionary in the __main__ of this file.

    :param opt_params: dictionary, containing the paramters controlling the hyper-paramter
        optimization, number of cross-validation splits, etc. See definition of the
        dictionary in the __main__ of this file.

    :param estimator: string, estimator used for the order learning. Currently available
        are: "ranksvm" and "svr".

    :return: dictionary, containing the aggregated evalation results:
        "correlation": pandas.DataFrame, rank correlation of predicted retention scores
            and observed retention times, for each (training, target)-system pair.
        "accuracy": pandas.DataFrame, pairwise order prediction accuracy, for each
            (training, target)-system pair.
        "simple_statistics": pandas.DataFrame, e.g., number of training and test
            examples respectively pairs (and other things), for each (training, target)-pair.
        "grid_search_results": pandas.DataFrame, scores for different hyper-parameters
            during the optimization for each (training, target)-pair
        "grid_search_best_params": pandas.DataFrame, best / selected hyper-parameters
            during the optimization for each (training, target)-pair
    """
    correlations, accuracies, simple_statistics = DataFrame(), DataFrame(), DataFrame()
    grid_search_results, grid_search_best_params = DataFrame(), DataFrame()

    for training_system in training_systems:
        _, correlations_sys, accuracies_sys, simple_statistics_sys, grid_search_results_sys, \
        grid_search_best_params_sys = \
            evaluate_on_target_systems (
                training_systems = training_system, target_systems = target_systems, predictor = predictor,
                n_jobs = n_jobs, input_dir = input_dir, pair_params = pair_params, kernel_params = kernel_params,
                opt_params = opt_params, estimator = estimator, feature_type = feature_type)

        correlations = concat ([correlations, correlations_sys], ignore_index = True)
        accuracies = concat ([accuracies, accuracies_sys], ignore_index = True)
        simple_statistics = concat ([simple_statistics, simple_statistics_sys], ignore_index = True)
        grid_search_results = concat ([grid_search_results, grid_search_results_sys], ignore_index = True)
        grid_search_best_params = concat ([grid_search_best_params, grid_search_best_params_sys],
                                          ignore_index = True)

    results = {"correlations": correlations, "accuracies": accuracies, "simple_statistics": simple_statistics,
               "grid_search_results": grid_search_results, "grid_search_best_params": grid_search_best_params}

    return results


def evaluate_all_on_one (
        training_systems, target_systems, leave_target_system_out, input_dir,
        predictor, n_jobs, feature_type, pair_params, kernel_params, opt_params, estimator,
        perc_for_training = 100):
    """
    This function iterates over all specified target systems and runs the evalation using
    them while a set of training systems is used to learn the model. Here the target system
    can be part of the training and is excluded if desired.

    :param training_systems: list of strings, system-ids used for training:
        E.g.: ["FEM_long", "RIKEN"]

    :param target_systems: list of strings, systems-ids used as target systems, i.e.,
        in which the performance is evaluted.

    :param leave_target_system_out: boolean, should the target system under evaluation
        be excluded from the training, if it is contained in that set?

    :param input_dir: string, directory containing the pre-processed dataset under
        investigation.
        E.g.: 'base_dir + "/data/processed/" + dataset + "/"', with dataset = "PredRet/v2"

    :param predictor: list of string, predictors used for the model training.
        E.g.: ["maccs", "circular"] --> both MACCS and circular fingerprints are used
              ["maccs"]             --> only MACCS fingerprints are used

    :param n_jobs: integer, number of jobs used for the parallelization. Several jobs are
        primarily used for hyper-parameter estimation of the order predictor. See
        'model_selection_cls.py' for details.

    :param feature_type: string, feature type that is used for the RankSVM. Currently
        only 'difference' features are supported, i.e., \phi_j - \phi_i is used for
        the decision. If the estimator is not RankSVM, but e.g. Support Vector Regression,
        than tis parameter can be set to None and is ignored.

    :param pair_params: dictionary, containing the paramters used for the creation of
        the RankSVM learning pairs, e.g. minimum and maximum oder distance.

    :param kernel_params: dictionary, containing the parameters for the kernels and
        generally for handling the input features / predictors. See definition of the
        dictionary in the __main__ of this file.

    :param opt_params: dictionary, containing the paramters controlling the hyper-paramter
        optimization, number of cross-validation splits, etc. See definition of the
        dictionary in the __main__ of this file.

    :param estimator: string, estimator used for the order learning. Currently available
        are: "ranksvm" and "svr".

    :param perc_for_training: scalar, percentage of the target systems data, that is
        used for the training, e.g., selected by simple random sub-sampling. This value
        only effects the training process, of the target system is in the set of training
        systems.

    :return: dictionary, containing the aggregated evalation results:
        "correlation": pandas.DataFrame, rank correlation of predicted retention scores
            and observed retention times, for each (training, target)-system pair.
        "accuracy": pandas.DataFrame, pairwise order prediction accuracy, for each
            (training, target)-system pair.
        "simple_statistics": pandas.DataFrame, e.g., number of training and test
            examples respectively pairs (and other things), for each (training, target)-pair.
        "grid_search_results": pandas.DataFrame, scores for different hyper-parameters
            during the optimization for each (training, target)-pair
        "grid_search_best_params": pandas.DataFrame, best / selected hyper-parameters
            during the optimization for each (training, target)-pair
    """

    correlations, accuracies, simple_statistics = DataFrame(), DataFrame(), DataFrame()
    grid_search_results, grid_search_best_params = DataFrame(), DataFrame()

    for target_system in target_systems:
        training_systems_c = training_systems.copy()
        if leave_target_system_out:
            try:
                training_systems_c.remove (target_system)
            except:
                pass

        _, correlations_sys, accuracies_sys, simple_statistics_sys, grid_search_results_sys, \
        grid_search_best_params_sys = \
            evaluate_on_target_systems (
                training_systems = training_systems_c, target_systems = target_system, predictor = predictor,
                n_jobs = n_jobs, input_dir = input_dir, pair_params = pair_params, kernel_params = kernel_params,
                opt_params = opt_params, estimator = estimator, perc_for_training = perc_for_training,
                feature_type = feature_type)

        correlations = concat ([correlations, correlations_sys], ignore_index = True)
        accuracies = concat ([accuracies, accuracies_sys], ignore_index = True)
        simple_statistics = concat ([simple_statistics, simple_statistics_sys], ignore_index = True)
        grid_search_results = concat ([grid_search_results, grid_search_results_sys], ignore_index = True)
        grid_search_best_params = concat ([grid_search_best_params, grid_search_best_params_sys],
                                          ignore_index = True)

    results = {"correlations": correlations, "accuracies": accuracies, "simple_statistics": simple_statistics,
               "grid_search_results": grid_search_results, "grid_search_best_params": grid_search_best_params}

    return results


def evaluate_single_on_one (
        systems, input_dir, predictor, n_jobs, feature_type, pair_params, kernel_params, opt_params,
        estimator, perc_for_training = 100):
    """
    Run the evaluation for a set of systems S1 for all pairs like:

        (training_system = s_i, target_system = s_i) with s_i \in S1,

    that means, the prediction performence of each system using it self for the model
    training is evalauted.

    :param systems: list of strings, system-ids used for training and as target:
        E.g.: ["FEM_long", "RIKEN"]

    :param input_dir: string, directory containing the pre-processed dataset under
        investigation.
        E.g.: 'base_dir + "/data/processed/" + dataset + "/"', with dataset = "PredRet/v2"

    :param predictor: list of string, predictors used for the model training.
        E.g.: ["maccs", "circular"] --> both MACCS and circular fingerprints are used
              ["maccs"]             --> only MACCS fingerprints are used

    :param n_jobs: integer, number of jobs used for the parallelization. Several jobs are
        primarily used for hyper-parameter estimation of the order predictor. See
        'model_selection_cls.py' for details.

    :param feature_type: string, feature type that is used for the RankSVM. Currently
        only 'difference' features are supported, i.e., \phi_j - \phi_i is used for
        the decision. If the estimator is not RankSVM, but e.g. Support Vector Regression,
        than tis parameter can be set to None and is ignored.

    :param pair_params: dictionary, containing the paramters used for the creation of
        the RankSVM learning pairs, e.g. minimum and maximum oder distance.

    :param kernel_params: dictionary, containing the parameters for the kernels and
        generally for handling the input features / predictors. See definition of the
        dictionary in the __main__ of this file.

    :param opt_params: dictionary, containing the paramters controlling the hyper-paramter
        optimization, number of cross-validation splits, etc. See definition of the
        dictionary in the __main__ of this file.

    :param estimator: string, estimator used for the order learning. Currently available
        are: "ranksvm" and "svr".

    :param perc_for_training: scalar, percentage of the target systems data, that is
        used for the training, e.g., selected by simple random sub-sampling. This value
        only effects the training process, of the target system is in the set of training
        systems.

    :return: dictionary, containing the aggregated evalation results:
        "correlation": pandas.DataFrame, rank correlation of predicted retention scores
            and observed retention times, for each (training, target)-system pair.
        "accuracy": pandas.DataFrame, pairwise order prediction accuracy, for each
            (training, target)-system pair.
        "simple_statistics": pandas.DataFrame, e.g., number of training and test
            examples respectively pairs (and other things), for each (training, target)-pair.
        "grid_search_results": pandas.DataFrame, scores for different hyper-parameters
            during the optimization for each (training, target)-pair
        "grid_search_best_params": pandas.DataFrame, best / selected hyper-parameters
            during the optimization for each (training, target)-pair
    """

    correlations, accuracies, simple_statistics = DataFrame(), DataFrame(), DataFrame()
    grid_search_results, grid_search_best_params = DataFrame(), DataFrame()

    for target_system in systems:
        _, correlations_sys, accuracies_sys, simple_statistics_sys, grid_search_results_sys, \
        grid_search_best_params_sys = \
            evaluate_on_target_systems (
                training_systems = target_system, target_systems = target_system, predictor = predictor,
                n_jobs = n_jobs, input_dir = input_dir,  pair_params = pair_params, kernel_params = kernel_params,
                opt_params = opt_params, estimator = estimator, perc_for_training = perc_for_training,
                feature_type = feature_type)

        correlations = concat ([correlations, correlations_sys], ignore_index = True)
        accuracies = concat ([accuracies, accuracies_sys], ignore_index = True)
        simple_statistics = concat ([simple_statistics, simple_statistics_sys], ignore_index = True)
        grid_search_results = concat ([grid_search_results, grid_search_results_sys], ignore_index = True)
        grid_search_best_params = concat ([grid_search_best_params, grid_search_best_params_sys], ignore_index = True)

    results = {"correlations": correlations, "accuracies": accuracies, "simple_statistics": simple_statistics,
               "grid_search_results": grid_search_results, "grid_search_best_params": grid_search_best_params}

    return results


def write_out_results (output_dir, ofile_prefix, param_suffixes, results):
    """
    Write the results to the disk. The output filename is compiled in the following way:

        output_dir + "/" + prefix + "_" + dataframe_type + "_" + suffix.csv,

    with:
        prefix: string, can be used to index results computed in parallel, e.g. "_00", "_01", ...
        dataframe_type: string, \in {"accuracies", "correlations", ...} (see 'evaluate_system_pairs', etc.)
        suffix: string, identifies the "flavor" of the experiment, e.g. "_embso=False" (are molecules
            excluded from the training those structure in the test), ...

    :param output_dir: string, output directory for the results.

    :param ofile_prefix: string, id of the result for example primarily used to allow
        parallelization

    :param param_suffixes: dictionary, containing the "flavors" of the experiment:
        E.g.: {"use_feature_scaler": True, "exclude_molecules_from_training_based_on_structure": False, ...}

    :param results: dictionary, containing the results to be written out (compare also
        e.g. 'evaluate_system_pairs'):
        keys: strings, type of result, e.g. "accuracies"
        value: pandas.DataFrane, containing the actual results in a table
    """
    CSV_SEP = "\t"
    CSV_FLOAT_FORMAT = "%.4f"

    for key, value in results.items():
        ofile = output_dir + "/" + ofile_prefix + key + "_" + dict2str (param_suffixes, sep = "_") + ".csv"
        value.to_csv (ofile, index = False, sep = CSV_SEP, float_format = CSV_FLOAT_FORMAT)


if __name__ == "__main__":
    # Directory structure of the results:
    # estimator                     {ranksvm, kernelridge, etc ...}
    #   pair_params                 {combn, order_graph, etc ...}
    #       feature_type            {difference, exterior product, etc ...}
    #           predictor           {fps_maccs, fps_circular, desc, etc ...}
    #               kernel          {tanimoto, minmax, gaussian, linear, etc ...}
    #                   scenario    {baseline, on_one, selected_scenarios, baseline_single}
    #                       * correlations.csv              - Rank-correlation, Spearman-correlation of the predicted target values
    #                       * accuracies.csv                - Pairwise prediction accuracies
    #                       * simple_statistics.csv         - Number of shared molecules between training and test, etc ...
    #                       * grid_search_results.csv
    #                       * grid_search_best_params.csv

    # Read parameters from the command line arguments and the config file
    scenario, sysset, tsysidx, base_dir, dataset, systems, pair_params, \
    feature_type, predictor, feature_scaler, kernel, n_jobs, debug, estimator, \
    excl_mol_by_struct_only, poly_feature_exp, slack_type, all_pairs_for_test, dp_weight_function, \
    use_log_reranking = parse_sysargs (sys.argv)

    ## Input dir
    # Define the input dir dependent on the chosen dataset
    input_dir = base_dir + "/data/processed/" + dataset + "/"

    ## Output dir
    # Define the base output dir dependent on the chosen dataset and whether we are running in debug mode.
    output_dir = base_dir + "/results/raw/" + dataset + "/"
    if debug:
        output_dir += "/debug/"
    else:
        output_dir += "/final/"
    output_dir += "_".join ([x for x in [estimator, dict2str ({"slacktype": slack_type}, sep = "_")] if x != ""]) + "/"

    # Create final output dir based on the settings for the experiment
    output_dir += "/".join ([x for x in [dict2str (pair_params, sep = "_"), feature_type] if x is not None]) + "/"
    output_dir += "_".join (predictor)
    if poly_feature_exp:
        output_dir += "_POLYD2"
    output_dir += "/" + kernel + "/" + scenario + "/"

    if not os.path.isdir (output_dir):
        os.makedirs (output_dir)

    ## Kernel and optimization parameters
    # Note:
    # - Outer splits used for model evaluation.
    # - Inner splits used for hyper-paramter optimization.
    if debug:
        opt_params = {"C":                       [0.1, 1, 10],              # regularization paramter grid
                      "epsilon":                 [0.025, 0.1, 0.5, 1.0],    # SVR error-tube paramter grid
                      "n_splits_shuffle":        3,                         # number of outer random splits
                                                                            # (< 75 test examples)
                      "n_splits_nshuffle":       3,                         # number of inner random splits
                                                                            # (< 75 test examples)
                      "n_splits_cv":             3,                         # number of outer cross-validation
                                                                            # (>= 75 test examples)
                      "n_splits_ncv":            3,                         # number of inner cross-validation
                                                                            # (>= 75 test examples)
                      "n_rep":                   2,                         # number of repetitions of the
                                                                            # model evaluation (averaged sub-
                                                                            # sequently).
                      "excl_mol_by_struct_only": excl_mol_by_struct_only,
                      "slack_type":              slack_type,
                      "all_pairs_for_test":      all_pairs_for_test}        # See 'find_hparan_ranksvm' in the
                                                                            # model_selection_cls.py.

        kernel_params = {"kernel": kernel, "gamma": [0.1, 0.25, 0.5, 1, 2, 3],
                         "scaler": feature_scaler, "poly_feature_exp": poly_feature_exp}

        reranking_params = {"D": [0, 5e-3, 7.5e-3, 1e-2],
                            "use_sign": [False], "topk": 1, "cut_off_n_cand": 300, "n_rep": 10,
                            "epsilon_rt": 0, "min_rt_delta_range": [0], "use_log": use_log_reranking}
    else:
        # Hyper-parameter grids, number of cross-validation / random split folds, number of repetitions
        # (e.g. evaluations) used in the paper.
        opt_params = {"C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                      "epsilon": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10],
                      "n_splits_shuffle": 25, "n_splits_nshuffle": 25,
                      "n_splits_cv": 10, "n_splits_ncv": 10,
                      "n_rep": 10, "excl_mol_by_struct_only": excl_mol_by_struct_only,
                      "slack_type": slack_type, "all_pairs_for_test": all_pairs_for_test}

        kernel_params = {"kernel": kernel, "gamma": [0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3],
                         "scaler": feature_scaler, "poly_feature_exp": poly_feature_exp}

        reranking_params = {"D": [0, 5e-4, 1e-3, 2.5e-3, 5e-3, 7.5e-3, 1e-2, 2.5e-2, 5e-2, 7.5e-2, 1e-1],
                            "use_sign": [False], "topk": 1, "cut_off_n_cand": 300, "n_rep": 1000,
                            "epsilon_rt": 0, "min_rt_delta_range": [0], "use_log": use_log_reranking}

    # In the following let us assume we are given a dataset with 3 different systems: s_1, s_2 and s_3
    if scenario == "baseline":
        # Baseline results: Single system is used for training the order predictor and the
        #                   performance is evaluated in all available target systems.
        #
        # E.g. pairwise prediction accuracy (matrix):
        #
        # training system / target system --->
        #  |
        #  V
        #       s_1     s_2     s_3
        # s_1 acc_11  acc_12  acc_13
        # s_2 acc_21  acc_22  acc_23
        # s_3 acc_31  acc_32  acc_33
        #
        # In the paper: Section 3.1 (3.1.3, 3.1.4), Table 3 (target system == training system),
        #               Table 4 (single system setting)
        #
        # Note: In the paper we only present the diagonal of the accuracy matrix.
        #       Therefore one can also use the "baseline_single" scenario in order
        #       to save computation time, for reproduction.

        # Run evaluation
        tsystems = systems if tsysidx == -1 else systems[tsysidx]
        results = evaluate_system_pairs (
            training_systems = systems, target_systems = tsystems, input_dir = input_dir, predictor = predictor, n_jobs = n_jobs,
            feature_type = feature_type, pair_params = pair_params, kernel_params = kernel_params, opt_params = opt_params,
            estimator = estimator)

        # Write out the results
        ofile_prefix = "" if tsysidx == -1 else "%02d_" % tsysidx
        param_suffixes = {"sysset": sysset, "featurescaler": feature_scaler, "embso": excl_mol_by_struct_only}
        write_out_results (output_dir, ofile_prefix, param_suffixes, results)

    elif scenario == "all_on_one":
        # Multiple training systems: All the available systems are used to train the order predictor
        #                            in a _single_ model. The evaluation is subsequently performed on
        #                            each available (target) system. In this scenario two cases are
        #                            considered jointly: leave-target-system-out (ltso) from training
        #                            [True, False].
        #
        # E.g. pairwise prediction accuracy (matrix):
        #
        # training system / target system --->
        #  |
        #  V
        #
        # If: ltso = False
        #
        #                 s_1     s_2     s_3
        # {s_1,s_2,s_3} acc_*1  acc_*2  acc_*3
        #
        # If: ltso = True
        #
        #                 s_1     s_2     s_3
        # {s_2,s_3}     acc_-11    -       -
        # {s_1,s_3}        -    acc_-22    -
        # {s_2,s_3}        -       -    acc_-33
        #
        # In the paper: In Section 3.1.4 we evaluate the performance of the RankSVM order prediction
        #               in the "Multiple systems for training" case. 'all_on_one' can be used to get
        #               the results presented in Table 4 (Multiple systems, no target data; Multiple
        #               systems, all data; ltso = True; ltso = False).

        for ltso in [True, False]:
            print ("Leave-target-system-out: %s" % str (ltso))

            # Run evaluation
            tsystems = systems if tsysidx == -1 else [systems[tsysidx]]
            results = evaluate_all_on_one (
                training_systems = systems, target_systems = tsystems, leave_target_system_out = ltso,
                input_dir = input_dir, predictor = predictor, n_jobs = n_jobs, feature_type = feature_type,
                pair_params = pair_params, kernel_params = kernel_params, opt_params = opt_params,
                estimator = estimator)

            # Write out the results
            ofile_prefix = "" if tsysidx == -1 else "%02d_" % tsysidx
            param_suffixes = {"sysset": sysset, "featurescaler": feature_scaler, "ltso": ltso,
                              "embso": excl_mol_by_struct_only}
            write_out_results (output_dir, ofile_prefix, param_suffixes, results)

    elif scenario == "all_on_one_perc":
        # Multiple training systems: All the available systems are used to train the order predictor
        #                            in a _single_ model. The evaluation is subsequently performed on
        #                            each available (target) system. In this scenario, one can vary the
        #                            percentage of training data, that contributs to the model, comming
        #                            from the target system.
        #
        # E.g. pairwise prediction accuracy (matrix):
        #
        # training system / target system --->
        #  |
        #  V
        #                            s_1     s_2     s_3
        # {s_1(  0%),s_2,s_3} acc_(  0%)1     -       -
        # {s_1( 10%),s_2,s_3} acc_( 10%)1     -       -
        #                               ...
        # {s_1(100%),s_2,s_3} acc_(100%)1     -       -
        # {s_1,(  0%)s_2,s_3}         -  acc_(  0%)2  -
        # {s_1,( 10%)s_2,s_3}         -  acc_( 10%)2  -
        #                               ...
        # {s_1,(100%)s_2,s_3}         -  acc_(100%)2  -
        # {s_1,s_2,(  0%)s_3}         -       - acc_(  0%)3
        # {s_1,s_2,( 10%)s_3}         -       - acc_( 10%)3
        #                               ...
        # {s_1,s_2,(100%)s_3}         -       - acc_(100%)3
        #
        # In the paper: In Section 3.1.4 we evaluate the performance of the RankSVM order prediction
        #               in the "Multiple systems for training" case. 'all_on_one' can be used to get
        #               the results presented in Figure 4 (multiple systems).

        # Run evaluation
        tsystems = systems if tsysidx == -1 else [systems[tsysidx]]

        for perc_for_training in range (0, 110, 10):
            results = evaluate_all_on_one (
                training_systems = systems, target_systems = tsystems, leave_target_system_out = False,
                input_dir = input_dir, predictor = predictor, n_jobs = n_jobs, feature_type = feature_type,
                pair_params = pair_params, kernel_params = kernel_params, opt_params = opt_params,
                estimator = estimator, perc_for_training = perc_for_training)

            # Write out the results
            ofile_prefix = "" if tsysidx == -1 else "%02d_" % tsysidx
            param_suffixes = {"sysset": sysset, "featurescaler": feature_scaler, "ltso": False,
                              "embso": excl_mol_by_struct_only, "percfortrain": perc_for_training}
            write_out_results (output_dir, ofile_prefix, param_suffixes, results)

    elif scenario == "baseline_single":
        # Baseline results: Single system is used for training the order predictor and the
        #                   performance is evaluated in the training systems.
        #
        # E.g. pairwise prediction accuracy (matrix):
        #
        # training system / target system --->
        #  |
        #  V
        #       s_1     s_2     s_3
        # s_1 acc_11     -       -
        # s_2    -    acc_22     -
        # s_3    -       -    acc_33
        #
        # In the paper: Section 3.1 (3.1.3, 3.1.4), Table 3 (target system == training system),
        #               Table 4 (single system setting)

        # Run evaluation
        tsystems = systems if tsysidx == -1 else [systems[tsysidx]]
        results = evaluate_single_on_one (
            systems = tsystems, input_dir = input_dir, predictor = predictor, n_jobs = n_jobs,
            feature_type = feature_type, pair_params = pair_params, kernel_params = kernel_params,
            opt_params = opt_params, estimator = estimator)

        # Write out the results
        ofile_prefix = "" if tsysidx == -1 else "%02d_" % tsysidx
        param_suffixes = {"sysset": sysset, "featurescaler": feature_scaler, "allpairsfortest": all_pairs_for_test}
        write_out_results (output_dir, ofile_prefix, param_suffixes, results)

    elif scenario == "baseline_single_perc":
        # Baseline results: Single system is used for training the order predictor and the
        #                   performance is evaluated in the training systems. We can vary
        #                   the percentage of target system data used for training
        #
        # E.g. pairwise prediction accuracy (matrix):
        #
        # training system / target system --->
        #  |
        #  V
        #                  s_1     s_2     s_3
        # ( 10%)s_1 acc_( 10%)1     -       -
        # ( 20%)s_1 acc_( 20%)1     -       -
        #                       ...
        # (100%)s_1 acc_(100%)1     -       -
        # ( 10%)s_2         -  acc_( 10%)2  -
        # ( 20%)s_2         -  acc_( 20%)2  -
        #                       ...
        # (100%)s_2         -  acc_(100%)2  -
        # ( 10%)s_3         -       -  acc_( 10%)3
        # ( 20%)s_3         -       -  acc_( 20%)3
        #                       ...
        # (100%)s_3         -       -  acc_(100%)3
        #
        # In the paper: Section 3.1 (3.1.3, 3.1.3), Figure 4 (Single system).

        # Run evaluation
        tsystems = systems if tsysidx == -1 else [systems[tsysidx]]
        for perc_for_training in range (10, 110, 10):
            results = evaluate_single_on_one (
                systems = tsystems, input_dir = input_dir, predictor = predictor, n_jobs = n_jobs,
                feature_type = feature_type, pair_params = pair_params, kernel_params = kernel_params,
                opt_params = opt_params, estimator = estimator,
                perc_for_training = perc_for_training)

            # Write out the results
            ofile_prefix = "" if tsysidx == -1 else "%02d_" % tsysidx
            param_suffixes = {"sysset": sysset, "featurescaler": feature_scaler, "percfortrain": perc_for_training}
            write_out_results (output_dir, ofile_prefix, param_suffixes, results)

    elif scenario == "met_ident_perf_GS_BS": # GS_BS means simple grid-search with boot-strapping
        param_suffixes = {"sysset": sysset, "featscaler": feature_scaler, "usecoldesc": False}

        # Model input / output directory
        model_output_dir = output_dir + "/model/"
        if not os.path.isdir (model_output_dir):
            os.makedirs (model_output_dir)

        # Get model filename and check whether it already has been trained:
        model_fn = model_output_dir + "/ranking_model_" + dict2str (param_suffixes, sep = "_") + ".mdl"

        if os.path.isfile (model_fn) and os.path.isfile (training_data_fn) and os.path.isfile (best_params_fn)\
                and os.path.isfile (kernel_params_fn):
            print ("Load ranking model ...")

            # 1) Load an existing model
            ranking_model = joblib.load (model_fn)
        else:
            print ("Compute ranking model ...")
            # 1) Train a model using all the available data and store the model to disk
            ranking_model, best_params = train_model_using_all_data (
                training_systems = systems, predictor = predictor, pair_params = pair_params,
                estimator = estimator, kernel_params = kernel_params, opt_params = opt_params,
                input_dir = input_dir, feature_type = feature_type, n_jobs = n_jobs)

            joblib.dump (ranking_model, model_fn)

        # Construct or load the candidate structure
        input_dir_candidates = base_dir + "/data/processed/Impact/candidates/"
        cand_data_fn = model_output_dir + "/cand_data_" + dict2str (param_suffixes, sep = "_") + ".data"

        if os.path.isfile (cand_data_fn):
            print ("Load candidate data ...")
            cand_data = joblib.load (cand_data_fn)
        else:
            print ("Construct candidate graph ...")
            cand_data = build_candidate_structure (
                model = {"ranking_model": ranking_model, "predictor": predictor},
                input_dir_candidates = input_dir_candidates,
                n_jobs = n_jobs, verbose = debug * 5)

            joblib.dump (cand_data, cand_data_fn)

        # Perform the reranking
        # Get the retention times and preference scores of the dataset
        rts = []
        wtx = []
        for data in cand_data:
            rts.append (data["rt_cand_list"])
            wtx.append (np.array(data["wtx"])[data["is_true_identification"]])
        rts = np.array (rts)
        wtx = np.array (wtx).flatten()
        assert (len (rts) == len (wtx))

        if dp_weight_function == "pwmax":
            wfun = _weight_func_max # used in the Paper (see Section 2.3.2)
        else:
            raise ValueError ("Invalid weight function for the dynamic programming: "
                              "%s" % dp_weight_function)

        rd_split = ShuffleSplit (n_splits = reranking_params["n_rep"], train_size = 2/3, random_state = 320)
        d_top1_acc = {D: [] for D in reranking_params["D"]}
        d_top1 = {D: [] for D in reranking_params["D"]}
        n_spectra = []

        for i_split, (train_set, _) in enumerate (rd_split.split (range (len (rts)))):
            print ("Process split: %d/%d." % (i_split + 1, rd_split.get_n_splits()))

            n_spectra.append (len (train_set))

            # Shuffle split does not preserve the order of the examples when sub-setting.
            train_set = np.sort (train_set)

            rts_train = rts[train_set]
            wtx_train = wtx[train_set]

            pairs, _ = get_pairs_single_system (
                rts_train, d_lower = 0, d_upper = np.inf, return_rt_differences = True)

            # Calculate the pairwise accuracy
            score = 0.0
            for i, j in pairs:
                if wtx_train[i] < wtx_train[j]:
                    score += 1.0
            if len (pairs) > 0:
                score /= len (pairs)

            print ("Kendall tau=%f, Spearmanr=%f, pairwise acc=%f"
                   % (sp.stats.kendalltau (wtx_train, rts_train)[0], sp.stats.spearmanr (wtx_train, rts_train)[0], score))

            # 3) Perform the reranking of the candidates
            l_res = Parallel (n_jobs = n_jobs, verbose = 20)(
                delayed (perform_reranking_of_candidates)(
                    cand_data = [cand_data[idx] for idx in train_set], weight_function = wfun,
                    cut_off_n_cand = reranking_params["cut_off_n_cand"], topk = reranking_params["topk"],
                    **{"D": D, "use_sign": False, "epsilon_rt": reranking_params["epsilon_rt"], "use_log": reranking_params["use_log"]})
                for D in reranking_params["D"])

            # 4) Aggregate the top-k accuracies of the different repetitions
            max_top1 = -np.inf
            max_top1_idx = -1
            for idx, D in enumerate (reranking_params["D"]):
                d_top1_acc[D].append (l_res[idx][0][0])
                d_top1[D].append (l_res[idx][0][0] / 100 * len (train_set))

                if l_res[idx][0][0] > max_top1:
                    max_top1 = l_res[idx][0][0]
                    max_top1_idx = idx

            print ("Best params:", reranking_params["D"][max_top1_idx], "Top-1 acc:", max_top1)

        # 5) Write out the results
        ofile_prefix = ""

        for D in reranking_params["D"]:
            param_suffixes_c = copy.deepcopy (param_suffixes)
            param_suffixes_c["D"] = D
            param_suffixes_c["uselog"] = reranking_params["use_log"]
            param_suffixes_c["wfun"] = dp_weight_function
            param_suffixes_c["epsrt"] = reranking_params["epsilon_rt"]
            param_suffixes_c["nrds"] = reranking_params["n_rep"]

            write_out_results (output_dir, ofile_prefix, param_suffixes_c,
                               {"topk_acc": DataFrame ({"top1_acc": np.array (d_top1_acc[D]),
                                                        "top1": np.array (d_top1[D]),
                                                        "n_spectra": np.array (n_spectra)})})

    else:
        raise ValueError ("Invalid scenario: '%s'." % scenario)