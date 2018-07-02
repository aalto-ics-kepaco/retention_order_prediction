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
from evaluation_scenarios_cls import perform_reranking_of_candidates, _weight_func_max, \
    build_candidate_structure, train_model_using_all_data

from rank_svm_cls import get_pairs_single_system

## Load some scikit learn helper functions
from sklearn.externals import joblib
from sklearn.model_selection import ShuffleSplit

## Load some helper functions
from helper_cls import dict2str

def evaluate_system_pairs (
        training_systems, target_systems, input_dir, predictor, n_jobs, feature_type, pair_params,
        kernel_params, opt_params, estimator):
    """
    Run the evaluation on pairs (s_1, s_2) of systems. For that each system
    is used as training and target system.

    :param systems: list of string, names of the chromatographic systems
    :param input_dir: string, directory containing the input features and retention times
    :param output_dir: string, directory to store the results
    :param predictor:
    :param n_jobs:
    :param feature_type:
    :param pair_params:
    :param kernel_params:
    :return:
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

def parse_sysargs (arguments):
    """
    Task: Parser for the system arguments passed during the script-call. If no arguments are provided
          the experimental parameters are set to some default values, e.g. suitable for debugging.

    :return:
    """
    # Which estimator should be used, e.g. ranksvm (default), kernelridge, svr, ...?
    estimator = arguments[1]

    ## Which _evaluation_ should done?
    scenario = arguments[2]          # {baseline, ...}
    sysset = arguments[3]            # {set=all, set=1, ...}
    tsysidx = eval (arguments[4])    # {-1,0,...,|sysset|-1},
                                     #     If -1 all target systems are considered without parallelization.

    # Leave-target-system-out: This only effects the scenarios 'on_one' and 'selected_scenarios'.
    # Therefore we will run 'ltso' in {True, False} by default, i.e. both settings are ran when
    # the evaluation is started.

    # Read the config file that defines the 'data' and the 'model'.
    with open (arguments[5]) as config_file:
        d_config = json.load (config_file)

        ## Which _data_ should be used?
        base_dir = d_config["data"]["base_dir"]
        dataset = d_config["data"]["dataset"]
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

    # How many jobs can we run in parallel?
    n_jobs = eval (arguments[6])

    # Do we run in debug-mode?
    debug = eval (arguments[7])

    return scenario, sysset, tsysidx, base_dir, dataset, systems, pair_params, feature_type, predictor,\
           feature_scaler, kernel, n_jobs, debug, estimator, excl_mol_by_struct_only, poly_feature_exp,\
           slack_type, all_pairs_for_test, dp_weight_function, use_log_reranking

def write_out_results (output_dir, ofile_prefix, param_suffixes, results):
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
    output_dir += predictor
    if poly_feature_exp:
        output_dir += "_POLYD2"
    output_dir += "/" + kernel + "/" + scenario + "/"

    if not os.path.isdir (output_dir):
        os.makedirs (output_dir)

    ## Kernel and optimization parameters
    if debug:
        opt_params = {"C": [0.1, 1, 10], "epsilon": [0.025, 0.1, 0.5, 1.0],
                      "n_splits_shuffle": 3, "n_splits_nshuffle": 3,
                      "n_splits_cv": 3, "n_splits_ncv": 3,
                      "n_rep": 2, "excl_mol_by_struct_only": excl_mol_by_struct_only,
                      "slack_type": slack_type, "all_pairs_for_test": all_pairs_for_test}

        kernel_params = {"kernel": kernel, "gamma": [0.1, 0.25, 0.5, 1, 2, 3],
                         "scaler": feature_scaler, "poly_feature_exp": poly_feature_exp}

        reranking_params = {"D": [0, 1e-3, 10], "use_sign": [False], "topk": 1, "cut_off_n_cand": 100,
                            "n_rep": 3, "epsilon_rt": [-1],
                            "min_rt_delta_range": np.arange (0, 0.20, 0.05)}
    else:
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

    if scenario == "baseline":
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
        best_params_fn = model_output_dir + "/best_params_" + dict2str (param_suffixes, sep = "_") + ".prm"
        training_data_fn = model_output_dir + "/training_data_" + dict2str (param_suffixes, sep = "_") + ".data"
        kernel_params_fn = model_output_dir + "/kernel_params_" + dict2str (param_suffixes, sep = "_") + ".prm"

        if os.path.isfile (model_fn) and os.path.isfile (training_data_fn) and os.path.isfile (best_params_fn)\
                and os.path.isfile (kernel_params_fn):
            print ("Load ranking model ...")

            # 1) Load an existing model
            ranking_model = joblib.load (model_fn)
            best_params = joblib.load (best_params_fn)
            training_data = joblib.load (training_data_fn)
            kernel_params_out = joblib.load (kernel_params_fn)
        else:
            print ("Compute ranking model ...")
            # 1) Train a model using all the available data and store the model to disk
            ranking_model, best_params, training_data, kernel_params_out = train_model_using_all_data (
                training_systems = systems, predictor = predictor, pair_params = pair_params, estimator = estimator,
                kernel_params = kernel_params, opt_params = opt_params, input_dir = input_dir, feature_type = feature_type, n_jobs = n_jobs)

            joblib.dump (ranking_model, model_fn)
            joblib.dump (best_params, best_params_fn)
            joblib.dump (training_data, training_data_fn)
            joblib.dump (kernel_params_out, kernel_params_fn)

        # Construct or load the candidate structure
        input_dir_candidates = base_dir + "/data/processed/Impact/candidates/"
        cand_data_fn = model_output_dir + "/cand_data_" + dict2str (param_suffixes, sep = "_") + ".data"

        if os.path.isfile (cand_data_fn):
            print ("Load candidate data ...")
            cand_data = joblib.load (cand_data_fn)
        else:
            print ("Construct candidate graph ...")
            cand_data = build_candidate_structure (
                model = {"ranking_model": ranking_model, "training_data": training_data,
                         "kernel_params": kernel_params_out, "predictor": predictor},
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
            wfun = _weight_func_max # used in the Paper
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
