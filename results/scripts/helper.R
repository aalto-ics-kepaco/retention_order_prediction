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

require (data.table)

#' Make string out of list by concatenating the list names
#' and their corresponding list values.
#' 
#' @param list input list
#' @param sep character, used as seperator betweeen the list names
#' 
#' @return string, if list is not NULL, otherwise NULL
#' 
#' @example 
#'   l <- list (A = 7, B = "HALLO")
#'   s <- list2str (l)
#'   print (s)
#'   > "A=7-B=HALLO"
#'   
#'   print (list2str (NULL))
#'   > NULL
list2str <- function (list, sep = "-", sort_names = TRUE) {
    if (is.null (list)) {
        return (NULL)
    }
    
    str <- vector()
    
    list_names <- names (list)
    if (sort_names) {
        list_names <- sort (list_names)
    }
    
    for (name in list_names) {
        if (is.null (list[[name]])) {
            next
        }
        
        if (name == " ") {
            str <- c(str, list[[name]])
        } else {
            str <- c(str, paste (name, list[[name]], sep = "=")) 
        }
    }
    
    str <- paste (str, collapse = sep)
    
    return (str)
}

#' Create the path to the input directory, specified by the experimental
#' setting.
#' 
#' @param base_dir string, path to the directory under which all 
#'                 experimental results are stored.
get_input_dir <- function (
    base_dir = "./", pair_params = list (type = "combn"), 
    feature_type = "difference", predictor = paste (c("desc"), sep = "_"),
    kernel = "rbf", scenario = "baseline", predictor_column = NULL, 
    kernel_column = NULL)
{
    input_dir <- paste (
        c(base_dir, list2str (pair_params, sep = "_"), feature_type, 
          list2str (list (" " = predictor, "column" = predictor_column), 
                    sep = "_", sort_names = FALSE),
          list2str (list (" " = kernel, "column" = kernel_column), sep = "_",
                    sort_names = FALSE),
          scenario), collapse = "/")
    return (input_dir)
}

#' Create the filename of a particular output file based on its flavor and, 
#' if needed, its set of systems.
#' 
#' @param prefix string, specifying the output file to load, e.g. accuracies_*,
#'               simple_statistics_*, ...
#' @param system_set string, specifying the set used for learning, e.g. set=1, 
#'                   set=all, etc, ...
#' @param flavor list, of "experimental flavors", e.g. leave-target-system-out
#'               or normalized feature vectors. 
#' @return string, filename of the desired output file
#' 
#' @examples 
#'   1) accuracies_ltso=False_normalized_features=False.csv
#'   2) simple_statistics_set=all_ltso=False_normalized_features=False.csv
#'   ...
get_result_fn <- function (
    prefix = "accuracies", system_set = NULL,
    flavor = list (ltso = "False", normfeat = "False")) 
{
    flavor_str <- list2str (flavor, sep = "_", sort_names = FALSE)

    if (is.null (system_set)) {
        result_fn <- paste (prefix, flavor_str, sep = "_")
    } else {
        result_fn <- paste (prefix, system_set, flavor_str, sep = "_")
    }
        
    result_fn <- paste (result_fn, "csv", sep = ".")
    return (result_fn)
}

#' Load the results for different settings
#'   
#' @references 'get_input_dir': Based on the experimental setting the correct
#'             input directory is created. This one can than be used here.
#'             
#' @param base_dir string, string, path to the directory under which all 
#'                 experimental results are stored.
#' @param flavor list, of "experimental flavors", e.g. leave-target-system-out
#'               or normalized feature vectors. 
#' @param measure string, measure to load:
#'                'accurarcy': Pairwise prediction accuracy of the RankSVM
#'                'rank_corr': Rank-correlation of the pseudo target-values
#'                'spear_corr': Spearman-correlation of -"-
load_results <- function (
    scenario, measure = "accuracy", base_dir = "./", 
    pair_params = list (type = "combn"), feature_type = "difference",
    predictor = paste (c("desc"), sep = "_"), kernel = "rbf", 
    flavor = list (ltso = "False", normfeat = "False"), kernel_column = NULL,
    predictor_column = NULL)
{
    # Determine the correct input dir based on the experimental setting
    input_dir <- get_input_dir (
        base_dir, pair_params, feature_type, predictor, kernel, scenario, 
        predictor_column, kernel_column)
    
    # Build the result filename
    prefix <- switch (measure[1],
                      "accuracy" =, "accuracy_w" = "accuracies",
                      "rt_diff" = "rt_diffs",
                      "accuracy_concor" = , "accuracy_concor_w" = "accuracies_concorr",
                      "accuracy_discor" = , "accuracy_discor_w" = "accuracies_discor",
                      "rank_corr" =, "spear_corr" =, "rank_corr_w" = "correlations",
                      stop ("No valid measure provided."))
    result_fn <- get_result_fn (prefix = prefix, flavor = flavor)
    
    # Load the results into data.table
    fn <- paste (input_dir, result_fn, sep = "/")
    if (! file.exists (fn)) {
        stop (paste ("No such file:", fn))
    }
    
    dt <- data.table (read.csv (fn, sep = "\t"))
    setnames (dt, "target_system",   "target")
    setnames (dt, "training_system", "source")
    setkey (dt, "target", "source")
    
    # Subset columns based on measure
    if (length (measure) == 1) {
        measure_str <- switch (measure,
                               "accuracy" =, "accuracy_concor" =, "accuracy_discor" = "score",
                               "accuracy_w" =, "accuracy_concor_w" =, "accuracy_discor_w" = "score_w",
                               "rank_corr" =, "spear_corr" =, "rank_corr_w" =, "rt_diff" = measure,
                               stop ("No valid measure provided."))
        if (all (c("d_lower", "d_upper") %in% colnames (dt))) {
            dt <- dt[, .(measure = get (measure_str), target, source, d_lower, d_upper)]    
        }
        else if (all (c("test_size", "d_upper") %in% colnames (dt))) {
            dt <- dt[, .(measure = get (measure_str), target, source, test_size, d_upper)]    
        }
        else {
            dt <- dt[, .(measure = get (measure_str), target, source)]    
        }
        setnames (dt, "measure", measure)  
    } 
    else if (length (measure) == 2) {
        measure_str <- switch (measure[1],
                               "accuracy" =, "accuracy_concor" =, "accuracy_discor" = "score",
                               "accuracy_w" =, "accuracy_concor_w" =, "accuracy_discor_w" = "score_w",
                               "rank_corr" =, "spear_corr" =, "rank_corr_w" =, "rt_diff" = measure[1],
                               stop ("No valid measure provided."))
        measure_std_str <- switch (
            measure[2],
            "accuracy_std" =, "accuracy_concor_std" =, "accuracy_discor_std" = "score_std",
            "accuracy_w_std" =, "accuracy_concor_w_std" =, "accuracy_discor_w_std" = "score_w_std",
            "rank_corr_std" =, "spear_corr_std" =, "rank_corr_w_std" =, "rt_diff_std" = measure[2],
            stop ("No valid measure provided."))
        if (all (c("d_lower", "d_upper") %in% colnames (dt))) {
            dt <- dt[, .(measure = get (measure_str), measure_std = get (measure_std_str), 
                         target, source, d_lower, d_upper)]    
        }
        else if (all (c("test_size", "d_upper") %in% colnames (dt))) {
            dt <- dt[, .(measure = get (measure_str), measure_std = get (measure_std_str), 
                         target, source, test_size, d_upper)]    
        }
        else {
            dt <- dt[, .(measure = get (measure_str), measure_std = get (measure_std_str), 
                         target, source)]    
        }
        setnames (dt, "measure", measure[1])  
        setnames (dt, "measure_std", measure[2])
    }
    
    return (dt)
}

#' Load the results for the baseline setting:
#'   Performance of the RankSVM for each system pair (s_i,s_j).
load_baseline_results <- function (...)
{
    return (load_results (scenario = "baseline", ...))
}

#' Load the results for the centering and normalization comparison
load_test_centering_and_normalizing_results <- function (
    scaler = "unscaled", centernormfeat = "False", ...)
{
    return (load_results (scenario = "test_centering_and_normalizing", 
                          flavor = list (scaler = scaler, centernormfeat = centernormfeat), ...))
}

#' Load the results for the different rank differences
load_test_different_rank_differences_for_pairs <- function (
    scaler = "minmax", d = 1, ...)
{
    return (load_results (
        scenario = "test_different_rank_difference_for_pairs", 
        pair_params = list (type = "combn", d = d),
        flavor = list (scaler = scaler), ...))
}

#' Load the results for the all_on_one setting:
load_all_on_one_results <- function (...)
{
    return (load_results (scenario = "all_on_one", ...))
}

#' Load the results for the all_on_one_2 setting:
load_all_on_one_2_results <- function (...)
{
    return (load_results (scenario = "all_on_one_2", ...))
}

#' Load the results for the all_on_one_perc setting:
load_all_on_one_perc_results <- function (...)
{
    return (load_results (scenario = "all_on_one_perc", ...))
}

#' Load the results for the baseline_single setting:
load_baseline_single_results <- function (...)
{
    return (load_results (scenario = "baseline_single", ...))
}

#' Load the results for the baseline_single_perc setting:
load_baseline_single_perc_results <- function (...)
{
    return (load_results (scenario = "baseline_single_perc", ...))
}

#' Load the results for the evaluation of the column selectivity descriptors
load_evaluation_column_selectivity_descriptors <- function (...)
{
    return (load_results (scenario = "evaluate_column_selectivity_descriptors", ...))
}

#' Load the results for the metabolite identification reranking
load_topkacc_of_reranked_molecules <- function (
    scenario = "evaluate_metabolite_identification_performance",
    prefix = "topk_acc", base_dir, pair_params, feature_type, predictor, kernel, 
    flavor, predictor_column, kernel_column) 
{
    # Determine the correct input dir based on the experimental setting
    input_dir <- get_input_dir (
        base_dir, pair_params, feature_type, predictor, kernel, scenario, 
        predictor_column, kernel_column)
    
    # Construct the filename of the result-file
    result_fn <- get_result_fn (prefix = prefix, flavor = flavor)
    
    # Load the results into data.table
    fn <- paste (input_dir, result_fn, sep = "/")
    if (! file.exists (fn)) {
        stop (paste ("No such file:", fn))
    }
    
    return (data.table (read.csv (fn, sep = "\t")))
}

load_topkacc_of_reranked_molecules_GS_BS <- function (...) {
    return (load_topkacc_of_reranked_molecules (
        scenario = "met_ident_perf_GS_BS", ...))
}

#' Load the results for the pairwise prediction accuracy of the target system
#' in the metabolite identification setting.
load_pwacc_in_reranking_target_system <- function (...) 
{
    return (load_results (scenario = "evaluate_column_selectivity_descriptors", ...))
}

#' Load the results for the retention time difference of the target system
#' in the metabolite identification setting.
load_rtdiff_in_reranking_target_system <- function (...) 
{
    return (load_results (scenario = "evaluate_column_selectivity_descriptors", ...))
}


#' Load the simple statistics 
#'   
load_statistics <- function (
    scenario, base_dir = "./", prefix = "simple_statistics",
    pair_params = list (type = "combn"), feature_type = "difference",
    predictor = paste (c("desc"), sep = "_"), kernel = "rbf", 
    flavor = list (ltso = "False", normfeat = "False"))
{
    # Determine the correct input dir based on the experimental setting
    input_dir <- get_input_dir (
        base_dir, pair_params, feature_type, 
        predictor, kernel, scenario)
    
    # Build the result filename
    result_fn <- get_result_fn (prefix = prefix, flavor = flavor)
    
    fn <- paste (input_dir, result_fn, sep = "/")
    if (! file.exists (fn)) {
        stop (paste ("No such file:", fn))
    }
    
    # Load the results into data.table
    dt_baseline <- data.table (read.csv (paste (input_dir, result_fn, sep = "/"), sep = "\t"))
    setnames (dt_baseline, "target_system",   "target")
    setnames (dt_baseline, "training_systems", "source")
    setkey (dt_baseline, "target", "source")
    
    return (dt_baseline)
}

load_baseline_simple_statistics <- function (...)
{
    return (load_simple_statistics (scenario = "baseline", ...))
}

get_pairwise_correlation <- function (
    db, with_self_correlation = TRUE, rm_na = FALSE, with_both_directions = FALSE,
    same_mol_based_on = "inchi", system_id_col_str = "system", rt_col_str = "rt",
    min_n_inter = 5)
{
    # Create pairs of systems
    system_ids <- sort (unique (db[[system_id_col_str]]))
    system_pairs <- t(combn (system_ids, 2))
    system_pairs <- data.table (
        A = factor (system_pairs[, 1], levels = system_ids),
        B = factor (system_pairs[, 2], levels = system_ids))
    
    # Calculate rank-correlations and number of shared molecules
    df <- t(apply (system_pairs, MARGIN = 1, FUN = function (pair) {
        db_pair_merged <- merge (db[get(system_id_col_str) %in% pair[1]],
                                 db[get(system_id_col_str) %in% pair[2]],
                                 by = same_mol_based_on)
        n_inter <- nrow (db_pair_merged)
        
        # If the intersection is to small no rank-correlation can be
        # calculated
        if (n_inter >= min_n_inter) {
            rank_corr  <- cor (db_pair_merged[[paste0 (rt_col_str, ".x")]],
                               db_pair_merged[[paste0 (rt_col_str, ".y")]],
                               method = "kendall")
            spear_corr <- cor (db_pair_merged[[paste0 (rt_col_str, ".x")]], 
                               db_pair_merged[[paste0 (rt_col_str, ".y")]],
                               method = "spearman")
            pears_corr <- cor (db_pair_merged[[paste0 (rt_col_str, ".x")]], 
                               db_pair_merged[[paste0 (rt_col_str, ".y")]],
                               method = "pearson")
        } else {
            rank_corr  <- NA
            spear_corr <- NA
            pears_corr <- NA
        }
        
        return (c(rank_corr, spear_corr, pears_corr, n_inter))
    }))
    
    pairwise_corr <- data.table (
        from = system_pairs$A, to = system_pairs$B,
        rank_corr = df[, 1], spear_corr = df[, 2], pears_corr = df[, 3], n_inter = df[, 4])
    
    if (with_both_directions) {
        pairwise_corr <- rbind (
            pairwise_corr,
            data.table (
                from = system_pairs$B, to = system_pairs$A,
                rank_corr = df[, 1], spear_corr = df[, 2], 
                pears_corr = df[, 3], n_inter = df[, 4])
        )
    }
    
    if (rm_na) {
        pairwise_corr <- pairwise_corr[! is.na (pairwise_corr$rank_corr)]
    }
    
    # Add systems self-correlation
    if (with_self_correlation) {
        pairwise_corr <- rbind (
            pairwise_corr, 
            db[, list (to = unique (get(system_id_col_str)),
                       rank_corr = cor (get(rt_col_str), get(rt_col_str), 
                                        method = "kendall"),
                       spear_corr = cor (get(rt_col_str), get(rt_col_str),
                                         method = "spearman"),
                       pears_corr = cor (get(rt_col_str), get(rt_col_str),
                                         method = "pearson"),
                       n_inter = .N),
               keyby = .(from = get(system_id_col_str))])
    }
    
    return (pairwise_corr)
}

#' Plot the pairwise correlation between CSs. These correlations are calcualted
#' based on the retention-times of molecules measured with both (pairwise) CSs.
#' 
#' @param pairwise_corr data.table, containing the pairwise correlations of 
#'                      different CSs. 
#' @param measure string, which correlation should be used for plotting:
#'                - "kendall": Kendall's Tau rank correlation
#'                - "spearman": Spearmans correlation
#' @param label string, which information should be plotted in the cells:
#'              - "measure": correlation chosen by @param measure
#'              - "n_inter": number of molecules in the intersection
#' @param filename string, filename of the output-file to save the plot. The 
#'                 default value is 'NULL' which means, that a ggplot2 object
#'                 is returned and nothing is written to a file.
#'                 (default = 'NULL')
#' @param filepath string, path to the output-directory to store the plot.
#'                 (default = 'NULL')
#' @param form_str string, columnname containing the source systems, columns
#'                 (default = "from")
#' @param to_str string, columnname containing the target systems, rows 
#'               (default = "to")
#'                 
#' @return ggplot2 object if @param filepath or @param filename is.null.
#'                      
#' @seealso get_pairwise_correlation
plot_pairwise_correlation <- function (
    pairwise_corr, measure = "kendall", label = "measure", filename = NULL, 
    filepath = NULL, from_str = "from", to_str = "to", bold_text_if = NULL, 
    border_cell_if = NULL, label_rounding_digits = 3) 
{
    stopifnot (measure %in% c("kendall", "spearman", "accuracy", "accuracy_w"))
    
    if (is.null (bold_text_if)) {
        pairwise_corr$fontface <- "plain"
    } 
    else {
        pairwise_corr$fontface <- ifelse (pairwise_corr[[bold_text_if]],
                                          "bold", "plain")
    }
    
    pairwise_corr[[from_str]] <- factor (pairwise_corr[[from_str]])
    pairwise_corr[[to_str]] <- factor (pairwise_corr[[to_str]])
    
    measure_type_str <- switch (
        measure, "kendall" = "rank_corr", "spearman" = "spear_corr",
        "accuracy" = "accuracy", "accuracy_w" = "accuracy_w")
    cmidpoint <- switch (measure, "kendall" =, "spearman" = 0, "accuracy" =, "accuracy_w" = 0.5)
    
    p <- ggplot (pairwise_corr, aes(x = get (from_str), y = get (to_str))) +
        geom_raster (aes (fill  = get (measure_type_str),
                          alpha = ifelse (is.na (get (measure_type_str)), 0, 1))) +
        geom_text (aes (label = switch (label, 
                                        measure = round (get (measure_type_str), label_rounding_digits), 
                                        n_inter = n_inter),
                        fontface = fontface), size = 3.0) +
        scale_alpha_continuous (range = c(0, 1), guide = "none") +
        scale_fill_gradient2 (low  = "red", mid  = "white", high = "blue",
                              name = measure, midpoint = cmidpoint) +
        labs (x = NULL, y = NULL, title = NULL) +
        theme (axis.text.x     = element_text (angle = 45, hjust = 1),
               legend.position = "right")
    
    
    if (! is.null (border_cell_if)) {
        pairwise_corr$with_border <- pairwise_corr[[border_cell_if]]
        p <- p + geom_rect (aes (xmin  = as.numeric (get (from_str)) - 0.5, xmax = as.numeric (get (from_str)) + 0.5,
                                 ymin  = as.numeric (get (to_str)) - 0.5, ymax = as.numeric (get (to_str)) + 0.5),
                            alpha = 0, color = ifelse (pairwise_corr$with_border, "black", NA), size = 0.5)
    }
    
    if ("setting" %in% colnames (pairwise_corr)) {
        p <- p + facet_wrap (~ setting, ncol = 3, scales = "free")
    }
    
    if (! any (is.null(filename), is.null (filepath))) {
        ggsave (filename = filename, path = filepath,
                width = 15, height = 9)
    }
    
    return (p)
}
