#' Calculate the pairwise rank / spearman correlation of different CS based
#' on the retention-times of shared molecules.
#' 
#' @param db data.table, containing the retentio-time measurements. The table
#'           must contain the following columns:
#'           - "inchi": (or the name defined by @param same_mol_based_on)
#'           - "system": the CS a molecule has been measured with
#'           - "rt": the retention time of a molecule
#' @param rm_na boolean, should system pairs be excluded those correlation is
#'              NA. This value is assigned to system pairs those number of 
#'              shared molecules is less then @param min_n_inter
#' @param same_mol_based_on string, column in the table identifying a molecule,
#'                          e.g. the InChI of a molecule.
#' @param min_n_inter integer, minimum number of shared molecules between two
#'                    systems so that a correlation is calculated.
#'                    
#' @return data.table, pairwise correlations and number of shared molecules for 
#'         all pairs of systems that can be constructed from @param db.
get_pairwise_correlation <- function (
    db, with_self_correlation = TRUE, rm_na = FALSE, with_both_directions = FALSE,
    same_mol_based_on = "inchi", min_n_inter = 5)
{
    # Create pairs of systems
    systems <- sort (unique (db$system))
    system_pairs <- t(combn (systems, 2))
    system_pairs <- data.table (
        A = factor (system_pairs[, 1], levels = systems),
        B = factor (system_pairs[, 2], levels = systems))

    # Calculate rank-correlations and number of shared molecules
    df <- t(apply (system_pairs, MARGIN = 1, FUN = function (pair) {
        db_pair_merged <- merge (db[system %in% pair[1]],
                                 db[system %in% pair[2]],
                                 by = same_mol_based_on)
        n_inter <- nrow (db_pair_merged)

        # If the intersection is to small no rank-correlation can be
        # calculated
        if (n_inter >= min_n_inter) {
            rank_corr  <- cor (db_pair_merged$rt.x, db_pair_merged$rt.y,
                               method = "kendall")
            spear_corr <- cor (db_pair_merged$rt.x, db_pair_merged$rt.y,
                               method = "spearman")
        } else {
            rank_corr  <- NA
            spear_corr <- NA
        }

        return (c(rank_corr, spear_corr, n_inter))
    }))

    pairwise_corr <- data.table (
        from = system_pairs$A, to = system_pairs$B,
        rank_corr = df[, 1], spear_corr = df[, 2], n_inter = df[, 3])
    
    if (with_both_directions) {
        pairwise_corr <- rbind (
            pairwise_corr,
            data.table (
                from = system_pairs$B, to = system_pairs$A,
                rank_corr = df[, 1], spear_corr = df[, 2], n_inter = df[, 3])
        )
    }

    if (rm_na) {
        pairwise_corr <- pairwise_corr[! is.na (pairwise_corr$rank_corr)]
    }
        
    # Add systems self-correlation
    if (with_self_correlation) {
        pairwise_corr <- rbind (
            pairwise_corr, 
            db[, list (to = unique (system),
                       rank_corr = cor (rt, rt, method = "kendall"),
                       spear_corr = cor (rt, rt, method = "spearman"),
                       n_inter = .N),
               keyby = .(from = system)])
    }

    return (pairwise_corr)
}

get_pairwise_column_dissimilarity <- function (
    db, with_self_dissimilarity = TRUE, rm_na = FALSE, 
    with_both_directions = FALSE, weights = c(12.5, 100, 30, 143, 83))
{
    # Create pairs of systems
    systems <- sort (unique (db$system))
    system_pairs <- t(combn (systems, 2))
    system_pairs <- data.table (
        A = factor (system_pairs[, 1], levels = systems),
        B = factor (system_pairs[, 2], levels = systems))
    
    # Caculate the column dissimilarity
    df <- t(apply (system_pairs, MARGIN = 1, FUN = function (pair) {
        HSABC_1 <- as.matrix (db[system == pair[1], .(H, S., A, B, C2.8)])
        HSABC_2 <- as.matrix (db[system == pair[2], .(H, S., A, B, C2.8)])
        
        # This forumla is originating from:
        #   <http://www.hplccolumns.org/database/compare.php>.
        Fs_12 <- sqrt (sum (((HSABC_1 - HSABC_2) * weights) ** 2))
        
        return (Fs_12)
    }))
    
    db_column_dissimilarity <- data.table (
        from = system_pairs$A, to = system_pairs$B, Fs_12 = t(df)[,1])
    
    if (with_both_directions) {
        db_column_dissimilarity <- rbind (
            db_column_dissimilarity,
            data.table (from = system_pairs$B, 
                        to = system_pairs$A, Fs_12 = t(df)[,1]))
    }
    
    if (rm_na) {
        db_column_dissimilarity <- db_column_dissimilarity[! is.na (db_column_dissimilarity$Fs_12)]
    }

    # Add systems self-dissimilarity
    if (with_self_dissimilarity) {
        db_column_dissimilarity <- rbind (
            db_column_dissimilarity, 
            db[, list (to = unique (system), Fs_12 = 0), keyby = .(from = system)])
    }
    
    return (db_column_dissimilarity)
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

plot_pairwise_column_dissimilarity <- function (
    db_column_dissimilarity, filename = NULL, filepath = NULL,
    border_cell_if = NULL)
{
    p <- ggplot (db_column_dissimilarity, aes(x = from, y = to)) +
        geom_raster (aes (fill  = Fs_12,
                          alpha = ifelse (is.na (Fs_12), 0, 1))) +  
        geom_text (aes (label = round (Fs_12, 2)),
                   size     = 2.5) +
        scale_alpha_continuous (range = c(0, 1), guide = "none") +
        scale_fill_gradient2 (low  = "green",
                              high = "red",
                              name = "Dissimilarity") +
        labs (x = NULL, y = NULL, title = NULL) +
        theme (axis.text.x     = element_text (angle = 45, hjust = 1),
               legend.position = "right")
    
    if (! is.null (border_cell_if)) {
        db_column_dissimilarity$with_border <- db_column_dissimilarity[[border_cell_if]]
        p <- p + geom_rect (aes (xmin  = as.numeric (from) - 0.5, xmax = as.numeric (from) + 0.5,
                                 ymin  = as.numeric (to) - 0.5, ymax = as.numeric (to) + 0.5),
                            alpha = 0, color = ifelse (db_column_dissimilarity$with_border, "black", NA), size = 0.5)
    }
    
    if ("setting" %in% colnames (db_column_dissimilarity)) {
        p <- p + facet_wrap (~ setting, ncol = 3, scales = "free")
    }
    
    if (! any (is.null(filename), is.null (filepath))) {
        ggsave (filename = filename, path = filepath,
                width = 15, height = 9)
    }
    
    return (p)
}

get_rank_correlation_with_global_ordering <- function (db,
                                                       ordering,
                                                       db.id,
                                                       same_mol_based_on = "inchi")
{
    stopifnot (same_mol_based_on == "inchi")

    setkey (db.id, ID)
    mol.identifier.global.ordering <- db.id[ordering]$inchi

    systems <- sort (unique (db$system))

    rank.cors <- rep (NA, length (systems))
    n_inter   <- rep (NA, length (systems))
    p.values   <- rep (NA, length (systems))
    idx <- 1

    for (sys in systems) {
        db.system <- db[system == sys]

        setkey (db.system, rt)

        mol.identifier <- do.call (paste, db.system[, lapply (.SD, FUN = identity),
                                                    .SDcols = same_mol_based_on])

        res <- cor.test (1:nrow (db.system),
                         match (mol.identifier, mol.identifier.global.ordering),
                         measure = "kendall", exact = FALSE)

        rank.cors[idx] <- res$estimate
        p.values[idx]  <- res$p.value
        n_inter[idx]   <- nrow (db.system)
        idx <- idx + 1
    }

    return (data.table (rank_corr = rank.cors, p.value.kend = p.values,
                        system = systems, n_inter = n_inter))
}

get_rank_correlation_with_equivalence_classes <- function (db,
                                                           db.ranks)
{
    # The column "inchi" can be brought to db.ranks using the "merge"
    # function
    stopifnot ("inchi" %chin% colnames (db.ranks))
    setkey (db.ranks, inchi)

    # Output-vectors for the rank-correlations
    rank.cors <- n_inter <- p.values <- vector ()

    # Calculate the rank-correlation for all available systems
    systems <- unique (db$system)
    for (sys in systems) {
        db.system <- db[system == sys]

        db.rt_rank <- merge (db.system, db.ranks, by = "inchi")

        if (nrow (db.rt_rank) >= 3) {
            res <- cor.test (db.rt_rank$rt, db.rt_rank$rank,
                             measure = "kendall", exact = FALSE)

            rank.cors <- c(rank.cors, res$estimate)
            p.values  <- c(p.values, res$p.value)
        } else {
            rank.cors <- c(rank.cors, NA)
            p.values  <- c(p.values, NA)
        }

        n_inter   <- c(n_inter, nrow (db.rt_rank))

        # setkey (db.system, inchi)



        # Get the rank for each entry with respect to it's retention-time
        # ranks.system <- frank (db.system, rt, ties.measure = "dense")

        # Get the rank for each entry with respect to it's equivalence
        # class
        # ranks.equcla <- db.ranks[db.system$inchi]$rank

        # Calculate the rank-correlation
        # res <- cor.test (ranks.system, ranks.equcla,
                         # measure = "kendall", exact = FALSE)


    }

    return (data.table (rank_corr = rank.cors, p.value.kend = p.values,
                        system = systems, n_inter = n_inter))
}







plot_rank_correlation_with_global_ordering <- function (global.order.cor, measure = "kendall",
                                                        dependency.n.mol = FALSE, group.by = NULL,
                                                        filename = NULL, filepath = NULL, ylim = c(NA, NA))
{
    stopifnot (measure %chin% c("kendall", "spearman"))
    corr_type_str <- switch (measure, "kendall" = "rank_corr", "spearman" = "rank.cors.spear")

    if (dependency.n.mol) {
        p <- ggplot (global.order.cor)
        p <- p + geom_point (aes (x = system, y = n_inter, color = get (corr_type_str)),
                             size = 5, alpha = 1)
        p <- p + scale_color_gradient2 (low = "red", mid = "white", high = "blue",
                                        name = paste0 ("Rank correlation (", measure, ")"))
        p <- p + labs (x = "System", y = "# of molecules",
                       title = "Rank correlation of the different systems with global ordering")
        # p <- p + theme (axis.text.x = element_text (angle = 45, hjust = 1))
    } else {
        p <- ggplot (global.order.cor, aes (system, get (corr_type_str)))

        if (! is.null (group.by))
            p <- p + geom_bar (aes (fill = get (group.by)),
                               stat = 'identity', position = "dodge")
        else
            p <- p + geom_bar (stat = 'identity')

        p <- p + scale_fill_discrete ("Algorithm / Graph")

        p <- p + labs (x = "System", y = paste0 ("Rank correlation (", measure, ")"),
                       title = "Rank correlation of the different systems with global ordering")
        p <- p + lims (y = ylim)

    }
    p <- p + theme (axis.text.x = element_text (angle = 45, hjust = 1),
                    legend.position = "bottom")


    if (! is.null(filename) & ! is.null (filepath))
        ggsave (filename = filename, path = filepath,
                width = 12, height = 9)

    return (p)
}

# plot_rank_correlation_with_global_ordering <- function (global.order.cor, measure = "kendall",
#                                                         filename = NULL, filepath = NULL)
# {
#     stopifnot (measure %chin% c("kendall", "spearman"))
#     corr_type_str <- switch (measure, "kendall" = "rank_corr", "spearman" = "rank.cors.spear")
#
#     p <- ggplot (global.order.cor, aes (system, get (corr_type_str)))
#     p <- p + geom_bar (stat = 'identity')
#     p <- p + labs (x = "System", y = paste0 ("Rank correlation (", measure, ")"),
#                    title = "Rank correlation of the different systems with global ordering")
#     p <- p + theme (axis.text.x = element_text (angle = 45, hjust = 1))
#
#     return (p)
# } 
