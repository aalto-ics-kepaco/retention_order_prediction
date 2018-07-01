# Get the intersection between the different systems:
tmp <- data.table (read.table ("../input/Stanstrup_exp_Aug16_aggregated_with_smiles_only_reversed.csv", sep = ",", header = TRUE))
systems <- as.character (unique (tmp$system))
setkey (tmp, "system")
pairs <- combn (systems, 2, simplify = TRUE)
pairs <- rbind (pairs, -1)
for (idx in 1:ncol (pairs)) {
    pairs[3, idx] <- length (intersect (tmp[pairs[1, idx], inchi], tmp[pairs[2, idx], inchi]))
}
pairs <- cbind (pairs, pairs[c(2, 1, 3),])
for (idx in seq_along (systems)) {
    pairs <- cbind (pairs, c(systems[idx], systems[idx], length (tmp[systems[idx], inchi])))
}
pairs <- t(pairs)
colnames (pairs) <- c("target", "source", "n.inter")

# Read the correlation results for the pairwise setting
dt <- data.table (read.table ("system_pairs/system_pairs-correlations.csv", sep = ",", header = TRUE), stringsAsFactors = TRUE)
setnames (dt, "target_system", "target")
setnames (dt, "training_system", "source")

# dt <- merge (dt, data.table (pairs), by = c("target", "source"))
# dt$n.inter <- as.numeric (dt$n.inter)


plot.pairwise.measure (dt, "rank_corr", title = "Molecular descriptors",
                       ofile = "/m/cs/scratch/kepaco/bache1/data/retention_time_prediction/rank_svm/output/rankcorr_desc_order-none_with_values.png")

dt.mean_by_source <- dt[, .(rank_corr = mean (rank_corr), spear_corr = mean (spear_corr)), by = "source"]
dt.mean_by_target <- dt[, .(rank_corr = mean (rank_corr), spear_corr = mean (spear_corr)), by = "target"]

dt.ordered <- dt
dt.ordered$source <- factor (dt.ordered$source, 
                             levels = dt.mean_by_source$source[order (dt.mean_by_source$rank_corr, decreasing = TRUE)])
dt.ordered$target <- factor (dt.ordered$target, 
                             levels = dt.mean_by_target$target[order (dt.mean_by_target$rank_corr, decreasing = TRUE)])
plot.pairwise.measure (dt.ordered, "rank_corr", title = "Molecular descriptors, ordered by source and target performance",
                       ofile = "/m/cs/scratch/kepaco/bache1/data/retention_time_prediction/rank_svm/output/rankcorr_desc_order-both_with_values.png")


# Read the correlation results for the pairwise setting
dt <- data.table (read.table ("system_pairs-correlations-pred=fps.csv", sep = ",", header = TRUE), stringsAsFactors = TRUE)
setnames (dt, "target_system", "target")
setnames (dt, "training_system", "source")

# dt <- merge (dt, data.table (pairs), by = c("target", "source"))
# dt$n.inter <- as.numeric (dt$n.inter)


plot.pairwise.measure (dt, "rank_corr", title = "Molecular fingerprints",
                       ofile = "/m/cs/scratch/kepaco/bache1/data/retention_time_prediction/rank_svm/output/rankcorr_fps_order-none.png")

dt.mean_by_source <- dt[, .(rank_corr = mean (rank_corr), spear_corr = mean (spear_corr)), by = "source"]
dt.mean_by_target <- dt[, .(rank_corr = mean (rank_corr), spear_corr = mean (spear_corr)), by = "target"]

dt.ordered <- dt
dt.ordered$source <- factor (dt.ordered$source, 
                             levels = dt.mean_by_source$source[order (dt.mean_by_source$rank_corr, decreasing = TRUE)])
dt.ordered$target <- factor (dt.ordered$target, 
                             levels = dt.mean_by_target$target[order (dt.mean_by_target$rank_corr, decreasing = TRUE)])
plot.pairwise.measure (dt.ordered, "rank_corr", title = "Molecular fingerprints, ordered by source and target performance",
                       ofile = "/m/cs/scratch/kepaco/bache1/data/retention_time_prediction/rank_svm/output/rankcorr_fps_order-both.png")


# # Read the accuracy results for the pairwise setting
# dt <- data.table (read.table ("system_pairs-mean_accuracies.csv", sep = ",", header = TRUE), stringsAsFactors = TRUE)
# setnames (dt, "target_system", "target")
# setnames (dt, "training_system", "source")
# 
# dt <- merge (dt, data.table (pairs), by = c("target", "source"))
# dt$n.inter <- as.numeric (dt$n.inter)
# 
# 
# plot.pairwise.measure (dt, "score", title = "Molecular descriptors",
#                        ofile = "/m/cs/scratch/kepaco/bache1/data/retention_time_prediction/rank_svm/output/accuracy_desc_order-none.png")
# 
# dt.mean_by_source <- dt[, .(score = mean (score)), by = "source"]
# dt.mean_by_target <- dt[, .(score = mean (score)), by = "target"]
# 
# dt.ordered <- dt
# dt.ordered$source <- factor (dt.ordered$source, 
#                              levels = dt.mean_by_source$source[order (dt.mean_by_source$score, decreasing = TRUE)])
# dt.ordered$target <- factor (dt.ordered$target, 
#                              levels = dt.mean_by_target$target[order (dt.mean_by_target$score, decreasing = TRUE)])
# plot.pairwise.measure (dt.ordered, "score", title = "Molecular descriptors, ordered by source and target performance",
#                        ofile = "/m/cs/scratch/kepaco/bache1/data/retention_time_prediction/rank_svm/output/accuracy_desc_order-both.png")
# 



# # Sort occording to predictive performance of a particular system (source)
# dt.ordered_by_source <- dt

# dt.ordered_by_source$source <- factor (dt.ordered_by_source$source, 
#                                        levels = dt.mean_by_source$source[order (dt.mean_by_source$rank_corr, decreasing = TRUE)])
# plot.pairwise.measure (dt.ordered_by_source, "rank_corr", title = "Molecular descriptors, ordered by source performance")
# 
# # Sort occording to achieveable performance in a particular target system
# dt.ordered_by_target <- dt
# dt.mean_by_target    <- dt.ordered_by_target[, .(rank_corr = mean (rank_corr), spear_corr = mean (spear_corr)), 
#                                              by = "target"]
# dt.ordered_by_target$target <- factor (dt.ordered_by_target$target, 
#                                        levels = dt.mean_by_target$target[order (dt.mean_by_target$rank_corr, decreasing = TRUE)])
# plot.pairwise.measure (dt.ordered_by_target, "rank_corr", title = "Molecular descriptors, ordered by target performance")
# 
# #