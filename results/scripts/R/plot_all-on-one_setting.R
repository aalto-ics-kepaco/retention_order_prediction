setwd ("/m/cs/scratch/kepaco/bache1/data/retention_time_prediction/rank_svm/output/")

dt.pairs <- data.table (read.table ("system_pairs/system_pairs-correlations.csv", sep = ",", header = TRUE), 
                        stringsAsFactors = TRUE)
setnames (dt.pairs, "target_system", "target")
setnames (dt.pairs, "training_system", "source")
setkey (dt.pairs, "target", "source")

dt.others <- data.table (read.table ("others_on_one/others_on_one-correlations-pred=desc.csv", sep = ",", header = TRUE))
setnames (dt.others, "target_system", "target")
setnames (dt.others, "training_system", "source")

systems <- unique (dt.pairs$target)

tmp <- rbind (dt.others[, .(rank_corr, target, scenario = "others_on_one")], 
              dt.pairs[.(systems, systems), .(rank_corr, target, scenario = "system_pairs")])