library (data.table)
library (foreach)
library (doParallel)

#args <- commandArgs (trailingOnly = TRUE)
#if (length (args) > 0) {
#    basedir_project <- args[1] 
#    num_cluster <- args[2]
#} else {
#    basedir_project <- "/home/bach/Documents/studies/doctoral/projects/rt_prediction_ranksvm/"
#    num_cluster <- 2
#}

basedir_project <- "/home/bach/Documents/studies/doctoral/projects/rt_prediction_ranksvm/"
# basedir_project <- "/scratch/cs/kepaco/bache1//projects/rt_prediction_ranksvm/"
basedir_candidates <- paste0 (basedir_project, "/data/processed/Impact/candidates/")

# Let us start a cluster to do the work
cl <- makeCluster (2)
registerDoParallel (cl)
on.exit (stopCluster(cl))


prepare_candidate_data <- function (basedir_project, basedir_candidates) {
    print ("enter function")
    # Scoring files contain the InChI's for all the candidates
    scoring_files <- list.files (paste0 (basedir_candidates, "/scorings/before_cleaning_up/"), 
                                 pattern = "scoring_list*")
    n_scoring_files <- length (scoring_files)

    # for (idx_sfile in 1:n_scoring_files) {
    
    foreach (idx_sfile = 1:n_scoring_files) %dopar% {
        print (paste0 ("Process: ", scoring_files[idx_sfile], " (", idx_sfile, "/", n_scoring_files, ")"))

        fps_file <- gsub("scoring_", paste0 ("fps_maccs_count_"), scoring_files[idx_sfile])
        if (file.exists(paste0(basedir_candidates, "/fingerprints/maccs_count/", fps_file))) {
            print ("Already ready!")
            next
        }
        
        # Load some tools: Needs to be done in the parallel loop, as otherwise
        # the functions are not visible.
        source (paste0 (basedir_project, "/results/scripts/R/helper.R"))
        source (paste0 (basedir_project, "/data/scripts/R/tools/fingerprint_tools.R"))

        # Calculate candidates fingerprints
        candidates <- data.table (read.csv (paste0 (basedir_candidates, "/scorings/before_cleaning_up/", scoring_files[idx_sfile])))
        setnames (candidates, "id1", "inchi")
        print (paste0 ("Number of candidates: ", nrow (candidates)))

        # Store the fingerprints into a data table
        fps <- calculate_fingerprints_from_inchi (
            candidates$inchi, fps_definitions = "maccs", fps_type = "raw")
        fps <- data.table (
            cat_fingerprint_list_to_matrix (fps), keep.rownames = TRUE)

        setnames (fps, "rn", "inchi")
        stopifnot (all (fps$inchi == candidates$inchi))

        # Write the fingerprints to the disk
        write.csv (fps, row.names = FALSE,
                   file = paste0 (basedir_candidates, "/fingerprints/maccs_count/", fps_file))
    }
}

# prepare_candidate_data (basedir_project, basedir_candidates)
