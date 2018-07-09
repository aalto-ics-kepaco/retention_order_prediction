#
# Calclate the MACCS counting fingerprints for all the molecular candidates of
# the Impact dataset. 
#
# The function can be changed to handle binary FPs as well.
#
# Compare also the 'clean_up_impact_candidate_lists.R' script.
#

require (data.table)

base_dir <- stop ("DEFINE BASE-PATH CONTAINING INPUT DATA!") 
# Eg: base_dir <- "~/Documents/studies/doctoral/projects/rt_prediction_ranksvm/method_publishing/data"

source (paste (base_dir, "scripts/fingerprint_tools.R", sep = "/"))

basedir_candidates <- paste (base_dir, "/data/processed/Impact/candidates/", sep = "/")

# Scoring files contain the InChI's for all the candidates
scoring_files <- list.files (paste0 (basedir_candidates, "/scorings/before_cleaning_up/"), 
                             pattern = "scoring_list*")
n_scoring_files <- length (scoring_files)

for (idx_sfile in 1:n_scoring_files) {
    print (paste0 ("Process: ", scoring_files[idx_sfile], " (", idx_sfile, "/", n_scoring_files, ")"))

    fps_file <- gsub("scoring_", paste0 ("fps_maccs_count_"), scoring_files[idx_sfile])
    if (file.exists(paste0(basedir_candidates, "/fingerprints/maccs_count/", fps_file))) {
        print ("Already ready!")
        next
    }
    
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
