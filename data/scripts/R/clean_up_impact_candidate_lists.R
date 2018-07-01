require (data.table)

# Task: Compare the compounds in the scoring lists for the Impact candidates 
#       with the compounds for which we could calculate the fingerprints. This
#       sometimes fails for example when a SMILES (descripting a compound) could
#       not be parsed. If not for all compounds the fingerprints could be 
#       computed, we remove those compounds from the candidates list. We have to
#       ensure that the removed compounds to not contain the correct 
#       identification.

# 0) Set up the working directories

# 0-a) Which fingerprint defintion should be considered?
fps_def <- "maccs_count"

sdir <- "/home/bach/Documents/studies/doctoral/projects/rt_prediction_ranksvm/data/processed/Impact/candidates/"
fps_dir <- paste0 (sdir, "/fingerprints/", fps_def, "/")
scorings_dir <- paste0 (sdir, "/scorings/before_cleaning_up/")
scorings_dir_out <- paste0 (sdir, "/scorings/", fps_def, "/")
if (! dir.exists (scorings_dir_out)) {
    dir.create (scorings_dir_out)
}

# 0-b) Load the list that maps the InChIs in the candidate and fingerprint lists
#      to the InChI-keys that are used to identify the scoring lists. This needs
#      to be done as we do not have only the 2D InChIs in the candidate and 
#      fingerprint lists. 
map_inchikey2inchi <- data.table (read.csv (paste0 (sdir, "/inchikey2inchi.csv")))

# 2) Iterate over all scoring lists and load the corresponding fingerprints:
n_succ_fps <- 0
idx_scr_file <- 1
scoring_files <- list.files (scorings_dir, pattern = "scoring_list*")
for (scr_file in scoring_files) {
    print (paste0 ("Process: ", scr_file, " (", idx_scr_file, "/", 
                   length (scoring_files), ")"))
    idx_scr_file <- idx_scr_file + 1
    
    # 2-a) Check whether the corresponding fingerprints have been already computed.
    fps_file <- gsub("scoring_", paste0 ("fps_", fps_def, "_"), scr_file)
    
    if (! file.exists (paste0 (fps_dir, "/", fps_file))) {
        print (paste ("Missing fingerprints:", fps_file))
        next
    } 
    
    n_succ_fps <- n_succ_fps + 1
    
    scores <- data.table (read.csv (paste0 (scorings_dir, "/", scr_file)))
    fps <- data.table (read.csv (paste0 (fps_dir, "/", fps_file)))
    
    # Extract the InChI-key of the true / correct candidates
    true_cand_inchikey <- strsplit (strsplit (scr_file, "=")[[1]][4], "[.]")[[1]][1]
    stopifnot (map_inchikey2inchi[inchikey == true_cand_inchikey, inchi] %in% scores$id1)
    
    # Remove entries in the fps-list that contain NA values
    # Note: Here we only check the first fp as the remaining ones should be NA
    #       as well in this case.
    if (any (is.na (fps$maccs_1))) {
        fps <- fps[! is.na(maccs_1)]
        rewrite_fps <- TRUE
    } else {
        rewrite_fps <- FALSE
    }
    
    # 3) Compare the compounds contained in both lists
    # 3-a) If all InChIs are the same --> everything all right --> next
    if (all (as.character (scores$id1) %in% as.character (fps$inchi))) { next }
    
    # 4) Remove those compounds from the scoring list, for which we do not have
    #    any fingerprints.
    inchi_shared <- intersect (scores$id1, fps$inchi)
    
    # This check would fail, if InChIs are repeated within the scoring or fps
    # list. 'intersect' would remove those repetitions.
    stopifnot (length (inchi_shared) == length (fps$inchi))
    
    scores_out <- scores[id1 %in% inchi_shared]
    stopifnot (all (sort (as.character (scores_out$id1)) == sort (as.character (fps$inchi))))
    # Check that the true / correct candidate remains after the clean up.
    stopifnot (map_inchikey2inchi[inchikey == true_cand_inchikey, inchi] %in% scores_out$id1)
    
    # 4-a) Write out the new scoring list:
    scr_file_out <- gsub("scoring_list", "scoring_list_cleaned", scr_file)
    write.csv (scores_out, file = paste0 (scorings_dir_out, "/", scr_file_out),
               row.names = FALSE, quote = TRUE)
    
    # 4-b) Write out the new fps list (if we had to remove NA entries):
    if (rewrite_fps) {
        fps_file_out <- gsub(paste0 ("fps_", fps_def), paste0 ("fps_", fps_def, "_cleaned"), fps_file)
        write.csv (fps, file = paste0 (fps_dir, "/", fps_file_out), 
                   row.names = FALSE, quote = TRUE)
    }
}

print (paste0 ("Fingeprints successfully calcualte: ", n_succ_fps, "/", 
               length (scoring_files)))
