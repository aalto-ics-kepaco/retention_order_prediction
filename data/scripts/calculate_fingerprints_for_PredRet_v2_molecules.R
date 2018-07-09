#
# Script to calculate molecular fingerprints for the PredRet dataset.
# 

require (data.table)

base_dir <- stop ("DEFINE BASE-PATH CONTAINING INPUT DATA!") 
# Eg: base_dir <- "~/Documents/studies/doctoral/projects/rt_prediction_ranksvm/data"

# Load the fingerprint tools
source (paste (base_dir, "/scripts/fingerprint_tools.R", sep = "/"))

# Set the source directory for the data
sdir <- paste (base_dir, "processed/PredRet/v2/", sep = "/")

# Get inchis
rts <- data.table (read.csv (paste0 (sdir, "rts.csv")))

calculate_binary <- FALSE
calculate_count <- FALSE

# Calculate fingerprints (binary)
if (calculate_binary) {
    print ("Calculate binary fingerprints")
    
    fps <- calculate_fingerprints_from_inchi (
        unique (rts$inchi), 
        fps_definitions = c("shortestpath", "standard", "pubchem", 
                            "extended", "circular", "maccs", "kr", "estate"))
    fps <- data.table (cat_fingerprint_list_to_matrix (fps), keep.rownames = TRUE)
    setnames (fps, "rn", "inchi")

    # Save fingerprints
    write.csv (fps, file = paste0 (sdir, "fps.csv"), row.names = FALSE)
}

# Calculate fingerprints (MACCS, counting)
# TODO: Update this part to be compatible with the new MACCS count in CDK!
if (calculate_count) {
    print ("Calculate counting fingerprints")
    
    fps <- calculate_fingerprints_from_inchi (unique (rts$inchi),
                                              fps_definitions = "maccs",
                                              fps_type = "raw")
    fps <- data.table (cat_fingerprint_list_to_matrix (fps), keep.rownames = TRUE)
    setnames (fps, "rn", "inchi")
    
    write.csv (fps, file = paste0 (sdir, "fps_maccs_count.csv"), row.names = FALSE)
}
