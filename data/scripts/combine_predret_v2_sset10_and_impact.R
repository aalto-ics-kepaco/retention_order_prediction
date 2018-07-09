#
# Script to combine the PredRet and Impact (from Massbank) RTs / FPs. 
#
# MACCS counting FPs for the Impact data (only the structures without MSMS) are
# calculated as well.
#
# The combined set is needed for the experiments regarding the metabolite 
# identification presented in Section 3.2. There, we refer to the combined 
# dataset as "Others & target", with "others" encompassing the data from PredRet
# and target encompassing the data from Impact for which _no_ MSMS spectra is
# available. 
#

require (data.table)

#' Calculate the inchikey for a list of inchis
#' 
#' @param inchi list of strings, inchis describing the molecules
#' @param only_2D binary, remove the 3D and polarization information from the key
#'                E.g.: XX2DXX-XX3DXX-POL --> XX2DXX
#' @param verbose
#' @return list of strings, inchikeys
inchi2inchikey <- function (inchi, only_2D = FALSE, verbose = FALSE) {
    inchi <- split (inchi, ceiling (seq_along (inchi) / 100))
    
    output <- sapply (inchi, function(x) {
        string <- paste (as.character(x), collapse = '" -:"')
        system (paste0 ('obabel -iinchi -:"', string, '" -oinchikey') , intern = TRUE, ignore.stderr = !verbose)
    })
    
    output <- as.character (unlist (output)) 
    
    if (only_2D) {
        output <- sapply (output, FUN = function (x) strsplit (x, "-")[[1]][1], USE.NAMES = FALSE)
    }
    
    return (output)
}

base_dir <- stop ("DEFINE BASE-PATH CONTAINING INPUT DATA!") 
# Eg: base_dir <- "~/Documents/studies/doctoral/projects/rt_prediction_ranksvm/method_publishing/data"

# Load the fingerprint tools
source (paste (base_dir, "scripts/fingerprint_tools.R", sep = "/"))

# Set the source directory for the data
sdir <- paste (base_dir, "processed/s10_imp_no3D/", sep = "/")

#### Retention times ####

# Load Predret (v2, sset10) retention times
sset10 <- c("Eawag_XBridgeC18", "FEM_long", "RIKEN", "UFZ_Phenomenex", "LIFE_old")
rts_predret_set10 <- data.table (read.csv (paste0 (sdir, "rts_predret.csv")))
rts_predret_set10 <- rts_predret_set10[system %in% sset10]

# Load impact retention times
rts_impact <- data.table (read.csv (paste0 (sdir, "rts_impact.csv")))
# Remove charge and stereo information from the compounds
rts_impact$inchikey <- inchi2inchikey (rts_impact$inchi)
rts_impact$inchikey_1 <- inchi2inchikey (rts_impact$inchi, only_2D = TRUE)
rts_impact$system <- "Impact"

# Remove the molecular structures from the Impact dataset that have MS/MS spectra.
rts_impact_with_msms <- data.table (read.csv (paste0 (sdir, "rts_impact_with_msms.csv")))
rts_impact_with_msms$inchikey_1 <- inchi2inchikey (rts_impact_with_msms$inchi, only_2D = TRUE)
rts_impact <- rts_impact[! (inchikey_1 %in% rts_impact_with_msms$inchikey_1)]

# Put PretRed and Impact retention times together.
rts <- rbind (rts_predret_set10, rts_impact)
write.csv (rts, file = paste0 (sdir, "rts.csv"), row.names = FALSE)

#### Fingerprints ####

calculate_count <- FALSE

if (calculate_count) {
    # Load the Predret (counting, maccs) fingerprints
    fps_count_predret <- data.table (read.csv (paste0 (sdir, "fps_maccs_count_predret.csv")))
    fps_count_predret <- fps_count_predret[inchi %in% rts_predret_set10$inchi]
    
    # Calculate the (counting, maccs) fingerprints for the Impact dataset.
    fps_count_impact <- calculate_fingerprints_from_inchi (
        unique (rts_impact$inchi), fps_definitions = "maccs", fps_type = "raw")
    fps_count_impact <- data.table (
        cat_fingerprint_list_to_matrix (fps_count_impact), keep.rownames = TRUE)
    setnames (fps_count_impact, "rn", "inchi")
    
    # Put PretRed and Impact (counting, maccs) fingerprints together.
    fps_count <- unique (rbind (fps_count_predret, fps_count_impact), key = "inchi")
    setkey (fps_count, "inchi")
    stopifnot (all (rts$inchi %in% fps_count$inchi))
    
    write.csv (fps_count, file = paste0 (sdir, "fps_maccs_count.csv"), row.names = FALSE)
}
