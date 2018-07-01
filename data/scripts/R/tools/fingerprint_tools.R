require (Matrix)
require (igraph)
require (obabel2R)
require (rcdk)
require (fingerprint)
require (data.table)

# Script to calculate the molecular fingerprints for each unique entry in
# the retention-time database containing the molecules at the moment of
# the publication.

#' Calculate molecular fingerprints from inchi
#' 
#' @description 
#' This function takes a set of inchis and calculates for those the different 
#' sets of fingerprints. Those fps are defined in the \texttt{rcdk}-package.
#' 
#' For a list of valid fingerprint definitions see \texttt{get.fingerpint}.
#' 
#' @note 
#' The function is a wrapper for \texttt{calculate_fingerprints_from_smiles}.
#' 
#' @param inchi list of strings (1 x n_samples)
#' @param fps_definitions list of strings (1 x n_definitions) containing the
#'                        desired fps definitions
#' @param fps_type string, which fingerprint type should be returned:
#'                 - "bit": binary fingerprint
#'                 - "raw": (key, value)-pairs:
#'                    * key represents the fp definition, e.g. SMARTS string
#'                    * value represents the number of occurances fp
#'                 - "count": counting fingerprint (calculated from raw ones)#'                        
#' @param verbose boolean, should the function be verbose
#' @return list of fingerprints (1 x n_definitions)
#'         list[[i]]: object of class \texttt{fingerprint} or NULL if any error
#'                    occured
calculate_fingerprints_from_inchi <- function (
    inchi,
    fps_definitions = c("pubchem", "maccs", "kr", "circular"),
    fps_type = "bit",
    verbose = TRUE)
{
    smiles <- inchi2smile (inchi)
    fps <- calculate_fingerprints_from_smiles (smiles, fps_definitions, 
                                               fps_type, verbose)
    
    for (fps_definition in fps_definitions) {
        if (is.null (fps[[fps_definition]])) {
            next
        }
        
        stopifnot (names (fps[[fps_definition]]) == smiles)
        names (fps[[fps_definition]]) <- inchi
    }
    
    return (fps)    
}

#' Calculate molecular fingerprints from smiles
#' 
#' @description 
#' This function takes a set of smiles and calculates for those the different 
#' sets of fingerprints. Those fps are defined in the \texttt{rcdk}-package.
#' 
#' For a list of valid fingerprint definitions see \texttt{get.fingerpint}.
#' 
#' @param smiles list of strings (1 x n_samples)
#' @param fps_definitions list of strings (1 x n_definitions) containing the
#'                        desired fps definitions
#' @param fps_type string, which fingerprint type should be returned:
#'                 - "bit": binary fingerprint
#'                 - "raw": (key, value)-pairs:
#'                    * key represents the fp definition, e.g. SMARTS string
#'                    * value represents the number of occurances fp
#'                 - "count": counting fingerprint (calculated from raw ones)
#' @param verbose boolean, should the function be verbose                        
#' @return list of fingerprints (1 x n_definitions)
#'         list[[i]]: list \texttt{fingerprint} objects
#'                    If a smiles could not be parsed, the corresponding finger-
#'                    print is NULL.
calculate_fingerprints_from_smiles <- function (
    smiles,
    fps_definitions = c("pubchem", "maccs", "kr", "circular"),
    fps_type = "bit",
    verbose = TRUE) 
{
    # Parse all smiles and perform configuration
    tictoc::tic ("Parsing and configuration")

    smiles.parsed <- parse.smiles (smiles)
    n.mol <- length (smiles)
    
    verbose <- ifelse (n.mol > 1, verbose, FALSE)

    if (verbose) {
        pb <- txtProgressBar (0, n.mol - 1, style = 3)
    }
    
    for (idx in 1:n.mol) {
        # Sometimes the parsing of a smiles failes. For those molecules
        # no descriptors are calculated.
        if (is.null (smiles.parsed[[idx]])) {
            next
        }
        
        do.aromaticity (smiles.parsed[[idx]])
        do.typing      (smiles.parsed[[idx]])
        do.isotopes    (smiles.parsed[[idx]])
        convert.implicit.to.explicit (smiles.parsed[[idx]])
        do.isotopes    (smiles.parsed[[idx]])
        
        if (verbose) {
            setTxtProgressBar (pb, idx)
        }
    }

    if (verbose) {
        close (pb)
    }
    
    tictoc::toc (log = TRUE, quiet = ! verbose)

    # Calculate all the desired fingerprints
    fps <- list()

    for (fps_definition in fps_definitions) {
        if (verbose) {
            tictoc::tic (sprintf ("\nCalculate the '%s' fingerprints", fps_definition))
            pb <- txtProgressBar (0, n.mol - 1, style = 3)
        }

        fps[[fps_definition]] <- lapply (smiles.parsed,
            function (x, fps_definition) {
                if (is.null (x)) {
                    NULL
                } else {
                    if (verbose){
                        setTxtProgressBar (pb, which (sapply (smiles.parsed, '==', x)))
                    }
                    get.fingerprint (x, type = fps_definition, fp.mode = fps_type)    
                }
            }, fps_definition)
        
        if (is.null (fps[[fps_definition]])) {
            warning (sprintf ("Could not calculate definition '%s'.", fps_definition))
        }

        if (verbose) {
            tictoc::toc (log = TRUE, quiet = ! verbose)
            close (pb)
        }
    }

    return (fps)
}

#' Function to parse one line fingerprint file created using MAYACHEMTOOLS
#'
#' @description 
#'   
#' @references
#' MAYACHEMTOOLS: http://www.mayachemtools.org/
#' 
#' @param line string, one line from the fingerprint file
#' @return matrix-row (1 x n_fingerprints), numeric, rowname corresponds to the
#'         molecule id extracted from the line.
parse_maya_fps_line <- function (line) {
    stopifnot (is.character (line))
    
    line <- strsplit (line, ',')[[1]]

    fps <- string_to_numeric_for_maya_feature_strings (line[2])
    fps <- matrix (fps, nrow = 1)
    
    sample_id <- line[1]
    rownames (fps) <- sample_id
    
    return (fps)
}

#' Functon to parse the feature string in a MAYACHEMTOOLS fingerprint line.
#' 
#' @references 
#' MAYACHEMTOOLS: http://www.mayachemtools.org/
#' 
#' @return numeric vector of length n_fingerprints
string_to_numeric_for_maya_feature_strings <- function (feature_str) {
    fps <- strsplit (feature_str, ';')[[1]]
    
    is_binary <- (fps[1] == "FingerprintsBitVector") # Count: FingerprintsVector
    
    fps <- fps[length (fps)]
    fps <- as.numeric (unlist (strsplit (fps, ifelse (is_binary, "", " "))))
    
    return (fps)
}

#' Function to read a list of fingerprints (csv-file) produced by the 
#' MAYACHEMTOOLS. Each fingerprint will be parsed. A matrix containing the 
#' fingerprints is returned and each row is named after using the first column 
#' of the csv-file, e.g. the InChI.
#' 
#' @references 
#' MAYACHEMTOOLS: http://www.mayachemtools.org/
#' 
#' @param filename string, filename of the csv-file
#' @return matrix (n_samples x n_fingerprints), with rownames corresponding to 
#'         the molecule ids.
read_maya_fps_file_to_matrix <- function (filename) {
    fps_df <- read.csv (filename, stringsAsFactors = FALSE)
    n_samples <- nrow (fps_df)
    
    if (n_samples == 0) {
        fps_mat <- matrix (NA, nrow = 0, ncol = 0)
    } else {
        n_fingerprints <- length (string_to_numeric_for_maya_feature_strings (fps_df[1, 2]))
        fps_mat <- matrix (NA, nrow = n_samples, ncol = n_fingerprints)
        for (idx in 1:n_samples) {
            fps_mat[idx, ] <- string_to_numeric_for_maya_feature_strings (fps_df[idx, 2])
        }
        rownames (fps_mat) <- fps_df[, 1]
    }
    
    return (fps_mat)
}

#' Concatenate list of fingerprints into a matrix
#' 
#' 
#' @param fps list of fingerprints (1 x m) (with m = n_definitions)
#'            list[[i]]: object of class \texttt{fingerprint}
#' @param masks list of fingerprint masks (1 x m)
#'              list[[i]]: binary vector (1 x n_fingerpints):
#'                         TRUE:  keep fingerprints
#'                         FALSE: exclude fingerprint
#' @param as_sparse binary, should the ouput matrix be of class \texttt{sparseMatrix}
#' @return matrix (or \texttt{sparseMatrix}) (n_samples x n_fingerpints') 
#'         n_fingerpints' = n_fingerprints_1 + ... + n_fingerprints_m
cat_fingerprint_list_to_matrix <- function (
    fps, masks = NULL, as_sparse = FALSE) 
{
    stopifnot (is.list (fps))
    
    if (! is.null (masks)) {
        stopifnot (is.list (masks))
        stopifnot (names (masks) == names (fps))
    }
    
    if (length (fps) == 0) {
        if (as_sparse) {
            fps_cat <- sparseMatrix (NA, nrow = 0, ncol = 0)
        } else {
            fps_cat <- matrix (NA, nrow = 0, ncol = 0)
        } 
    } else {
        fps_definitions <- names (fps)    
        sample_ids <- names (fps[[1]])
        
        if (as_sparse) {
            fps_cat <- sparseMatrix (NA, nrow = length (sample_ids), ncol = 0)
        } else {
            fps_cat <- matrix (NA, nrow = length (sample_ids), ncol = 0)    
        }
        
        rownames (fps_cat) <- sample_ids
        
        for (fps_definition in fps_definitions) {
            if (is.null (fps[[fps_definition]])) {
                warning (sprintf ("Fingerprint definition '%s' is NULL and will be skipped.",
                                  fps_definition))
                next
            }
            
            stopifnot (sample_ids == names (fps[[fps_definition]]))
            
            fps_def <- fingerprints_to_matrix (fps[[fps_definition]], as_sparse)
            colnames (fps_def) <- paste (fps_definition, 1:ncol (fps_def), sep = "_")
            
            if (! is.null (masks) && ! is.null (masks[[fps_definition]])) {
                fps_def <- fps_def[, masks[[fps_definition]]]
            }
            
            fps_cat <- cbind (fps_cat, fps_def)
        }
    }
    
    return (fps_cat)
}

#' Fingerprints to matrix class
#' 
#' @description 
#' Convert the \texttt{fingerprints} class into a matrix object. 
#' 
#' @param fps list of \texttt{fingerprints} objects (1 x n_samples)
#' @param as_sparse binary, should the output matrix be of class 
#'                  \texttt{sparseMatrix}
#' @return matrix object (can be of class \texttt{sparseMatrix}),
#'         (n_samples x n_fingerprints)
fingerprints_to_matrix <- function (
    fps, as_sparse = FALSE)
{
    sample_ids <- names (fps)
    
    fps <- fp.to.matrix2 (fps)
    
    if (as_sparse) {
        TRUE_ids <- which (fps > 0, arr.ind = TRUE)
        fps <- sparseMatrix (TRUE_ids[, 1], TRUE_ids[, 2], fps[TRUE_ids])    
    } 
    
    rownames (fps) <- sample_ids
    
    return (fps)
}

fp.to.matrix2 <- function (fplist) {
    get_value_from_feature <- function (feature) { attributes(feature)$count }
    get_key_from_feature <- function (feature) { attributes(feature)$feature }
    
    # Need to distinguish, whether we got a binary fp-representation or a feature
    # that for example represents the fp count.
    if ("features" %in% names (attributes (fplist[[1]]))) {
        # Got raw or counting fingeprints
        size <- length (attributes (fplist[[1]])$features)
        is_binary <- FALSE
    } else {
        # Got binary
        size <- fplist[[1]]@nbit    
        is_binary <- TRUE
    }
    
    m <- matrix (0, nrow = length (fplist), ncol = size)
    cnt <- 1
    for (fp in fplist) {
        if (is.null (fp)) {
            m[cnt, ] <- NA
        } else {
            if (is_binary) {
                m[cnt, fp@bits] <- 1    
            } else {
                # Sort the keys to ensure same order of fingerprint definitions
                # for all molecules.
                srtres <- sort (sapply (attributes (fp)$feature, 
                                        get_key_from_feature),
                                index.return = TRUE)
                values <- sapply (attributes (fp)$feature, get_value_from_feature)
                m[cnt, ] <- values[srtres$ix]
            }
        }
        cnt <- cnt + 1
    }
    return (m)
}

#' Determine the fingerprint dimension 
#' 
#' @description 
#' Given a list of different fingerprint defintions this function determines
#' the dimension of each definition. 
#' 
#' @param fps list of fingerprints (1 x m) (with m = n_definitions)
#'            list[[i]]: object of class \texttt{fingerprint}
#' @return list of intengers (1 x n_definitions) with the fingerprint dimensions
get_fingerprint_dimensions <- function (fps) {
    stopifnot (is.list (fps))
    
    fps_definitions <- names (fps)
    fps_dimensions <- lapply (rep (NA, length (fps_definitions)), FUN = identity)
    names (fps_dimensions) <- fps_definitions
    
    for (fps_definition in fps_definitions) {
        if (is.null (fps[[fps_definition]])) {
            next
        }
        if (is.null (fps[[fps_definition]][[1]])) {
            next
        }
        fps_dimensions[[fps_definition]] <- attributes (fps[[fps_definition]][[1]])$nbit
    }
    
    return (fps_dimensions)
}

#' Produce a binary mask to exclude molecular fingerpints
#' 
#' @description 
#' Given a binary fingerprint matrix a binary mask is produced,
#' to exclude fingerprints:
#' - with always the same value
#' - which are redunant
#' 
#' @param fps binary matrix (or -1,1), shape (n_samples x n_fingerprints)
#'        OR  list of fingerprints (1 x m) (with m = n_definitions)
#'            list[[i]]: object of class \texttt{fingerprint}
#' @param remove_single_value binary, exclude fps with always the same value
#' @param remove_redundant binary, exclude redundant fps
#' 
#' @return binary vector (1 x n_fingerpints):
#'         TRUE:  keep fingerprints
#'         FALSE: exclude fingerprint
get_fingerprint_mask <- function (
    fps,
    remove_single_value = TRUE,
    remove_redundant    = TRUE) 
{
    # Fingerprint lists need to be converted to binary matrices first
    if (is.list (fps)) {
        fps <- fingerprints_to_matrix (fps)
    }
    
    stopifnot (all (unique (as.vector (fps)) %in% c(TRUE, FALSE)) ||
                   all (unique (as.vector (fps)) %in% c(-1, 1)))
    
    n_samples <- nrow (fps)
    n_fps     <- ncol (fps)
    
    fps[fps == 0] <- -1
    
    # Find the columns in which all the fingerprints are either -1 or 1
    if (remove_single_value) {
        is_all_TRUE  <- apply (fps, MARGIN = 2, FUN = function (x) all (x ==  1))
        is_all_FALSE <- apply (fps, MARGIN = 2, FUN = function (x) all (x == -1))
    } else {
        is_all_TRUE  <- rep (FALSE, n_fps)
        is_all_FALSE <- rep (FALSE, n_fps)
    }
    
    # Find all the redundant fingerprints:
    #   fp_i = -1 <--> fp_j = -1
    #   fp_i =  1 <--> fp_j =  1
    if (remove_redundant) {
        fps_cor <- (t(fps) %*% fps) / n_samples
        
        # Build up an undirected graph with edges between all redundant
        # components
        fps_graph <- graph_from_adjacency_matrix (fps_cor == 1, mode = "undirected")
        # Find the connected components in this graph. For each component only
        # one fingerprint definition needs to be kept. We keep the one with the
        # lowest fingerprint index.
        fps_comp <- components (fps_graph)
        n_comp <- fps_comp$no
        cols_not_redundant <- sapply (
            1:n_comp, 
            FUN = function (idx, fps_comp) { which (fps_comp$membership == idx)[1] }, 
            fps_comp
        )
        is_redundant <- rep (TRUE, n_fps)
        is_redundant[cols_not_redundant] <- FALSE
    } else {
        is_redundant <- rep (FALSE, n_fps)
    }
    
    return (! (is_all_TRUE | is_all_FALSE | is_redundant))
}

#' Produce a binary mask to exclude molecular fingerprints
get_count_fingerprint_mask <- function (count_fps) 
{
    is_all_ZERO <- apply (count_fps, MARGIN = 2, FUN = function (x) all (x == 0))
    
    return (! is_all_ZERO)
}

#' Function to remove certain redundant and unused fingerprints
#' @param: fps, matrix (n_samples x n_fingerprints)
#' @return matrix (n_samples x n_fingerprints')
clean_up_molecular_fingerprints <- function (
    bin_fps,
    remove_single_value = TRUE,
    remove_redundant    = TRUE) 
{
    tictoc::tic ("Pre-process the fingerprints")
    
    n_samples <- nrow (fps)
    n_fps     <- ncol (fps)

    # Find the columns in which all the fingerprints are either -1 or 1
    fps[fps == 0] <- -1

    cols_all_TRUE  <- apply (fps, MARGIN = 2, FUN = function (x) all (x ==  1))
    cols_all_FALSE <- apply (fps, MARGIN = 2, FUN = function (x) all (x == -1))

    stopifnot (length (cols_all_TRUE)  == n_fps)
    stopifnot (length (cols_all_FALSE) == n_fps)

    fps <- fps[, ! (cols_all_TRUE | cols_all_FALSE)]
    n_fps <- ncol (fps)

    # Find all the redundant fingerprints:
    #   fp_i = -1 <--> fp_j = -1
    #   fp_i =  1 <--> fp_j =  1
    fps_cor <- (t(fps) %*% fps) / n_samples

    stopifnot (all (diag (fps_cor) == 1))
    stopifnot (all (dim (fps_cor) == n_fps))
    # 'fps.ALL.cor'_ij is equal one iff fp_i and fp_j are redundant

    # Build up an undirected graph with edges between all redundant
    # components
    fps.ALL.graph <- graph_from_adjacency_matrix (mat.fps.ALL.cor == 1, mode = "undirected")
    # Find the connected components in this graph. For each component only
    # one fingerprint definition needs to be kept. We keep the one with the
    # lowest fingerprint index.
    fps.ALL.comp <- components (fps.ALL.graph)
    fps.ALL.selection <- sapply (1:fps.ALL.comp$no,
                                 FUN = function (c.idx, fps.ALL.comp) { which (fps.ALL.comp$membership == c.idx)[1] },
                                 fps.ALL.comp)
    mat.fps.ALL <- mat.fps.ALL[, fps.ALL.selection]
    n.fps <- ncol (mat.fps.ALL)

    # Let us check whether the desired properties are fullfilled
    stopifnot (! any (apply (mat.fps.ALL, MARGIN = 2, FUN = function (x) all (x ==  1))))
    stopifnot (! any (apply (mat.fps.ALL, MARGIN = 2, FUN = function (x) all (x == -1))))

    mat.fps.ALL.cor <- (t(mat.fps.ALL) %*% mat.fps.ALL) / n.mol

    stopifnot (all (diag (mat.fps.ALL.cor) == 1))
    stopifnot (all (dim (mat.fps.ALL.cor) == n.fps))
    stopifnot (sum (mat.fps.ALL.cor == 1) == n.fps)

    mat.fps.ALL[mat.fps.ALL == -1] <- 0

    tictoc::toc (log = TRUE)

    stopifnot (strcmp (rownames (mat.fps.ALL), smiles))

    return (data.table (mat.fps.ALL, smiles = smiles))
}
