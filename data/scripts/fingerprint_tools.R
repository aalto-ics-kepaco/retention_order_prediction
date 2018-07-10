####
#
# The MIT License (MIT)
#
# Copyright 2017,2018 Eric Bach <eric.bach@aalto.fi>
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

#
# Function to calculate molecular fingerprints from SMILES and InChIs.
#

require (Matrix)
require (obabel2R)
require (rcdk)
require (fingerprint)
require (data.table)

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
