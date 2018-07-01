require (rcdk)
require (tictoc)
require (obabel2R)
require (data.table)

# Script to calculate the molecular descriptors for each unique entry in
# the retention-time database containing the molecules at the moment of
# the publication.

calculate_molecular_descriptors_from_inchi <-function (
    inchi, desc.rcdk = get.desc.names()) {
    
    smiles <- inchi2smile (inchi)
    mat.desc <- calculate_molecular_descriptors_from_smiles (smiles, desc.rcdk)
    
    # Check that the oder is still correct.
    stopifnot (all (rownames (mat.desc) == smiles))
    
    rownames (mat.desc) <- inchi
    
    return (mat.desc)
}    

calculate_molecular_descriptors_from_smiles <- function (
    smiles,
    desc.rcdk = get.desc.names()) {

    # Parse all smiles and perform configuration
    tictoc::tic ("Parsing and configuration")

    smiles.parsed <- parse.smiles (smiles)
    n.mol <- length (smiles)
    
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
    }

    tictoc::toc (log = TRUE)

    # Parse list of desired descriptors and calculate those for each
    # molecule.
    tictoc::tic ("Calculate descriptors")

    # Remove the redundant part of the descriptor names for readability.
    desc.names <- sapply (desc.rcdk,
                          FUN = function (str) unlist (strsplit (str, split = "[.]"))[7],
                          USE.NAMES = FALSE)
    n.desc   <- length (desc.names)
    

    stopifnot (length (desc.rcdk) == n.desc)

    # Determine the number of descriptors, e.g. some descriptors returns
    # several numerical values.
    desc.val <- list()
    for (idx in 1:n.desc) {
        desc.val[[desc.names[idx]]] <- eval.desc (smiles.parsed[[1]], desc.rcdk[idx])
    }
    desc.val <- unlist (desc.val)

    mat.desc <-  matrix (NA, nrow = n.mol, ncol = length (desc.val))
    rownames (mat.desc) <- smiles
    colnames (mat.desc) <- names (desc.val)

    for (idx in 1:n.mol) {
        if (is.null (smiles.parsed[[idx]])) {
            print (sprintf ("Molecule %05d/%05d; Could not be parsed!", idx, n.mol))
            next
        }
        
        start_time <- Sys.time()

        mat.desc[idx, ] <- unlist (sapply (desc.rcdk,
                                   FUN = function (desc, smiles) eval.desc (smiles, desc),
                                   smiles.parsed[[idx]],
                                   USE.NAMES = FALSE))

        print (sprintf ("Molecule %05d/%05d; %.3fsec", 
                        idx, n.mol, as.numeric(Sys.time() - start_time)))
    }

    tictoc::toc (log = TRUE)

    return (mat.desc)
}

get_descriptor_mask <- function (
    desc, 
    remove_na_desc = TRUE)
{
    n_desc <- ncol (desc)
    
    if (remove_na_desc) {
        is_na <- apply (desc, MARGIN = 2, FUN = function (x) any (is.na(x)))
    } else {
        is_na <- rep (FALSE, n_desc)
    }
    
    return (! is_na)
}
