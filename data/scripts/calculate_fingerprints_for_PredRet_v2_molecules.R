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
