####
#
# The MIT License (MIT)
#
# Copyright 2018 Eric Bach <eric.bach@aalto.fi>
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

args <- commandArgs (trailingOnly = TRUE)

if (length (args) > 0) {
    df <- read.csv (file = args[1], sep = "\t")
    
    for (ifn in args[-1]) {
        df <- rbind (df, read.csv (file = ifn, sep = "\t"))
    }
    
    # ofile <- strsplit (args[1], "[0-9]+_")[[1]][2]
    ofile <- substring (args[1], 4)
    
    write.table (df, file = ofile, sep = "\t", row.names = FALSE, quote = FALSE)
}
