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
