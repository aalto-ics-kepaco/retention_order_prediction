# Combining evaluation results from parallel runs

The implementation of the [main evaluation script](../../../src/) allows to run the evaluation
on each target system in a different triton job. 

Following the directory structure 
separating different experimental settings, in each output directory there is several
results files created for each target system:

00_accuracies*.csv, 01_accuracies*.csv, ...

Those can be combined using the Makefile in 'combine_results' together with the r-script
'combine_results.R'.
