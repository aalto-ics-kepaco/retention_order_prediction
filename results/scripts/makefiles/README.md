# Combining evaluation results from parallel runs

The implementation of the [main evaluation script](../../../src/evaluation_scenarios_main.py)
allows to run the evaluation, e.g., the experiments related to the order prediction 
presented in Section 3.1, on each target system in a different job, e.g., on a
cluster system. 

## Example

Let's assume we want to evaluate the order prediction performance on the dataset 
[```PredRet/v2```](../../../data/processed/PredRet/v2/). In the paper we only 
consider five chromatographic systems (can be though of as sub(data)sets), i.e.,
```"Eawag_XBridgeC18"```, ```"FEM_long"```, ```"RIKEN"```, ```"UFZ_Phenomenex"```
and ```"LIFE_old"```

Following the directory structure 
separating different experimental settings, in each output directory there is several
results files created for each target system:

00_accuracies*.csv, 01_accuracies*.csv, ...

Those can be combined using the Makefile in 'combine_results' together with the r-script
'combine_results.R'.
