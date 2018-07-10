# Combining evaluation results from parallel runs

The implementation of the [main evaluation script](src/evaluation_scenarios_main.py)
allows to run the evaluation, e.g., the experiments related to the order prediction 
presented in Section 3.1, on each target system in a different job, e.g., on a
cluster system. 

## Results per target system

Let's assume we want to evaluate the order prediction performance on the dataset 
[```PredRet/v2```](data/processed/PredRet/v2/). In the paper we only 
consider five chromatographic systems (can be though of as sub(data)sets), i.e.,
```"Eawag_XBridgeC18"```, ```"FEM_long"```, ```"RIKEN"```, ```"UFZ_Phenomenex"```
and ```"LIFE_old"```. Each of this system can serve as _target system_ for the 
analysis, i.e, it is the system in which the performence is evaluated, and the 
corresponding performance measures are stored for each target system separetly 
([Click here for an example output.](results/raw/PredRet/v2/final/ranksvm_slacktype=on_pairs/allow_overlap=True_d_lower=0_d_upper=16_ireverse=False_type=order_graph/difference/maccs/tanimoto/baseline_single)):

- ```00_accuracies_allpairsfortest=False_featurescaler=noscaling_sysset=10.csv```
- ```01_accuracies_allpairsfortest=False_featurescaler=noscaling_sysset=10.csv```,
- ...

where the numbers correspond to the target systems.

## Combining the results using the Makefile

The different results can be combined using the [Makefile](Makefile) in combination
with the R-script [combine_results.R](../combine_results.R).

Those can be combined using the Makefile in 'combine_results' together with the r-script
'combine_results.R'.
