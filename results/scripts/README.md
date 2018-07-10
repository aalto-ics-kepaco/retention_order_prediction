# Script to analyse the results

## ```ECCB2018.Rmd```

R-Markdown that produces all the Tables and Figures presented in the paper.

## ```helper.R```

Set of functions that allows to load the results of the experiments into R [```data.table```](https://cran.r-project.org/web/packages/data.table/)s.

### Example 

Load the results of the experiment evaluating the __pairwise prediction__ (```"accuracy"```)
performance using __RankSVM__ (```"ranksvm_slacktype=on_pairs"```) when trained 
on a single system and applied to a single target system:

```R
sdir_results <- "results/raw/PredRet/v2/"

load_baseline_single_results (
    measure = c("accuracy", "accuracy_std"), 
    base_dir = paste0 (sdir_results, "ranksvm_slacktype=on_pairs/"),
    predictor = "maccs",
    kernel = "tanimoto",  
    pair_params = list (allow_overlap = "True", d_lower = 0, d_upper = 16, ireverse = "False", type = "order_graph"), 
    feature_type = "difference", 
    flavor = list (allpairsfortest = "True", featurescaler = "noscaling", sysset = 10))
```

Parameters:
- ```measure```: Which evaluation measure to load, e.g., accuracy, correlation, ... (see also: [evaluation_scenarios_cls.py](src/evaluation_scenarios_cls.py#L464))
- ```base_dir```: Directory of the processed input data of a certain dataset, e.g. ```PredRet/v2```
- ```predictor```: Which feature was used to represent the molecules, e.g., MACCS fingerprints.
- ```kernel```: Which kernel was used on top of the molecular features, e.g., Tanimoto kernel.
- ```pair_params```: Paramters for the training pair generation from the retention times for the RankSVM (see function [get_pairs_from_order_graph](src/rank_svm_cls.py#L60) for details.)