# Scripts to analyse the results

## ```ECCB2018.Rmd```: Reproduce figures and tables from the paper

R-Markdown that produces all the Tables and Figures presented in the paper.
(Re-)compilation of the [HTML report](../ECCB2018.html) containing the paper results:

```bash
R -e "rmarkdown::render('ECCB2018.Rmd',output_file='../ECCB2018.html')"
```

## ```helper.R```: Load results in to R

Set of functions that allows to load the results of the experiments into R [```data.table```](https://cran.r-project.org/web/packages/data.table/)s.

### Example: Load results

Load the results of the experiment evaluating the __pairwise prediction__ (```"accuracy"```)
performance using __RankSVM__ (```"ranksvm_slacktype=on_pairs"```) when trained 
on a single system and applied to a single target system:

```R
sdir_results <- "results/raw/PredRet/v2/final/" # example when dataset PredRet/v2 is used

res <- load_baseline_single_results (
    measure = c("accuracy", "accuracy_std"), 
    base_dir = paste0 (sdir_results, "ranksvm_slacktype=on_pairs/"),
    predictor = "maccs",
    kernel = "tanimoto",  
    pair_params = list (allow_overlap = "True", d_lower = 0, d_upper = 16, ireverse = "False", type = "order_graph"), 
    feature_type = "difference", 
    flavor = list (allpairsfortest = "True", featurescaler = "noscaling", sysset = 10))
```

Parameters:
- ```measure```: Which evaluation measure to load, e.g., accuracy, correlation, ... (see also: [evaluation_scenarios_cls.py](/src/evaluation_scenarios_cls.py#L464))
- ```base_dir```: Directory of the processed input data of a certain dataset, e.g. ```PredRet/v2```
  - For RankSVM this paramter is set to [```paste0 (sdir_results, "ranksvm_slacktype=on_pairs/")```](/results/raw/PredRet/v2/final_ECCB2018_paper/ranksvm_slacktype=on_pairs)
  - For SVR this parameter is set to [```paste0 (sdir_results, "svr/")```](/results/raw/PredRet/v2/final_ECCB2018_paper/svr)
  - If the evaluation script is run in _debug_ mode, than replace ```final``` by ```debug```.
- ```predictor```: Which feature was used to represent the molecules, e.g., MACCS fingerprints.
- ```kernel```: Which kernel was used on top of the molecular features, e.g., Tanimoto kernel.
- ```pair_params```: Paramters for the training pair generation from the retention times for the RankSVM (see for example function [```get_pairs_from_order_graph```](src/rank_svm_cls.py#L60) for details)
  - In the paper all the results are calculated using the paramters shown in the example
  - For SVR this paramter can be set to ```NULL```
- ```feature_type```: Feature type used in the RankSVM. 
  - Only ```"difference"``` is supported and used in the paper.
  - For SVR this paramter can be set to ```NULL```
- ```flavor```: List of parameters used to identify the some settings during the evaluation:
  - ```allpairsfortest```: See parameter documentation of [```find_hparan_ranksvm```](/src/model_selection_cls.py#L198).
  - ```featurescaler```: Feature scaler used for the molecular features. (see also: [```evaluate_on_target_systems```](/src/evaluation_scenarios_cls.py#L209))
  - ```sysset```: Which (sub)set of systems from the specified dataset should be used, e.g., [the used in the paper (=10)](/results/raw/PredRet/v2/config_local.json#L7).

The different experiments evaluated in the paper require different [```load_*```](helper.R#L246)
functions. Those are provided in the [helper.R](helper.R) script. Further examples
how to load the results can be found in the report / summary [R-markdown script](ECCB2018.Rmd).

### Example: Access results

```R
> res[d_lower == 0 & d_upper == Inf]

    accuracy accuracy_std           target           source d_lower d_upper
 1:   0.8439       0.0053 Eawag_XBridgeC18 Eawag_XBridgeC18       0     Inf
 2:   0.9048       0.0075         FEM_long         FEM_long       0     Inf
 3:   0.8623       0.0173         LIFE_old         LIFE_old       0     Inf
 4:   0.8484       0.0083            RIKEN            RIKEN       0     Inf
 5:   0.8019       0.0086   UFZ_Phenomenex   UFZ_Phenomenex       0     Inf
````

- ```source``` refers in the result files to the system(s) used for training. 
- ```d_lower``` and ```d_upper``` refers here to the paramters used to calculate the __test__ pairs for evaluation.
    - ```0``` and ```Inf``` means, that all possible pairs are used for testing (as the paper defines the _Pairwise accuracy_ in Section 3.1.2)
    - The result files also contain other ```d_lower``` and ```d_upper``` pairs.
    - Those can be used to, e.g., evaluate the pairwise prediction accuracy for nearby eluting molecules, i.e. with small retention time difference: ```d_lower = 0``` and ```d_upper = 4```.
    - Please look at the [source code generating the pairwise accuracies](/src/evaluation_scenarios_cls.py#L457).
