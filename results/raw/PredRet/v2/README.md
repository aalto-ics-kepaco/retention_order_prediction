# Experiment settings

The model, data and feature parameters for different experimental settings are 
summarized in json-files, that are specific to each data set, e.g. _results/raw/PredRet/v2/\*.json_).

## ```"data"```

```json
"base_dir": "PATH/TO/MAIN/PROJECT/DIRECTORY/",
"dataset": "PredRet/v2",
"excl_mol_by_struct_only": true,
"systems": {
  "10": ["Eawag_XBridgeC18", "FEM_long", "RIKEN", "UFZ_Phenomenex", "LIFE_old"],
  "10_imp": ["Eawag_XBridgeC18", "FEM_long", "RIKEN", "UFZ_Phenomenex", "LIFE_old", "Impact"],
  "imp": ["Impact"]
}
```

The data section configures which dataset should be used and which systems are
considered for the evaluation. For example in the paper the experiments regarding
the retention order prediction are performed using system set (sset) ```"10"```. 
As described in the paper, molecular structures measured with another system are
not used for training if they are also in the test set of the (current) target
system: ```"excl_mol_by_struct_only": true```. 

## ```"model"```

```json
"modelselection": {
  "all_pairs_for_test": true
}
```

Parameters configuring the model selection implemented in [model_selection_cls.py](src/model_selection_cls.py). 
```all_pairs_for_test: true``` means that all available test pairs should be generated
to calculate the test-score for each hyper-parameter (see also [documentation](src/model_selection_cls.py#L246)).

### Case: Ranking SVM (```"ranksvm"```)

```json
"ranksvm": {
  "pair_params": {
    "type": "order_graph",
    "d_upper": 16,
    "d_lower": 0,
    "allow_overlap": true,
    "ireverse": false
  },
  "feature_type": "difference",
  "slack_type": "on_pairs"
}
```

Parameters configuring the RankSVM:

- [```pair_params```](src/model_selection_cls.py#L230):
    - _type_: String, which function should be used to calculate pairs: [```"order_graph"```](src/rank_svm_cls.py#L60) or [```"multiple_system"```](src/rank_svm_cls.py#L225) (equivalent with _ireverse_ is false) 
    - _ireverse_: If true, cross chromatographic-system elution transitivity pairs are used for training. (in paper = false)
    - _allow_overlap_: If true, contradicting elution order pairs from different systems are included in the training. (in paper = true)
    - _d_upper_: Scalar, maximum retention order difference of two molecules from one systems to be considered as __training pair__. (in paper = 16) 
    - _d_lower_: Scalar, minimum retention order difference of two molecules from one systems to be considered as __training pair__. (in paper = 0) 
- ```feature_type```: Feature type used for the RankSVM. In the paper we use ```difference``` which means: \phi_j - \phi_i. 
- ```slack_type```: In the paper we use a slack-variable for training pair (= ```on_pairs```).

Some remarks regarding _d_upper_ and _d_lower_: This two paramter are only effecting 
which and especially how many training pairs are used for the RankSVM. If _d_lower_ 
is set to 0, than no restrictions are made to the minimum elution order distance. 
If _d_upper_ is less then Infinity, than this means, that we use for example only 
pairs of molecular structure, where are not more than 15 molecules eluting between 
them. So this includes elution order distance 0, 1, 2, ..., 15. We have found out
that including more pairs does not increase the performence. This could be explaint 
by the fact, that far apart eluting molecules are easier to distinguish the nearly,
i.e. molecules with small retention time difference, molecules. Please read the
source code of the function [```get_pairs_single_system```](src/rank_svm_cls.py#L300)
for the most simple realisation of such a filtering.

### Case: Support Vector Regression (SVR) (```"svr"```)

```json
"svr": {}
```

No paramters are set here.

## ```"molecular_representation"```

```json
"molecule_representation": {
  "kernel": "minmax",
  "predictor": ["maccsCount_f2dcf0b3"],
  "feature_scaler": "noscaling",
  "poly_feature_exp": false
}
```

Parameters defining the molecular representation and the kernels used to measure
the similarity:

- _kernel_: String, which kernel to use. Currently supported are ```"tanimoto"``` and ```"minmax"``` for fingerprints.
- _predictor_: list of strings, which fingerprints should be used. In the paper we use ```["maccs"]``` and ```["maccsCount_f2dcf0b3"]```.
- _feature_scaler_: How the input molecular features, e.g. fingerprint, should be scaled before use. (in paper = ```"noscaling"```)
- _poly_feature_exp_: [See code](src/evaluation_scenarios_cls.py#L241) (in paper = false)

## ```"application"```

```json
"application": {
    "candidate_reranking": {
      "dp_weight_function": "pwmax",
      "use_log": false
    }
  }
```

Paramters used for the metabolite identification experiments: 

- _dp_weight_function_: [```"pwmax"```](src/metabolite_identification_cls.py#L501) refers to the edge weight definition used in the paper.
- _use_log_: If true, than the logarithm of the order penalty term is used. (in paper = false)

Note: Some paramters used for the metabolite identification setting are defined
directly in the [main evaluation script](src/evaluation_scenarios_main.py#L525).