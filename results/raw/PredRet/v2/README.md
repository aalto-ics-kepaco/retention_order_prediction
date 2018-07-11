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

#### Case: Ranking SVM (```"ranksvm"```)

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

Parameters 