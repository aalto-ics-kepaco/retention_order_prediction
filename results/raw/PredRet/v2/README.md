# Experiment settings

The settings, e.g. kernel and molecular features, of an experiment are set in a 
[json-file](https://en.wikipedia.org/wiki/JSON).

## ```"data"```

```json
  "data": {
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