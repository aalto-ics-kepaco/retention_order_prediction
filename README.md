# Overview

# Installation

## Required R packages
- data.table
- Matrix
- obabel2R
- rcdk
- fingerprint

## Required programms
- open babel

## Required python libraries

Code has only be tested with python 3.5 and 3.6.

# Usage

All experiments of the paper can be reproduced by using the calling the [evaluation_scenarios_main.py](src/evaluation_scenarios_main.py)
script with the propper paramters:

```bash
usage: evaluation_scenarios_main.py <ESTIMATOR> <SCENARIO> <SYSSET> <TSYSIDX> <PATH/TO/CONFIG.JSON> <NJOBS> <DEBUG>
  ESTIMATOR:           {'ranksvm', 'svr'}, which order predictor to use.
  SCENARIO:            {'baseline', 'baseline_single', 'baseline_single_perc', 'all_on_one', 'all_on_one_perc', 'met_ident_perf_GS_BS'}, which experiment to run.
  SYSSET:              {10, imp, 10_imp}, which set of systems to train on.
  TSYSIDX:             {-1, 0, ..., |sysset| - 1}, which target system to use for evaluation.
  PATH/TO/CONFIG.JSON: configuration file, e.g. PredRet/v2/config.json
  NJOBS:               How many jobs should run in parallel for hyper-parameter estimation?
  DEBUG:               {True, False}, should we run a smoke test.
```

For example: To reproduce Table 3 in the paper the following function calls are 
need:

__MACCS counting fingerprints:__

```bash
python src/evaluation_scenarios_main.py ranksvm baseline_single 10 -1 results/raw/PredRet/v2/config.json 2 False
```

- [```baseline_single```](src/evaluation_scenarios_main.py#L708): Single system used for training and testing.
- [```10```](results/raw/PredRet/v2/config.json#L7): Use "Eawag_XBridgeC18", "FEM_long", "RIKEN", "UFZ_Phenomenex", "LIFE_old" for training and testing.
- ```-1```: By setting TSYSIDX to -1, we run all target systems in a single job. [This parameter can be used for parallelization](results/scripts/makefiles#combining-evaluation-results-from-parallel-runs).
- [```results/raw/PredRet/v2/config.json```](results/raw/PredRet/v2/config.json): Configuration of the experiment, e.g. [molecular features and kernels](results/raw/PredRet/v2/config.json#L28).
- ```2```: Number of jobs/cpus used for the [hyper-parameter search](src/model_selection_cls.py#L370).
- ```False```: Not running in debug-mode. Results will be stored in the [final](results/raw/PredRet/v2/final) directory.

__MACCS binary fingerprints:__

Modify the [```results/raw/PredRet/v2/config.json```](results/raw/PredRet/v2/config.json)
configuration file:

```json
"molecule_representation": {
  "kernel": "minmax",
  "predictor": ["maccsCount_f2dcf0b3"],
  "feature_scaler": "noscaling",
  "poly_feature_exp": false
}
```

becomes

```json
"molecule_representation": {
  "kernel": "tanimoto",
  "predictor": ["maccs"],
  "feature_scaler": "noscaling",
  "poly_feature_exp": false
}
```

Then run:

```bash
python src/evaluation_scenarios_main.py ranksvm baseline_single 10 -1 results/raw/PredRet/v2/config.json 2 False
```

The results will go into a different folder.

# Citation