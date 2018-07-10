## Experimental settings

In the following the phrase *performance* encompasses the pairwise prediction 
accuracy and  the rank correlation of the predicted preference value with the 
true retention time.

### Performance assessment using cross-validation

For every setting the performance is acceset using cross-validation and the 
hyper-parameter, e.g. C and gamma (rbf-kernel), etc., are estimated using nested
cross-validation. As several for several chromatographic systems only a few 
retention times are available, the following distinction is made during the 
evaluation:

1. Less than 75 (moleucles, retention-time)-tuples are available for training
2. More than 75 (moleucles, retention-time)-tuples are available for training

In the first case a [ShuffleSplit](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html#sklearn.model_selection.ShuffleSplit)
is used to get the training and test sets whereby 75% of the available data is 
used for training. The shuffled split is repeated 25 times as different test 
sets might overlap. 

In the second case a [KFold](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html#sklearn.model_selection.KFold)
is used to get the training and test sets whereby 10 splits are created.

#### Hyper-parameter estimation

The regularization parameter C, the kernel parameters (e.g. gamma), ... are 
optimized using the nested cross-validation realized by [GridSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV).
The objective of the rankSVM tries to minimize the number of miss-classified 
object pairs and this pairwise accuracies is used to choose the best set of
hyper-parameters. However, when we do the nested cross-validation we leave out
molecular structures not pairs. The latter case might be more intuitive, as this
is what we would like to achive - a better pairwise performance. But, experiments 
have shown, that if we optimize the hyper-parameters using different subset of 
training pairs rather than subsets of training molecular structures, we do not 
find hyper-parameters also performing well on the independent test-set. A 
possible explanation could be that as we include all available pairs from the
training (see 'combn'), we actually by leaving out certain pairwise realtions,
still consider a likely all training molecular structures. So the SVM gets 
actually better on the training strctures, but for new test structures it fails.

![On molecules](doc/parameter_estimation_on_molecules.png)
![On pairs](doc/parameter_estimation_on_pairs.png)

### 'baseline_single'
In this setting the peformance of the RankSVM is calculated for each 
chromatographic system separetly. Each system is only used to predict it self.
This setting can be useful to analyse different kernel settings etc.

### 'baseline'
In this setting the peformance of the RankSVM is calculated for each *pair* of 
(s_i,s_j) chromatographic systems. Thereby system s_i serves as training and s_j
as target system. 

### 'on_one'
In this setting *all* the chromatographic systems are considered jointly. That 
means that all systems are used to train and predict all other systems.

### 'selected_scenarios'
This setting encompasses different subsets of chromatographic systems, which are
than evaluated in the same way as 'on_one'. This setting allows to preselect the
systems used for mutual prediction, e.g. systems those columns are very similar,
etc.

### Leave-target-system-out
Each setting can be run in leave-target-system-out mode (*ltse=True*). That 
means that, if applicable, the target system is not used for training. In that
way one can analyse what happens if, we do not have any information about the 
target system during the training, but stil want to make prediction. At this 
point it is important to note, that a training example *(molecule, RT)* is 
always considered to come from one particular system. That means than even if we
leave the target system out from the training, we can have its molecular 
structures in another system and those would be used for training. That means,
that a molecular structure is always considered together with its target system.

## Todo list

- Include column features into the prediction
- Make [KernelRidge](http://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html#sklearn.kernel_ridge.KernelRidge) 
  compatible with the evaluation framework:
    - ~~Overwrite KernelRidge 'score' function to return pairwise prediction 
      accuracy~~.
    - make 'find_hparam_cv2' function compatible with KernelRidge
    - ~~write a 'map_value' function for KernelRidge (essentially only a renaming
      of prediction).~~
    - Add target value (e.g. retention time) scaler, i.e. centering, normalization.
    - Add input feature (e.g. kernel feature vector) scaler, i.e. centering, normalization.

## JSON Configuration file for the experimental settings

The model, data and feature parameters for different experimental settings are summarized in 
a set of json-files, that are specific to each data set (e.g. *results/raw/PredRet/v2/\*.json*).

### Molecular representation

```json
{
    "molecule_representation": {
      "predictor": ["maccs"],
      "feature_scaler": "noscaling"
    }
}
```

The molecular representation defines how a molecule is represented in the computer,
e.g. molecular fingerprints or molecular descriptors. 

#### Case: Molecular descriptors

Together with the molecular
description, we also need to define, how we standardize the molecular descriptors.
This is necessary as molecular descriptors can have very different range, e.g. [0,1] 
vs. [-100,1000]. At the moment the following standardizations are supported: 
 
- **noscaling**: No standardization is applied
- **minmax**: All descriptors are scaled to the intervall [0,1]
- **std**: All descriptors are scaled to have *zero* mean and standard deviation *one*.

When the necessary statistics for the scaling are calculated, than those are only 
calculated based on the data contained in the training and target systems. If for
example we are learning a model using *MTBLS52* and predict for *FEM_long* than only
from those two systems the statistics are calculated, but not from any other like
*LIFE_old*. The idea behind this is, to not have an information leak from not involved
systems during the evaluation.

### Model

#### Case: Ranking SVM (*ranksvm*)

#### Case: Kernel Ridge Regression (*kernelridge*)

```json
{
    "kernelridge": {
      "centering": true,
      "normalization": true
    }
}
```

For regression models the bias in the data, e.g. the row mean, should be considered
when doing predictions. The bias-term is modelled by centering the data in the feature
space of the used kernel, e.g. by directly operating on the kernel matrix. The 
normalization after the centering can is usefull for the model estimation.

#### Case: Support Vector Regression (*svr*) 

```json
{
    "svr": {
    }
}
```

For regression models the bias in the data, e.g. the row mean, should be considered
when doing predictions. In the Support Vector Regression model the ```intercept_```
term (sklearn) is taking care about it.
    
