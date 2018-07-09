# Retention time data used for the retention order prediction experiments

We evaluated the retention order prediction approach using retention times extracted
from the publicly available retention time database provided by [Stanstrup et al. (2015)][@predret_paper].
We downloaded a snapshot ([Stanstrup_exp_Aug16.csv](Stanstrup_exp_Aug16.csv)) of 
the database in August 2016 using the open-source tool [PredRet](https://github.com/stanstrup/PredRet):
```R
PredRet_get_db (exp_pred = "exp")
```
Any changes made to the database after that date are _not reflected_ in this repository. 

## Data filtering

We perform the following pre-processing steps (see [pre-processing script](data/scripts/R/preprocessing/preprocessing_PredRet_v2.Rmd))
to the data:

- Consider only reversed phase columns
- Consider only datasets included in the publication by [Stanstrup et al. (2015)][@predret_paper]
- Early eluting molecules are excluded
- Minimum retention time is used when multiple RTs for a single molecular structures
- When relative difference of multiple RTs is larger than 5%, the structure will be removed.

## References

TODO: List references

[@predret_paper]: https://pubs.acs.org/doi/abs/10.1021/acs.analchem.5b02287 "PredRet: Prediction of Retention Time by Direct Mapping between Multiple Chromatographic Systems, Stanstrup, J.; Neumann, S. & Vrhovšek, U., Analytical Chemistry, 2015"