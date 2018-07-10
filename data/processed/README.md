# Processed data 

Each folder contains a processed dataset that is ready to be used with different
experiments:

## Retention order prediction 

Datasets ```PredRet``` and ```s10_imp_no3D``` contain retention times data and molecular
fingerprints (fps) to train the RankSVM respectively the SVR:

- ```rts.csv```: Molecular structures and corresponding retention times
- ```fps.csv```: Binary fingerprints  for each molecular structure in the dataset
- ```fps_maccs_count.csv```: MACCS counting fps 

Note: The ```rts.csv``` contains also the information which chromatographic system
was used to measure a certain structure. One dataset, e.g. ```PredRet```, encompasses
possible several (chromatographic) systems.

There are scripts for each dataset to be created from the [raw](../raw) files.