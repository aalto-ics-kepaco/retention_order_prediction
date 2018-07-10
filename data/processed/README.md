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

There are [scripts](../scripts) for each dataset to be created from the [raw](../raw) files:

- ```PredRet```:
  - Pre-processing: [preprocessing_PredRet_v2.Rmd](../scripts/preprocessing_PredRet_v2.Rmd)
  - Fingerprint calculation: [calculate_fingerprints_for_PredRet_v2_molecules.R](../scripts/calculate_fingerprints_for_PredRet_v2_molecules.R)
- ```s10_imp_no3D```: Combination of the PredRet and Impact (without MSMS) data
  - Data-set combination and fingerprint calculation: [combine_predret_v2_sset10_and_impact.R](../scripts/combine_predret_v2_sset10_and_impact.R) 

## Metabolite identification using tandem Mass Spectrometry (MS/MS) data

Dataset ```Impact``` contains the data used in the metabolite identification
experiment:

- ```candidates/rts_msms.csv```: Retention times for each MS/MS-spectra with molecular structure
- ```candidates/fingerprints/```: Fingerprints for the molecular candidates of each MS/MS-spectra
- ```candidates/scorings/```: MS/MS-spectra based scores for all molecular candidates

Note: 

- asd