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
- ```s10_imp_no3D```: Combination of the PredRet and Impact (without MS/MS-spectrum) data
  - Data-set combination and fingerprint calculation: [combine_predret_v2_sset10_and_impact.R](../scripts/combine_predret_v2_sset10_and_impact.R) 

## Metabolite identification using tandem Mass Spectrometry (MS/MS) data

Dataset ```Impact``` contains the data used in the metabolite identification
experiment:

- ```candidates/rts_msms.csv```: Retention times for each MS/MS-spectrum with molecular structure
- ```candidates/fingerprints/```: Fingerprints for the molecular candidates of each MS/MS-spectrum
- ```candidates/scorings/```: MS/MS-spectra based scores for all molecular candidates

Note: The MS/MS-spectra are not included in this repository. The MS/MS-based scores
have been calculated using Input Output Kernel Regression (IOKR) [(Brouard et al., 2016)][@iokr_paper]
as described in the paper.

The repository also contains a toy-dataset ```Metabolite_identification_toy_dataset```
than can be used as template for the required structure by the software and for
debugging perposes.

### ```candidates/rts_msms.csv```

Csv-file with three columns: "inchikey", "inchi", "rt". The "inchikey" serves as
identifier for the MS/MS-spectra. The "rt" column contains the retention time of 
each MS/MS-spectra.

### ```candidates/fingerprints/maccs_count/fps_maccs_count_list=ML_spec=CCMSLIB_id=INCHIKEY.csv```

- ML: Molecular formula of the molecular structure used to query the candidate database, e.g. [Pubchem](https://pubchem.ncbi.nlm.nih.gov/).
- CCMSLIB: [GNPS](https://gnps.ucsd.edu/ProteoSAFe/static/gnps-splash.jsp) ID if the MS/MS-spectrum mapped to the molecular structure from the ```Impact``` dataset.
- INCHIKEY: InChI-key of the molecular structure from the ```Impact``` dataset. Internally used as identifier for the spectra.

Csv-file with "inchi", "fp1", "fp2", ... columns. The "inchi" identifies each 
molecular candidate. The candidates' fingerprints can be calculated using the 
[calculate_fingerprints_for_impact_candidates.R](../scripts/calculate_fingerprints_for_impact_candidates.R)
script. For some molecular structure the calculation of the molecular fingerprints
might fail. In this case the [clean_up_impact_candidate_lists.R](../scripts/clean_up_impact_candidate_lists.R)
script can be used, to filter those candidates and modify the corresponding scoring 
files (see next Section).

### ```candidates/scoring/maccs_count/scoring=ML_spec=CCMSLIB_id=INCHIKEY.csv```

Csv-file with two columns: "id1" and "score". "id1" referes here to the InChI of
each molecular structure and "score" to the MS/MS-based IOKR score.

## References

[@iokr_paper]: https://academic.oup.com/bioinformatics/article/32/12/i28/2288626 "Fast metabolite identification with Input Output Kernel Regression, Brouard, C., Shen, H., Dührkop, K., d’Alché-Buc, F., Böcker, S., and Rousu, J. Bioinformatics, 2016"