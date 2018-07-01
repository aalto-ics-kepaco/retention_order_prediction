For some candidates not all the fingerprints could be calculated. We therefore remove
those candidates from the candidate lists / scoring files. This is done using the 
script: 

  - **data/scripts/R/clean_up_impact_candidate_lists.R**

The scoring files, that needed to be modifed, i.e. examples have been removed, are
in the subdirectory **before_clean_up/**. 

The cleanup is done for each fingerprint definition separetly and the cleaned up 
scoring files are in the respective subdirectories:

  - **maccs_binary/**
  - **maccs_counting/** 
  - ...
