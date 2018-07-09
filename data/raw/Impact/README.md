# Massbank entries used for the metabolite identification experiments

For the metabolite identification experiments we downloaded the retention times 
provided by the Department of Chemistry of the University of Athens from Massbank.

## Data filtering

We only consider entries where:

- AC$MASS_SPECTROMETRY: MS_TYPE == MS2
- AC$MASS_SPECTROMETRY: ION_MODE == POSITIVE
- retentiom time >= 1min.

When multiple retention times for single compound where available, than we only
added the lowest retention time to our dataset. We removed molecular structures
for which the relative retention time difference of multiple measurements was 
greater than 5%. 

We furthermore removed 8 compounds due to difficulties calculating the molecular
fingerprints from their structure. 

The described data filtering left us with 342 (molecular structure, retention-times)
-tuples. 

## References

[1] MassBank: a public repository for sharing mass spectral data for life sciences,
    Horai, H.; Arita, M.; Kanaya, S.; Nihei, Y.; Ikeda, T.; Suwa, K.; Ojima, Y.; Tanaka, K.; Tanaka, S.; Aoshima, K. & others