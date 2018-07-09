# Massbank entries used for the metabolite identification experiments

For the metabolite identification experiments we extracted the retention times 
provided by the [Department of Chemistry of the University of Athens][@AU_in_massbank]
from [Massbank][@massbank]. The dataset encompasses entries made before 31st of May 2016. Any 
changes that happend after this date are _not reflected_ in this repository.

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

The described data filtering left us with __342 (molecular structure, retention-times)-tuples__. 

## Accession IDs

The file [massbank_entry_list.txt](massbank_entry_list.txt) contains all the accession
ids of the massbank entries we used. Lines that contain several ids indicate, that
their corresponding retention times have been merged, as the ids corresponded to 
the same molecular structure.

## References

TODO: Print references here.

[@AU_in_massbank]: https://massbank.eu/MassBank/jsp/Result.jsp?type=rcdidx&idxtype=site&srchkey=32&sortKey=name&sortAction=1&pageNo=1&exec=
[@massbank]: https://onlinelibrary.wiley.com/doi/abs/10.1002/jms.1777 "Massbank: a public repository for sharing mass spectral data for life sciences, Horai, H.; Arita, M.; Kanaya, S.; Nihei, Y.; Ikeda, T.; Suwa, K.; Ojima, Y.; Tanaka, K.; Tanaka, S.; Aoshima, K. & others, Journal of mass spectrometry, 2010"

