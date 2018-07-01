# Define sources
SOURCE_ACC=$(wildcard [0-9]*_accuracies*.csv)
SOURCE_COR=$(wildcard [0-9]*_correlations*.csv)
SOURCE_GSBP=$(wildcard [0-9]*_grid_search_best_params*.csv)
SOURCE_GSR=$(wildcard [0-9]*_grid_search_results*.csv)
SOURCE_SS=$(wildcard [0-9]*_simple_statistics*.csv)

.PHONY: all accuracies correlations grid_search_best_params grid_search_results simple_statistics clean

all: accuracies correlations grid_search_best_params grid_search_results simple_statistics

accuracies: $(SOURCE_ACC)
	Rscript combine_results.R $^
correlations: $(SOURCE_COR)
	Rscript combine_results.R $^
grid_search_best_params: $(SOURCE_GSBP)
	Rscript combine_results.R $^
grid_search_results: $(SOURCE_GSR)
	Rscript combine_results.R $^
simple_statistics: $(SOURCE_SS)
	Rscript combine_results.R $^

clean:
	rm -f $(SOURCE_ACC)
	rm -f $(SOURCE_COR)
	rm -f $(SOURCE_GSBP)
	rm -f $(SOURCE_GSR)
	rm -f $(SOURCE_SS)
