## Accept the following parameters
# - allpairsfortest: APFT {True, False}
# - embso: EMBSO {True, False}
# - featurescaler: FEATSCAL {, ...}
# - ltso: LTSO {True, False}
# - percfortrain: PERC {0, 10, ..., 100}
# - sysset: SYSSET {10, imp, ...}

# Define sources

# Set up the wildcards based on the 
SOURCE_ACC_WC=[0-9]*_accuracies
SOURCE_COR_WC=[0-9]*_correlations
SOURCE_GSBP_WC=[0-9]*_grid_search_best_params
SOURCE_GSR_WC=[0-9]*_grid_search_results
SOURCE_SS_WC=[0-9]*_simple_statistics

ifeq ($(APFT),)
else
SOURCE_ACC_WC:=$(SOURCE_ACC_WC)_allpairsfortest=$(APFT)
SOURCE_COR_WC:=$(SOURCE_COR_WC)_allpairsfortest=$(APFT)
SOURCE_GSBP_WC:=$(SOURCE_GSBP_WC)_allpairsfortest=$(APFT)
SOURCE_GSR_WC:=$(SOURCE_GSR_WC)_allpairsfortest=$(APFT)
SOURCE_SS_WC:=$(SOURCE_SS_WC)_allpairsfortest=$(APFT)
endif

ifeq ($(EMBSO),)
else
SOURCE_ACC_WC:=$(SOURCE_ACC_WC)_embso=$(EMBSO)
SOURCE_COR_WC:=$(SOURCE_COR_WC)_embso=$(EMBSO)
SOURCE_GSBP_WC:=$(SOURCE_GSBP_WC)_embso=$(EMBSO)
SOURCE_GSR_WC:=$(SOURCE_GSR_WC)_embso=$(EMBSO)
SOURCE_SS_WC:=$(SOURCE_SS_WC)_embso=$(EMBSO)
endif

ifeq ($(FEATSCAL),)
SOURCE_ACC_WC:=$(SOURCE_ACC_WC)_featurescaler=noscaling
SOURCE_COR_WC:=$(SOURCE_COR_WC)_featurescaler=noscaling
SOURCE_GSBP_WC:=$(SOURCE_GSBP_WC)_featurescaler=noscaling
SOURCE_GSR_WC:=$(SOURCE_GSR_WC)_featurescaler=noscaling
SOURCE_SS_WC:=$(SOURCE_SS_WC)_featurescaler=noscaling
else
SOURCE_ACC_WC:=$(SOURCE_ACC_WC)_featurescaler=$(FEATSCAL)
SOURCE_COR_WC:=$(SOURCE_COR_WC)_featurescaler=$(FEATSCAL)
SOURCE_GSBP_WC:=$(SOURCE_GSBP_WC)_featurescaler=$(FEATSCAL)
SOURCE_GSR_WC:=$(SOURCE_GSR_WC)_featurescaler=$(FEATSCAL)
SOURCE_SS_WC:=$(SOURCE_SS_WC)_featurescaler=$(FEATSCAL)
endif

ifeq ($(LTSO),)
else
SOURCE_ACC_WC:=$(SOURCE_ACC_WC)_ltso=$(LTSO)
SOURCE_COR_WC:=$(SOURCE_COR_WC)_ltso=$(LTSO)
SOURCE_GSBP_WC:=$(SOURCE_GSBP_WC)_ltso=$(LTSO)
SOURCE_GSR_WC:=$(SOURCE_GSR_WC)_ltso=$(LTSO)
SOURCE_SS_WC:=$(SOURCE_SS_WC)_ltso=$(LTSO)
endif

ifeq ($(PERC),)
else
SOURCE_ACC_WC:=$(SOURCE_ACC_WC)_percfortrain=$(PERC)
SOURCE_COR_WC:=$(SOURCE_COR_WC)_percfortrain=$(PERC)
SOURCE_GSBP_WC:=$(SOURCE_GSBP_WC)_percfortrain=$(PERC)
SOURCE_GSR_WC:=$(SOURCE_GSR_WC)_percfortrain=$(PERC)
SOURCE_SS_WC:=$(SOURCE_SS_WC)_percfortrain=$(PERC)
endif

ifeq ($(SYSSET),)
else
SOURCE_ACC_WC:=$(SOURCE_ACC_WC)_sysset=$(SYSSET)
SOURCE_COR_WC:=$(SOURCE_COR_WC)_sysset=$(SYSSET)
SOURCE_GSBP_WC:=$(SOURCE_GSBP_WC)_sysset=$(SYSSET)
SOURCE_GSR_WC:=$(SOURCE_GSR_WC)_sysset=$(SYSSET)
SOURCE_SS_WC:=$(SOURCE_SS_WC)_sysset=$(SYSSET)
endif

# Get the source files
SOURCE_ACC=$(wildcard $(SOURCE_ACC_WC).csv)
SOURCE_COR=$(wildcard $(SOURCE_COR_WC).csv)
SOURCE_GSBP=$(wildcard $(SOURCE_GSBP_WC).csv)
SOURCE_GSR=$(wildcard $(SOURCE_GSR_WC).csv)
SOURCE_SS=$(wildcard $(SOURCE_SS_WC).csv)

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
