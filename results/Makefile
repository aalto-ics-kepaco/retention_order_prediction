# Parameter specifying the configuration of the data for which the
# results should be analyzed.
DATACONFIG=PredRet/v1

processed/plot_test_centering_and_normalizing.html: scripts/R/plot_test_centering_and_normalizing.Rmd $(shell find $(find raw/$(DATACONFIG)/final/ -name test_centering_and_normalizing) -name *.csv)
	R -e "rmarkdown::render('scripts/R/plot_test_centering_and_normalizing.Rmd',params=list(dataconfig='$(DATACONFIG)'),output_file='plot_test_centering_and_normalizing.html')"
	mv scripts/R/plot_test_centering_and_normalizing.html processed/

.PHONY: clean
clean:
	rm -rf raw/debug/*
	rm -rf raw/PredRet/v1/debug/*
	rm -rf raw/PredRet/v2/debug/*

