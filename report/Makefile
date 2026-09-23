# Build the internal report.  `make` -> PDF ; `make clean` -> remove aux files
TEX = multireference_eagleeye_report

all: $(TEX).pdf

$(TEX).pdf: $(TEX).tex
	pdflatex -interaction=nonstopmode $(TEX).tex
	pdflatex -interaction=nonstopmode $(TEX).tex

clean:
	rm -f $(TEX).aux $(TEX).log $(TEX).out $(TEX).toc *.log

distclean: clean
	rm -f $(TEX).pdf

.PHONY: all clean distclean
