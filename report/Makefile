# ======================================================================
# Targets:
#   make            build both the report and the slides
#   make report     build only the report   (main.pdf)
#   make slides     build only the slides   (presentation.pdf)
#   make clean      remove aux files and both PDFs
# ======================================================================

.PHONY: all report slides clean

all: report slides

# ---------------------------------------------------------------- report
report: main.pdf

main.pdf: main.tex main.bib
	pdflatex main
	bibtex main
	pdflatex main
	pdflatex main

# ---------------------------------------------------------------- slides
slides: presentation.pdf

presentation.pdf: presentation.tex
	pdflatex presentation
	pdflatex presentation

# ---------------------------------------------------------------- clean
clean:
	rm -f *.aux *.bbl *.blg *.log *.toc *.dvi *.out *.tdo *.snm *.nav \
	      *.gx.table *.gx.gnuplot *.fdb_latexmk *.fls main.pdf presentation.pdf *~
