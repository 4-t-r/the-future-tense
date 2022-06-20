.RECIPEPREFIX = >

.ONESHELL:

.PHONY: help
help:
> @echo "Targets:"
> @echo "... expose"
> @echo "... paper"
> @echo "... paper-full"
> @echo "... clean"

.PHONY: expose
expose: clean
> pdflatex expose

.PHONY: paper
paper: clean
> cd paper
> pdflatex paper
> mv -f paper.pdf ../paper.pdf

.PHONY: paper-full
paper-full: clean
> cd paper
> pdflatex paper
> bibtex paper
> pdflatex paper
> pdflatex paper
> pdflatex paper
> mv -f paper.pdf ../paper.pdf

.PHONY: clean
clean:
> rm -f \
      *.aux \
      *.log \
      *.out \
      *.pdf \
      *.blg \
      *.bbl \
      ./paper/*.aux \
      ./paper/*.log \
      ./paper/*.out \
      ./paper/*.pdf \
      ./paper/*.blg \
      ./paper/*.bbl \
