.RECIPEPREFIX = >

.ONESHELL:

.PHONY: help
help:
> @echo "Targets:"
> @echo "... clean"
> @echo "... expose"
> @echo "... expose-full"
> @echo "... paper"
> @echo "... paper-full"

.PHONY: expose
expose: clean
> cd expose
> pdflatex expose
> mv -f expose.pdf ../expose.pdf

.PHONY: expose-full
expose-full: clean
> cd expose
> pdflatex expose
> bibtex expose
> pdflatex expose
> pdflatex expose
> pdflatex expose
> mv -f expose.pdf ../expose.pdf

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
      *.bbl \
      *.blg \
      *.log \
      *.out \
      ./expose/*.aux \
      ./expose/*.bbl \
      ./expose/*.blg \
      ./expose/*.log \
      ./expose/*.out \
      ./expose/*.pdf \
      ./paper/*.aux \
      ./paper/*.bbl \
      ./paper/*.blg \
      ./paper/*.log \
      ./paper/*.out \
      ./paper/*.pdf \
      paper.pdf \
