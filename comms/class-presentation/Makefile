TEX    = pdflatex -interaction=nonstopmode
SRC    = mlmd-presentation.tex
PDF    = $(SRC:.tex=.pdf)

MD_SRCS  := $(shell find . -name '*.md')
MD_PDFS  := $(MD_SRCS:.md=.pdf)

.PHONY: all clean

all: $(PDF) $(MD_PDFS)

$(PDF): $(SRC) $(wildcard images/*)
	$(TEX) $(SRC)
	$(TEX) $(SRC)

%.pdf: %.md
	pandoc $< --pdf-engine=xelatex -o $@

clean:
	rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb
	rm -f $(MD_PDFS)
