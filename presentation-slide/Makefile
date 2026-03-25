TEX = pdflatex -interaction=nonstopmode
SRC = mlmd-presentation.tex
PDF = $(SRC:.tex=.pdf)

.PHONY: all clean

all: $(PDF)

$(PDF): $(SRC) $(wildcard images/*)
	$(TEX) $(SRC)
	$(TEX) $(SRC)

clean:
	rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb
