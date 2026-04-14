.PHONY: pdf clean view

BUILD_DIR := _output
OUTPUT_PDF := final-report.pdf
SOURCE := final-report.tex

pdf: $(BUILD_DIR)
	BSTINPUTS=./styles:$(BSTINPUTS) latexmk -pdf -output-directory=$(BUILD_DIR) -jobname=final-report $(SOURCE)
	cp $(BUILD_DIR)/final-report.pdf ./$(OUTPUT_PDF)

clean:
	latexmk -C -output-directory=$(BUILD_DIR)
	rm -f $(BUILD_DIR)/*
	rm -f $(OUTPUT_PDF)

view:
	open ./$(OUTPUT_PDF)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)
