TEX   := main_area/paper.tex
OUTD  := out
BIN   := pdflatex
FLAGS := -interaction=nonstopmode -halt-on-error -output-directory=$(OUTD)

.PHONY: all clean watch

all: $(OUTD)/paper.pdf

$(OUTD):
	mkdir -p $@

$(OUTD)/paper.pdf: $(TEX) | $(OUTD)
	$(BIN) $(FLAGS) $(TEX)
	@if grep -q '\\bibliography' $(TEX); then \
	    bibtex $(OUTD)/paper; \
	    $(BIN) $(FLAGS) $(TEX); \
	    $(BIN) $(FLAGS) $(TEX); \
	fi

clean:
	rm -rf $(OUTD)

watch:
	while true; do \
	    inotifywait -e modify $(TEX); \
	    $(MAKE) all; \
	done
