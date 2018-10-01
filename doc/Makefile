all: lambda.pdf

%.pdf: %.tex 
	pdflatex $<
	@grep -e 'may have changed' $(basename $<).log && rm $@; $(MAKE) $@ || echo "done."

.PHONY: clean

clean:
	rm *.pdf *.aux *.log *.out
