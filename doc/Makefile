all: lambda.pdf

%.pdf: %.tex 
	pdflatex $<
	@grep -e 'may have changed' $(basename $<).log && rm $@; $(MAKE) $@ || echo "done."
	test -f copy.sh && sh copy.sh

.PHONY: clean

clean:
	rm *.pdf *.aux *.log *.out
