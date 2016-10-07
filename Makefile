# ------------------------------------------------
# Makefile
# ------------------------------------------------

FABFILE = fabfile.py
NETEMSRC = $(shell find ./netem -name '*.py')
PYSRC = $(FABFILE) $(NETEMSRC)
PYCFILE = $(shell find ./ -name '*.pyc')

all: codecheck

codecheck: $(PYSRC)
	-echo "Running code check"
	python2 -m pyflakes $(PYSRC)
	python2 -m pylint $(PYSRC)
	python2 -m pep8 $(PYSRC)

errcheck: $(PYSRC)
	-echo "Running check for errors only"
	python2 -m pyflakes $(PYSRC)
	python2 -m pylint $(PYSRC)

clean:
	echo $(PYCFILE)
	rm -rf $(PYCFILE)
