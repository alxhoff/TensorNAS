PYFILES := $(shell find . -name '*.py')

clean:
	bash -c "find . | grep -E '(__pycache__|\.pyc|\.pyo$$)' | xargs rm -rf";

format:
	black $(PYFILES) 

install:
	pip install -r requirements.txt

