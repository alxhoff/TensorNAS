PYFILES := $(shell find . -name '*.py')

format:
	black $(PYFILES) 

install:
	pip install -r requirements.txt
