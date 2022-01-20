## setup

virtualenv -p python3 venv
pip --no-cache-dir install tensorflow
pip --no-cache-dir install deap
pip --no-cache-dir install matplotlib

## run
. venv/bin/activate
python demos/DemoEASimple.py

