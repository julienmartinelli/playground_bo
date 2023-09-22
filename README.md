## Basics of Bayesian Optimization using BoTorch

This repository contains:

* `utils.py` and `test_functions.py` in `playground_bo/`, the necessary functions to run vanilla BO
* `scripts/main.py`, a script for running experiments from the command line
* `notebooks/results.ipynb`, a notebook to plot the results
* `notebooks/demo.ipynb`, a introductory notebook for running BO using GPytorch and BoTorch

Install as a regular python package: `python -m pip install -e .` (`-e` for editable mode to continue develop as you run)

To perform an experiment, run in a terminal the following command (at the root)

```
mkdir results
python ./scripts/main.py -h  # for help with arguments
python ./scripts/main.py -ni 6 -b 5 -se 666 -e Zhou -a UCB -k RBF -s results
```
