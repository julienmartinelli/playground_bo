## Basics of Bayesian Optimization using BoTorch

This repository contains:

* `utils.py`, `test_functions.py` and `main.py`, the necessary functions to run vanilla BO
* `script_main.sh`, a shell script for parallelization across multiple repetitions and settings
* `results.ipynb`, a notebook to plot the results
* `demo.ipynb`, a introductory notebook for running BO using GPytorch and BoTorch

To perform an experiment, run in a terminal the following command (at the root)

```
chmod 777 script_main.sh && ./script_main.sh
```

Kernels, test functions, acquisition function, budget etc can be tweaked in the shell file. See the parser defined in `utils.py` for more information.