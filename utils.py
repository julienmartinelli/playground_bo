import argparse
import itertools

import matplotlib as mpl
import torch
from botorch.acquisition import qMaxValueEntropy
from botorch.acquisition.analytic import ExpectedImprovement, UpperConfidenceBound
from botorch.test_functions import Branin, Hartmann, Rosenbrock
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel

from test_functions import Zhou


def set_matplotlib_params():

    """Set matplotlib params."""

    mpl.rcParams.update(mpl.rcParamsDefault)
    mpl.rc("font", family="serif")
    mpl.rcParams.update(
        {
            "font.size": 24,
            "lines.linewidth": 2,
            "axes.labelsize": 24,  # fontsize for x and y labels
            "axes.titlesize": 24,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 20,
            "axes.linewidth": 2,
            "pgf.texsystem": "pdflatex",  # change this if using xetex or lautex
            "text.usetex": True,  # use LaTeX to write all text
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
        }
    )


def adapt_save_fig(fig, filename="test.pdf"):

    """Remove right and top spines, set bbox_inches and dpi."""

    for ax in fig.get_axes():
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
    fig.savefig(filename, bbox_inches="tight", dpi=300)


def parser_bo():

    """
    Parser used to run the algorithm from an already known crn.
    - Output:
        * parser: ArgumentParser object.
    """

    parser = argparse.ArgumentParser(description="Command description.")

    parser.add_argument(
        "-n", "--N_REP", help="int, number of reps for stds", type=int, default=1
    )
    parser.add_argument(
        "-ni", "--N_INIT", help="int, size of initial dataset", type=int, default=1
    )
    parser.add_argument(
        "-se", "--seed", default=None, help="int, random seed", type=int
    )
    parser.add_argument(
        "-s", "--savefolder", default=None, type=str, help="Name of saving directory."
    )
    parser.add_argument(
        "-b",
        "--budget",
        help="BO Budget",
        default=10,
        type=int,
    )
    parser.add_argument(
        "-k",
        "--kernels",
        nargs="*",
        type=str,
        default=["RBF"],
        help="list of kernels to try.",
    )
    parser.add_argument(
        "-a",
        "--acqfs",
        nargs="*",
        type=str,
        default=["MES"],
        help="list of BO acquisition function to try.",
    )
    parser.add_argument(
        "-e",
        "--experiments",
        nargs="*",
        type=str,
        default=["Forrester"],
        help="list of test functions to optimize.",
    )
    return parser


def build_combinations(N_REP, experiments, kernels, acqfs, n_init, seed):

    """Construct the list of combination settings to run."""

    combi = []
    li = [experiments, kernels, acqfs, [n_init], [seed + n for n in range(N_REP)]]
    combi.append(list(itertools.product(*li)))
    return sum(combi, [])


def pick_acqf(acqf, data, gpr, bounds):

    "Instantiate the given acqf."

    if acqf == "UCB":
        beta = 0.2  # basic value for normalized data
        af = UpperConfidenceBound(gpr, beta)
    elif acqf == "MES":
        Ncandids = 100  # size of candidate set to approximate MES
        candidate_set = torch.rand(
            Ncandids, bounds.size(1), device=bounds.device, dtype=bounds.dtype
        )
        candidate_set = bounds[0] + (bounds[1] - bounds[0]) * candidate_set
        af = qMaxValueEntropy(gpr, candidate_set)
    else:
        af = ExpectedImprovement(gpr, data["train_Y"].max())
    return af


def pick_kernel(ker, dim):

    "Instantiate the given kernel."

    # ScaleKernel adds the amplitude hyperparameter
    if ker == "RBF":
        K = ScaleKernel(RBFKernel(ard_num_dims=dim))
    elif ker == "Matern":
        K = ScaleKernel(MaternKernel(ard_num_dims=dim))
    return K


def pick_test_function(func):

    "Instantiate the given function to optimize."

    if func == "Zhou":
        testfunc = Zhou()
    elif func == "Hartmann":
        testfunc = Hartmann(negate=True)
    elif func == "Branin":
        testfunc = Branin(negate=True)
    elif func == "Rosenbrock":
        testfunc = Rosenbrock(dim=2, negate=True, bounds=[(-5.0, 5.0), (-5.0, 5.0)])
    return testfunc


def embed_test_function(testfunc, x):

    lb, ub = torch.tensor(testfunc._bounds).T
    return testfunc(lb + (ub - lb) * x)