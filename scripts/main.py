import torch
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

from playground_bo.utils import (
    parser_bo,
    pick_acqf,
    pick_kernel,
    pick_test_function,
)

# warnings.filterwarnings("ignore") # removes negative variances rounding errors and failed fitting attempts
torch.set_default_dtype(torch.double)


def main(N_INIT, budget, kernel, acqf, experiment, seed, savefolder):
    torch.manual_seed(seed)

    problem = pick_test_function(experiment)
    bounds = torch.tensor(problem._bounds).T
    dim = bounds.shape[1]
    sigma = 0.01  # hardcoded noise level, but should be function-dependent.

    ##### INITIAL DATASET AND GP FITTING
    data = {}
    data["train_X"] = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(N_INIT, dim)
    train_Y = problem(data["train_X"]).view(-1, 1)
    data["train_Y"] = train_Y + sigma * torch.randn(size=train_Y.shape)

    gpr = SingleTaskGP(
        data["train_X"], data["train_Y"], covar_module=pick_kernel(kernel, dim)
    )
    mll = ExactMarginalLogLikelihood(gpr.likelihood, gpr)
    fit_gpytorch_model(mll, max_retries=10)

    ##### BO LOOP
    data["regrets"] = torch.zeros(budget + 1)
    data["regrets"][0] = problem.optimal_value - data["train_Y"].max()
    for b in range(budget):
        af = pick_acqf(acqf, data, gpr, bounds)
        candidates, _ = optimize_acqf(
            acq_function=af,
            bounds=bounds,
            q=1,  # batch size, i.e. we only query one point
            num_restarts=10,
            raw_samples=512,
        )
        y = problem(candidates)
        data["train_X"] = torch.cat((data["train_X"], candidates))
        data["train_Y"] = torch.cat((data["train_Y"], y.view(-1, 1)))
        data["regrets"][b + 1] = problem.optimal_value - data["train_Y"].max()
        gpr = SingleTaskGP(
            data["train_X"], data["train_Y"], covar_module=pick_kernel(kernel, dim)
        )
        mll = ExactMarginalLogLikelihood(gpr.likelihood, gpr)
        fit_gpytorch_model(mll, max_retries=10)
    return data


if __name__ == "__main__":
    parser = parser_bo()
    conf = vars(parser.parse_args())

    results = main(**conf)

    results["conf"] = conf
    path = f"{conf['savefolder']}/{conf['experiment']}_{conf['kernel']}_{conf['acqf']}_{conf['seed']}.pt"

    torch.save(results, path)
