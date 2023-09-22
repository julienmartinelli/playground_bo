from typing import List, Optional, Tuple

import torch
from botorch.test_functions.synthetic import SyntheticTestFunction
from torch import Tensor

### Test functions that are not implemented in BoTorch


class Zhou(SyntheticTestFunction):
    optimal_value = 2.002595246981888

    def __init__(
        self,
        noise_std: Optional[float] = None,
        negate: bool = False,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        r"""
        Args:
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
            bounds: Custom bounds for the function specified as (lower, upper) pairs.
        """
        self.dim = 1
        self._bounds = [(-0.0, 1.0) for _ in range(self.dim)]
        self._optimizers = [tuple(1 / 3 for _ in range(self.dim))]
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds)

    def evaluate_true(self, X: Tensor) -> Tensor:
        def phi_zou(X: Tensor) -> Tensor:
            return (2 * torch.pi) ** (-0.5) * torch.exp(-0.5 * X**2)

        part1 = 10 * (X - 1 / 3)
        part2 = 10 * (X - 2 / 3)
        return 5 * (phi_zou(part1) + phi_zou(part2))
