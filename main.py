import argparse
import hashlib
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from bencherscaffold.client import BencherClient
from bencherscaffold.protoclasses.bencher_pb2 import Value, ValueType

from turbo import Turbo1


class Levy:
    def __init__(self, dim=10):
        self.dim = dim
        self.lb = -5 * np.ones(dim)
        self.ub = 10 * np.ones(dim)

    def __call__(self, x):
        assert len(x) == self.dim
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        w = 1 + (x - 1.0) / 4.0
        val = np.sin(np.pi * w[0]) ** 2 + \
              np.sum((w[1:self.dim - 1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[1:self.dim - 1] + 1) ** 2)) + \
              (w[self.dim - 1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[self.dim - 1])**2)
        return val


BENCHER_BENCHMARK_DIMS = {
    "lasso-dna": 180,
    "lasso-hard": 1000,
    "mopta08": 124,
    "svm": 388,
    "mujoco-ant": 888,
    "mujoco-humanoid": 6392,
    "rover": 60,
}


class BencherObjective:
    def __init__(self, benchmark_name: str, max_tries: int = 5):
        assert benchmark_name in BENCHER_BENCHMARK_DIMS

        self.benchmark_name = benchmark_name
        self.dim = BENCHER_BENCHMARK_DIMS[benchmark_name]
        self.lb = np.zeros(self.dim)
        self.ub = np.ones(self.dim)
        self.max_tries = max_tries
        self.client = BencherClient(hostname="localhost", max_retries=max_tries)
        self._Value = Value
        self._ValueType = ValueType

    def __call__(self, x: np.ndarray) -> float:
        assert x.ndim in [1, 2], "x must be 1D or 2D"
        _x = x
        if x.ndim == 2:
            assert x.shape[0] == 1, "x has to be essentially 1D"
            _x = x.squeeze(0)
        assert _x.shape[0] == self.dim
        assert np.all(_x >= 0.0) and np.all(_x <= 1.0)

        value_list = [
            self._Value(
                type=self._ValueType.CONTINUOUS,
                value=float(val),
            )
            for val in _x
        ]
        res = self.client.evaluate_point(
            benchmark_name=self.benchmark_name,
            point=value_list,
        )
        return float(res)


def parse_args():
    parser = argparse.ArgumentParser(description="Run TuRBO-1 on Levy or bencher benchmarks.")
    parser.add_argument("--dim", type=int, default=10, help="Dimension for the Levy benchmark.")
    parser.add_argument("--max-evals", type=int, default=1000, help="Maximum number of evaluations.")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size.")
    parser.add_argument("--n-init", type=int, default=20, help="Number of initial points.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for deterministic runs.")
    parser.add_argument(
        "--bencher-benchmark",
        type=str,
        choices=sorted(BENCHER_BENCHMARK_DIMS.keys()),
        help="Name of a bencher benchmark to run instead of Levy.",
        default=None
    )
    parser.add_argument(
        "--disable-trust-region",
        action="store_true",
        help="Disable trust region adaptation/clipping.",
    )
    return parser.parse_args()


def _set_seed(seed: Optional[int]):
    if seed is None:
        return
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _create_run_directory(args) -> Path:
    args_json = json.dumps(vars(args), sort_keys=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    digest = hashlib.sha256(args_json.encode("utf-8")).hexdigest()[:8]
    run_dir = Path("results") / f"{timestamp}_{digest}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _save_results(run_dir: Path, args, turbo1: Turbo1):
    payload = {
        "args": vars(args),
        "X": turbo1.X.tolist(),
        "y": turbo1.fX.ravel().tolist(),
    }
    with open(run_dir / "results.json", "w", encoding="utf-8") as fp:
        json.dump(payload, fp)


def main():
    args = parse_args()
    _set_seed(args.seed)

    if args.bencher_benchmark:
        bencher = BencherObjective(args.bencher_benchmark)
        f, lb, ub = bencher, bencher.lb, bencher.ub
        dim = bencher.dim
    else:
        levy = Levy(args.dim)
        f, lb, ub = levy, levy.lb, levy.ub
        dim = levy.dim

    turbo1 = Turbo1(
        f=f,  # Handle to objective function
        lb=lb,  # Numpy array specifying lower bounds
        ub=ub,  # Numpy array specifying upper bounds
        n_init=args.n_init,  # Number of initial bounds from an Latin hypercube design
        max_evals=args.max_evals,  # Maximum number of evaluations
        batch_size=args.batch_size,  # How large batch size TuRBO uses
        verbose=True,  # Print information from each batch
        use_ard=True,  # Set to true if you want to use ARD for the GP kernel
        max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
        n_training_steps=50,  # Number of steps of ADAM to learn the hypers
        min_cuda=1024,  # Run on the CPU for small datasets
        device="cpu",  # "cpu" or "cuda"
        dtype="float64",  # float64 or float32
        use_trust_region=not args.disable_trust_region,
    )

    print(f"Running TuRBO-1 on {args.bencher_benchmark or 'Levy'} with dim={dim}")
    turbo1.optimize()
    run_dir = _create_run_directory(args)
    _save_results(run_dir, args, turbo1)


if __name__ == "__main__":
    main()
