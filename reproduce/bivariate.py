"""Shared bivariate evaluation pipeline.

The default registry contains the 42 PNL and MultiRank settings used by the
public reproduction commands.  The commented block below can be enabled when
the bundled standard benchmarks also need to be rerun.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable

import numpy as np
from sklearn.preprocessing import StandardScaler

from data import bivariate as benchmark


datasets = {
    # PNL-AN (15 settings)
    "pnl-an-gaussian-identity": benchmark.PNLANGaussianIdentity,
    "pnl-an-gaussian-cube": benchmark.PNLANGaussianCube,
    "pnl-an-gaussian-sigmoid": benchmark.PNLANGaussianSigmoid,
    "pnl-an-gaussian-exp": benchmark.PNLANGaussianExp,
    "pnl-an-gaussian-tanh": benchmark.PNLANGaussianTanh,
    "pnl-an-uniform-identity": benchmark.PNLANUniformIdentity,
    "pnl-an-uniform-cube": benchmark.PNLANUniformCube,
    "pnl-an-uniform-sigmoid": benchmark.PNLANUniformSigmoid,
    "pnl-an-uniform-exp": benchmark.PNLANUniformExp,
    "pnl-an-uniform-tanh": benchmark.PNLANUniformTanh,
    "pnl-an-beta-identity": benchmark.PNLANBetaIdentity,
    "pnl-an-beta-cube": benchmark.PNLANBetaCube,
    "pnl-an-beta-sigmoid": benchmark.PNLANBetaSigmoid,
    "pnl-an-beta-exp": benchmark.PNLANBetaExp,
    "pnl-an-beta-tanh": benchmark.PNLANBetaTanh,
    # PNL-LSNM (15 settings)
    "pnl-lsnm-gaussian-identity": benchmark.PNLLSNMGaussianIdentity,
    "pnl-lsnm-gaussian-cube": benchmark.PNLLSNMGaussianCube,
    "pnl-lsnm-gaussian-sigmoid": benchmark.PNLLSNMGaussianSigmoid,
    "pnl-lsnm-gaussian-exp": benchmark.PNLLSNMGaussianExp,
    "pnl-lsnm-gaussian-tanh": benchmark.PNLLSNMGaussianTanh,
    "pnl-lsnm-uniform-identity": benchmark.PNLLSNMUniformIdentity,
    "pnl-lsnm-uniform-cube": benchmark.PNLLSNMUniformCube,
    "pnl-lsnm-uniform-sigmoid": benchmark.PNLLSNMUniformSigmoid,
    "pnl-lsnm-uniform-exp": benchmark.PNLLSNMUniformExp,
    "pnl-lsnm-uniform-tanh": benchmark.PNLLSNMUniformTanh,
    "pnl-lsnm-beta-identity": benchmark.PNLLSNMBetaIdentity,
    "pnl-lsnm-beta-cube": benchmark.PNLLSNMBetaCube,
    "pnl-lsnm-beta-sigmoid": benchmark.PNLLSNMBetaSigmoid,
    "pnl-lsnm-beta-exp": benchmark.PNLLSNMBetaExp,
    "pnl-lsnm-beta-tanh": benchmark.PNLLSNMBetaTanh,
    # MultiRank (12 settings)
    "multirank-k2-gaussian": benchmark.MultiRankK2Gaussian,
    "multirank-k2-uniform": benchmark.MultiRankK2Uniform,
    "multirank-k2-beta": benchmark.MultiRankK2Beta,
    "multirank-k3-gaussian": benchmark.MultiRankK3Gaussian,
    "multirank-k3-uniform": benchmark.MultiRankK3Uniform,
    "multirank-k3-beta": benchmark.MultiRankK3Beta,
    "multirank-k4-gaussian": benchmark.MultiRankK4Gaussian,
    "multirank-k4-uniform": benchmark.MultiRankK4Uniform,
    "multirank-k4-beta": benchmark.MultiRankK4Beta,
    "multirank-k5-gaussian": benchmark.MultiRankK5Gaussian,
    "multirank-k5-uniform": benchmark.MultiRankK5Uniform,
    "multirank-k5-beta": benchmark.MultiRankK5Beta,

    # Optional standard benchmarks (24 settings).
    # Uncomment this block only when rerunning the standard benchmarks.  Use
    # separate output filenames so one rerun is not replaced by another.
    # "an": benchmark.AN,
    # "ans": benchmark.ANs,
    # "ls": benchmark.LS,
    # "lss": benchmark.LSs,
    # "mnu": benchmark.MNU,
    # "sim": benchmark.SIM,
    # "simc": benchmark.SIMc,
    # "simg": benchmark.SIMG,
    # "simln": benchmark.SIMln,
    # "tue": benchmark.Tuebingen,
    # "cha": benchmark.Cha,
    # "net": benchmark.Net,
    # "multi": benchmark.Multi,
    # "d4s1": benchmark.D4S1,
    # "d4s2a": benchmark.D4S2A,
    # "d4s2b": benchmark.D4S2B,
    # "d4s2c": benchmark.D4S2C,
    # "per": benchmark.PER,
    # "sig": benchmark.SIG,
    # "vex": benchmark.VEX,
    # "qdv": benchmark.QDV,
    # "sigv": benchmark.SIGV,
    # "rbfv": benchmark.RBFV,
    # "nnv": benchmark.NNV,
}


EXCLUDED_PAIR_IDS = {
    "tue": {47, 52, 53, 54, 55, 70, 71, 105, 107},
}


def set_seed(seed: int) -> None:
    """Reset the random generators used by the public runners."""

    random.seed(seed)
    np.random.seed(seed)


def run_bivariate(
    dataset_id: str,
    pair_id: int,
    method: Callable[[np.ndarray, np.ndarray], int],
    shuffle: bool = True,
) -> tuple[int, float] | None:
    """Run one pair, optionally hiding its stored column order from a method."""

    if pair_id in EXCLUDED_PAIR_IDS.get(dataset_id, set()):
        return None

    dataset = datasets[dataset_id](
        pair_id=pair_id,
        double=True,
        preprocessor=StandardScaler(),
    )
    cause = dataset.cause[:, :1]
    effect = dataset.effect[:, :1]

    start_time = time.time()
    if shuffle and random.random() <= 0.5:
        direction = -method(effect, cause)
    else:
        direction = method(cause, effect)
    elapsed = time.time() - start_time
    return int(direction), elapsed


# Backward-compatible alias for older local scripts that used the misspelling.
run_bivairate = run_bivariate


__all__ = [
    "datasets",
    "EXCLUDED_PAIR_IDS",
    "set_seed",
    "run_bivariate",
    "run_bivairate",
]
