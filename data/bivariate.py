"""Dataset readers used by the LRQS reproduction scripts.

The public classes expose two NumPy arrays, ``cause`` and ``effect``, whose
columns are ordered according to the ground-truth direction.  Each reader is
implemented for the on-disk format of one benchmark family.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent / "bivariate"


class CausalDataset:
    """Base class for a collection of ground-truth cause/effect pairs."""

    n_datasets: int
    relative_folder: str

    def __init__(
        self,
        pair_id: int | None = None,
        path: str | Path = DATA_DIR,
        double: bool = False,
        preprocessor: Any | None = None,
    ) -> None:
        if pair_id is None or not 1 <= pair_id <= self.n_datasets:
            raise AssertionError(
                f"pair_id must be between 1 and {self.n_datasets}, got {pair_id}"
            )

        cause, effect = self._read_pair(Path(path), pair_id)
        if preprocessor is not None:
            cause = preprocessor.fit_transform(cause)
            effect = preprocessor.fit_transform(effect)

        dtype = np.float64 if double else np.float32
        self.cause = np.asarray(cause, dtype=dtype)
        self.effect = np.asarray(effect, dtype=dtype)

    @property
    def folder_name(self) -> str:
        return self.relative_folder

    def _read_pair(self, data_root: Path, pair_id: int) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class ANLSMN(CausalDataset):
    """Reader for comma-separated pair files with a separate 0/1 label file."""

    n_datasets = 100
    relative_folder = "ANLSMN_pairs"

    def _read_pair(self, data_root: Path, pair_id: int) -> tuple[np.ndarray, np.ndarray]:
        directory = data_root / self.relative_folder
        pair = pd.read_csv(directory / f"pair_{pair_id}.txt")
        label = pd.read_csv(directory / "pairs_gt.txt", header=None).iloc[
            pair_id - 1, 0
        ]
        first = pair.iloc[:, 1:2].to_numpy()
        second = pair.iloc[:, 2:3].to_numpy()
        if label == 1:
            return first, second
        if label == 0:
            return second, first
        raise ValueError(f"Invalid ground-truth label {label!r} for pair {pair_id}")


class AN(ANLSMN):
    relative_folder = "ANLSMN_pairs/AN"


class ANs(ANLSMN):
    relative_folder = "ANLSMN_pairs/AN-s"


class LS(ANLSMN):
    relative_folder = "ANLSMN_pairs/LS"


class LSs(ANLSMN):
    relative_folder = "ANLSMN_pairs/LS-s"


class MNU(ANLSMN):
    relative_folder = "ANLSMN_pairs/MN-U"


_META_COLUMNS = (
    "id",
    "cause_start",
    "cause_end",
    "effect_start",
    "effect_end",
    "weight",
)


def _read_pairmeta(directory: Path) -> pd.DataFrame:
    return pd.read_csv(
        directory / "pairmeta.txt",
        sep=r"\s+",
        header=None,
        names=_META_COLUMNS,
        index_col=0,
    ).astype(int)


def _slice_ground_truth_columns(
    pair: pd.DataFrame, metadata_row: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    cause = pair.iloc[
        :,
        int(metadata_row["cause_start"]) - 1 : int(metadata_row["cause_end"]),
    ].to_numpy()
    effect = pair.iloc[
        :,
        int(metadata_row["effect_start"]) - 1 : int(metadata_row["effect_end"]),
    ].to_numpy()
    return cause, effect


class Tuebingen(CausalDataset):
    n_datasets = 108
    relative_folder = "Tuebingen"

    def _read_pair(self, data_root: Path, pair_id: int) -> tuple[np.ndarray, np.ndarray]:
        directory = data_root / self.relative_folder
        pair = pd.read_csv(
            directory / f"pair{pair_id:04d}.txt", sep=r"\s+", header=None
        )
        metadata = _read_pairmeta(directory)
        return _slice_ground_truth_columns(pair, metadata.loc[pair_id])


class BenchmarkSimulated(Tuebingen):
    n_datasets = 100
    relative_folder = "Benchmark_simulated"


class SIM(BenchmarkSimulated):
    relative_folder = "Benchmark_simulated/SIM"


class SIMc(BenchmarkSimulated):
    relative_folder = "Benchmark_simulated/SIM-c"


class SIMG(BenchmarkSimulated):
    relative_folder = "Benchmark_simulated/SIM-G"


class SIMln(BenchmarkSimulated):
    relative_folder = "Benchmark_simulated/SIM-ln"


def _space_separated_vector(value: str) -> np.ndarray:
    return np.asarray([float(item) for item in value.strip().split(" ")]).reshape(
        -1, 1
    )


def _order_string_pair(
    first: str, second: str, target: int
) -> tuple[np.ndarray, np.ndarray]:
    a = _space_separated_vector(first)
    b = _space_separated_vector(second)
    if target == 1:
        return a, b
    if target == -1:
        return b, a
    raise ValueError(f"Invalid target: {target!r}")


class Dataverse(CausalDataset):
    n_datasets = 300
    relative_folder = "Dataverse_pairs"
    file_stem: str

    def _read_pair(self, data_root: Path, pair_id: int) -> tuple[np.ndarray, np.ndarray]:
        prefix = data_root / self.relative_folder / self.file_stem
        pairs = pd.read_csv(f"{prefix}_pairs.csv")
        targets = pd.read_csv(f"{prefix}_targets.csv")
        row = pair_id - 1
        return _order_string_pair(
            pairs.loc[row, "A"],
            pairs.loc[row, "B"],
            int(targets.loc[row, "Target"]),
        )

    @property
    def file_name(self) -> str:
        return self.file_stem


class Cha(Dataverse):
    file_stem = "CE-Cha"


class Multi(Dataverse):
    file_stem = "CE-Multi"


class Net(Dataverse):
    file_stem = "CE-Net"


class DREAMChallenge(CausalDataset):
    relative_folder = "DREAM_challenges"
    file_stem: str

    def _read_pair(self, data_root: Path, pair_id: int) -> tuple[np.ndarray, np.ndarray]:
        path = data_root / self.relative_folder / f"{self.file_stem}.csv"
        pairs = pd.read_csv(path)
        pairs = pairs.loc[pairs["Target"].isin((-1, 1))].reset_index(drop=True)
        row = pairs.iloc[pair_id - 1]
        return _order_string_pair(row["A"], row["B"], int(row["Target"]))

    @property
    def file_name(self) -> str:
        return self.file_stem


class D4S1(DREAMChallenge):
    n_datasets = 24
    file_stem = "D4S1"


class D4S2A(DREAMChallenge):
    n_datasets = 352
    file_stem = "D4S2A"


class D4S2B(DREAMChallenge):
    n_datasets = 498
    file_stem = "D4S2B"


class D4S2C(DREAMChallenge):
    n_datasets = 386
    file_stem = "D4S2C"


class FlowSimulated(CausalDataset):
    n_datasets = 100
    relative_folder = "Flow_simulated"

    def _read_pair(self, data_root: Path, pair_id: int) -> tuple[np.ndarray, np.ndarray]:
        directory = data_root / self.relative_folder
        pair = pd.read_csv(
            directory / f"pair{pair_id:04d}.txt", sep=r"\s+", header=None
        )
        metadata = _read_pairmeta(directory).reset_index(drop=True)
        metadata.index = metadata.index + 1
        return _slice_ground_truth_columns(pair, metadata.loc[pair_id])


class PER(FlowSimulated):
    relative_folder = "Flow_simulated/periodic"


class SIG(FlowSimulated):
    relative_folder = "Flow_simulated/sigmoid"


class VEX(FlowSimulated):
    relative_folder = "Flow_simulated/velocity_exp"


class QPEConstrained(CausalDataset):
    n_datasets = 100
    relative_folder = "QPE_constrained"

    def _read_pair(self, data_root: Path, pair_id: int) -> tuple[np.ndarray, np.ndarray]:
        path = data_root / self.relative_folder / f"pair{pair_id:04d}.txt"
        pair = pd.read_csv(path, sep=r"\s+", header=None)
        return pair.iloc[:, 0:1].to_numpy(), pair.iloc[:, 1:2].to_numpy()


class QDV(QPEConstrained):
    relative_folder = "QPE_constrained/qdv"


class SIGV(QPEConstrained):
    relative_folder = "QPE_constrained/sigv"


class RBFV(QPEConstrained):
    relative_folder = "QPE_constrained/rbfv"


class NNV(QPEConstrained):
    relative_folder = "QPE_constrained/nnv"


def _register_anlsmn_subclass(name: str, folder: str) -> None:
    globals()[name] = type(
        name,
        (ANLSMN,),
        {"relative_folder": folder, "__module__": __name__},
    )


_CUSTOM_DATASET_FOLDERS = {
    "PNLANGaussianIdentity": "PNL/PNL-AN-Gaussian-identity",
    "PNLANGaussianCube": "PNL/PNL-AN-Gaussian-cube",
    "PNLANGaussianSigmoid": "PNL/PNL-AN-Gaussian-sigmoid",
    "PNLANGaussianExp": "PNL/PNL-AN-Gaussian-exp",
    "PNLANGaussianTanh": "PNL/PNL-AN-Gaussian-tanh",
    "PNLANUniformIdentity": "PNL/PNL-AN-Uniform-identity",
    "PNLANUniformCube": "PNL/PNL-AN-Uniform-cube",
    "PNLANUniformSigmoid": "PNL/PNL-AN-Uniform-sigmoid",
    "PNLANUniformExp": "PNL/PNL-AN-Uniform-exp",
    "PNLANUniformTanh": "PNL/PNL-AN-Uniform-tanh",
    "PNLANBetaIdentity": "PNL/PNL-AN-Beta-identity",
    "PNLANBetaCube": "PNL/PNL-AN-Beta-cube",
    "PNLANBetaSigmoid": "PNL/PNL-AN-Beta-sigmoid",
    "PNLANBetaExp": "PNL/PNL-AN-Beta-exp",
    "PNLANBetaTanh": "PNL/PNL-AN-Beta-tanh",
    "PNLLSNMGaussianIdentity": "PNL/PNL-LSNM-Gaussian-identity",
    "PNLLSNMGaussianCube": "PNL/PNL-LSNM-Gaussian-cube",
    "PNLLSNMGaussianSigmoid": "PNL/PNL-LSNM-Gaussian-sigmoid",
    "PNLLSNMGaussianExp": "PNL/PNL-LSNM-Gaussian-exp",
    "PNLLSNMGaussianTanh": "PNL/PNL-LSNM-Gaussian-tanh",
    "PNLLSNMUniformIdentity": "PNL/PNL-LSNM-Uniform-identity",
    "PNLLSNMUniformCube": "PNL/PNL-LSNM-Uniform-cube",
    "PNLLSNMUniformSigmoid": "PNL/PNL-LSNM-Uniform-sigmoid",
    "PNLLSNMUniformExp": "PNL/PNL-LSNM-Uniform-exp",
    "PNLLSNMUniformTanh": "PNL/PNL-LSNM-Uniform-tanh",
    "PNLLSNMBetaIdentity": "PNL/PNL-LSNM-Beta-identity",
    "PNLLSNMBetaCube": "PNL/PNL-LSNM-Beta-cube",
    "PNLLSNMBetaSigmoid": "PNL/PNL-LSNM-Beta-sigmoid",
    "PNLLSNMBetaExp": "PNL/PNL-LSNM-Beta-exp",
    "PNLLSNMBetaTanh": "PNL/PNL-LSNM-Beta-tanh",
    "MultiRankK2Gaussian": "MultiRank/MultiRank-K2-Gaussian",
    "MultiRankK2Uniform": "MultiRank/MultiRank-K2-Uniform",
    "MultiRankK2Beta": "MultiRank/MultiRank-K2-Beta",
    "MultiRankK3Gaussian": "MultiRank/MultiRank-K3-Gaussian",
    "MultiRankK3Uniform": "MultiRank/MultiRank-K3-Uniform",
    "MultiRankK3Beta": "MultiRank/MultiRank-K3-Beta",
    "MultiRankK4Gaussian": "MultiRank/MultiRank-K4-Gaussian",
    "MultiRankK4Uniform": "MultiRank/MultiRank-K4-Uniform",
    "MultiRankK4Beta": "MultiRank/MultiRank-K4-Beta",
    "MultiRankK5Gaussian": "MultiRank/MultiRank-K5-Gaussian",
    "MultiRankK5Uniform": "MultiRank/MultiRank-K5-Uniform",
    "MultiRankK5Beta": "MultiRank/MultiRank-K5-Beta",
}

for _class_name, _folder in _CUSTOM_DATASET_FOLDERS.items():
    _register_anlsmn_subclass(_class_name, _folder)


__all__ = [
    "DATA_DIR",
    "CausalDataset",
    "ANLSMN",
    "AN",
    "ANs",
    "LS",
    "LSs",
    "MNU",
    "Tuebingen",
    "BenchmarkSimulated",
    "SIM",
    "SIMc",
    "SIMG",
    "SIMln",
    "Dataverse",
    "Cha",
    "Multi",
    "Net",
    "DREAMChallenge",
    "D4S1",
    "D4S2A",
    "D4S2B",
    "D4S2C",
    "FlowSimulated",
    "PER",
    "SIG",
    "VEX",
    "QPEConstrained",
    "QDV",
    "SIGV",
    "RBFV",
    "NNV",
    *_CUSTOM_DATASET_FOLDERS.keys(),
]
