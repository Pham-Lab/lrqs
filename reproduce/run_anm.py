"""Run the causal-learn ANM baseline on the 42 custom settings."""

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from anm import predict_causal_direction_anm
from reproduce.bivariate import datasets, run_bivariate, set_seed
from reproduce.metric import audrc, prec


SEED = 0
OUTPUT_DIR = SCRIPT_DIR / "results" / "rerun"
warnings.filterwarnings("ignore", category=ConvergenceWarning)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    metrics = []

    for dataset_id, dataset_class in datasets.items():
        set_seed(SEED)
        for pair_id in range(1, dataset_class.n_datasets + 1):
            pair_result = run_bivariate(
                dataset_id, pair_id, predict_causal_direction_anm
            )
            if pair_result is None:
                continue
            direction, elapsed = pair_result
            results.append(
                {
                    "dataset_id": dataset_id,
                    "pair_id": pair_id,
                    "direction": direction,
                    "delta_time": elapsed,
                }
            )

        subset = [row for row in results if row["dataset_id"] == dataset_id]
        directions = np.asarray([row["direction"] for row in subset])
        runtimes = np.asarray([row["delta_time"] for row in subset])
        summary = {
            "dataset_id": dataset_id,
            "prec": prec(directions),
            "audrc": audrc(directions),
            "dt_mean": float(np.mean(runtimes)),
            "dt_std": float(np.std(runtimes)),
        }
        print(summary)
        metrics.append(summary)

        pd.DataFrame(results).to_csv(
            OUTPUT_DIR / "anm_custom_results.csv", index=False
        )
        pd.DataFrame(metrics).to_csv(
            OUTPUT_DIR / "anm_custom_metrics.csv", index=False
        )
